# Copyright (c) 2024, Tri Dao, Albert Gu.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update, selective_state_update_2step, selective_state_update_Nstep
except ImportError:
    selective_state_update = None

from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated
from mamba_ssm.ops.triton.causal_conv1d_varlen import causal_conv1d_varlen_update, causal_conv1d_varlen_states_update_v2

from mamba_ssm.distributed.tensor_parallel import ColumnParallelLinear, RowParallelLinear
from mamba_ssm.distributed.distributed_utils import all_reduce, reduce_scatter

from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
from mamba_ssm.ops.triton.ssd_combined import mamba_split_conv1d_scan_combined
from mamba_ssm.ops.triton.state_indexing import inplace_state_indexing
from mamba_ssm.ops.triton.tree_scan import tree_scan_mask_combined_fwd, tree_mask_conv1d, mask_index_gather
from mamba_ssm.utils.profile import cuda_time


class Mamba2(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=128,
        d_conv=4,
        conv_init=None,
        expand=2,
        headdim=64,
        d_ssm=None,  # If not None, we only apply SSM on this many dimensions, the rest uses gated MLP
        ngroups=1,
        A_init_range=(1, 16),
        D_has_hdim=False,
        rmsnorm=True,
        norm_before_gate=False,
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        dt_limit=(0.0, float("inf")),
        bias=False,
        conv_bias=True,
        # Fused kernel and sharding options
        chunk_size=256,
        use_mem_eff_path=True,
        layer_idx=None,  # Absorb kwarg for general module
        process_group=None,
        sequence_parallel=True,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.conv_init = conv_init
        self.expand = expand
        self.process_group = process_group
        self.sequence_parallel = sequence_parallel
        self.world_size = 1 if process_group is None else process_group.size()
        self.local_rank = 0 if process_group is None else process_group.rank()
        self.d_inner = (self.expand * self.d_model) // self.world_size
        assert self.d_inner * self.world_size == self.expand * self.d_model
        self.headdim = headdim
        self.d_ssm = self.d_inner if d_ssm is None else d_ssm // self.world_size
        assert ngroups % self.world_size == 0
        self.ngroups = ngroups // self.world_size
        assert self.d_ssm % self.headdim == 0
        self.nheads = self.d_ssm // self.headdim
        self.D_has_hdim = D_has_hdim
        self.rmsnorm = rmsnorm
        self.norm_before_gate = norm_before_gate
        self.dt_limit = dt_limit
        self.activation = "silu"
        self.chunk_size = chunk_size
        self.use_mem_eff_path = use_mem_eff_path
        self.layer_idx = layer_idx

        # Order: [z, x, B, C, dt]
        d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        if self.process_group is None:
            self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=bias, **factory_kwargs)
        else:
            self.in_proj = ColumnParallelLinear(self.d_model, d_in_proj * self.world_size, bias=bias,
                                                process_group=self.process_group, sequence_parallel=self.sequence_parallel,
                                                **factory_kwargs)

        conv_dim = self.d_ssm + 2 * self.ngroups * self.d_state
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=conv_dim,
            padding=d_conv - 1,
            **factory_kwargs,
        )
        if self.conv_init is not None:
            nn.init.uniform_(self.conv1d.weight, -self.conv_init, self.conv_init)

        self.act = nn.SiLU()

        # Initialize log dt bias
        dt = torch.exp(
            torch.rand(self.nheads, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        # Just to be explicit. Without this we already don"t put wd on dt_bias because of the check
        # name.endswith("bias") in param_grouping.py
        self.dt_bias._no_weight_decay = True

        assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
        A = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(*A_init_range)
        A_log = torch.log(A).to(dtype=dtype)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_ssm if self.D_has_hdim else self.nheads, device=device))
        self.D._no_weight_decay = True

        if self.rmsnorm:
            assert RMSNormGated is not None
            self.norm = RMSNormGated(self.d_ssm, eps=1e-5, norm_before_gate=self.norm_before_gate,
                                     group_size=self.d_ssm // ngroups, **factory_kwargs)

        if self.process_group is None:
            self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        else:
            self.out_proj = RowParallelLinear(self.d_inner * self.world_size, self.d_model, bias=bias,
                                              process_group=self.process_group, sequence_parallel=self.sequence_parallel,
                                              **factory_kwargs)

    def forward(self, u, seqlen=None, seq_idx=None, inference_params=None, mask=None):
        """
        u: (batch, seqlen, hidden_dim) if seqlen=None.
            If seqlen is not None, u is (batch * seqlen, hidden_dim). This is so that when we
            split u during sequence parallel, we split the batch * seqlen dimension
            (in case batch is small).
        mask: (batch, seqlen) 
            If mask is provided, the embedding at the masked position will get zeroed out
        Returns: same shape as u
        """
        seqlen_og = seqlen
        if seqlen is None:
            batch, seqlen, dim = u.shape
        else:
            batch_seqlen, dim = u.shape
            batch = batch_seqlen // seqlen
    
        # adding masking: if a position is masked, we zero out the hidden_dim for that position,
        # This will zero out B, C, making the output useless, but it is ok since we are not going to use the output at a padded token
        # This will preserve the ssm_state as dA is 1 (identify) and B is zero.
        if mask is not None:
            # for tree scan kernel we pass in a causal attention mask instead of token level mask so it is handled differetly.
            if not inference_params.use_tree_scan_kernel:
                assert len(u.shape) == 3 
                u = u * mask[:, :, None]

        # At inference time, if u has seqlen greater than 1 we always want to use the 
        conv_state, ssm_state = None, None
        if inference_params is not None:
            conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
            use_selective_scan = (not inference_params.use_Nstep_kernel) 
            # This branch executes the original Mamba-2's single tokens decoding step with selective scan
            if inference_params.seqlen_offset > 0 and u.shape[1] == 1 and use_selective_scan:
                # The states are updated inplace
                out, _, _ = self.step(u, conv_state, ssm_state, inference_params=inference_params)
                return out

        zxbcdt = self.in_proj(u)  # (B, L, d_in_proj) or (B * L, d_in_proj)
        if seqlen_og is not None:
            zxbcdt = rearrange(zxbcdt, "(b l) d -> b l d", l=seqlen)
        # If the model is loaded in fp16, without the .float() here, A might be -inf
        A = -torch.exp(self.A_log.float())  # (nheads) or (d_inner, d_state)
        dt_limit_kwargs = {} if self.dt_limit == (0.0, float("inf")) else dict(dt_limit=self.dt_limit)
        # This branch is a giant fused kernel used for training 
        if self.use_mem_eff_path and inference_params is None:
            out = mamba_split_conv1d_scan_combined(
                zxbcdt,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.dt_bias,
                A,
                D=rearrange(self.D, "(h p) -> h p", p=self.headdim) if self.D_has_hdim else self.D,
                chunk_size=self.chunk_size,
                seq_idx=seq_idx,
                activation=self.activation,
                rmsnorm_weight=self.norm.weight if self.rmsnorm else None,
                rmsnorm_eps=self.norm.eps if self.rmsnorm else 1e-6,
                outproj_weight=self.out_proj.weight,
                outproj_bias=self.out_proj.bias,
                headdim=None if self.D_has_hdim else self.headdim,
                ngroups=self.ngroups,
                norm_before_gate=self.norm_before_gate,
                **dt_limit_kwargs,
            )
            if seqlen_og is not None:
                out = rearrange(out, "b l d -> (b l) d")
            if self.process_group is not None:
                reduce_fn = reduce_scatter if self.sequence_parallel else all_reduce
                out = reduce_fn(out, self.process_group)
        # This branch is used by original Mamba-2 to process prompts (i.e. the first iteration during generation)
        # The main difference between this branch and the previous branch is that the conv_state and ssm_state needs to be saved for future here
        else:
            d_mlp = (zxbcdt.shape[-1] - 2 * self.d_ssm - 2 * self.ngroups * self.d_state - self.nheads) // 2
            z0, x0, z, xBC, dt = torch.split(
                zxbcdt,
                [d_mlp, d_mlp, self.d_ssm, self.d_ssm + 2 * self.ngroups * self.d_state, self.nheads],
                dim=-1
            )
            if conv_state is not None:
                # If we just take xBC[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
                # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
                xBC_t = rearrange(xBC, "b l d -> b d l")
                # The original implementation assumes we are at the start of the generation where there no xBC in front 
                # Now that we are also using this for mutliple token decoding we need to manually shift the conv_state
                if inference_params.seqlen_offset == 0:
                    #TODO: This doesn"t support padding from the right
                    conv_state.copy_(repeat(F.pad(xBC_t, (self.d_conv - xBC_t.shape[-1], 0)), "b ... -> (b r) ...", r=inference_params.ndraft))  # Update state (B D W)
                    if inference_params.activation_replay:
                        conv_state_idx = torch.arange(inference_params.num_input_seq, device=conv_state.device) * inference_params.ndraft
                        inference_params.verified_key_value_memory_dict["state"][self.layer_idx][0].copy_(conv_state[conv_state_idx,...])

            assert self.activation in ["silu", "swish"]
            # The original implementation doesn't consider a second pass, so it is not using conv_state
            # by right you need to use the conv_state
            # state copying 
            if inference_params.seqlen_offset > 0:
                if inference_params.jit_state_copy:
                    if inference_params.activation_replay:
                        if inference_params.use_tree_scan_kernel:
                            # replay state during tree scan
                            # Getting the information for the verified tokens
                            if not inference_params.first_iteration:
                                idx = inference_params.verified_key_value_memory_dict["indices"]
                                m = inference_params.verified_key_value_memory_dict["mask"].clone() # m here is the mask 
                                verified_conv_state = inference_params.verified_key_value_memory_dict["state"][self.layer_idx][0]
                                # xBC is the xBC extended by the conv_state from the last iteration, with this we can determine the latest conv_state
                                verified_xBC = inference_params.value_cache[self.layer_idx]["xBC"]
                                # Choosing tokens to save
                                verified_xBC = torch.index_select(verified_xBC, 0, idx)
                                # The state indices here are not continuous anymore
                                # extending the mask by 4 1s to handled the index for the states
                                m = torch.nn.functional.pad(m, (4, 0), value=1)
                                state_indices = mask_index_gather(m, self.d_conv).squeeze(0)
                                # adjusting indices for multi-sequence generation
                                state_indices = state_indices + \
                                    torch.arange(inference_params.num_input_seq, device=verified_xBC.device)[:, None] * verified_xBC.shape[1]
                                # Savinge the verified state into verified_conv_state
                                verified_conv_state.copy_(
                                    F.embedding(state_indices, verified_xBC.view(-1, verified_xBC.shape[2])).transpose(1,2))
                                # Broad-casting the verified state to all other batch position.
                                conv_state.copy_(repeat(verified_conv_state, "b ... -> (b r) ...", r=inference_params.ndraft))
                        else:
                            # The first iteration we do not need to copy state since all states are correct
                            # The following iterations are then run through CUDA graph
                            if not inference_params.first_iteration:
                                # Getting the information for the verified tokens
                                idx = inference_params.verified_key_value_memory_dict["indices"]
                                l = inference_params.verified_key_value_memory_dict["mask"].sum(dim=1, keepdim=True) # this is the number of verified tokens
                                verified_conv_state = inference_params.verified_key_value_memory_dict["state"][self.layer_idx][0]
                                # xBC is the xBC extended by the conv_state from the last iteration, with this we can determine the latest conv_state
                                verified_xBC = inference_params.value_cache[self.layer_idx]["xBC"]
                                # Choosing tokens to save
                                verified_xBC = torch.index_select(verified_xBC, 0, idx)
                                state_indices = torch.arange(4, device=verified_xBC.device)[None, :] + l
                                # adjusting indices for multi-sequence generation
                                state_indices = state_indices + torch.arange(inference_params.num_input_seq, device=verified_xBC.device)[:, None] * verified_xBC.shape[1]
                                # Savinge the verified state into verified_conv_state
                                verified_conv_state.copy_(
                                    F.embedding(state_indices, verified_xBC.view(-1, verified_xBC.shape[2])).transpose(1,2))
                                # Broad-casting the verified state to all other batch position.
                                conv_state.copy_(repeat(verified_conv_state, "b ... -> (b r) ...", r=inference_params.ndraft))

                    else:
                        # if not doing activation replay, we simply choose the correct state and do the broad-casting. 
                        idx = inference_params.verified_key_value_memory_dict["indices"]
                        updated_conv_state = conv_state.index_select(0, idx)
                        conv_state.copy_(repeat(updated_conv_state, "b ... -> (b r) ...", r=inference_params.ndraft))
                
                assert causal_conv1d_update is not None, "causal_conv1d not installed"

                if inference_params.activation_replay:
                    if inference_params.use_tree_scan_kernel:
                        inference_params.value_cache[self.layer_idx]["xBC"][:, :self.d_conv, :].copy_(conv_state.transpose(1,2))
                        inference_params.value_cache[self.layer_idx]["xBC"][:, self.d_conv:self.d_conv+xBC.shape[1], :].copy_(xBC)
                        xBC = tree_mask_conv1d(
                            xBC.transpose(1,2),
                            mask,
                            conv_state, 
                            rearrange(self.conv1d.weight, "d 1 w -> d w"),
                            self.conv1d.bias,
                            self.activation).transpose(1, 2)
                    else:
                        # If we assume there is no padding at the start of xBC, The following commented lines would also work
                        # inference_params.value_cache[self.layer_idx]["xBC"][:, self.d_conv:, :].copy_(xBC)
                        # inference_params.value_cache[self.layer_idx]["xBC"][:, :self.d_conv, :].copy_(conv_state.transpose(1,2))
                        # else, we use the customized kernel to copy states sequence by sequence in the batch.
                        if inference_params.unroll_tree:
                            # this probably has problem as the update_v2 function can only handle padding from the left but not from the right
                            # in a tree case there might be padding from the right
                            # inference_params.value_cache[self.layer_idx]["xBC"][:batch, ...].copy_(causal_conv1d_varlen_states_update_v2(xBC, mask, conv_state).transpose(1,2))
                            # need to use the mask_index_gather 
                            m = torch.nn.functional.pad(mask, (4, 0), value=1)
                            # print(m, xBC.shape[1]+conv_state.shape[2], m.shape)
                            parent_matrix = mask_index_gather(m, xBC.shape[1]+conv_state.shape[2], right_align=False).squeeze(0)
                            extended_state = torch.cat([conv_state.transpose(1,2), xBC], dim=1).view((-1, xBC.shape[2]))
                            state_mask = parent_matrix != -1
                            state_indices = parent_matrix + torch.arange(xBC.shape[0], device=xBC.device)[:, None] * (xBC.shape[1]+conv_state.shape[2])
                            state_indices = torch.where(state_mask==1, state_indices, 0)
                            # print(state_indices, extended_state.shape)
                            new_xBC = F.embedding(state_indices, extended_state)
                            # print(new_xBC.shape)
                            inference_params.value_cache[self.layer_idx]["xBC"][:batch, ...].copy_(new_xBC)
                            xBC = causal_conv1d_varlen_update(
                                xBC.transpose(1,2), 
                                mask, 
                                conv_state, 
                                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                                self.conv1d.bias,
                                self.activation
                            ).transpose(1, 2)
                        else:
                            inference_params.value_cache[self.layer_idx]["xBC"].copy_(causal_conv1d_varlen_states_update_v2(xBC, mask, conv_state).transpose(1,2))
                            xBC = causal_conv1d_update(
                                xBC.transpose(1,2),
                                conv_state,
                                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                                self.conv1d.bias,
                                self.activation).transpose(1, 2)
                else:
                    if mask is None:
                        mask = torch.ones((u.shape[0], u.shape[1]), dtype=u.dtype, device=u.device)
                    xBC = causal_conv1d_varlen_update(
                        xBC.transpose(1,2), 
                        mask, 
                        conv_state, 
                        rearrange(self.conv1d.weight, "d 1 w -> d w"),
                        self.conv1d.bias,
                        self.activation
                    ).transpose(1, 2)
                    
            else:
                if inference_params.activation_replay:
                    # The pad here trims/pad xBC to npad length 
                    padded_xBC = repeat(F.pad(xBC, (0, 0, inference_params.npad+1 - xBC.shape[1], 0)), "b ... -> (b r) ...", r=inference_params.ndraft)
                    inference_params.value_cache[self.layer_idx]["xBC"][:, self.d_conv:, :].copy_(padded_xBC)
                if causal_conv1d_fn is None or self.activation not in ["silu", "swish"]:
                    xBC = self.act(
                        self.conv1d(xBC.transpose(1, 2)).transpose(1, 2)
                    )  # (B, L, self.d_ssm + 2 * ngroups * d_state)
                else:
                    xBC = causal_conv1d_fn(
                        xBC.transpose(1, 2),
                        rearrange(self.conv1d.weight, "d 1 w -> d w"),
                        bias=self.conv1d.bias,
                        activation=self.activation,
                    ).transpose(1, 2)

            # masking out the post convolution output to ensure B is 0 for padded tokens again
            # since dt is passed through a softplus activation, we set dt to -inf so that post-activation dt is 0
            if mask is not None:
                if not inference_params.use_tree_scan_kernel:
                    xBC = xBC[:, -mask.shape[1]:, :]
                    xBC = xBC * mask[:, :, None]
                    dt = dt.masked_fill(mask[:, :, None] == 0, float("-1e4"))


            x, B, C = torch.split(xBC, [self.d_ssm, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)

            # This branch is the actual Nstep kernel for decoding, activation replay and joint attainment are both implemented here
            if inference_params.seqlen_offset > 0 and inference_params.use_Nstep_kernel:
                # ssm_state copying
                if inference_params.jit_state_copy:
                    if inference_params.activation_replay:
                        # Adding this case to handle first iteration after prefill stage 
                        if not inference_params.first_iteration:
                            # Selecting and masking the correct sequence based on the verification result 
                            idx = inference_params.verified_key_value_memory_dict["indices"]
                            m = inference_params.verified_key_value_memory_dict["mask"]
                            verified_state = inference_params.verified_key_value_memory_dict["state"][self.layer_idx][1]
                            verified_x = torch.index_select(inference_params.value_cache[self.layer_idx]["x"], 0, idx)
                            verified_x = verified_x * m[:, :, None]
                            verified_dt = torch.index_select(inference_params.value_cache[self.layer_idx]["dt"], 0, idx)
                            verified_dt = verified_dt.masked_fill(m[:, :, None]==0, float("-1e4"))
                            verified_B = torch.index_select(inference_params.value_cache[self.layer_idx]["B"], 0, idx)
                            verified_B = verified_B * m[:, :, None]
                            # Doing a forward pass through kernel to recompute the correct state, the output is not needed here
                            _ = selective_state_update_Nstep(
                                verified_state,
                                rearrange(verified_x, "b l (h p) -> b l h p", p=self.headdim),
                                repeat(verified_dt, "b l h -> b l h p", p=self.headdim),
                                repeat(A, "h -> h p n", p=self.headdim, n=self.d_state).to(dtype=torch.float32), 
                                rearrange(verified_B, "b l (g n) -> b l g n", g=self.ngroups),
                                rearrange(torch.zeros_like(verified_B), "b l (g n) -> b l g n", g=self.ngroups), 
                                D=repeat(self.D, "h -> h p", p=self.headdim), 
                                z=rearrange(torch.zeros_like(verified_x), "b l (h p) -> b l h p", p=self.headdim) if not self.rmsnorm else None, 
                                dt_bias=repeat(self.dt_bias, "h -> h p", p=self.headdim),
                                dt_softplus=True 
                            )
                            ssm_state.copy_(repeat(verified_state, "b ... -> (b r) ...", r=inference_params.ndraft))
                    else:
                        idx = inference_params.verified_key_value_memory_dict["indices"]
                        idx = repeat(idx, "b ... -> (b r) ...", r=inference_params.ndraft)
                        inplace_state_indexing(ssm_state, idx)
                
                # The actual forward pass
                y = selective_state_update_Nstep(
                    ssm_state,
                    rearrange(x, "b l (h p) -> b l h p", p=self.headdim),
                    repeat(dt, "b l h -> b l h p", p=self.headdim),
                    repeat(A, "h -> h p n", p=self.headdim, n=self.d_state).to(dtype=torch.float32), 
                    rearrange(B, "b l (g n) -> b l g n", g=self.ngroups),
                    rearrange(C, "b l (g n) -> b l g n", g=self.ngroups), 
                    D=repeat(self.D, "h -> h p", p=self.headdim), 
                    z=rearrange(z, "b l (h p) -> b l h p", p=self.headdim) if not self.rmsnorm else None, 
                    dt_bias=repeat(self.dt_bias, "h -> h p", p=self.headdim),
                    dt_softplus=True 
                )
                # saving x, dt, B for recomputing state
                # C, z doesn't affect state so not needed, A is constant through time so not needed.
                if inference_params.activation_replay: 
                    if inference_params.unroll_tree:
                        inference_params.value_cache[self.layer_idx]["x"][:batch,...].copy_(x)
                        inference_params.value_cache[self.layer_idx]["dt"][:batch,...].copy_(dt)
                        inference_params.value_cache[self.layer_idx]["B"][:batch,...].copy_(B)
                    else:
                        inference_params.value_cache[self.layer_idx]["x"].copy_(x)
                        inference_params.value_cache[self.layer_idx]["dt"].copy_(dt)
                        inference_params.value_cache[self.layer_idx]["B"].copy_(B)
                        # inference_params.value_cache[self.layer_idx]["C"].copy_(C)
                        # inference_params.value_cache[self.layer_idx]["z"].copy_(z)
            elif inference_params.seqlen_offset > 0 and inference_params.use_tree_scan_kernel:
                # Adding this case to handle first iteration after prefill stage 
                if not inference_params.first_iteration:
                    # Selecting and masking the correct sequence based on the verification result 
                    idx = inference_params.verified_key_value_memory_dict["indices"]
                    m = inference_params.verified_key_value_memory_dict["mask"]
                    m = torch.nn.functional.pad(m, (0, inference_params.npad+1-m.shape[1]), value=0)
                    verified_state = inference_params.verified_key_value_memory_dict["state"][self.layer_idx][1]
                    verified_state.copy_(verified_state[idx, ...])

                    verified_x = torch.index_select(inference_params.value_cache[self.layer_idx]["x"], 0, idx)
                    verified_x = verified_x * m[:, :, None]

                    verified_dt = torch.index_select(inference_params.value_cache[self.layer_idx]["dt"], 0, idx)
                    verified_dt = verified_dt.masked_fill(m[:, :, None]==0, float("-1e4"))

                    verified_B = torch.index_select(inference_params.value_cache[self.layer_idx]["B"], 0, idx)
                    verified_B = verified_B * m[:, :, None]
                    # Doing a forward pass through kernel to recompute the correct state, the output is not needed here
                    _ = selective_state_update_Nstep(
                        verified_state,
                        rearrange(verified_x, "b l (h p) -> b l h p", p=self.headdim),
                        repeat(verified_dt, "b l h -> b l h p", p=self.headdim),
                        repeat(A, "h -> h p n", p=self.headdim, n=self.d_state).to(dtype=torch.float32), 
                        rearrange(verified_B, "b l (g n) -> b l g n", g=self.ngroups),
                        rearrange(torch.zeros_like(verified_B), "b l (g n) -> b l g n", g=self.ngroups), 
                        D=repeat(self.D, "h -> h p", p=self.headdim), 
                        z=rearrange(torch.zeros_like(verified_x), "b l (h p) -> b l h p", p=self.headdim) if not self.rmsnorm else None, 
                        dt_bias=repeat(self.dt_bias, "h -> h p", p=self.headdim),
                        dt_softplus=True 
                    )
                    ssm_state.copy_(repeat(verified_state, "b ... -> (b r) ...", r=inference_params.ndraft))
                # The actual forward pass
                y, _, _, _ = tree_scan_mask_combined_fwd(
                    rearrange(x, "b l (h p) -> b l h p", p=self.headdim),
                    dt,
                    A,
                    rearrange(B, "b l (g n) -> b l g n", g=self.ngroups),
                    rearrange(C, "b l (g n) -> b l g n", g=self.ngroups),
                    dA_mask=mask,
                    chunk_size=self.chunk_size,
                    D=rearrange(self.D, "(h p) -> h p", p=self.headdim) if self.D_has_hdim else self.D,
                    z=rearrange(z, "b l (h p) -> b l h p", p=self.headdim) if not self.rmsnorm else None,
                    dt_bias=self.dt_bias,
                    initial_states=ssm_state,
                    seq_idx=seq_idx,
                    dt_softplus=True,
                    **dt_limit_kwargs)
                # saving x, dt, B for recomputing state
                # C, z doesn't affect state so not needed, A is constant through time so not needed.
                if inference_params.activation_replay: 
                    inference_params.value_cache[self.layer_idx]["x"][:, :x.shape[1], ...].copy_(x)
                    inference_params.value_cache[self.layer_idx]["dt"][:, :dt.shape[1], ...].copy_(dt)
                    inference_params.value_cache[self.layer_idx]["B"][:, :B.shape[1], ...].copy_(B)
            # This branch is for decoding using chunk_scan_kernel and prompt preprocessing with chunk_scan_kernel
            # Chunk_scan kernel is optimzed for long sequence processing
            # It does not support activation replay as well
            else:
                # ssm_state copying, only needed for decoding steps and not prompt preprocessing steps
                if inference_params.seqlen_offset > 0 and inference_params.jit_state_copy:
                    idx = inference_params.verified_key_value_memory_dict["indices"]
                    idx = repeat(idx, "b ... -> (b r) ...", r=inference_params.ndraft)
                    inplace_state_indexing(ssm_state, idx)
                    # updated_ssm_state = ssm_state.index_select(0, idx)
                    # ssm_state.copy_(repeat(updated_ssm_state, "b ... -> (b r) ...", r=inference_params.ndraft))
                # For decoding steps we need initial_state for the chunk_scan to be the ssm_state
                # For prompt preprocessing, the initial state is none and is thus 0
                y = mamba_chunk_scan_combined(
                    rearrange(x, "b l (h p) -> b l h p", p=self.headdim),
                    dt,
                    A,
                    rearrange(B, "b l (g n) -> b l g n", g=self.ngroups),
                    rearrange(C, "b l (g n) -> b l g n", g=self.ngroups),
                    chunk_size=self.chunk_size,
                    D=rearrange(self.D, "(h p) -> h p", p=self.headdim) if self.D_has_hdim else self.D,
                    z=rearrange(z, "b l (h p) -> b l h p", p=self.headdim) if not self.rmsnorm else None,
                    dt_bias=self.dt_bias,
                    dt_softplus=True,
                    initial_states=ssm_state if inference_params.seqlen_offset > 0 else None,
                    seq_idx=seq_idx,
                    **dt_limit_kwargs,
                    return_final_states=ssm_state is not None,
                )
                if ssm_state is not None:
                    y, last_state = y
                    if inference_params.seqlen_offset > 0:
                        ssm_state.copy_(last_state)
                    else:
                        ssm_state.copy_(repeat(last_state, "b ... -> (b r) ...", r=inference_params.ndraft))
                if inference_params.activation_replay:
                    ssm_state_idx = torch.arange(inference_params.num_input_seq, device=conv_state.device) * inference_params.ndraft
                    inference_params.verified_key_value_memory_dict["state"][self.layer_idx][1].copy_(ssm_state[ssm_state_idx, ...])
            y = rearrange(y, "b l h p -> b l (h p)")
            if self.rmsnorm:
                y = self.norm(y, z)
            if d_mlp > 0:
                y = torch.cat([F.silu(z0) * x0, y], dim=-1)
            if seqlen_og is not None:
                y = rearrange(y, "b l d -> (b l) d")
            out = self.out_proj(y)
        return out

    def step(self, hidden_states, conv_state, ssm_state, inference_params=None):
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
        zxbcdt = self.in_proj(hidden_states.squeeze(1))  # (B 2D)
        d_mlp = (zxbcdt.shape[-1] - 2 * self.d_ssm - 2 * self.ngroups * self.d_state - self.nheads) // 2
        z0, x0, z, xBC, dt = torch.split(
            zxbcdt,
            [d_mlp, d_mlp, self.d_ssm, self.d_ssm + 2 * self.ngroups * self.d_state, self.nheads],
            dim=-1
        )

        # introduced to speedup beam search and mamba drafting model
        if inference_params.jit_state_copy:
            idx = inference_params.verified_key_value_memory_dict["indices"]
            updated_conv_state = conv_state.index_select(0, idx)
            conv_state.copy_(repeat(updated_conv_state, "b ... -> (b r) ...", r=inference_params.ndraft))
        # Conv step
        if causal_conv1d_update is None:
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))  # Update state (B D W)
            conv_state[:, :, -1] = xBC
            xBC = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)  # (B D)
            if self.conv1d.bias is not None:
                xBC = xBC + self.conv1d.bias
            xBC = self.act(xBC).to(dtype=dtype)
        else:
            xBC = causal_conv1d_update(
                xBC,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )

        x, B, C = torch.split(xBC, [self.d_ssm, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)
        A = -torch.exp(self.A_log.float())  # (nheads,)

        # introduced to speedup beam search and mamba drafting model
        if inference_params.jit_state_copy:
            idx = inference_params.verified_key_value_memory_dict["indices"]
            idx = repeat(idx, "b ... -> (b r) ...", r=inference_params.ndraft)
            inplace_state_indexing(ssm_state, idx)
            # updated_ssm_state = ssm_state.index_select(0, idx)
            # ssm_state.copy_(repeat(updated_ssm_state, "b ... -> (b r) ...", r=inference_params.ndraft))
        # SSM step
        if selective_state_update is None:
            assert self.ngroups == 1, "Only support ngroups=1 for this inference code path"
            # Discretize A and B
            dt = F.softplus(dt + self.dt_bias.to(dtype=dt.dtype))  # (batch, nheads)
            dA = torch.exp(dt * A)  # (batch, nheads)
            x = rearrange(x, "b (h p) -> b h p", p=self.headdim)
            dBx = torch.einsum("bh,bn,bhp->bhpn", dt, B, x)
            ssm_state.copy_(ssm_state * rearrange(dA, "b h -> b h 1 1") + dBx)
            y = torch.einsum("bhpn,bn->bhp", ssm_state.to(dtype), C)
            y = y + rearrange(self.D.to(dtype), "h -> h 1") * x
            y = rearrange(y, "b h p -> b (h p)")
            if not self.rmsnorm:
                y = y * self.act(z)  # (B D)
        else:
            A = repeat(A, "h -> h p n", p=self.headdim, n=self.d_state).to(dtype=torch.float32)
            dt = repeat(dt, "b h -> b h p", p=self.headdim)
            dt_bias = repeat(self.dt_bias, "h -> h p", p=self.headdim)
            D = repeat(self.D, "h -> h p", p=self.headdim)
            B = rearrange(B, "b (g n) -> b g n", g=self.ngroups)
            C = rearrange(C, "b (g n) -> b g n", g=self.ngroups)
            x_reshaped = rearrange(x, "b (h p) -> b h p", p=self.headdim)
            if not self.rmsnorm:
                z = rearrange(z, "b (h p) -> b h p", p=self.headdim)
            y = selective_state_update(
                ssm_state, x_reshaped, dt, A, B, C, D, z=z if not self.rmsnorm else None,
                dt_bias=dt_bias, dt_softplus=True
            )
            y = rearrange(y, "b h p -> b (h p)")
        if self.rmsnorm:
            y = self.norm(y, z)
        if d_mlp > 0:
            y = torch.cat([F.silu(z0) * x0, y], dim=-1)
        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size, self.conv1d.weight.shape[0], self.d_conv, device=device, dtype=conv_dtype
        )
        ssm_dtype = self.in_proj.weight.dtype if dtype is None else dtype
        ssm_state = torch.zeros(
            batch_size, self.nheads, self.headdim, self.d_state, device=device, dtype=ssm_dtype
        )
        return conv_state, ssm_state
    
    def allocate_value_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        dtype = self.in_proj.weight.dtype if dtype is None else dtype
        x_cache = torch.zeros(
            batch_size, max_seqlen, self.nheads * self.headdim, device=device, dtype=dtype
        )
        dt_cache = torch.zeros(
            batch_size, max_seqlen, self.nheads, device=device, dtype=dtype
        )
        B_cache = torch.zeros(
            batch_size, max_seqlen, self.ngroups * self.d_state, device=device, dtype=dtype
        )
        xBC_cache = torch.zeros(
            batch_size, max_seqlen+self.d_conv, self.d_ssm + 2*self.ngroups*self.d_state, device=device, dtype=dtype
        )
        return {"x": x_cache, "dt":dt_cache, "B":B_cache, "xBC":xBC_cache}


    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            batch_shape = (batch_size,)
            conv_state = torch.zeros(
                batch_size,
                self.conv1d.weight.shape[0],
                self.d_conv,
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            )
            ssm_state = torch.zeros(
                batch_size,
                self.nheads,
                self.headdim,
                self.d_state,
                device=self.in_proj.weight.device,
                dtype=self.in_proj.weight.dtype,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            if inference_params.unroll_tree and inference_params.seqlen_offset != 0:
                conv_state, ssm_state = conv_state[:batch_size, ...], ssm_state[:batch_size, ...]
            # TODO: What if batch size changes between generation, and we reuse the same states?
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state
