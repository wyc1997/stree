# Copyright (c) 2023, Albert Gu, Tri Dao.

import math
from functools import partial
import json
import os
import copy

from collections import namedtuple
from typing import Optional
from transformers.generation import GreedySearchDecoderOnlyOutput, SampleDecoderOnlyOutput, TextStreamer

import torch
import torch.nn as nn
import torch.multiprocessing as mp
from einops import repeat
mp.set_start_method('spawn', force=True)

from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.modules.mamba2 import Mamba2
from mamba_ssm.modules.mha import MHA
from mamba_ssm.modules.mlp import GatedMLP
from mamba_ssm.modules.block import Block
from mamba_ssm.utils.generation import GenerationMixin
from mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf
from mamba_ssm.utils.generation_utils import update_graph_cache, modify_logit_for_repetition_penalty, InferenceParams, sample
from mamba_ssm.utils.profile import cuda_time

try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


def create_block(
    d_model,
    d_intermediate,
    ssm_cfg=None,
    attn_layer_idx=None,
    attn_cfg=None,
    norm_epsilon=1e-5,
    rms_norm=False,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
    device=None,
    dtype=None,
):
    if ssm_cfg is None:
        ssm_cfg = {}
    if attn_layer_idx is None:
        attn_layer_idx = []
    if attn_cfg is None:
        attn_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    if layer_idx not in attn_layer_idx:
        # Create a copy of the config to modify
        ssm_cfg = copy.deepcopy(ssm_cfg) if ssm_cfg is not None else {}
        ssm_layer = ssm_cfg.pop("layer", "Mamba1")
        if ssm_layer not in ["Mamba1", "Mamba2"]:
            raise ValueError(f"Invalid ssm_layer: {ssm_layer}, only support Mamba1 and Mamba2")
        mixer_cls = partial(
            Mamba2 if ssm_layer == "Mamba2" else Mamba,
            layer_idx=layer_idx,
            **ssm_cfg,
            **factory_kwargs
        )
    else:
        mixer_cls = partial(MHA, layer_idx=layer_idx, **attn_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    if d_intermediate == 0:
        mlp_cls = nn.Identity
    else:
        mlp_cls = partial(
            GatedMLP, hidden_features=d_intermediate, out_features=d_model, **factory_kwargs
        )
    block = Block(
        d_model,
        mixer_cls,
        mlp_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block


# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


class MixerModel(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_layer: int,
        d_intermediate: int,
        vocab_size: int,
        ssm_cfg=None,
        attn_layer_idx=None,
        attn_cfg=None,
        norm_epsilon: float = 1e-5,
        rms_norm: bool = False,
        initializer_cfg=None,
        fused_add_norm=False,
        residual_in_fp32=False,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32

        self.embedding = nn.Embedding(vocab_size, d_model, **factory_kwargs)

        # We change the order of residual and layer norm:
        # Instead of LN -> Attn / MLP -> Add, we do:
        # Add -> LN -> Attn / MLP / Mixer, returning both the residual branch (output of Add) and
        # the main branch (output of MLP / Mixer). The model definition is unchanged.
        # This is for performance reason: we can fuse add + layer_norm.
        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model,
                    d_intermediate=d_intermediate,
                    ssm_cfg=ssm_cfg,
                    attn_layer_idx=attn_layer_idx,
                    attn_cfg=attn_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs
        )

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
                n_residuals_per_layer=1 if d_intermediate == 0 else 2,  # 2 if we have MLP
            )
        )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }
    
    def allocate_value_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_value_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    def forward(self, input_ids,  inference_params=None, mask=None, **mixer_kwargs):
        hidden_states = self.embedding(input_ids)
        # end.record()
        # torch.cuda.synchronize()
        # print("embedding time:", start.elapsed_time(end))
        residual = None
        self.hidden_residual_pairs = []
        # if inference_params.layers is None:
        #     run_layers = self.layers 
        # else:
        #     run_layers = self.layers[inference_params.layers[0]:inference_params.layers[1]]
        for layer in self.layers:
            # start = torch.cuda.Event(enable_timing=True)
            # end = torch.cuda.Event(enable_timing=True)
            # start.record()
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params, mask=mask
            )
            self.hidden_residual_pairs.append((hidden_states, residual))
            # end.record()
            # torch.cuda.synchronize()
            # if layer.mixer.layer_idx == 0:
            #     print("mixer 1 layers time:", start.elapsed_time(end))
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            hidden_states = layer_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
                is_rms_norm=isinstance(self.norm_f, RMSNorm)
            )
        return hidden_states


class MambaLMHeadModel(nn.Module, GenerationMixin):

    def __init__(
        self,
        config: MambaConfig,
        initializer_cfg=None,
        device=None,
        dtype=None,
    ) -> None:
        self.config = config
        d_model = config.d_model
        n_layer = config.n_layer
        d_intermediate = config.d_intermediate
        vocab_size = config.vocab_size
        ssm_cfg = config.ssm_cfg
        attn_layer_idx = config.attn_layer_idx
        attn_cfg = config.attn_cfg
        rms_norm = config.rms_norm
        residual_in_fp32 = config.residual_in_fp32
        fused_add_norm = config.fused_add_norm
        pad_vocab_size_multiple = config.pad_vocab_size_multiple
        factory_kwargs = {"device": device, "dtype": dtype}


        super().__init__()
        if vocab_size % pad_vocab_size_multiple != 0:
            vocab_size += pad_vocab_size_multiple - (vocab_size % pad_vocab_size_multiple)
        self.backbone = MixerModel(
            d_model=d_model,
            n_layer=n_layer,
            d_intermediate=d_intermediate,
            vocab_size=vocab_size,
            ssm_cfg=ssm_cfg,
            attn_layer_idx=attn_layer_idx,
            attn_cfg=attn_cfg,
            rms_norm=rms_norm,
            initializer_cfg=initializer_cfg,
            fused_add_norm=fused_add_norm,
            residual_in_fp32=residual_in_fp32,
            **factory_kwargs,
        )
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False, **factory_kwargs)
        self.generation_hist = []

        # Initialize weights and apply final processing
        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )
        self.tie_weights()

    def tie_weights(self):
        if self.config.tie_embeddings:
            self.lm_head.weight = self.backbone.embedding.weight

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.backbone.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
    
    def allocate_value_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.backbone.allocate_value_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

    def forward(self, input_ids, position_ids=None, inference_params=None, num_last_tokens=0, mask=None, **mixer_kwargs):
        """
        "position_ids" is just to be compatible with Transformer generation. We don't use it.
        num_last_tokens: if > 0, only return the logits for the last n tokens
        """
        hidden_states = self.backbone(input_ids, inference_params=inference_params, mask=mask, **mixer_kwargs)
        if num_last_tokens > 0:
            hidden_states = hidden_states[:, -num_last_tokens:]
        lm_logits = self.lm_head(hidden_states)
        CausalLMOutput = namedtuple("CausalLMOutput", ["logits"])
        return CausalLMOutput(logits=lm_logits)

    @classmethod
    def from_pretrained(cls, pretrained_model_name, device=None, dtype=None, **kwargs):
        config_data = load_config_hf(pretrained_model_name)
        config = MambaConfig(**config_data)
        model = cls(config, device=device, dtype=dtype, **kwargs)
        model.load_state_dict(load_state_dict_hf(pretrained_model_name, device=device, dtype=dtype))
        return model

    def save_pretrained(self, save_directory):
        """
        Minimal implementation of save_pretrained for MambaLMHeadModel.
        Save the model and its configuration file to a directory.
        """
        # Ensure save_directory exists
        os.makedirs(save_directory, exist_ok=True)

        # Save the model's state_dict
        model_path = os.path.join(save_directory, 'pytorch_model.bin')
        torch.save(self.state_dict(), model_path)

        # Save the configuration of the model
        config_path = os.path.join(save_directory, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config.__dict__, f, indent=4)


class PartialMixerModel(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_layer: int,
        d_intermediate: int,
        vocab_size: int,
        ssm_cfg=None,
        attn_layer_idx=None,
        attn_cfg=None,
        norm_epsilon: float = 1e-5,
        rms_norm: bool = False,
        initializer_cfg=None,
        fused_add_norm=False,
        residual_in_fp32=False,
        device=None,
        dtype=None,
        model_layer_split=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32

        self.embedding = nn.Embedding(vocab_size, d_model, **factory_kwargs)

        # We change the order of residual and layer norm:
        # Instead of LN -> Attn / MLP -> Add, we do:
        # Add -> LN -> Attn / MLP / Mixer, returning both the residual branch (output of Add) and
        # the main branch (output of MLP / Mixer). The model definition is unchanged.
        # This is for performance reason: we can fuse add + layer_norm.
        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model,
                    d_intermediate=d_intermediate,
                    ssm_cfg=ssm_cfg,
                    attn_layer_idx=attn_layer_idx,
                    attn_cfg=attn_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )

        self.model_layer_split = model_layer_split
        # used for block 2 to pass info back to block 1 about verified tokens
        self.verified_tokens = None
        # Acting as a spin lock for block 2 to wait
        self.block1_output = None
        # shared variable to terminate subprocess for block 1
        self.block1_subprocess_terminate = False

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs
        )

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
                n_residuals_per_layer=1 if d_intermediate == 0 else 2,  # 2 if we have MLP
            )
        )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    def allocate_value_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_value_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    def forward(self, input_ids, inference_params=None, **mixer_kwargs):
        hidden_states = self.embedding(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params
            )
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            hidden_states = layer_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
                is_rms_norm=isinstance(self.norm_f, RMSNorm)
            )
        return hidden_states


# def update_verified_memory_dict(inference_params, )
@torch.inference_mode()
def block1_subprocess(model, input_ids, inference_params, temperature, repetition_penalty, output_queue: mp.Queue, input_queue: mp.Queue):
    drafted_outputs = [input_ids]
    verified_outputs = input_ids
    verified_length = input_ids.shape[1]
    current_token_verified = True
    drafted_length = 0
    max_draft_length = 1
    should_terminate = False
    # keep running until main process notify us to stop
    while True:
        if should_terminate:
            input_queue.close()
            # print("Terminating block1 while loop")
            break
        residual = None
        hidden_states = model.backbone.embedding(drafted_outputs[-1])
        while True:
            if max_draft_length > drafted_length or not input_queue.empty():
                # print("waiting")
                break
        b1_start = torch.cuda.Event(enable_timing=True)
        b1_end = torch.cuda.Event(enable_timing=True)
        b1_start.record()
        for i in range(model.backbone.model_layer_split+1):
            if not input_queue.empty(): 
                verified_tokens = input_queue.get()
                if verified_tokens[0] is None and verified_tokens[1] is None:
                    should_terminate = True
                    break
                # print(verified_tokens, drafted_outputs[-1])
                # print((verified_tokens == drafted_outputs[-1]))
                matches = (verified_tokens == drafted_outputs[-1]).squeeze()
                if matches.any():
                    current_token_verified = True
                    # print(hidden_states.shape)
                    hidden_states = hidden_states[matches]
                    # print(hidden_states.shape)
                    residual = residual[matches] if residual is not None else None
                    verified_outputs = torch.cat([verified_outputs, verified_tokens], dim=1)
                    verified_length += verified_tokens.shape[1]
                    drafted_length = 0
                    # selecting the correct ssm_state and conv_state to store into a verified cache
                    for j in range(i):
                        inference_params.verified_key_value_memory_dict[j][0] = inference_params.key_value_memory_dict[j][0][matches].clone()
                        inference_params.verified_key_value_memory_dict[j][1] = inference_params.key_value_memory_dict[j][1][matches].clone()
                        inference_params.key_value_memory_dict[j] = (inference_params.key_value_memory_dict[j][0][matches],
                                                                     inference_params.key_value_memory_dict[j][1][matches])
                else:
                    # drafted token doesn't match, we need to start over from the closest verified token
                    drafted_outputs[-1] = verified_tokens
                    verified_outputs = torch.cat([verified_outputs, verified_tokens], dim=1)
                    verified_length += verified_tokens.shape[1]
                    drafted_length = 0
                    # resetting mamba layer internal states.
                    for j in range(model.backbone.model_layer_split+1):
                        inference_params.key_value_memory_dict[i] = (inference_params.verified_key_value_memory_dict[i][0].clone(),
                                                                    inference_params.verified_key_value_memory_dict[i][1].clone())
                    break
            
            # if batch size increases from last iteration, we copy the internal states to match the batch size
            if i in inference_params.key_value_memory_dict and \
                hidden_states.shape[0] > inference_params.key_value_memory_dict[i][1].shape[0] and \
                inference_params.key_value_memory_dict[i][1].shape[0] == 1:
                inference_params.key_value_memory_dict[i] = (torch.repeat_interleave(inference_params.key_value_memory_dict[i][0], hidden_states.shape[0], dim=0),
                                                            torch.repeat_interleave(inference_params.key_value_memory_dict[i][1], hidden_states.shape[0], dim=0))
            hidden_states, residual = model.backbone.layers[i](hidden_states, residual, inference_params=inference_params)
            # if we know beforehand that current token have already been verified, we can directly store the internal states
            if current_token_verified:
                inference_params.verified_key_value_memory_dict[i] = (inference_params.key_value_memory_dict[i][0].clone(), 
                                                                      inference_params.key_value_memory_dict[i][1].clone())
            if i == model.backbone.model_layer_split:
                b1_end.record()
                torch.cuda.synchronize()
                print('b1 execution time: ', b1_start.elapsed_time(b1_end))
                # notify block2
                # print("sending to block2")
                b1_send_start = torch.cuda.Event(enable_timing=True)
                b1_send_end = torch.cuda.Event(enable_timing=True)
                b1_send_start.record()
                output_queue.put([hidden_states, residual])
                b1_send_end.record()
                torch.cuda.synchronize()
                print("b1 send time: ", b1_send_start.elapsed_time(b1_send_end))
                # go on to draft tokens
                hidden_states = layer_norm_fn(
                    hidden_states,
                    model.backbone.norm_f.weight,
                    model.backbone.norm_f.bias,
                    eps=model.backbone.norm_f.eps,
                    residual=residual,
                    prenorm=False,
                    residual_in_fp32=model.backbone.residual_in_fp32,
                    is_rms_norm=isinstance(model.backbone.norm_f, RMSNorm))
                hidden_states = hidden_states[:, -1, :]
                lm_logits = model.lm_head(hidden_states)

                inference_params.seqlen_offset += drafted_outputs[-1].shape[1]
                if repetition_penalty == 1.0:
                    lm_logits /= temperature
                    _, sampled_tokens = torch.topk(lm_logits, model.drafting_topk, dim=-1)
                    sampled_tokens = sampled_tokens.view(model.drafting_topk, 1)
                else:
                    logits = modify_logit_for_repetition_penalty(
                        lm_logits.clone(), verified_outputs, repetition_penalty
                    )
                    logits /= temperature
                    _, sampled_tokens = torch.topk(logits, model.drafting_topk, dim=-1)
                    sampled_tokens = sampled_tokens.view(model.drafting_topk, 1)
                drafted_outputs.append(sampled_tokens)
                drafted_length += 1
                current_token_verified = False
    return 
class MambaSelfDraftLMHeadModel(nn.Module, GenerationMixin):

    def __init__(
        self,
        config: MambaConfig,
        initializer_cfg=None,
        device=None,
        dtype=None,
        self_draft_layer=None,
        drafting_topk=10,
    ) -> None:
        self.config = config
        d_model = config.d_model
        n_layer = config.n_layer
        d_intermediate = config.d_intermediate
        vocab_size = config.vocab_size
        ssm_cfg = config.ssm_cfg
        attn_layer_idx = config.attn_layer_idx
        attn_cfg = config.attn_cfg
        rms_norm = config.rms_norm
        residual_in_fp32 = config.residual_in_fp32
        fused_add_norm = config.fused_add_norm
        pad_vocab_size_multiple = config.pad_vocab_size_multiple
        factory_kwargs = {"device": device, "dtype": dtype}
        self.self_draft_layer = self_draft_layer
        self.drafting_topk = drafting_topk

        super().__init__()
        if vocab_size % pad_vocab_size_multiple != 0:
            vocab_size += pad_vocab_size_multiple - (vocab_size % pad_vocab_size_multiple)
        self.backbone = PartialMixerModel(
            d_model=d_model,
            n_layer=n_layer,
            d_intermediate=d_intermediate,
            vocab_size=vocab_size,
            ssm_cfg=ssm_cfg,
            attn_layer_idx=attn_layer_idx,
            attn_cfg=attn_cfg,
            rms_norm=rms_norm,
            initializer_cfg=initializer_cfg,
            fused_add_norm=fused_add_norm,
            residual_in_fp32=residual_in_fp32,
            model_layer_split=self_draft_layer,
            **factory_kwargs,
        )
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False, **factory_kwargs)
        self.backbone.share_memory()
        self.lm_head.share_memory()

        # Initialize weights and apply final processing
        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )
        self.tie_weights()

    def tie_weights(self):
        if self.config.tie_embeddings:
            self.lm_head.weight = self.backbone.embedding.weight

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.backbone.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

    def forward(self, input_ids, position_ids=None, inference_params=None, num_last_tokens=0, **mixer_kwargs):
        """
        "position_ids" is just to be compatible with Transformer generation. We don't use it.
        num_last_tokens: if > 0, only return the logits for the last n tokens
        """
        hidden_states = self.backbone(input_ids, inference_params=inference_params, **mixer_kwargs)
        if num_last_tokens > 0:
            hidden_states = hidden_states[:, -num_last_tokens:]
        lm_logits = self.lm_head(hidden_states)
        
        CausalLMOutput = namedtuple("CausalLMOutput", ["logits"])
        return CausalLMOutput(logits=lm_logits)

    @classmethod
    def from_pretrained(cls, pretrained_model_name, device=None, dtype=None, **kwargs):
        config_data = load_config_hf(pretrained_model_name)
        config = MambaConfig(**config_data)
        model = cls(config, device=device, dtype=dtype, **kwargs)
        model.load_state_dict(load_state_dict_hf(pretrained_model_name, device=device, dtype=dtype))
        return model

    def save_pretrained(self, save_directory):
        """
        Minimal implementation of save_pretrained for MambaLMHeadModel.
        Save the model and its configuration file to a directory.
        """
        # Ensure save_directory exists
        os.makedirs(save_directory, exist_ok=True)

        # Save the model's state_dict
        model_path = os.path.join(save_directory, 'pytorch_model.bin')
        torch.save(self.state_dict(), model_path)

        # Save the configuration of the model
        config_path = os.path.join(save_directory, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config.__dict__, f, indent=4)
    
    def generate(self, 
                 input_ids, 
                 max_length, 
                 top_k=1, 
                 top_p=0, 
                 min_p=0, 
                 temperature=1, 
                 return_dict_in_generate=False, 
                 output_scores=False,
                 cg=False,
                 teacher_outputs=None,
                 vocab_size=None,
                 enable_timing=False,   
                 repetition_penalty=1.0,
                 eos_token_id=None,
                 streamer: Optional[TextStreamer] = None,
                 **kwargs):
        if streamer is not None:
            streamer.put(input_ids.cpu())

        batch_size, seqlen_og = input_ids.shape
        teacher_output_len = teacher_outputs.shape[1] if teacher_outputs is not None else 0
        if cg:
            if not hasattr(self, "_decoding_cache"):
                self._decoding_cache = None
            self._decoding_cache = update_graph_cache(
                self,
                self._decoding_cache,
                batch_size,
                seqlen_og,
                max_length,
            )
            inference_params = self._decoding_cache.inference_params
            inference_params.reset(max_length, batch_size)
        else:
            inference_params = InferenceParams(max_seqlen=max_length, max_batch_size=batch_size)

        def get_logits(input_ids, inference_paramsm, input_queue: mp.Queue):
            decoding = inference_params.seqlen_offset > 0
            if decoding:
                position_ids = torch.full(
                    (batch_size, 1),
                    inference_params.seqlen_offset,
                    dtype=torch.long,
                    device=input_ids.device,
                )
            else:
                position_ids = None
            if not cg or not decoding:
                # spinning lock to wait for block1 output 
                get_wait_time_start = torch.cuda.Event(enable_timing=True)
                get_wait_time_end = torch.cuda.Event(enable_timing=True)
                get_wait_time_start.record()
                block1_output = input_queue.get()
                get_wait_time_end.record()
                torch.cuda.synchronize()
                print("b2 get wait time: ", get_wait_time_start.elapsed_time(get_wait_time_end) )
                # print("got block1 result")
                hidden_states, residual = block1_output
                for i in range(self.backbone.model_layer_split+1, len(self.backbone.layers)):
                    hidden_states, residual = self.backbone.layers[i](hidden_states, residual, inference_params=inference_params)
                # go on to draft tokens
                hidden_states = layer_norm_fn(
                    hidden_states,
                    self.backbone.norm_f.weight,
                    self.backbone.norm_f.bias,
                    eps=self.backbone.norm_f.eps,
                    residual=residual,
                    prenorm=False,
                    residual_in_fp32=self.backbone.residual_in_fp32,
                    is_rms_norm=isinstance(self.backbone.norm_f, RMSNorm))
                hidden_states = hidden_states[:, -1, :]
                logits = self.lm_head(hidden_states).squeeze(dim=1)
            else:
                logits = self._decoding_cache.run(
                    input_ids, position_ids, inference_params.seqlen_offset
                ).squeeze(dim=1)
            return logits[..., :vocab_size] if vocab_size is not None else logits

        def sample_tokens(logits, inference_params):
            if teacher_outputs is None or teacher_output_len <= inference_params.seqlen_offset:
                token = sample(logits, top_k=top_k, top_p=top_p, min_p=min_p, temperature=temperature)
            else:
                token = teacher_outputs[:, inference_params.seqlen_offset]
            # return rearrange(token, "b -> b 1")
            return token.unsqueeze(1)

        def should_stop(current_token, inference_params):
            if inference_params.seqlen_offset == 0:
                return False
            if eos_token_id is not None and (current_token == eos_token_id).all():
                return True
            if inference_params.seqlen_offset >= max_length - 1:
                return True
            return False

        start = torch.cuda.Event(enable_timing=enable_timing)
        end = torch.cuda.Event(enable_timing=enable_timing)

        if enable_timing:
            start.record()
        scores, sequences = [], [input_ids]
        sequences_cat = input_ids
        block1_result_queue = mp.Queue()
        block2_verified_token_queue = mp.Queue()
        b1_sp = mp.Process(target=block1_subprocess, args=(self, 
                                                           input_ids, 
                                                           inference_params, 
                                                           temperature, 
                                                           repetition_penalty, 
                                                           block1_result_queue, 
                                                           block2_verified_token_queue))
        b1_sp.start()
        while not should_stop(sequences[-1], inference_params):
            # print("here")
            block2_start = torch.cuda.Event(enable_timing=True)
            block2_end = torch.cuda.Event(enable_timing=True)
            block2_start.record()
            scores.append(get_logits(sequences[-1], inference_params, block1_result_queue))
            block2_end.record()
            torch.cuda.synchronize()
            print("block2 execution time: ", block2_start.elapsed_time(block2_end))
            # print("after get logit")
            inference_params.seqlen_offset += sequences[-1].shape[1]
            if repetition_penalty == 1.0:
                sampled_tokens = sample_tokens(scores[-1], inference_params)
            else:
                logits = modify_logit_for_repetition_penalty(
                    scores[-1].clone(), sequences_cat, repetition_penalty
                )
                sampled_tokens = sample_tokens(logits, inference_params)
                sequences_cat = torch.cat([sequences_cat, sampled_tokens], dim=1)
            sequences.append(sampled_tokens)
            b2_send_start = torch.cuda.Event(enable_timing=True)
            b2_send_end = torch.cuda.Event(enable_timing=True)
            b2_send_start.record()
            block2_verified_token_queue.put(sampled_tokens)
            b2_send_end.record()
            torch.cuda.synchronize()
            print("queue sending time: ", b2_send_start.elapsed_time(b2_send_end))
            if streamer is not None:
                streamer.put(sampled_tokens.cpu())
        # a signal to terminate the subprocess
        block2_verified_token_queue.put([None, None])
        b1_sp.join()
        if streamer is not None:
            streamer.end()
        if enable_timing:
            end.record()
            torch.cuda.synchronize()
            print(f"Prompt processing + decoding time: {(start.elapsed_time(end)):.0f}ms")
        output_cls = GreedySearchDecoderOnlyOutput if top_k == 1 else SampleDecoderOnlyOutput
        output = output_cls(sequences=torch.cat(sequences, dim=1), scores=tuple(scores))
        if not output_scores:
            output.scores = None
        return output if return_dict_in_generate else output.sequences
    
