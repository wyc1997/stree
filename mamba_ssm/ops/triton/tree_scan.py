
import math
import torch
import torch.nn.functional as F

import triton
import triton.language as tl

from einops import rearrange, repeat

from mamba_ssm.ops.triton.softplus import softplus
from mamba_ssm.ops.triton.ssd_chunk_state import _chunk_state_fwd
from mamba_ssm.ops.triton.ssd_state_passing import _state_passing_fwd
from mamba_ssm.ops.triton.ssd_bmm import _bmm_chunk_fwd
from mamba_ssm.ops.triton.ssd_chunk_scan import _chunk_scan_fwd
from packaging import version

TRITON_22 = version.parse(triton.__version__) >= version.parse('2.2.0')

def init_to_zero(names):
    return lambda nargs: [nargs[name].zero_() for name in names if nargs[name] is not None]

def tree_cum_ref(A, dt, chunk_size, NUM_CHILD, dt_bias=None, dt_softplus=False):
    '''
    A: (nheads)
    dt: (batch, seqlen, nheads)
    Perform a tree scan a cross A and also multiple by dt, 
    assuming we have a complete tree, with each node having NUM_CHILD child node 
    The parent node of any node at index X have the index of (X - 1) / NUM_CHILD
    '''
    batch, seqlen, nheads = dt.shape
    if seqlen % chunk_size != 0:
        dt = F.pad(dt, (0, 0, 0, chunk_size - seqlen % chunk_size))
    dt = rearrange(dt, "b (c l) h -> b h c l", l=chunk_size)
    dt = dt.float()  # We want high precision for this before cumsum
    if dt_bias is not None:
        dt = dt + rearrange(dt_bias, "h -> h 1 1")
    if dt_softplus:
        dt = F.softplus(dt)

    dA = dt * rearrange(A, "h -> h 1 1")

    # chunk_index = torch.arange(dA.shape[-1])
    # parent_index = (chunk_index - 1) / NUM_CHILD
    
    tree_scan_dA = []
    # IS THERE A WAY TO PARALLELIZE A SCAN?
    for s in range(dA.shape[-1]):
        if s == 0:
            tree_scan_dA.append(dA[...,[0]])
        else:
            parent_index = int((s - 1) / NUM_CHILD)
            tree_scan_dA.append(tree_scan_dA[parent_index] + dA[...,[s]])

    return torch.cat(tree_scan_dA, dim=-1)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_H': 1}),
        triton.Config({'BLOCK_SIZE_H': 2}),
        triton.Config({'BLOCK_SIZE_H': 4}),
        triton.Config({'BLOCK_SIZE_H': 8}),
        triton.Config({'BLOCK_SIZE_H': 16}),
        triton.Config({'BLOCK_SIZE_H': 32}),
        triton.Config({'BLOCK_SIZE_H': 64}),
    ],
    key=['chunk_size', 'nheads'],
)
@triton.jit
def _chunk_cumsum_kernel(
    # Pointers to matrices
    dt_ptr, A_ptr, dt_bias_ptr, dt_out_ptr, dA_ts_ptr,
    # Matrix dimension
    batch, seqlen, nheads, chunk_size,
    dt_min, dt_max,
    # Strides
    stride_dt_batch, stride_dt_seqlen, stride_dt_head,
    stride_A_head,
    stride_dt_bias_head,
    stride_dt_out_batch, stride_dt_out_chunk, stride_dt_out_head, stride_dt_out_csize,
    stride_dA_cs_batch, stride_dA_cs_chunk, stride_dA_cs_head, stride_dA_cs_csize,
    # Meta-parameters
    NUM_CHILD: tl.constexpr,
    DT_SOFTPLUS: tl.constexpr,
    HAS_DT_BIAS: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_CHUNK: tl.constexpr,
):
    pid_b = tl.program_id(axis=0)
    pid_c = tl.program_id(axis=1)
    pid_h = tl.program_id(axis=2)
    dt_ptr += pid_b * stride_dt_batch + pid_c * chunk_size * stride_dt_seqlen
    dt_out_ptr += pid_b * stride_dt_out_batch + pid_c * stride_dt_out_chunk
    dA_ts_ptr += pid_b * stride_dA_cs_batch + pid_c * stride_dA_cs_chunk

    offs_h = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    offs_c = tl.arange(0, BLOCK_SIZE_CHUNK)
    dt_ptrs = dt_ptr + (offs_h[:, None] * stride_dt_head + offs_c[None, :] * stride_dt_seqlen)
    A_ptrs = A_ptr + offs_h * stride_A_head
    dt_out_ptrs = dt_out_ptr + (offs_h[:, None] * stride_dt_out_head + offs_c[None, :] * stride_dt_out_csize)
    dA_ts_ptrs = dA_ts_ptr + (offs_h[:, None] * stride_dA_cs_head + offs_c[None, :] * stride_dA_cs_csize)
    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)

    dt = tl.load(dt_ptrs, mask=(offs_h[:, None] < nheads) & (offs_c[None, :] < chunk_size_limit), other=0.0).to(tl.float32)
    if HAS_DT_BIAS:
        dt_bias = tl.load(dt_bias_ptr + offs_h * stride_dt_bias_head, mask=offs_h < nheads, other=0.0).to(tl.float32)
        dt += dt_bias[:, None]
    if DT_SOFTPLUS:
        dt = softplus(dt)
    # As of Triton 2.2.0, tl.clamp is not available yet
    # dt = tl.clamp(dt, dt_min, dt_max)
    dt = tl.minimum(tl.maximum(dt, dt_min), dt_max)
    dt = tl.where((offs_h[:, None] < nheads) & (offs_c[None, :] < chunk_size_limit), dt, 0.0)
    tl.store(dt_out_ptrs, dt, mask=(offs_h[:, None] < nheads) & (offs_c[None, :] < chunk_size))
    A = tl.load(A_ptrs, mask=offs_h < nheads, other=0.0).to(tl.float32)
    dA = dt * A[:, None]
    dA_cs = tl.cumsum(dA, axis=1)
    tl.store(dA_ts_ptrs, dA_cs, mask=(offs_h[:, None] < nheads) & (offs_c[None, :] < chunk_size))

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_H': 1}),
        triton.Config({'BLOCK_SIZE_H': 2}),
        triton.Config({'BLOCK_SIZE_H': 4}),
        triton.Config({'BLOCK_SIZE_H': 8}),
        triton.Config({'BLOCK_SIZE_H': 16}),
        triton.Config({'BLOCK_SIZE_H': 32}),
        triton.Config({'BLOCK_SIZE_H': 64}),
    ],
    key=['chunk_size', 'nheads'],
)
@triton.jit
def _tree_cum_kernel(
    # Pointers to matrices
    dt_ptr, A_ptr, dt_bias_ptr, dt_out_ptr, dA_ts_ptr, dA_mask_ptr,
    # Matrix dimension
    batch, seqlen, nheads, chunk_size,
    dt_min, dt_max,
    # Strides
    stride_dt_batch, stride_dt_seqlen, stride_dt_head,
    stride_A_head,
    stride_dt_bias_head,
    stride_dt_out_batch, stride_dt_out_chunk, stride_dt_out_head, stride_dt_out_csize,
    stride_dA_ts_batch, stride_dA_ts_chunk, stride_dA_ts_head, stride_dA_ts_csize,
    stride_dA_mask_batch, stride_dA_mask_chunk, stride_dA_mask_csize1, stride_dA_mask_csize2,
    # Meta-parameters
    NUM_CHILD: tl.constexpr,
    DT_SOFTPLUS: tl.constexpr,
    HAS_DT_BIAS: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_CHUNK: tl.constexpr,
):
    pid_b = tl.program_id(axis=0)
    pid_c = tl.program_id(axis=1)
    pid_h = tl.program_id(axis=2)
    dt_ptr += pid_b * stride_dt_batch + pid_c * chunk_size * stride_dt_seqlen
    dt_out_ptr += pid_b * stride_dt_out_batch + pid_c * stride_dt_out_chunk
    dA_ts_ptr += pid_b * stride_dA_ts_batch + pid_c * stride_dA_ts_chunk
    dA_mask_ptr += pid_b * stride_dA_mask_batch + pid_c * stride_dA_mask_chunk

    offs_h = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    offs_c = tl.arange(0, BLOCK_SIZE_CHUNK)
    dt_ptrs = dt_ptr + (offs_h * stride_dt_head)
    A_ptrs = A_ptr + offs_h * stride_A_head
    dt_out_ptrs = dt_out_ptr + (offs_h * stride_dt_out_head)
    dA_ts_ptrs = dA_ts_ptr + (offs_h * stride_dA_ts_head)
    dA_mask_ptrs = dA_mask_ptr
    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)
    A = tl.load(A_ptrs, mask=offs_h < nheads, other=0.0).to(tl.float32)

    dt = tl.load(dt_ptrs, mask=(offs_h < nheads), other=0.0).to(tl.float32)

    if HAS_DT_BIAS:
        dt_bias = tl.load(dt_bias_ptr + offs_h * stride_dt_bias_head, mask=offs_h < nheads, other=0.0).to(tl.float32)
        dt += dt_bias
    if DT_SOFTPLUS:
        dt = softplus(dt)
    # As of Triton 2.2.0, tl.clamp is not available yet
    # dt = tl.clamp(dt, dt_min, dt_max)
    dt = tl.minimum(tl.maximum(dt, dt_min), dt_max)
    dt = tl.where((offs_h < nheads), dt, 0.0)
    tl.store(dt_out_ptrs, dt, mask=(offs_h < nheads))
    dA_ts = dt * A
    tl.store(dA_ts_ptrs, dA_ts, mask=(offs_h < nheads))

    tl.store(dA_mask_ptrs, 1)

    for s in range(1, chunk_size):
        curr_dt_ptrs = dt_ptrs + s * stride_dt_seqlen
        curr_dt_out_ptrs = dt_out_ptrs + s * stride_dt_out_csize
        curr_dA_ts_ptrs = dA_ts_ptrs + s * stride_dA_ts_csize
        curr_dA_mask_ptrs = dA_mask_ptrs + s * stride_dA_mask_csize1 + tl.arange(0, BLOCK_SIZE_CHUNK) * stride_dA_mask_csize2
        curr_dA_mask_idt = dA_mask_ptrs + s * stride_dA_mask_csize1 + s * stride_dA_mask_csize2
        
        
        parent_idx = (s - 1) // NUM_CHILD
        parent_dA_ts_ptrs = dA_ts_ptrs + parent_idx * stride_dA_ts_csize
        parent_dA_mask_ptrs = dA_mask_ptrs + parent_idx * stride_dA_mask_csize1 + tl.arange(0, BLOCK_SIZE_CHUNK) * stride_dA_mask_csize2

        parent_dA_ts = tl.load(parent_dA_ts_ptrs, mask=(offs_h < nheads), other=0.0)
        dt = tl.load(curr_dt_ptrs, mask=(offs_h < nheads), other=0.0).to(tl.float32)

        # getting and updating masks 
        parent_dA_mask = tl.load(parent_dA_mask_ptrs)
        tl.store(curr_dA_mask_ptrs, parent_dA_mask)
        tl.store(curr_dA_mask_idt, 1)

        if HAS_DT_BIAS:
            dt_bias = tl.load(dt_bias_ptr + offs_h * stride_dt_bias_head, mask=offs_h < nheads, other=0.0).to(tl.float32)
            dt += dt_bias
        if DT_SOFTPLUS:
            dt = softplus(dt)
        # As of Triton 2.2.0, tl.clamp is not available yet
        # dt = tl.clamp(dt, dt_min, dt_max)
        dt = tl.minimum(tl.maximum(dt, dt_min), dt_max)
        dt = tl.where((offs_h < nheads), dt, 0.0)
        tl.store(curr_dt_out_ptrs, dt, mask=(offs_h < nheads))
        dA = dt * A
        dA_ts = dA + parent_dA_ts
        tl.store(curr_dA_ts_ptrs, dA_ts, mask=(offs_h < nheads))

def tree_cum(dt, A, chunk_size, dt_bias=None, dt_softplus=False, dt_limit=(0.0, float("inf")), num_child=2):
    batch, seqlen, nheads = dt.shape
    assert A.shape == (nheads,)
    assert seqlen < chunk_size # testing for 1 chunk only
    if dt_bias is not None:
        assert dt_bias.shape == (nheads,)
    nchunks = math.ceil(seqlen / chunk_size)
    dt_out = torch.empty(batch, nheads, nchunks, chunk_size, device=dt.device, dtype=torch.float32)
    dA_ts = torch.empty(batch, nheads, nchunks, chunk_size, device=dt.device, dtype=torch.float32)
    dA_mask = torch.zeros((batch, nchunks, chunk_size, chunk_size), device=dt.device, dtype=torch.float32)
    grid_chunk_cs = lambda META: (batch, nchunks, triton.cdiv(nheads, META['BLOCK_SIZE_H']))
    with torch.cuda.device(dt.device.index):
        _tree_cum_kernel[grid_chunk_cs](
            dt, A, dt_bias, dt_out, dA_ts, dA_mask, 
            batch, seqlen, nheads, chunk_size,
            dt_limit[0], dt_limit[1],
            dt.stride(0), dt.stride(1), dt.stride(2),
            A.stride(0),
            dt_bias.stride(0) if dt_bias is not None else 0,
            dt_out.stride(0), dt_out.stride(2), dt_out.stride(1), dt_out.stride(3),
            dA_ts.stride(0), dA_ts.stride(2), dA_ts.stride(1), dA_ts.stride(3),
            dA_mask.stride(0), dA_mask.stride(1), dA_mask.stride(2), dA_mask.stride(3),
            num_child,
            dt_softplus,
            HAS_DT_BIAS=dt_bias is not None,
            BLOCK_SIZE_CHUNK=triton.next_power_of_2(chunk_size),
        )
    return dA_ts, dt_out, dA_mask

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_H': 1}),
        triton.Config({'BLOCK_SIZE_H': 2}),
        triton.Config({'BLOCK_SIZE_H': 4}),
        triton.Config({'BLOCK_SIZE_H': 8}),
        triton.Config({'BLOCK_SIZE_H': 16}),
        triton.Config({'BLOCK_SIZE_H': 32}),
        triton.Config({'BLOCK_SIZE_H': 64}),
    ],
    key=['chunk_size', 'nheads'],
)
@triton.jit
def _tree_cum_mask_kernel(
    # Pointers to matrices
    dt_ptr, A_ptr, dt_bias_ptr, dt_out_ptr, dA_ts_ptr, dA_mask_ptr, dA_mask_out_ptr,
    # Matrix dimension
    batch, seqlen, nheads, chunk_size,
    dt_min, dt_max,
    # Strides
    stride_dt_batch, stride_dt_seqlen, stride_dt_head,
    stride_A_head,
    stride_dt_bias_head,
    stride_dt_out_batch, stride_dt_out_chunk, stride_dt_out_head, stride_dt_out_csize,
    stride_dA_ts_batch, stride_dA_ts_chunk, stride_dA_ts_head, stride_dA_ts_csize,
    stride_dA_mask_batch, stride_dA_mask_csize1, stride_dA_mask_csize2,
    stride_dA_mask_out_batch, strde_dA_mask_out_chunk, stride_dA_mask_out_csize1, stride_dA_mask_out_csize2,
    # Meta-parameters
    DT_SOFTPLUS: tl.constexpr,
    HAS_DT_BIAS: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_CHUNK: tl.constexpr,
):
    # assuming that there is only one chunk
    pid_b = tl.program_id(axis=0)
    pid_l = tl.program_id(axis=1)
    pid_h = tl.program_id(axis=2)
    dt_ptr += pid_b * stride_dt_batch
    dt_out_ptr += pid_b * stride_dt_out_batch 
    dA_ts_ptr += pid_b * stride_dA_ts_batch + pid_l * stride_dA_ts_csize
    dA_mask_ptr += pid_b * stride_dA_mask_batch + pid_l * stride_dA_mask_csize1
    dA_mask_out_ptr += pid_b * stride_dA_mask_out_batch + pid_l * stride_dA_mask_out_csize1

    offs_h = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    offs_l = tl.arange(0, BLOCK_SIZE_CHUNK)
    # offs_c = tl.arange(0, BLOCK_SIZE_CHUNK)
    dt_ptrs = dt_ptr + (offs_h[None, :] * stride_dt_head) + (offs_l[:, None] * stride_dt_seqlen)
    A_ptrs = A_ptr + offs_h[None, :] * stride_A_head
    dt_out_ptrs = dt_out_ptr + (offs_h[None, :] * stride_dt_out_head) + (offs_l[:, None] * stride_dt_out_csize)
    dA_ts_ptrs = dA_ts_ptr + (offs_h[None, :] * stride_dA_ts_head)
    dA_mask_ptrs = dA_mask_ptr + offs_l[:, None] * stride_dA_mask_csize2
    dA_mask_out_ptrs = dA_mask_out_ptr + offs_l[:, None] * stride_dA_mask_out_csize2
    # chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)
    A = tl.load(A_ptrs, mask=(offs_h[None,:] < nheads), other=0.0).to(tl.float32) # 1, nhead 

    dt = tl.load(dt_ptrs, mask=(offs_h[None,:] < nheads) & (offs_l[:,None] < seqlen), other=0.0).to(tl.float32) # seqlen, nhead

    dA_mask = tl.load(dA_mask_ptrs, mask=(pid_l < seqlen) & (offs_l[:,None] < seqlen), other=0.0).to(tl.int1) # seqlen, 1
    tl.store(dA_mask_out_ptrs, dA_mask, mask=(offs_l[:,None] < seqlen))

    if HAS_DT_BIAS:
        dt_bias = tl.load(dt_bias_ptr + offs_h * stride_dt_bias_head, mask=offs_h < nheads, other=0.0).to(tl.float32)
        dt += dt_bias
    if DT_SOFTPLUS:
        dt = softplus(dt)
    # As of Triton 2.2.0, tl.clamp is not available yet
    # dt = tl.clamp(dt, dt_min, dt_max)
    dt = tl.minimum(tl.maximum(dt, dt_min), dt_max)
    dt = tl.where((offs_h[None,:] < nheads), dt, 0.0)
    # dt_sum = tl.sum(dt, axis=0, keep_dims=True) # 1, nhead
    tl.store(dt_out_ptrs, dt, mask=((offs_h[None,:] < nheads) & (offs_l[:,None] == pid_l) & (offs_l[:,None] < seqlen)))
    dA_ts = tl.where(dA_mask, dt * A, 0.0)
    dA_ts_sum = tl.sum(dA_ts, axis=0, keep_dims=True)
    tl.store(dA_ts_ptrs, dA_ts_sum, mask=(offs_h[None,:] < nheads))

def tree_cum_mask(dt, A, dA_mask, chunk_size, dt_bias=None, dt_softplus=False, dt_limit=(0.0, float("inf"))):
    batch, seqlen, nheads = dt.shape
    assert A.shape == (nheads,)
    assert seqlen < chunk_size # testing for 1 chunk only
    if dt_bias is not None:
        assert dt_bias.shape == (nheads,)
    nchunks = math.ceil(seqlen / chunk_size)
    dt_out = torch.zeros(batch, nheads, nchunks, chunk_size, device=dt.device, dtype=torch.float32)
    dA_ts = torch.zeros(batch, nheads, nchunks, chunk_size, device=dt.device, dtype=torch.float32)
    dA_mask_out = torch.zeros(batch, nchunks, chunk_size, chunk_size, device=dt.device, dtype=torch.long)
    grid_chunk_cs = lambda META: (batch, chunk_size, triton.cdiv(nheads, META['BLOCK_SIZE_H']))
    with torch.cuda.device(dt.device.index):
        _tree_cum_mask_kernel[grid_chunk_cs](
            dt, A, dt_bias, dt_out, dA_ts, dA_mask, dA_mask_out,
            batch, seqlen, nheads, chunk_size,
            dt_limit[0], dt_limit[1],
            dt.stride(0), dt.stride(1), dt.stride(2),
            A.stride(0),
            dt_bias.stride(0) if dt_bias is not None else 0,
            dt_out.stride(0), dt_out.stride(2), dt_out.stride(1), dt_out.stride(3),
            dA_ts.stride(0), dA_ts.stride(2), dA_ts.stride(1), dA_ts.stride(3),
            dA_mask.stride(0), dA_mask.stride(1), dA_mask.stride(2),
            dA_mask_out.stride(0), dA_mask_out.stride(1), dA_mask_out.stride(2), dA_mask_out.stride(3),
            dt_softplus,
            HAS_DT_BIAS=dt_bias is not None,
            BLOCK_SIZE_CHUNK=triton.next_power_of_2(chunk_size),
        )
    return dA_ts, dt_out, dA_mask_out



def tree_scan_combined_fwd(x, dt, A, B, C, chunk_size, D=None, z=None, dt_bias=None, initial_states=None, seq_idx=None, dt_softplus=False, dt_limit=(0.0, float("inf")), num_child=2):
    batch, seqlen, nheads, headdim = x.shape
    _, _, ngroups, dstate = B.shape
    assert nheads % ngroups == 0
    assert B.shape == (batch, seqlen, ngroups, dstate)
    assert x.shape == (batch, seqlen, nheads, headdim)
    assert dt.shape == (batch, seqlen, nheads)
    assert A.shape == (nheads,)
    assert C.shape == B.shape
    assert seqlen < chunk_size # assuming that we only have 1 chunk
    if z is not None:
        assert z.shape == x.shape
    if D is not None:
        assert D.shape == (nheads, headdim) or D.shape == (nheads,)
    if seq_idx is not None:
        assert seq_idx.shape == (batch, seqlen)
    if B.stride(-1) != 1:
        B = B.contiguous()
    if C.stride(-1) != 1:
        C = C.contiguous()
    if x.stride(-1) != 1 and x.stride(1) != 1:  # Either M or K dimension should be contiguous
        x = x.contiguous()
    if z is not None and z.stride(-1) != 1 and z.stride(1) != 1:  # Either M or K dimension should be contiguous
        z = z.contiguous()
    if D is not None and D.stride(-1) != 1:
        D = D.contiguous()
    if initial_states is not None:
        assert initial_states.shape == (batch, nheads, headdim, dstate)
    else:
        initial_states = torch.zeros((batch, 1, nheads, headdim, dstate), device=B.device, dtype=B.dtype)
    # # (batch, nchunks, chunk_size, chunk_size) or (batch, nchunks, nheads, chunk_size, chunk_size)
    # dA_cumsum_tmp0, dt_tmp0 = _chunk_cumsum_fwd(dt[:, :147], A, chunk_size, dt_bias=dt_bias, dt_softplus=dt_softplus)
    # dA_cumsum_tmp1, dt_tmp1 = _chunk_cumsum_fwd(dt[:, 147:], A, chunk_size, dt_bias=dt_bias, dt_softplus=dt_softplus)
    # dA_cumsum_tmp2, dt_tmp2 = _chunk_cumsum_fwd(dt[:, 147:256], A, chunk_size, dt_bias=dt_bias, dt_softplus=dt_softplus)
    dA_cumsum, dt, dA_mask = tree_cum(dt, A, chunk_size, dt_bias=dt_bias, dt_softplus=dt_softplus, dt_limit=dt_limit, num_child=num_child)
    # The chunk_state_forward here computes the end state of every chunk, assuming each chunk starts
    # a zero-state 
    # states = _chunk_state_fwd(B, x, dt, dA_cumsum, seq_idx=seq_idx, states_in_fp32=True)
    # states_tmp0 = _chunk_state_fwd(B[:, :147], x[:, :147], dt_tmp0, dA_cumsum_tmp0, states_in_fp32=True)
    # states_tmp1 = _chunk_state_fwd(B[:, 147:], x[:, 147:], dt_tmp1, dA_cumsum_tmp1, states_in_fp32=True)
    # states_tmp2 = _chunk_state_fwd(B[:, 147:256], x[:, 147:256], dt_tmp2, dA_cumsum_tmp2, states_in_fp32=True)
    # state passing takes the end state of every chunk and passes it to the state of the next chunk
    # After this state of each chunk is the state up until the end of that chunk
    # states, final_states = _state_passing_fwd(rearrange(states, "... p n -> ... (p n)"), dA_cumsum[:, :, :, -1],
                                            #   initial_states=rearrange(initial_states, "... p n -> ... (p n)") if initial_states is not None else None,
                                            #   seq_idx=seq_idx, chunk_size=chunk_size, out_dtype=C.dtype)
    # states, final_states = [rearrange(t, "... (p n) -> ... p n", n=dstate) for t in [states, final_states]]
    # states_tmp0 = rearrange(_state_passing_fwd(rearrange(states_tmp0, "... p n -> ... (p n)"), dA_cumsum_tmp0[:, :, :, -1], chunk_size=chunk_size), "... (p n) -> ... p n", n=dstate)
    # states_tmp1 = rearrange(_state_passing_fwd(rearrange(states_tmp1, "... p n -> ... (p n)"), dA_cumsum_tmp1[:, :, :, -1], chunk_size=chunk_size), "... (p n) -> ... p n", n=dstate)
    # Creates a matrix of C and B where M_ij = Ci * Bj
    CB = _bmm_chunk_fwd(C, B, chunk_size, seq_idx=seq_idx, output_dtype=torch.float32)
    # Since we have only one chunk, chunk state fwd and state passing shouldn't do anything and we don't
    # have to consider the output states from there 
    out, out_x = _tree_scan_fwd(CB, x, dt, dA_cumsum, C, initial_states, dA_mask, D=D, z=z, seq_idx=seq_idx)
    return out, out_x, dt, dA_cumsum

def tree_scan_mask_combined_fwd(x, dt, A, B, C, dA_mask, chunk_size, D=None, z=None, dt_bias=None, initial_states=None, seq_idx=None, dt_softplus=False, dt_limit=(0.0, float("inf"))):
    batch, seqlen, nheads, headdim = x.shape
    _, _, ngroups, dstate = B.shape
    assert nheads % ngroups == 0
    assert B.shape == (batch, seqlen, ngroups, dstate)
    assert x.shape == (batch, seqlen, nheads, headdim)
    assert dt.shape == (batch, seqlen, nheads)
    assert A.shape == (nheads,)
    assert C.shape == B.shape
    assert dA_mask.shape == (batch, seqlen, seqlen)
    assert seqlen <= chunk_size # assuming that we only have 1 chunk
    if z is not None:
        assert z.shape == x.shape
    if D is not None:
        assert D.shape == (nheads, headdim) or D.shape == (nheads,)
    if seq_idx is not None:
        assert seq_idx.shape == (batch, seqlen)
    if B.stride(-1) != 1:
        B = B.contiguous()
    if C.stride(-1) != 1:
        C = C.contiguous()
    if x.stride(-1) != 1 and x.stride(1) != 1:  # Either M or K dimension should be contiguous
        x = x.contiguous()
    if z is not None and z.stride(-1) != 1 and z.stride(1) != 1:  # Either M or K dimension should be contiguous
        z = z.contiguous()
    if D is not None and D.stride(-1) != 1:
        D = D.contiguous()
    if initial_states is not None:
        assert initial_states.shape == (batch, nheads, headdim, dstate)
        
    else:
        initial_states = torch.zeros((batch, 1, nheads, headdim, dstate), device=B.device, dtype=B.dtype)
    # # (batch, nchunks, chunk_size, chunk_size) or (batch, nchunks, nheads, chunk_size, chunk_size)
    # dA_cumsum_tmp0, dt_tmp0 = _chunk_cumsum_fwd(dt[:, :147], A, chunk_size, dt_bias=dt_bias, dt_softplus=dt_softplus)
    # dA_cumsum_tmp1, dt_tmp1 = _chunk_cumsum_fwd(dt[:, 147:], A, chunk_size, dt_bias=dt_bias, dt_softplus=dt_softplus)
    # dA_cumsum_tmp2, dt_tmp2 = _chunk_cumsum_fwd(dt[:, 147:256], A, chunk_size, dt_bias=dt_bias, dt_softplus=dt_softplus)
    dA_cumsum, dt, dA_mask_out = tree_cum_mask(dt, A, dA_mask, chunk_size, dt_bias=dt_bias, dt_softplus=dt_softplus, dt_limit=dt_limit)
    # The chunk_state_forward here computes the end state of every chunk, assuming each chunk starts
    # a zero-state 
    # states = _chunk_state_fwd(B, x, dt, dA_cumsum, seq_idx=seq_idx, states_in_fp32=True)
    # states_tmp0 = _chunk_state_fwd(B[:, :147], x[:, :147], dt_tmp0, dA_cumsum_tmp0, states_in_fp32=True)
    # states_tmp1 = _chunk_state_fwd(B[:, 147:], x[:, 147:], dt_tmp1, dA_cumsum_tmp1, states_in_fp32=True)
    # states_tmp2 = _chunk_state_fwd(B[:, 147:256], x[:, 147:256], dt_tmp2, dA_cumsum_tmp2, states_in_fp32=True)
    # state passing takes the end state of every chunk and passes it to the state of the next chunk
    # After this state of each chunk is the state up until the end of that chunk
    # states, final_states = _state_passing_fwd(rearrange(states, "... p n -> ... (p n)"), dA_cumsum[:, :, :, -1],
                                            #   initial_states=rearrange(initial_states, "... p n -> ... (p n)") if initial_states is not None else None,
                                            #   seq_idx=seq_idx, chunk_size=chunk_size, out_dtype=C.dtype)
    # states, final_states = [rearrange(t, "... (p n) -> ... p n", n=dstate) for t in [states, final_states]]
    # states_tmp0 = rearrange(_state_passing_fwd(rearrange(states_tmp0, "... p n -> ... (p n)"), dA_cumsum_tmp0[:, :, :, -1], chunk_size=chunk_size), "... (p n) -> ... p n", n=dstate)
    # states_tmp1 = rearrange(_state_passing_fwd(rearrange(states_tmp1, "... p n -> ... (p n)"), dA_cumsum_tmp1[:, :, :, -1], chunk_size=chunk_size), "... (p n) -> ... p n", n=dstate)
    # Creates a matrix of C and B where M_ij = Ci * Bj
    CB = _bmm_chunk_fwd(C, B, chunk_size, seq_idx=seq_idx, output_dtype=torch.float32)
    # Since we have only one chunk, chunk state fwd and state passing shouldn't do anything and we don't
    # have to consider the output states from there 
    out, out_x = _tree_scan_fwd(CB, x, dt, dA_cumsum, C, initial_states.unsqueeze(1), dA_mask_out, D=D, z=z, seq_idx=seq_idx)
    return out, out_x, dt, dA_cumsum

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=2),
    ],
    key=['chunk_size', 'hdim', 'dstate'],
)
@triton.jit
def _tree_scan_fwd_kernel(
    # Pointers to matrices
    cb_ptr, x_ptr, z_ptr, out_ptr, out_x_ptr, dt_ptr, dA_cumsum_ptr, seq_idx_ptr, C_ptr, prev_states_ptr, D_ptr, tree_mask_ptr,
    # Matrix dimensions
    chunk_size, hdim, dstate,
    batch, seqlen, nheads_ngroups_ratio,
    # Strides
    stride_cb_batch, stride_cb_chunk, stride_cb_head, stride_cb_csize_m, stride_cb_csize_k,
    stride_x_batch, stride_x_seqlen, stride_x_head, stride_x_hdim,
    stride_z_batch, stride_z_seqlen, stride_z_head, stride_z_hdim,
    stride_out_batch, stride_out_seqlen, stride_out_head, stride_out_hdim,
    stride_dt_batch, stride_dt_chunk, stride_dt_head, stride_dt_csize,
    stride_dA_cs_batch, stride_dA_cs_chunk, stride_dA_cs_head, stride_dA_cs_csize,
    stride_seq_idx_batch, stride_seq_idx_seqlen,
    stride_C_batch, stride_C_seqlen, stride_C_head, stride_C_dstate,
    stride_states_batch, stride_states_chunk, stride_states_head, stride_states_hdim, stride_states_dstate,
    stride_D_head,
    stride_tm_batch, stride_tm_chunk, stride_tm_csize1, stride_tm_csize2,
    # Meta-parameters
    HAS_D: tl.constexpr,
    D_HAS_HDIM: tl.constexpr,
    HAS_Z: tl.constexpr,
    HAS_SEQ_IDX: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_DSTATE: tl.constexpr,
    IS_TRITON_22: tl.constexpr,
):
    pid_bc = tl.program_id(axis=1)
    pid_c = pid_bc // batch
    pid_b = pid_bc - pid_c * batch
    pid_h = tl.program_id(axis=2)
    num_pid_n = tl.cdiv(hdim, BLOCK_SIZE_N)
    pid_m = tl.program_id(axis=0) // num_pid_n
    pid_n = tl.program_id(axis=0) % num_pid_n
    cb_ptr += pid_b * stride_cb_batch + pid_c * stride_cb_chunk + (pid_h // nheads_ngroups_ratio) * stride_cb_head
    x_ptr += pid_b * stride_x_batch + pid_c * chunk_size * stride_x_seqlen + pid_h * stride_x_head
    dt_ptr += pid_b * stride_dt_batch + pid_c * stride_dt_chunk + pid_h * stride_dt_head
    dA_cumsum_ptr += pid_b * stride_dA_cs_batch + pid_c * stride_dA_cs_chunk + pid_h * stride_dA_cs_head
    C_ptr += pid_b * stride_C_batch + pid_c * chunk_size * stride_C_seqlen + (pid_h // nheads_ngroups_ratio) * stride_C_head
    prev_states_ptr += pid_b * stride_states_batch + pid_c * stride_states_chunk + pid_h * stride_states_head
    tree_mask_ptr += pid_b * stride_tm_batch + pid_c * stride_tm_chunk
    if HAS_SEQ_IDX:
        seq_idx_ptr += pid_b * stride_seq_idx_batch + pid_c * chunk_size * stride_seq_idx_seqlen

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    dA_cs_m = tl.load(dA_cumsum_ptr + offs_m * stride_dA_cs_csize, mask=offs_m < chunk_size, other=0.0).to(tl.float32)

    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)
    if HAS_SEQ_IDX:
        seq_idx_prev = tl.load(seq_idx_ptr - stride_seq_idx_seqlen, mask=pid_c >= 1, other=0)
        seq_idx_m = tl.load(seq_idx_ptr + offs_m * stride_seq_idx_seqlen, mask=offs_m < chunk_size_limit, other=-1)
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Without the if (pid_c > -1), with Triton 2.1.0, I get
    # Assertion `!(srcMmaLayout && dstMmaLayout) && "Unexpected mma -> mm a layout conversion"' failed.
    # With Triton 2.2.0, this works
    if IS_TRITON_22 or pid_c > -1:
        # Faster to just do 1 iteration with larger BLOCK_SIZE_K, up to block size 128
        offs_k_dstate = tl.arange(0, BLOCK_SIZE_DSTATE if BLOCK_SIZE_DSTATE <= 128 else BLOCK_SIZE_K)
        C_ptrs = C_ptr + (offs_m[:, None] * stride_C_seqlen + offs_k_dstate[None, :] * stride_C_dstate)
        prev_states_ptrs = prev_states_ptr + (offs_n[None, :] * stride_states_hdim + offs_k_dstate[:, None] * stride_states_dstate)
        if not HAS_SEQ_IDX:
            scale_m = tl.exp(dA_cs_m)
        else:
            scale_m = tl.where(seq_idx_m == seq_idx_prev, tl.exp(dA_cs_m), 0.0)
        if BLOCK_SIZE_DSTATE <= 128:
            C = tl.load(C_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_k_dstate[None, :] < dstate), other=0.0)
            prev_states = tl.load(prev_states_ptrs, mask=(offs_k_dstate[:, None] < dstate) & (offs_n[None, :] < hdim), other=0.0)
            prev_states = prev_states.to(C_ptr.dtype.element_ty)
            acc = tl.dot(C, prev_states) * scale_m[:, None]
        else:
            for k in range(0, dstate, BLOCK_SIZE_K):
                C = tl.load(C_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_k_dstate[None, :] < dstate - k), other=0.0)
                # C = (C * scale_m[:, None]).to(C_ptr.dtype.element_ty)
                prev_states = tl.load(prev_states_ptrs, mask=(offs_k_dstate[:, None] < dstate - k) & (offs_n[None, :] < hdim), other=0.0)
                prev_states = prev_states.to(C_ptr.dtype.element_ty)
                acc += tl.dot(C, prev_states)
                C_ptrs += BLOCK_SIZE_K
                prev_states_ptrs += BLOCK_SIZE_K
            acc *= scale_m[:, None]

    offs_k = tl.arange(0, BLOCK_SIZE_K)
    cb_ptrs = cb_ptr + (offs_m[:, None] * stride_cb_csize_m + offs_k[None, :] * stride_cb_csize_k)
    x_ptrs = x_ptr + (offs_k[:, None] * stride_x_seqlen + offs_n[None, :] * stride_x_hdim)
    dt_ptrs = dt_ptr + offs_k * stride_dt_csize
    dA_cumsum_ptrs = dA_cumsum_ptr + offs_k * stride_dA_cs_csize
    K_MAX = min((pid_m + 1) * BLOCK_SIZE_M, chunk_size_limit)
    tree_mask_ptrs = tree_mask_ptr + (offs_m[:, None] * stride_tm_csize1 + offs_k[None, :] * stride_tm_csize2)

    # We don't need to iterate K since tree scan sequence length is probably short. 
    for k in range(0, K_MAX, BLOCK_SIZE_K):
        cb = tl.load(cb_ptrs, mask=(offs_m[:, None] < chunk_size) & (offs_k[None, :] < chunk_size - k), other=0.0).to(tl.float32)
        dA_cs_k = tl.load(dA_cumsum_ptrs, mask=offs_k < chunk_size - k, other=0.0).to(tl.float32)
        # If there's seq_idx, we already set cb[i, j] = 0 for seq_idx[i] != seq_idx[j].
        # So we don't need masking wrt seq_idx here.
        cb *= tl.exp((dA_cs_m[:, None] - dA_cs_k[None, :]))
        dt_k = tl.load(dt_ptrs, mask=offs_k < chunk_size - k, other=0.0).to(tl.float32)
        cb *= dt_k

        # apply tree masking here
        tree_mask = tl.load(tree_mask_ptrs, mask=(offs_m[:, None] < chunk_size) & (offs_k[None, :] < chunk_size - k), other=0.0).to(tl.int1)
        cb = tl.where(tree_mask, cb, 0)

        cb = cb.to(x_ptr.dtype.element_ty)
        x = tl.load(x_ptrs, mask=(offs_k[:, None] < chunk_size_limit - k) & (offs_n[None, :] < hdim), other=0.0)
        acc += tl.dot(cb, x)
        cb_ptrs += BLOCK_SIZE_K * stride_cb_csize_k
        x_ptrs += BLOCK_SIZE_K * stride_x_seqlen
        dt_ptrs += BLOCK_SIZE_K * stride_dt_csize
        dA_cumsum_ptrs += BLOCK_SIZE_K * stride_dA_cs_csize
        tree_mask_ptrs += BLOCK_SIZE_K * stride_tm_csize2

    offs_out_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_out_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    if HAS_D:
        if D_HAS_HDIM:
            D = tl.load(D_ptr + pid_h * stride_D_head + offs_n, mask=offs_n < hdim, other=0.0).to(tl.float32)
        else:
            D = tl.load(D_ptr + pid_h * stride_D_head).to(tl.float32)
        x_residual = tl.load(x_ptr + (offs_m[:, None] * stride_x_seqlen + offs_n[None, :] * stride_x_hdim),
                             mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < hdim), other=0.0).to(tl.float32)
        acc += x_residual * D

    if HAS_Z:
        out_x_ptr += pid_b * stride_out_batch + pid_c * chunk_size * stride_out_seqlen + pid_h * stride_out_head
        out_x_ptrs = out_x_ptr + (stride_out_seqlen * offs_out_m[:, None] + offs_out_n[None, :])
        tl.store(out_x_ptrs, acc, mask=(offs_out_m[:, None] < chunk_size_limit) & (offs_out_n[None, :] < hdim))

        z_ptr += pid_b * stride_z_batch + pid_c * chunk_size * stride_z_seqlen + pid_h * stride_z_head
        z_ptrs = z_ptr + (stride_z_seqlen * offs_out_m[:, None] + stride_z_hdim * offs_out_n[None, :])
        z = tl.load(z_ptrs, mask=(offs_out_m[:, None] < chunk_size_limit) & (offs_out_n[None, :] < hdim), other=0.0).to(tl.float32)
        acc *= z * tl.sigmoid(z)

    out_ptr += pid_b * stride_out_batch + pid_c * chunk_size * stride_out_seqlen + pid_h * stride_out_head
    out_ptrs = out_ptr + (stride_out_seqlen * offs_out_m[:, None] + offs_out_n[None, :] * stride_out_hdim)
    tl.store(out_ptrs, acc, mask=(offs_out_m[:, None] < chunk_size_limit) & (offs_out_n[None, :] < hdim))

def _tree_scan_fwd(cb, x, dt, dA_cumsum, C, states, mask, D=None, z=None, seq_idx=None):
    batch, seqlen, nheads, headdim = x.shape
    _, _, nchunks, chunk_size = dt.shape
    _, _, ngroups, dstate = C.shape
    assert nheads % ngroups == 0
    assert C.shape == (batch, seqlen, ngroups, dstate)
    assert cb.shape == (batch, nchunks, ngroups, chunk_size, chunk_size)
    if z is not None:
        assert z.shape == x.shape
    if D is not None:
        assert D.shape == (nheads, headdim) or D.shape == (nheads,)
    assert dt.shape == (batch, nheads, nchunks, chunk_size)
    assert dA_cumsum.shape == (batch, nheads, nchunks, chunk_size)
    assert states.shape == (batch, nchunks, nheads, headdim, dstate)
    assert mask.shape == (batch, nchunks, chunk_size, chunk_size)
    if seq_idx is not None:
        assert seq_idx.shape == (batch, seqlen)
    # Allocates output.
    out = torch.empty(batch, seqlen, nheads, headdim, device=x.device, dtype=x.dtype)
    if z is not None:
        out_x = torch.empty(batch, seqlen, nheads, headdim, device=x.device, dtype=x.dtype)
        assert out_x.stride() == out.stride()
    else:
        out_x = None
    grid = lambda META: (triton.cdiv(chunk_size, META['BLOCK_SIZE_M']) * triton.cdiv(headdim, META['BLOCK_SIZE_N']),
                    batch * nchunks, nheads)
    z_strides = ((z.stride(0), z.stride(1), z.stride(2), z.stride(3))
                  if z is not None else (0, 0, 0, 0))
    _tree_scan_fwd_kernel[grid](
        cb, x, z, out, out_x, dt, dA_cumsum, seq_idx, C, states, D, mask,
        chunk_size, headdim, dstate,
        batch, seqlen, nheads // ngroups,
        cb.stride(0), cb.stride(1), cb.stride(2), cb.stride(3), cb.stride(4),
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        z_strides[0], z_strides[1], z_strides[2], z_strides[3],
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        dt.stride(0), dt.stride(2), dt.stride(1), dt.stride(3),
        dA_cumsum.stride(0), dA_cumsum.stride(2), dA_cumsum.stride(1), dA_cumsum.stride(3),
        *((seq_idx.stride(0), seq_idx.stride(1)) if seq_idx is not None else (0, 0)),
        C.stride(0), C.stride(1), C.stride(2), C.stride(3),
        states.stride(0), states.stride(1), states.stride(2), states.stride(3), states.stride(4),
        D.stride(0) if D is not None else 0,
        mask.stride(0), mask.stride(1), mask.stride(2), mask.stride(3),
        D is not None,
        D.dim() == 2 if D is not None else True,
        BLOCK_SIZE_DSTATE=max(triton.next_power_of_2(dstate), 16),
        HAS_Z=z is not None,
        HAS_SEQ_IDX=seq_idx is not None,
        IS_TRITON_22=TRITON_22,
    )
    return out, out_x

def tree_conv1d_ref(x: torch.Tensor, state: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, activation: str, num_child: int, depth: int):
    '''
        x: (batch, seqlen, dim)
        state: (batch, dim, state_len)
        weight: (dim, state_len)
        bias: (dim,)
        num_child: number of children per node 
        depth: depth of tree, a tree with only root node is of depth 1
        Performs a conv1d on the token tree respecting the sequences in the token tree. 
        Does NOT update the state since ther can be multiple copies of state since there are multiple sequences in x 
    '''
    batch, seqlen, dim = x.shape
    _, _, state_len = state.shape
    num_leaf = num_child ** (depth - 1)
    leaf_indices = torch.arange(seqlen - num_leaf, seqlen, 1).to(x.device)
    parent_matrix = [leaf_indices]
    for _ in range(depth-1):
        parent_matrix.insert(0, ((parent_matrix[0] - 1)/num_child).to(torch.int64))
    
    parent_matrix = torch.stack(parent_matrix, dim=1) # num_leaf, depth
    x_t = x.permute(1, 0, 2) # seqlen, batch, dim
    x_t = x_t.reshape(seqlen, batch * dim)
    selected_x_t = F.embedding(parent_matrix, x_t) # num_leaf, depth, batch*dim
    selected_x = selected_x_t.view(num_leaf, depth, batch, dim).permute(2, 0, 3, 1) # batch, num_leaf, dim, depth
    new_state = torch.cat([repeat(state, 'b ... -> (b r) ...', r=num_leaf), selected_x.reshape(batch*num_leaf, dim, depth)], dim=-1)

    conv_out = F.conv1d(new_state, weight.unsqueeze(1), bias, padding=0, groups=dim)[:,  :, -depth:].view(batch, num_leaf, dim, depth)

    out = torch.zeros_like(x)
    for i in range(num_leaf):
        out[:, parent_matrix[i], :]= conv_out[:, i, :, :].permute(0, 2, 1)
    
    if activation == "silu":
        output = torch.nn.functional.silu(out)

    return output

@triton.jit
def _tree_conv1d_kernel(
    X,
    STATES,
    WEIGHT,
    BIAS,
    OUTPUT,
    PARENT,
    dim,
    seqlen,
    stride_x_batch, stride_x_seqlen, stride_x_dim,
    stride_states_batch, stride_states_seqlen, stride_states_dim,
    stride_weight_dim, stride_weight_len,
    stride_bias_dim,
    stride_output_batch, stride_output_seqlen, stride_output_dim,
    stride_parent_batch, stride_parent_seqlen, stride_parent_depth,
    STATE_LEN: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    USE_SILU: tl.constexpr
):
    batch_idx = tl.program_id(2)
    seq_pos = tl.program_id(1) # this is the reverse sequence index (0 -> end of sequence)

    # Finding 
    PARENT_ptr = PARENT + (seqlen - seq_pos - 1) * stride_parent_seqlen + batch_idx * stride_parent_batch
    
    # temp = seq_pos
    # seq_idx = tl.zeros((STATE_LEN, ), dtype=tl.int64)
    # for i in tl.static_range(STATE_LEN):
    #     seq_idx[STATE_LEN-1-i] = temp
    #     temp = (temp - 1) // NUM_CHILD

    STATES_ptr = STATES + batch_idx * stride_states_batch
    OUTPUT_ptr = OUTPUT + batch_idx * stride_output_batch + (seqlen - seq_pos - 1) * stride_output_seqlen
    X_ptr = X + batch_idx * stride_x_batch

    state_idx = tl.arange(0, BLOCK_M)
    # rows = seqlen - seq_pos - BLOCK_M + state_idx
    seq_idx = tl.load(PARENT_ptr + state_idx * stride_parent_depth, mask=state_idx < STATE_LEN, other=0.0).to(tl.int64)
    valid_len = tl.sum((seq_idx >= 0).to(tl.int64))

    cols = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    # loading bias
    bias = tl.load(BIAS + cols * stride_bias_dim).to(tl.float32)
    # loading weight
    weight = tl.load(WEIGHT + state_idx[:, None] * stride_weight_len + cols[None, :] * stride_weight_dim,
                     mask=(state_idx[:, None] < STATE_LEN) & (cols[None, :] < dim),
                     other=0).to(tl.float32)
    
    # loading x, 
    x = tl.load(X_ptr + seq_idx[:, None] * stride_x_seqlen + cols[None, :] * stride_x_dim,
                mask=(seq_idx[:, None]>=0) & (cols[None, :] < dim),
                other=0).to(tl.float32)
    previous_state = tl.load(STATES_ptr + (state_idx[:, None] + valid_len) * stride_states_seqlen + cols[None, :] * stride_states_dim, 
                             mask=((state_idx[:, None] + valid_len) < STATE_LEN) & (seq_idx[:,None] < 0) & (cols[None, :] < dim),
                             other=0).to(tl.float32)
    # print(seq_pos, valid_len, seq_idx, x, previous_state)
    new_state = x + previous_state # (BLOCK_M, BLOCK_N)

    output = tl.sum(weight * new_state, axis=0) + bias # (BLOCK_N,)
    output = output.cast(tl.float32)
    if USE_SILU:
        output = output / (1 + tl.exp(-output))

    tl.store(OUTPUT_ptr + cols * stride_output_dim, 
             output, 
             mask=(cols < dim))


def tree_conv1d(x: torch.Tensor, state: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, activation: str, num_child: int, depth: int):
    '''
        x: (batch, dim, seqlen)
        state: (batch, dim, state_len)
        weight: (dim, state_len)
        bias: (dim,)
        num_child: number of children per node 
        depth: depth of tree, a tree with only root node is of depth 1
        Performs a conv1d on the token tree respecting the sequences in the token tree. 
        Does NOT update the state since ther can be multiple copies of state since there are multiple sequences in x 
    '''
    batch, dim, seqlen = x.shape
    _, _, state_len = state.shape
    output = torch.zeros_like(x)
    BLOCK_M = min(triton.next_power_of_2(state_len), 16)
    BLOCK_N = min(triton.next_power_of_2(dim), 256)
    grid = (triton.cdiv(dim, BLOCK_N), seqlen, batch)
    parent_matrix = [torch.arange(seqlen, device=x.device)]
    for _ in range(state_len - 1):
        parent_matrix.insert(0, torch.floor((parent_matrix[0]-1) / num_child))
    parent_matrix = torch.stack(parent_matrix, dim=1).unsqueeze(0)
    use_silu = activation == "silu"
    with torch.cuda.device(x.device.index):
        _tree_conv1d_kernel[grid](
            x,
            state,
            weight, 
            bias,
            output,
            parent_matrix,
            dim,
            seqlen,
            x.stride(0), x.stride(2), x.stride(1),
            state.stride(0), state.stride(2), state.stride(1),
            weight.stride(0), weight.stride(1),
            bias.stride(0),
            output.stride(0), output.stride(2), output.stride(1),
            parent_matrix.stride(0), parent_matrix.stride(1), parent_matrix.stride(2),
            STATE_LEN=state_len,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, USE_SILU=use_silu
        )
    return output

def tree_mask_conv1d(x: torch.Tensor, mask: torch.Tensor, state: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, activation: str):
    '''
        x: (batch, dim, seqlen)
        mask: (batch, seqlen, seqlen): the tree mask to indicate the structure of the tree
        state: (batch, dim, state_len)
        weight: (dim, state_len)
        bias: (dim,)
        num_child: number of children per node 
        depth: depth of tree, a tree with only root node is of depth 1
        Performs a conv1d on the token tree respecting the sequences in the token tree. 
        Does NOT update the state since ther can be multiple copies of state since there are multiple sequences in x 
    '''
    batch, dim, seqlen = x.shape
    _, _, state_len = state.shape
    output = torch.zeros_like(x)
    BLOCK_M = min(triton.next_power_of_2(state_len), 16)
    BLOCK_N = min(triton.next_power_of_2(dim), 256)
    grid = (triton.cdiv(dim, BLOCK_N), seqlen, batch)
    parent_matrix = mask_index_gather(mask, state_len)
    use_silu = activation == "silu"
    with torch.cuda.device(x.device.index):
        _tree_conv1d_kernel[grid](
            x,
            state,
            weight, 
            bias,
            output,
            parent_matrix,
            dim,
            seqlen,
            x.stride(0), x.stride(2), x.stride(1),
            state.stride(0), state.stride(2), state.stride(1),
            weight.stride(0), weight.stride(1),
            bias.stride(0),
            output.stride(0), output.stride(2), output.stride(1),
            parent_matrix.stride(0), parent_matrix.stride(1), parent_matrix.stride(2),
            STATE_LEN=state_len,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, USE_SILU=use_silu
        )
    return output


def mask_index_gather_ref(mask, state_len):
    '''
    mask (batch, seqlen, seqlen): tree mask
    state_len: length of conv1d state use to pad or trim the indices
    This is something we need to gather the index element needed for conv1d
    return b_out (batch, seqlen, state_len)
    '''
    b_out = []
    for b in range(mask.shape[0]):
        l_out = []
        for l in range(mask.shape[1]):
            curr = mask[b, l]
            indices = torch.nonzero(curr).view(-1)
            indices = torch.nn.functional.pad(indices, (state_len - len(indices), 0), value=-1)
            l_out.append(indices)
        l_out = torch.stack(l_out, dim=0)
        b_out.append(l_out)
    
    b_out = torch.stack(b_out, dim=0)
    return b_out

@triton.jit
def _mask_index_gather_kernel(
    MASK,
    OUT,
    state_len,
    stride_mask_batch, stride_mask_seqlen1, stride_mask_seqlen2,
    stride_out_batch, stride_out_seqlen, stride_out_statelen,
    seqlen:tl.constexpr,
    RIGHT_ALIGN: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_l = tl.program_id(1)

    # offs_m = tl.arange(0, BLOCK_M)
    MASK_ptr = MASK + pid_b * stride_mask_batch + pid_l * stride_mask_seqlen1
    OUT_ptr = OUT + pid_b * stride_out_batch + pid_l * stride_out_seqlen
    
    # MASK_ptrs = MASK_ptr + offs_m * stride_mask_seqlen2

    # mask = tl.load(MASK_ptrs, mask=(offs_m < seqlen), other=0.0).to(tl.int1)

    # count = tl.full((1,), state_len, dtype=tl.int64)
    # value = tl.full((1,), 1, dtype=tl.int64)
    if RIGHT_ALIGN:
        count = state_len - 1
        # print(pid_b, pid_l)

        for s in tl.static_range(seqlen):
            store_ptr = OUT_ptr + count * stride_out_statelen
            mask_ptrs = MASK_ptr + (seqlen - 1 - s) * stride_mask_seqlen2
            m = tl.load(mask_ptrs, mask=(count>=0),other=0.0).to(tl.int1)
            # When running with interpreter mask argument seems to get bypassed and store happens regardless.
            tl.store(store_ptr, (seqlen - 1 - s), mask=(m==1))
            count = tl.where(m==1, count - 1, count)
    else:
        count = state_len - 1

        for s in tl.static_range(seqlen):
            store_ptr = OUT_ptr + (state_len - 1 - count) * stride_out_statelen
            mask_ptrs = MASK_ptr + s * stride_mask_seqlen2
            m = tl.load(mask_ptrs, mask=(count>=0), other=0.0).to(tl.int1)
            tl.store(store_ptr, (s), mask=(m==1))
            count = tl.where(m==1, count-1, count)


def mask_index_gather(mask: torch.Tensor, state_len: int, right_align=True):
    if len(mask.shape) == 2:
        mask = mask.unsqueeze(0)
    batch, num_seq, seqlen = mask.shape
    grid = (batch, num_seq)
    out = -torch.ones((batch, num_seq, state_len), device=mask.device, dtype=torch.long)
    with torch.cuda.device(mask.device.index):
        _mask_index_gather_kernel[grid](
            mask, 
            out, 
            state_len,
            mask.stride(0), mask.stride(1), mask.stride(2),
            out.stride(0), out.stride(1), out.stride(2),
            seqlen, 
            RIGHT_ALIGN=right_align
        )
    return out