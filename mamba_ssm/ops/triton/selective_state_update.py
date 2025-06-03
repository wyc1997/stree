# Copyright (c) 2024, Tri Dao, Albert Gu.

"""We want triton==2.1.0 or triton==2.2.0 or triton==2.3.0 for this
"""

import math
import torch
import torch.nn.functional as F

import triton
import triton.language as tl

from einops import rearrange, repeat

from mamba_ssm.ops.triton.softplus import softplus
from mamba_ssm.ops.triton.tree_scan import mask_index_gather
from mamba_ssm.ops.triton.state_indexing import inplace_state_indexing
from mamba_ssm.ops.triton.tree_scan import mask_index_gather
import copy


@triton.heuristics({"HAS_DT_BIAS": lambda args: args["dt_bias_ptr"] is not None})
@triton.heuristics({"HAS_D": lambda args: args["D_ptr"] is not None})
@triton.heuristics({"HAS_Z": lambda args: args["z_ptr"] is not None})
@triton.heuristics({"BLOCK_SIZE_DSTATE": lambda args: triton.next_power_of_2(args["dstate"])})
@triton.jit
def _selective_scan_update_kernel(
    # Pointers to matrices
    state_ptr, x_ptr, dt_ptr, dt_bias_ptr, A_ptr, B_ptr, C_ptr, D_ptr, z_ptr, out_ptr,
    # Matrix dimensions
    batch, nheads, dim, dstate, nheads_ngroups_ratio,
    # Strides
    stride_state_batch, stride_state_head, stride_state_dim, stride_state_dstate,
    stride_x_batch, stride_x_head, stride_x_dim,
    stride_dt_batch, stride_dt_head, stride_dt_dim,
    stride_dt_bias_head, stride_dt_bias_dim,
    stride_A_head, stride_A_dim, stride_A_dstate,
    stride_B_batch, stride_B_group, stride_B_dstate,
    stride_C_batch, stride_C_group, stride_C_dstate,
    stride_D_head, stride_D_dim,
    stride_z_batch, stride_z_head, stride_z_dim,
    stride_out_batch, stride_out_head, stride_out_dim,
    # Meta-parameters
    DT_SOFTPLUS: tl.constexpr,
    TIE_HDIM: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    HAS_DT_BIAS: tl.constexpr,
    HAS_D: tl.constexpr,
    HAS_Z: tl.constexpr,
    BLOCK_SIZE_DSTATE: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_b = tl.program_id(axis=1)
    pid_h = tl.program_id(axis=2)
    state_ptr += pid_b * stride_state_batch + pid_h * stride_state_head
    x_ptr += pid_b * stride_x_batch + pid_h * stride_x_head
    dt_ptr += pid_b * stride_dt_batch + pid_h * stride_dt_head
    if HAS_DT_BIAS:
        dt_bias_ptr += pid_h * stride_dt_bias_head
    A_ptr += pid_h * stride_A_head
    B_ptr += pid_b * stride_B_batch + (pid_h // nheads_ngroups_ratio) * stride_B_group
    C_ptr += pid_b * stride_C_batch + (pid_h // nheads_ngroups_ratio) * stride_C_group
    if HAS_Z:
        z_ptr += pid_b * stride_z_batch + pid_h * stride_z_head
    out_ptr += pid_b * stride_out_batch + pid_h * stride_out_head

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = tl.arange(0, BLOCK_SIZE_DSTATE)
    state_ptrs = state_ptr + (offs_m[:, None] * stride_state_dim + offs_n[None, :] * stride_state_dstate)
    x_ptrs = x_ptr + offs_m * stride_x_dim
    dt_ptrs = dt_ptr + offs_m * stride_dt_dim
    if HAS_DT_BIAS:
        dt_bias_ptrs = dt_bias_ptr + offs_m * stride_dt_bias_dim
    if HAS_D:
        D_ptr += pid_h * stride_D_head
    A_ptrs = A_ptr + (offs_m[:, None] * stride_A_dim + offs_n[None, :] * stride_A_dstate)
    B_ptrs = B_ptr + offs_n * stride_B_dstate
    C_ptrs = C_ptr + offs_n * stride_C_dstate
    if HAS_D:
        D_ptrs = D_ptr + offs_m * stride_D_dim
    if HAS_Z:
        z_ptrs = z_ptr + offs_m * stride_z_dim
    out_ptrs = out_ptr + offs_m * stride_out_dim

    state = tl.load(state_ptrs, mask=(offs_m[:, None] < dim) & (offs_n[None, :] < dstate), other=0.0)
    x = tl.load(x_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
    if not TIE_HDIM:
        dt = tl.load(dt_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
        if HAS_DT_BIAS:
            dt += tl.load(dt_bias_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
        if DT_SOFTPLUS:
            dt = softplus(dt)
        A = tl.load(A_ptrs, mask=(offs_m[:, None] < dim) & (offs_n[None, :] < dstate), other=0.0).to(tl.float32)
        dA = tl.exp(A * dt[:, None])
    else:
        dt = tl.load(dt_ptr).to(tl.float32)
        if HAS_DT_BIAS:
            dt += tl.load(dt_bias_ptr).to(tl.float32)
        if DT_SOFTPLUS:
            dt = softplus(dt)
        A = tl.load(A_ptr).to(tl.float32)
        dA = tl.exp(A * dt)  # scalar, not a matrix

    B = tl.load(B_ptrs, mask=offs_n < dstate, other=0.0).to(tl.float32)
    C = tl.load(C_ptrs, mask=offs_n < dstate, other=0.0).to(tl.float32)
    if HAS_D:
        D = tl.load(D_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
    if HAS_Z:
        z = tl.load(z_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)

    if not TIE_HDIM:
        dB = B[None, :] * dt[:, None]
    else:
        dB = B * dt  # vector of size (dstate,)
    state = state * dA + dB * x[:, None]
    tl.store(state_ptrs, state, mask=(offs_m[:, None] < dim) & (offs_n[None, :] < dstate))
    out = tl.sum(state * C[None, :], axis=1)
    if HAS_D:
        out += x * D
    if HAS_Z:
        out *= z * tl.sigmoid(z)
    tl.store(out_ptrs, out, mask=offs_m < dim)


def selective_state_update(state, x, dt, A, B, C, D=None, z=None, dt_bias=None, dt_softplus=False):
    """
    Argument:
        state: (batch, dim, dstate) or (batch, nheads, dim, dstate)
        x: (batch, dim) or (batch, nheads, dim)
        dt: (batch, dim) or (batch, nheads, dim)
        A: (dim, dstate) or (nheads, dim, dstate)
        B: (batch, dstate) or (batch, ngroups, dstate)
        C: (batch, dstate) or (batch, ngroups, dstate)
        D: (dim,) or (nheads, dim)
        z: (batch, dim) or (batch, nheads, dim)
        dt_bias: (dim,) or (nheads, dim)
    Return:
        out: (batch, dim) or (batch, nheads, dim)
    """
    has_heads = state.dim() > 3
    if state.dim() == 3:
        state = state.unsqueeze(1)
    if x.dim() == 2:
        x = x.unsqueeze(1)
    if dt.dim() == 2:
        dt = dt.unsqueeze(1)
    if A.dim() == 2:
        A = A.unsqueeze(0)
    if B.dim() == 2:
        B = B.unsqueeze(1)
    if C.dim() == 2:
        C = C.unsqueeze(1)
    if D is not None and D.dim() == 1:
        D = D.unsqueeze(0)
    if z is not None and z.dim() == 2:
        z = z.unsqueeze(1)
    if dt_bias is not None and dt_bias.dim() == 1:
        dt_bias = dt_bias.unsqueeze(0)
    batch, nheads, dim, dstate = state.shape
    assert x.shape == (batch, nheads, dim)
    assert dt.shape == x.shape
    assert A.shape == (nheads, dim, dstate)
    ngroups = B.shape[1]
    assert nheads % ngroups == 0, "nheads must be divisible by ngroups"
    assert B.shape == (batch, ngroups, dstate)
    assert C.shape == B.shape
    if D is not None:
        assert D.shape == (nheads, dim)
    if z is not None:
        assert z.shape == x.shape
    if dt_bias is not None:
        assert dt_bias.shape == (nheads, dim)
    out = torch.empty_like(x)
    grid = lambda META: (triton.cdiv(dim, META['BLOCK_SIZE_M']), batch, nheads)
    z_strides = ((z.stride(0), z.stride(1), z.stride(2)) if z is not None else (0, 0, 0))
    # We don't want autotune since it will overwrite the state
    # We instead tune by hand.
    BLOCK_SIZE_M, num_warps = ((32, 4) if dstate <= 16
                               else ((16, 4) if dstate <= 32 else
                                     ((8, 4) if dstate <= 64 else
                                      ((4, 4) if dstate <= 128 else
                                       ((4, 8))))))
    tie_hdim = A.stride(-1) == 0 and A.stride(-2) == 0 and dt.stride(-1) == 0 and dt_bias.stride(-1) == 0
    with torch.cuda.device(x.device.index):
        _selective_scan_update_kernel[grid](
            state, x, dt, dt_bias, A, B, C, D, z, out,
            batch, nheads, dim, dstate, nheads // ngroups,
            state.stride(0), state.stride(1), state.stride(2), state.stride(3),
            x.stride(0), x.stride(1), x.stride(2),
            dt.stride(0), dt.stride(1), dt.stride(2),
            *(dt_bias.stride(0), dt_bias.stride(1)) if dt_bias is not None else 0,
            A.stride(0), A.stride(1), A.stride(2),
            B.stride(0), B.stride(1), B.stride(2),
            C.stride(0), C.stride(1), C.stride(2),
            *(D.stride(0), D.stride(1)) if D is not None else 0,
            z_strides[0], z_strides[1], z_strides[2],
            out.stride(0), out.stride(1), out.stride(2),
            dt_softplus,
            tie_hdim,
            BLOCK_SIZE_M,
            num_warps=num_warps,
        )
    if not has_heads:
        out = out.squeeze(1)
    return out


def selective_state_update_ref(state, x, dt, A, B, C, D=None, z=None, dt_bias=None, dt_softplus=False):
    """
    Argument:
        state: (batch, dim, dstate) or (batch, nheads, dim, dstate)
        x: (batch, dim) or (batch, nheads, dim)
        dt: (batch, dim) or (batch, nheads, dim)
        A: (dim, dstate) or (nheads, dim, dstate)
        B: (batch, dstate) or (batch, ngroups, dstate)
        C: (batch, dstate) or (batch, ngroups, dstate)
        D: (dim,) or (nheads, dim)
        z: (batch, dim) or (batch, nheads, dim)
        dt_bias: (dim,) or (nheads, dim)
    Return:
        out: (batch, dim) or (batch, nheads, dim)
    """
    has_heads = state.dim() > 3
    if state.dim() == 3:
        state = state.unsqueeze(1)
    if x.dim() == 2:
        x = x.unsqueeze(1)
    if dt.dim() == 2:
        dt = dt.unsqueeze(1)
    if A.dim() == 2:
        A = A.unsqueeze(0)
    if B.dim() == 2:
        B = B.unsqueeze(1)
    if C.dim() == 2:
        C = C.unsqueeze(1)
    if D is not None and D.dim() == 1:
        D = D.unsqueeze(0)
    if z is not None and z.dim() == 2:
        z = z.unsqueeze(1)
    if dt_bias is not None and dt_bias.dim() == 1:
        dt_bias = dt_bias.unsqueeze(0)
    batch, nheads, dim, dstate = state.shape
    assert x.shape == (batch, nheads, dim)
    assert dt.shape == x.shape
    assert A.shape == (nheads, dim, dstate)
    ngroups = B.shape[1]
    assert nheads % ngroups == 0, "nheads must be divisible by ngroups"
    assert B.shape == (batch, ngroups, dstate)
    assert C.shape == B.shape
    if D is not None:
        assert D.shape == (nheads, dim)
    if z is not None:
        assert z.shape == x.shape
    if dt_bias is not None:
        assert dt_bias.shape == (nheads, dim)
        dt = dt + dt_bias
    dt = F.softplus(dt) if dt_softplus else dt
    dA = torch.exp(rearrange(dt, "b h d -> b h d 1") * A)  # (batch, nheads, dim, dstate)
    B = repeat(B, "b g n -> b (g h) n", h=nheads // ngroups)  # (batch, nheads, dstate)
    C = repeat(C, "b g n -> b (g h) n", h=nheads // ngroups)  # (batch, nheads, dstate)
    dB = rearrange(dt, "b h d -> b h d 1") * rearrange(B, "b h n -> b h 1 n")  # (batch, nheads, dim, dstate)
    state.copy_(state * dA + dB * rearrange(x, "b h d -> b h d 1"))  # (batch, dim, dstate
    out = torch.einsum("bhdn,bhn->bhd", state.to(C.dtype), C)
    if D is not None:
        out += (x * D).to(out.dtype)
    out = (out if z is None else out * F.silu(z)).to(x.dtype)
    if not has_heads:
        out = out.squeeze(1)
    return out

def selective_state_update_2step_ref(state, x, dt, A, B, C, D=None, z=None, dt_bias=None, dt_softplus=False):
    """
    Argument:
        state: (batch, dim, dstate) or (batch, nheads, dim, dstate)
        x: (batch, seqlen, dim) or (batch, seqlen, nheads, dim)
        dt: (batch, seqlen, dim) or (batch, seqlen, nheads, dim)
        A: (dim, dstate) or (nheads, dim, dstate)
        B: (batch, seqlen, dstate) or (batch, seqlen, ngroups, dstate)
        C: (batch, seqlen, dstate) or (batch, seqlen, ngroups, dstate)
        D: (seqlen, dim) or (nheads, seqlen, dim)
        z: (batch, seqlen, dim) or (batch, seqlen, nheads, dim)
        dt_bias: (dim,) or (nheads, dim)
    Return:
        out: (batch, seqlen, dim) or (batch, seqlen, nheads, dim)
    """
    has_heads = state.dim() > 3
    if state.dim() == 3:
        state = state.unsqueeze(1)
    if x.dim() == 3:
        x = x.unsqueeze(2)
    if dt.dim() == 3:
        dt = dt.unsqueeze(2)
    if A.dim() == 2:
        A = A.unsqueeze(1)
    if B.dim() == 3:
        B = B.unsqueeze(2)
    if C.dim() == 3:
        C = C.unsqueeze(2)
    if D is not None and D.dim() == 1:
        D = D.unsqueeze(0)
    if z is not None and z.dim() == 3:
        z = z.unsqueeze(2)
    if dt_bias is not None and dt_bias.dim() == 1:
        dt_bias = dt_bias.unsqueeze(0)
    batch, nheads, dim, dstate = state.shape
    seqlen = 2
    assert x.shape == (batch, seqlen, nheads, dim)
    assert dt.shape == x.shape
    assert A.shape == (nheads, dim, dstate)
    ngroups = B.shape[2]
    assert nheads % ngroups == 0, "nheads must be divisible by ngroups"
    assert B.shape == (batch, seqlen, ngroups, dstate)
    assert C.shape == B.shape
    if D is not None:
        assert D.shape == (nheads, dim)
    if z is not None:
        assert z.shape == x.shape
    if dt_bias is not None:
        assert dt_bias.shape == (nheads, dim)
        dt = dt + dt_bias
    dt = F.softplus(dt) if dt_softplus else dt # (batch, seqlen, nheads, dim)
    dA = torch.exp(rearrange(dt, "b l h d -> b l h d 1") * A)  # (batch, seqlen, nheads, dim, dstate)
    B = repeat(B, "b l g n -> b l (g h) n", h=nheads // ngroups)  # (batch, seqlen, nheads, dstate)
    C = repeat(C, "b l g n -> b l (g h) n", h=nheads // ngroups)  # (batch, seqlen, nheads, dstate)
    dB = rearrange(dt, "b l h d -> b l h d 1") * rearrange(B, "b l h n -> b l h 1 n")  # (batch, seqlen, nheads, dim, dstate)
    state.copy_(state * dA[:, 0, ...] + dB[:, 0, ...] * rearrange(x[:, 0, ...], "b h d -> b h d 1"))  
    out1 = torch.einsum("bhdn,bhn->bhd", state.to(C.dtype), C[:, 0, ...])
    state.copy_(state * dA[:, 1, ...] + dB[:, 1, ...] * rearrange(x[:, 1, ...], "b h d -> b h d 1"))
    out2 = torch.einsum("bhdn,bhn->bhd", state.to(C.dtype), C[:, 1, ...])
    out = torch.stack([out1, out2], dim=1)
    if D is not None:
        out += (x * D).to(out.dtype)
    out = (out if z is None else out * F.silu(z)).to(x.dtype)
    if not has_heads:
        out = out.squeeze(1)
    return out

# The 2 step kernel is similar to the original selective_state_update kernel. To perform a 2 step update in 1 kernel,
# we load the data for 2 time steps together. Then using .split() by triton, we split the data alone the seq_len dimension.
# Computation for A, B, C, dt, z can happen together for both time step since these computation doesn't have dependcy on each other
# The state is the only part that is sequential. 
@triton.heuristics({"HAS_DT_BIAS": lambda args: args["dt_bias_ptr"] is not None})
@triton.heuristics({"HAS_D": lambda args: args["D_ptr"] is not None})
@triton.heuristics({"HAS_Z": lambda args: args["z_ptr"] is not None})
@triton.heuristics({"BLOCK_SIZE_DSTATE": lambda args: triton.next_power_of_2(args["dstate"])})
@triton.jit
def _selective_scan_update_2step_kernel(
    # Pointers to matrices
    state_ptr, x_ptr, dt_ptr, dt_bias_ptr, A_ptr, B_ptr, C_ptr, D_ptr, z_ptr, out_ptr,
    # Matrix dimensions
    batch, nheads, dim, dstate, nheads_ngroups_ratio,
    # Strides
    stride_state_batch, stride_state_head, stride_state_dim, stride_state_dstate,
    stride_x_batch, stride_x_seqlen, stride_x_head, stride_x_dim,
    stride_dt_batch, stride_dt_seqlen, stride_dt_head, stride_dt_dim,
    stride_dt_bias_head, stride_dt_bias_dim,
    stride_A_head, stride_A_dim, stride_A_dstate,
    stride_B_batch, stride_B_seqlen, stride_B_group, stride_B_dstate,
    stride_C_batch, stride_C_seqlen, stride_C_group, stride_C_dstate,
    stride_D_head, stride_D_dim,
    stride_z_batch, stride_Z_seqlen, stride_z_head, stride_z_dim,
    stride_out_batch, stride_out_seqlen, stride_out_head, stride_out_dim,
    # Meta-parameters
    DT_SOFTPLUS: tl.constexpr,
    TIE_HDIM: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    HAS_DT_BIAS: tl.constexpr,
    HAS_D: tl.constexpr,
    HAS_Z: tl.constexpr,
    BLOCK_SIZE_DSTATE: tl.constexpr,
    SEQLEN: tl.constexpr
):
    pid_m = tl.program_id(axis=0)
    pid_b = tl.program_id(axis=1)
    pid_h = tl.program_id(axis=2)
    state_ptr += pid_b * stride_state_batch + pid_h * stride_state_head
    x_ptr += pid_b * stride_x_batch + pid_h * stride_x_head
    dt_ptr += pid_b * stride_dt_batch  + pid_h * stride_dt_head
    if HAS_DT_BIAS:
        dt_bias_ptr += pid_h * stride_dt_bias_head
    A_ptr += pid_h * stride_A_head
    B_ptr += pid_b * stride_B_batch + (pid_h // nheads_ngroups_ratio) * stride_B_group
    C_ptr += pid_b * stride_C_batch + (pid_h // nheads_ngroups_ratio) * stride_C_group
    if HAS_Z:
        z_ptr += pid_b * stride_z_batch + pid_h * stride_z_head
    out_ptr += pid_b * stride_out_batch + pid_h * stride_out_head

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = tl.arange(0, BLOCK_SIZE_DSTATE)
    offs_l = tl.arange(0, SEQLEN)
    state_ptrs = state_ptr + (offs_m[:, None] * stride_state_dim + offs_n[None, :] * stride_state_dstate) # (dim_block_size, dstate_block_size)
    x_ptrs = x_ptr + (offs_m[:, None] * stride_x_dim + offs_l[None, :] * stride_x_seqlen) # (dim_block_size, seqlen)
    dt_ptrs = dt_ptr + (offs_m[:, None] * stride_dt_dim + offs_l[None, :] * stride_dt_seqlen) # (dim_block_size, seqlen)
    if HAS_DT_BIAS:
        dt_bias_ptrs = dt_bias_ptr + offs_m * stride_dt_bias_dim
    if HAS_D:
        D_ptr += pid_h * stride_D_head
    A_ptrs = A_ptr + (offs_m[:, None] * stride_A_dim + offs_n[None, :] * stride_A_dstate) # (dim_block_size, dstate_block_size)
    B_ptrs = B_ptr + (offs_n[:, None] * stride_B_dstate + offs_l[None, :] * stride_B_seqlen) # (dstate_block_size, seqlen)
    C_ptrs = C_ptr + (offs_n[:, None] * stride_C_dstate + offs_l[None, :] * stride_C_seqlen) # (dstate_block_size, seqlen)
    if HAS_D:
        D_ptrs = D_ptr + offs_m * stride_D_dim # (dim_block_size,)
    if HAS_Z:
        z_ptrs = z_ptr + (offs_m[:, None] * stride_z_dim + offs_l[None, :] * stride_Z_seqlen)  # (dim_block_size, seqlen)
    out_ptrs = out_ptr + (offs_m[:, None] * stride_out_dim + offs_l[None, :] * stride_out_seqlen)  # (dim_block_size, seqlen)

    state = tl.load(state_ptrs, mask=(offs_m[:, None] < dim) & (offs_n[None, :] < dstate), other=0.0)
    x = tl.load(x_ptrs, mask=offs_m[:, None] < dim, other=0.0).to(tl.float32)
    if not TIE_HDIM:
        dt = tl.load(dt_ptrs, mask=offs_m[:, None] < dim, other=0.0).to(tl.float32) # (dim_block_size, seqlen)
        if HAS_DT_BIAS:
            dt += tl.load(dt_bias_ptrs, mask=offs_m < dim, other=0.0)[:, None].to(tl.float32)
        if DT_SOFTPLUS:
            dt = softplus(dt)
        A = tl.load(A_ptrs, mask=(offs_m[:, None] < dim) & (offs_n[None, :] < dstate), other=0.0).to(tl.float32)
        dA = tl.exp(A[:, :, None] * dt[:, None, :]) # (dim_block_size, dstate_block_size, seqlen)
    else:
        dt = tl.load(dt_ptr + (offs_l[None, :] * stride_dt_seqlen), mask=(offs_l[None, :] < SEQLEN), other=0.0).to(tl.float32) # (1, seqlen)
        if HAS_DT_BIAS:
            dt += tl.load(dt_bias_ptr).to(tl.float32)[None, :]
        if DT_SOFTPLUS:
            dt = softplus(dt)
        A = tl.load(A_ptr).to(tl.float32)
        dA = tl.exp(A[:,None] * dt)  # (1, seqlen)

    B = tl.load(B_ptrs, mask=offs_n[:, None] < dstate, other=0.0).to(tl.float32)
    C = tl.load(C_ptrs, mask=offs_n[:, None] < dstate, other=0.0).to(tl.float32)
    if HAS_D:
        D = tl.load(D_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
    if HAS_Z:
        z = tl.load(z_ptrs, mask=offs_m[:, None] < dim, other=0.0).to(tl.float32)
        z1, z2 = z.split()

    if not TIE_HDIM:
        dB = B[None, :, :] * dt[:, None, :] # (dim_block_size, dstate_block_size, seqlen)
    else:
        dB = B * dt  # vector of size (dstate,)

    dA1, dA2 = dA.split()
    dB1, dB2 = dB.split()
    x1, x2 = x.split()
    C1, C2 = C.split()
    state = state * dA1 + dB1 * x1[:, None]
    out1 = tl.sum(state * C1[None, :], axis=1)
    if HAS_D:
        out1 += x1 * D
    if HAS_Z:
        out1 *= z1 * tl.sigmoid(z1)
    state = state * dA2 + dB2 * x2[:, None]
    out2 = tl.sum(state * C2[None, :], axis=1)
    if HAS_D:
        out2 += x2 * D
    if HAS_Z:
        out2 *= z2 * tl.sigmoid(z2)
    out = tl.join(out1, out2)
    tl.store(state_ptrs, state, mask=(offs_m[:, None] < dim) & (offs_n[None, :] < dstate))
    tl.store(out_ptrs, out, mask=offs_m[:, None] < dim)

def selective_state_update_2step(state, x, dt, A, B, C, D=None, z=None, dt_bias=None, dt_softplus=False):
    """
    Argument:
        state: (batch, dim, dstate) or (batch, nheads, dim, dstate)
        x: (batch, seqlen, dim) or (batch, seqlen, nheads, dim)
        dt: (batch, seqlen, dim) or (batch, seqlen, nheads, dim)
        A: (dim, dstate) or (nheads, dim, dstate)
        B: (batch, seqlen, dstate) or (batch, seqlen, ngroups, dstate)
        C: (batch, seqlen, dstate) or (batch, seqlen, ngroups, dstate)
        D: (dim,) or (nheads, dim)
        z: (batch, seqlen, dim) or (batch, seqlen, nheads, dim)
        dt_bias: (dim,) or (nheads, dim)
    Return:
        out: (batch, seqlen, dim) or (batch, seqlen, nheads, dim)
    """
    has_heads = state.dim() > 3
    if state.dim() == 3:
        state = state.unsqueeze(1)
    if x.dim() == 3:
        x = x.unsqueeze(2)
    if dt.dim() == 3:
        dt = dt.unsqueeze(2)
    if A.dim() == 2:
        A = A.unsqueeze(0)
    if B.dim() == 3:
        B = B.unsqueeze(2)
    if C.dim() == 3:
        C = C.unsqueeze(2)
    if D is not None and D.dim() == 1:
        D = D.unsqueeze(0)
    if z is not None and z.dim() == 3:
        z = z.unsqueeze(2)
    if dt_bias is not None and dt_bias.dim() == 1:
        dt_bias = dt_bias.unsqueeze(0)
    batch, nheads, dim, dstate = state.shape
    SEQLEN = 2
    assert x.shape == (batch, SEQLEN, nheads, dim)
    assert dt.shape == x.shape
    assert A.shape == (nheads, dim, dstate)
    ngroups = B.shape[2]
    assert nheads % ngroups == 0, "nheads must be divisible by ngroups"
    assert B.shape == (batch, SEQLEN, ngroups, dstate)
    assert C.shape == B.shape
    if D is not None:
        assert D.shape == (nheads, dim)
    if z is not None:
        assert z.shape == x.shape
    if dt_bias is not None:
        assert dt_bias.shape == (nheads, dim)
    out = torch.empty_like(x)
    grid = lambda META: (triton.cdiv(dim, META['BLOCK_SIZE_M']), batch, nheads)
    z_strides = ((z.stride(0), z.stride(1), z.stride(2), z.stride(3)) if z is not None else (0, 0, 0, 0))
    # We don't want autotune since it will overwrite the state
    # We instead tune by hand.
    BLOCK_SIZE_M, num_warps = ((32, 4) if dstate <= 16
                               else ((16, 4) if dstate <= 32 else
                                     ((8, 4) if dstate <= 64 else
                                      ((4, 4) if dstate <= 128 else
                                       ((4, 8))))))
    tie_hdim = A.stride(-1) == 0 and A.stride(-2) == 0 and dt.stride(-1) == 0 and dt_bias.stride(-1) == 0
    with torch.cuda.device(x.device.index):
        _selective_scan_update_2step_kernel[grid](
            state, x, dt, dt_bias, A, B, C, D, z, out,
            batch, nheads, dim, dstate, nheads // ngroups,
            state.stride(0), state.stride(1), state.stride(2), state.stride(3),
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            dt.stride(0), dt.stride(1), dt.stride(2), dt.stride(3),
            *(dt_bias.stride(0), dt_bias.stride(1)) if dt_bias is not None else 0,
            A.stride(0), A.stride(1), A.stride(2),
            B.stride(0), B.stride(1), B.stride(2), B.stride(3),
            C.stride(0), C.stride(1), C.stride(2), C.stride(3),
            *(D.stride(0), D.stride(1)) if D is not None else 0,
            z_strides[0], z_strides[1], z_strides[2], z_strides[3],
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            dt_softplus,
            tie_hdim,
            BLOCK_SIZE_M,
            SEQLEN=SEQLEN,
            num_warps=num_warps,
        )
    if not has_heads:
        out = out.squeeze(1)
    return out

def selective_state_update_Nstep_ref(state, x, dt, A, B, C, D=None, z=None, dt_bias=None, dt_softplus=False):
    """
    Argument:
        state: (batch, dim, dstate) or (batch, nheads, dim, dstate)
        x: (batch, seqlen, dim) or (batch, seqlen, nheads, dim)
        dt: (batch, seqlen, dim) or (batch, seqlen, nheads, dim)
        A: (dim, dstate) or (nheads, dim, dstate)
        B: (batch, seqlen, dstate) or (batch, seqlen, ngroups, dstate)
        C: (batch, seqlen, dstate) or (batch, seqlen, ngroups, dstate)
        D: (seqlen, dim) or (nheads, seqlen, dim)
        z: (batch, seqlen, dim) or (batch, seqlen, nheads, dim)
        dt_bias: (dim,) or (nheads, dim)
    Return:
        out: (batch, seqlen, dim) or (batch, seqlen, nheads, dim)
    """
    has_heads = state.dim() > 3
    if state.dim() == 3:
        state = state.unsqueeze(1)
    if x.dim() == 3:
        x = x.unsqueeze(2)
    if dt.dim() == 3:
        dt = dt.unsqueeze(2)
    if A.dim() == 2:
        A = A.unsqueeze(1)
    if B.dim() == 3:
        B = B.unsqueeze(2)
    if C.dim() == 3:
        C = C.unsqueeze(2)
    if D is not None and D.dim() == 1:
        D = D.unsqueeze(0)
    if z is not None and z.dim() == 3:
        z = z.unsqueeze(2)
    if dt_bias is not None and dt_bias.dim() == 1:
        dt_bias = dt_bias.unsqueeze(0)
    batch, nheads, dim, dstate = state.shape
    seqlen = x.shape[1]
    assert x.shape == (batch, seqlen, nheads, dim)
    assert dt.shape == x.shape
    assert A.shape == (nheads, dim, dstate)
    ngroups = B.shape[2]
    assert nheads % ngroups == 0, "nheads must be divisible by ngroups"
    assert B.shape == (batch, seqlen, ngroups, dstate)
    assert C.shape == B.shape
    if D is not None:
        assert D.shape == (nheads, dim)
    if z is not None:
        assert z.shape == x.shape
    if dt_bias is not None:
        assert dt_bias.shape == (nheads, dim)
        dt = dt + dt_bias
    dt = F.softplus(dt) if dt_softplus else dt # (batch, seqlen, nheads, dim)
    dA = torch.exp(rearrange(dt, "b l h d -> b l h d 1") * A)  # (batch, seqlen, nheads, dim, dstate)
    B = repeat(B, "b l g n -> b l (g h) n", h=nheads // ngroups)  # (batch, seqlen, nheads, dstate)
    C = repeat(C, "b l g n -> b l (g h) n", h=nheads // ngroups)  # (batch, seqlen, nheads, dstate)
    dB = rearrange(dt, "b l h d -> b l h d 1") * rearrange(B, "b l h n -> b l h 1 n")  # (batch, seqlen, nheads, dim, dstate)
    out = []
    for i in range(dA.shape[1]):
        state.copy_(state * dA[:, i, ...] + dB[:, i, ...] * rearrange(x[:, i, ...], "b h d -> b h d 1"))  
        out.append(torch.einsum("bhdn,bhn->bhd", state.to(C.dtype), C[:, i, ...]))
    out = torch.stack(out, dim=1)
    if D is not None:
        out += (x * D).to(out.dtype)
    out = (out if z is None else out * F.silu(z)).to(x.dtype)
    if not has_heads:
        out = out.squeeze(1)
    return out

# The Nstep_kernel is trying to extend 2 step kernel to an arbitrary number of steps. The problem with triton is that
# it doesn't support indexing of already loaded tensors and .split() function only works if the dimension being split 
# is of size 2. Therefore, we choose to use a for loop instead and load and compute each time step sequentially. It 
# turns out that this is still pretty fast, potentially thanks to the parallel execution of memory loading and computation
@triton.heuristics({"HAS_DT_BIAS": lambda args: args["dt_bias_ptr"] is not None})
@triton.heuristics({"HAS_D": lambda args: args["D_ptr"] is not None})
@triton.heuristics({"HAS_Z": lambda args: args["z_ptr"] is not None})
@triton.heuristics({"BLOCK_SIZE_DSTATE": lambda args: triton.next_power_of_2(args["dstate"])})
@triton.jit
def _selective_scan_update_Nstep_kernel(
    # Pointers to matrices
    state_ptr, x_ptr, dt_ptr, dt_bias_ptr, A_ptr, B_ptr, C_ptr, D_ptr, z_ptr, out_ptr,
    # Matrix dimensions
    batch, nheads, dim, dstate, nheads_ngroups_ratio,
    # Strides
    stride_state_batch, stride_state_head, stride_state_dim, stride_state_dstate,
    stride_x_batch, stride_x_seqlen, stride_x_head, stride_x_dim,
    stride_dt_batch, stride_dt_seqlen, stride_dt_head, stride_dt_dim,
    stride_dt_bias_head, stride_dt_bias_dim,
    stride_A_head, stride_A_dim, stride_A_dstate,
    stride_B_batch, stride_B_seqlen, stride_B_group, stride_B_dstate,
    stride_C_batch, stride_C_seqlen, stride_C_group, stride_C_dstate,
    stride_D_head, stride_D_dim,
    stride_z_batch, stride_Z_seqlen, stride_z_head, stride_z_dim,
    stride_out_batch, stride_out_seqlen, stride_out_head, stride_out_dim,
    # Meta-parameters
    DT_SOFTPLUS: tl.constexpr,
    TIE_HDIM: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    HAS_DT_BIAS: tl.constexpr,
    HAS_D: tl.constexpr,
    HAS_Z: tl.constexpr,
    BLOCK_SIZE_DSTATE: tl.constexpr,
    SEQLEN: tl.constexpr
):
    pid_m = tl.program_id(axis=0)
    pid_b = tl.program_id(axis=1)
    pid_h = tl.program_id(axis=2)
    state_ptr += pid_b * stride_state_batch + pid_h * stride_state_head
    x_ptr += pid_b * stride_x_batch + pid_h * stride_x_head
    dt_ptr += pid_b * stride_dt_batch  + pid_h * stride_dt_head
    if HAS_DT_BIAS:
        dt_bias_ptr += pid_h * stride_dt_bias_head
    A_ptr += pid_h * stride_A_head
    B_ptr += pid_b * stride_B_batch + (pid_h // nheads_ngroups_ratio) * stride_B_group
    C_ptr += pid_b * stride_C_batch + (pid_h // nheads_ngroups_ratio) * stride_C_group
    if HAS_Z:
        z_ptr += pid_b * stride_z_batch + pid_h * stride_z_head
    out_ptr += pid_b * stride_out_batch + pid_h * stride_out_head

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = tl.arange(0, BLOCK_SIZE_DSTATE)
    state_ptrs = state_ptr + (offs_m[:, None] * stride_state_dim + offs_n[None, :] * stride_state_dstate) # (dim_block_size, dstate_block_size)
    x_ptrs = x_ptr + (offs_m * stride_x_dim ) # (dim_block_size, )
    dt_ptrs = dt_ptr + (offs_m * stride_dt_dim ) # (dim_block_size, )
    if HAS_DT_BIAS:
        dt_bias_ptrs = dt_bias_ptr + offs_m * stride_dt_bias_dim
    if HAS_D:
        D_ptr += pid_h * stride_D_head
    A_ptrs = A_ptr + (offs_m[:, None] * stride_A_dim + offs_n[None, :] * stride_A_dstate) # (dim_block_size, dstate_block_size)
    B_ptrs = B_ptr + (offs_n * stride_B_dstate ) # (dstate_block_size, )
    C_ptrs = C_ptr + (offs_n * stride_C_dstate ) # (dstate_block_size, )
    if HAS_D:
        D_ptrs = D_ptr + offs_m * stride_D_dim # (dim_block_size,)
    if HAS_Z:
        z_ptrs = z_ptr + (offs_m * stride_z_dim )  # (dim_block_size, )
    out_ptrs = out_ptr + (offs_m * stride_out_dim )  # (dim_block_size, )

    state = tl.load(state_ptrs, mask=(offs_m[:, None] < dim) & (offs_n[None, :] < dstate), other=0.0)

    for s in tl.static_range(SEQLEN):
        x_ptrs_seqlen = x_ptrs + s * stride_x_seqlen
        dt_ptrs_seqlen = dt_ptrs + s * stride_dt_seqlen
        B_ptrs_seqlen = B_ptrs + s * stride_B_seqlen
        C_ptrs_seqlen = C_ptrs + s * stride_C_seqlen
        if HAS_Z:
            z_ptrs_seqlen = z_ptrs + s * stride_Z_seqlen
        out_ptrs_seqlen = out_ptrs + s * stride_out_seqlen
        x = tl.load(x_ptrs_seqlen, mask=offs_m < dim, other=0.0).to(tl.float32)
        if not TIE_HDIM:
            dt = tl.load(dt_ptrs_seqlen, mask=offs_m < dim, other=0.0).to(tl.float32) # (dim_block_size,)
            if HAS_DT_BIAS:
                dt += tl.load(dt_bias_ptrs, mask=offs_m < dim, other=0.0)[:, None].to(tl.float32)
            if DT_SOFTPLUS:
                dt = softplus(dt)
            A = tl.load(A_ptrs, mask=(offs_m[:, None] < dim) & (offs_n[None, :] < dstate), other=0.0).to(tl.float32)
            dA = tl.exp(A * dt[:, None]) # (dim_block_size, dstate_block_size)
        else:
            dt = tl.load(dt_ptr + (s * stride_dt_seqlen)).to(tl.float32) # (1,)
            if HAS_DT_BIAS:
                dt += tl.load(dt_bias_ptr).to(tl.float32)[None, :]
            if DT_SOFTPLUS:
                dt = softplus(dt)
            A = tl.load(A_ptr).to(tl.float32)
            dA = tl.exp(A * dt)  # (1,)

        B = tl.load(B_ptrs_seqlen, mask=offs_n < dstate, other=0.0).to(tl.float32)
        C = tl.load(C_ptrs_seqlen, mask=offs_n < dstate, other=0.0).to(tl.float32)
        if HAS_D:
            D = tl.load(D_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
        if HAS_Z:
            z = tl.load(z_ptrs_seqlen, mask=offs_m < dim, other=0.0).to(tl.float32)

        if not TIE_HDIM:
            dB = B[None, :] * dt[:, None] # (dim_block_size, dstate_block_size, seqlen)
        else:
            dB = B * dt  # vector of size (dstate,)

        state = state * dA + dB * x[:, None]
        out = tl.sum(state * C[None, :], axis=1)
        if HAS_D:
            out += x * D
        if HAS_Z:
            out *= z * tl.sigmoid(z)
        tl.store(out_ptrs_seqlen, out, mask=offs_m < dim)

    tl.store(state_ptrs, state, mask=(offs_m[:, None] < dim) & (offs_n[None, :] < dstate))

def selective_state_update_Nstep(state, x, dt, A, B, C, D=None, z=None, dt_bias=None, dt_softplus=False):
    """
    Argument:
        state: (batch, dim, dstate) or (batch, nheads, dim, dstate)
        x: (batch, seqlen, dim) or (batch, seqlen, nheads, dim)
        dt: (batch, seqlen, dim) or (batch, seqlen, nheads, dim)
        A: (dim, dstate) or (nheads, dim, dstate)
        B: (batch, seqlen, dstate) or (batch, seqlen, ngroups, dstate)
        C: (batch, seqlen, dstate) or (batch, seqlen, ngroups, dstate)
        D: (dim,) or (nheads, dim)
        z: (batch, seqlen, dim) or (batch, seqlen, nheads, dim)
        dt_bias: (dim,) or (nheads, dim)
    Return:
        out: (batch, seqlen, dim) or (batch, seqlen, nheads, dim)
    """
    has_heads = state.dim() > 3
    if state.dim() == 3:
        state = state.unsqueeze(1)
    if x.dim() == 3:
        x = x.unsqueeze(2)
    if dt.dim() == 3:
        dt = dt.unsqueeze(2)
    if A.dim() == 2:
        A = A.unsqueeze(0)
    if B.dim() == 3:
        B = B.unsqueeze(2)
    if C.dim() == 3:
        C = C.unsqueeze(2)
    if D is not None and D.dim() == 1:
        D = D.unsqueeze(0)
    if z is not None and z.dim() == 3:
        z = z.unsqueeze(2)
    if dt_bias is not None and dt_bias.dim() == 1:
        dt_bias = dt_bias.unsqueeze(0)
    batch, nheads, dim, dstate = state.shape
    SEQLEN = x.shape[1]
    assert x.shape == (batch, SEQLEN, nheads, dim)
    assert dt.shape == x.shape
    assert A.shape == (nheads, dim, dstate)
    ngroups = B.shape[2]
    assert nheads % ngroups == 0, "nheads must be divisible by ngroups"
    assert B.shape == (batch, SEQLEN, ngroups, dstate)
    assert C.shape == B.shape
    if D is not None:
        assert D.shape == (nheads, dim)
    if z is not None:
        assert z.shape == x.shape
    if dt_bias is not None:
        assert dt_bias.shape == (nheads, dim)
    out = torch.empty_like(x)
    grid = lambda META: (triton.cdiv(dim, META['BLOCK_SIZE_M']), batch, nheads)
    z_strides = ((z.stride(0), z.stride(1), z.stride(2), z.stride(3)) if z is not None else (0, 0, 0, 0))
    # We don't want autotune since it will overwrite the state
    # We instead tune by hand.
    BLOCK_SIZE_M, num_warps = ((32, 4) if dstate <= 16
                               else ((16, 4) if dstate <= 32 else
                                     ((8, 4) if dstate <= 64 else
                                      ((4, 4) if dstate <= 128 else
                                       ((4, 8))))))
    tie_hdim = A.stride(-1) == 0 and A.stride(-2) == 0 and dt.stride(-1) == 0 and dt_bias.stride(-1) == 0
    with torch.cuda.device(x.device.index):
        _selective_scan_update_Nstep_kernel[grid](
            state, x, dt, dt_bias, A, B, C, D, z, out,
            batch, nheads, dim, dstate, nheads // ngroups,
            state.stride(0), state.stride(1), state.stride(2), state.stride(3),
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            dt.stride(0), dt.stride(1), dt.stride(2), dt.stride(3),
            *(dt_bias.stride(0), dt_bias.stride(1)) if dt_bias is not None else 0,
            A.stride(0), A.stride(1), A.stride(2),
            B.stride(0), B.stride(1), B.stride(2), B.stride(3),
            C.stride(0), C.stride(1), C.stride(2), C.stride(3),
            *(D.stride(0), D.stride(1)) if D is not None else 0,
            z_strides[0], z_strides[1], z_strides[2], z_strides[3],
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            dt_softplus,
            tie_hdim,
            BLOCK_SIZE_M,
            SEQLEN=SEQLEN,
            num_warps=num_warps,
        )
    if not has_heads:
        out = out.squeeze(1)
    return out

def selective_state_update_Nstep_arp_ref(state, x, dt, A, B, C, state_cache, x_cache, dt_cache, B_cache, replay_mask, state_index, D=None, z=None, dt_bias=None, dt_softplus=False):
    """
    Argument:
        state: (batch, dim, dstate) or (batch, nheads, dim, dstate)
        x: (batch, seqlen, dim) or (batch, seqlen, nheads, dim)
        dt: (batch, seqlen, dim) or (batch, seqlen, nheads, dim)
        A: (dim, dstate) or (nheads, dim, dstate)
        B: (batch, seqlen, dstate) or (batch, seqlen, ngroups, dstate)
        C: (batch, seqlen, dstate) or (batch, seqlen, ngroups, dstate)
        state_cache: (batch, dim, dstate) or (batch, nheads, dim, dstate)
        x_cache: (batch, seqlen, dim) or (batch, seqlen, nheads, dim)
        dt_cache: (batch, seqlen, dim) or (batch, seqlen, nheads, dim)
        B_cache: (batch, seqlen, dstate) or (batch, seqlen, ngroups, dstate)
        replay_mask: (batch, seqlen) use to select the tokens for forwarding. 
        state_index: (batch,) use to select state
        D: (seqlen, dim) or (nheads, seqlen, dim)
        z: (batch, seqlen, dim) or (batch, seqlen, nheads, dim)
        dt_bias: (dim,) or (nheads, dim)
    Return:
        out: (batch, seqlen, dim) or (batch, seqlen, nheads, dim)
    """
    has_heads = state.dim() > 3
    if state.dim() == 3:
        state = state.unsqueeze(1)
    if x.dim() == 3:
        x = x.unsqueeze(2)
    if dt.dim() == 3:
        dt = dt.unsqueeze(2)
    if A.dim() == 2:
        A = A.unsqueeze(1)
    if B.dim() == 3:
        B = B.unsqueeze(2)
    if C.dim() == 3:
        C = C.unsqueeze(2)
    if D is not None and D.dim() == 1:
        D = D.unsqueeze(0)
    if z is not None and z.dim() == 3:
        z = z.unsqueeze(2)
    if dt_bias is not None and dt_bias.dim() == 1:
        dt_bias = dt_bias.unsqueeze(0)
    batch, nheads, dim, dstate = state.shape
    seqlen = x.shape[1]
    assert x.shape == (batch, seqlen, nheads, dim)
    assert dt.shape == x.shape
    assert A.shape == (nheads, dim, dstate)
    ngroups = B.shape[2]
    assert nheads % ngroups == 0, "nheads must be divisible by ngroups"
    assert B.shape == (batch, seqlen, ngroups, dstate)
    assert C.shape == B.shape
    if D is not None:
        assert D.shape == (nheads, dim)
    if z is not None:
        assert z.shape == x.shape
    dt_before_sp = dt.clone()
    if dt_bias is not None:
        assert dt_bias.shape == (nheads, dim)
        dt = dt + dt_bias
    dt = F.softplus(dt) if dt_softplus else dt # (batch, seqlen, nheads, dim)
    dA = torch.exp(rearrange(dt, "b l h d -> b l h d 1") * A)  # (batch, seqlen, nheads, dim, dstate)
    B = repeat(B, "b l g n -> b l (g h) n", h=nheads // ngroups)  # (batch, seqlen, nheads, dstate)
    C = repeat(C, "b l g n -> b l (g h) n", h=nheads // ngroups)  # (batch, seqlen, nheads, dstate)
    dB = rearrange(dt, "b l h d -> b l h d 1") * rearrange(B, "b l h n -> b l h 1 n")  # (batch, seqlen, nheads, dim, dstate)
    out = []
    l = torch.sum(replay_mask, dim=1).max().item()
    mask_indices = mask_index_gather(replay_mask, l).squeeze(0)
    temp_state = copy.deepcopy(state_cache)[state_index, ...] # selecting the correct state
    dt_cache_post_sp = F.softplus(dt_cache + dt_bias)
    temp_dA = torch.exp(rearrange(dt_cache_post_sp[state_index,...], "b l h d -> b l h d 1") * A)
    temp_dB = rearrange(dt_cache_post_sp[state_index, ...], "b l h d -> b l h d 1") * rearrange(B_cache[state_index, ...], "b l h n -> b l h 1 n")
    temp_x = x_cache[state_index, ...]

    for j in range(batch):
        for i in range(mask_indices.shape[1]):
            if mask_indices[j, i] != -1:
                temp_state[[j],...] = (temp_state[[j],...] * temp_dA[[j],mask_indices[j,i],...] + \
                    temp_dB[[j],mask_indices[j,i],...] * rearrange(temp_x[[j],mask_indices[j,i],...], "b h d -> b h d 1")).to(torch.float16)

    state_cache.copy_(temp_state)

    for i in range(dA.shape[1]):
        temp_state = (temp_state * dA[:, i, ...] + dB[:, i, ...] * rearrange(x[:, i, ...], "b h d -> b h d 1"))  
        out.append(torch.einsum("bhdn,bhn->bhd", temp_state.to(C.dtype), C[:, i, ...]))
        x_cache[:, i, ...].copy_(x[:, i, ...])
        dt_cache[:, i, :, 0].copy_(dt_before_sp[:, i, :, 0])
        B_cache[:, i, ...].copy_(B[:, i, [0],...])
    out = torch.stack(out, dim=1)
    if D is not None:
        out += (x * D).to(out.dtype)
    out = (out if z is None else out * F.silu(z)).to(x.dtype)
    if not has_heads:
        out = out.squeeze(1)
    state.copy_(temp_state)
    return out

# The Nstep_kernel is trying to extend 2 step kernel to an arbitrary number of steps. The problem with triton is that
# it doesn't support indexing of already loaded tensors and .split() function only works if the dimension being split 
# is of size 2. Therefore, we choose to use a for loop instead and load and compute each time step sequentially. It 
# turns out that this is still pretty fast, potentially thanks to the parallel execution of memory loading and computation
@triton.heuristics({"HAS_DT_BIAS": lambda args: args["dt_bias_ptr"] is not None})
@triton.heuristics({"HAS_D": lambda args: args["D_ptr"] is not None})
@triton.heuristics({"HAS_Z": lambda args: args["z_ptr"] is not None})
@triton.heuristics({"BLOCK_SIZE_DSTATE": lambda args: triton.next_power_of_2(args["dstate"])})
@triton.jit
def _selective_scan_update_Nstep_arp_kernel(
    # Pointers to matrices
    state_ptr, x_ptr, dt_ptr, dt_bias_ptr, A_ptr, B_ptr, C_ptr, 
    # ARP pointers 
    state_cache_ptr, x_cache_ptr, dt_cache_ptr, B_cache_ptr, B_cache_out_ptr, mask_indices_ptr,
    D_ptr, z_ptr, out_ptr,
    # Matrix dimensions
    batch, nheads, dim, dstate, nheads_ngroups_ratio,
    # Strides
    stride_state_batch, stride_state_head, stride_state_dim, stride_state_dstate,
    stride_x_batch, stride_x_seqlen, stride_x_head, stride_x_dim,
    stride_dt_batch, stride_dt_seqlen, stride_dt_head, stride_dt_dim,
    stride_dt_bias_head, stride_dt_bias_dim,
    stride_A_head, stride_A_dim, stride_A_dstate,
    stride_B_batch, stride_B_seqlen, stride_B_group, stride_B_dstate,
    stride_C_batch, stride_C_seqlen, stride_C_group, stride_C_dstate,
    stride_state_cache_batch, stride_state_cache_head, stride_state_cache_dim, stride_state_cache_dstate,
    stride_x_cache_batch, stride_x_cache_seqlen, stride_x_cache_head, stride_x_cache_dim,
    stride_dt_cache_batch, stride_dt_cache_seqlen, stride_dt_cache_head, stride_dt_cache_dim,
    stride_B_cache_batch, stride_B_cache_seqlen, stride_B_cache_group, stride_B_cache_dstate,
    stride_B_cache_out_batch, stride_B_cache_out_seqlen, stride_B_cache_out_head, stride_B_cache_out_dstate,
    stride_mask_batch, stride_mask_statelen,
    stride_D_head, stride_D_dim,
    stride_z_batch, stride_Z_seqlen, stride_z_head, stride_z_dim,
    stride_out_batch, stride_out_seqlen, stride_out_head, stride_out_dim,
    # Meta-parameters
    DT_SOFTPLUS: tl.constexpr,
    TIE_HDIM: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    HAS_DT_BIAS: tl.constexpr,
    HAS_D: tl.constexpr,
    HAS_Z: tl.constexpr,
    BLOCK_SIZE_DSTATE: tl.constexpr,
    SEQLEN: tl.constexpr,
    MASK_LEN: tl.constexpr
):
    pid_m = tl.program_id(axis=0)
    pid_b = tl.program_id(axis=1)
    pid_h = tl.program_id(axis=2)
    state_ptr += pid_b * stride_state_batch + pid_h * stride_state_head
    x_ptr += pid_b * stride_x_batch + pid_h * stride_x_head
    dt_ptr += pid_b * stride_dt_batch  + pid_h * stride_dt_head
    state_cache_ptr += pid_b * stride_state_cache_batch + pid_h * stride_state_cache_head  
    x_cache_ptr += pid_b * stride_x_cache_batch + pid_h * stride_x_cache_head
    dt_cache_ptr += pid_b * stride_dt_cache_batch  + pid_h * stride_dt_cache_head
    mask_indices_ptr += pid_b * stride_mask_batch
    if HAS_DT_BIAS:
        dt_bias_ptr += pid_h * stride_dt_bias_head
    A_ptr += pid_h * stride_A_head
    B_ptr += pid_b * stride_B_batch + (pid_h // nheads_ngroups_ratio) * stride_B_group
    B_cache_ptr += pid_b * stride_B_cache_batch + (pid_h // nheads_ngroups_ratio) * stride_B_cache_group
    B_cache_out_ptr += pid_b * stride_B_cache_out_batch + pid_h * stride_B_cache_out_head
    C_ptr += pid_b * stride_C_batch + (pid_h // nheads_ngroups_ratio) * stride_C_group
    if HAS_Z:
        z_ptr += pid_b * stride_z_batch + pid_h * stride_z_head
    out_ptr += pid_b * stride_out_batch + pid_h * stride_out_head

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = tl.arange(0, BLOCK_SIZE_DSTATE)
    state_ptrs = state_ptr + (offs_m[:, None] * stride_state_dim + offs_n[None, :] * stride_state_dstate) # (dim_block_size, dstate_block_size)
    state_cache_ptrs = state_cache_ptr + (offs_m[:, None] * stride_state_cache_dim + offs_n[None, :] * stride_state_cache_dstate) # (dim_block_size, dstate_block_size)
    x_ptrs = x_ptr + (offs_m * stride_x_dim ) # (dim_block_size, )
    x_cache_ptrs = x_cache_ptr + (offs_m * stride_x_cache_dim ) # (dim_block_size, )
    dt_ptrs = dt_ptr + (offs_m * stride_dt_dim ) # (dim_block_size, )
    dt_cache_ptrs = dt_cache_ptr + (pid_m * stride_dt_cache_dim ) # (1, )
    if HAS_DT_BIAS:
        dt_bias_ptrs = dt_bias_ptr + offs_m * stride_dt_bias_dim
    if HAS_D:
        D_ptr += pid_h * stride_D_head
    A_ptrs = A_ptr + (offs_m[:, None] * stride_A_dim + offs_n[None, :] * stride_A_dstate) # (dim_block_size, dstate_block_size)
    B_ptrs = B_ptr + (offs_n * stride_B_dstate ) # (dstate_block_size, )
    B_cache_ptrs = B_cache_ptr + (offs_n * stride_B_cache_dstate ) # (dstate_block_size, )
    B_cache_out_ptrs = B_cache_out_ptr + (offs_n * stride_B_cache_out_dstate)
    C_ptrs = C_ptr + (offs_n * stride_C_dstate ) # (dstate_block_size, )
    if HAS_D:
        D_ptrs = D_ptr + offs_m * stride_D_dim # (dim_block_size,)
    if HAS_Z:
        z_ptrs = z_ptr + (offs_m * stride_z_dim )  # (dim_block_size, )
    out_ptrs = out_ptr + (offs_m * stride_out_dim )  # (dim_block_size, )

    state = tl.load(state_cache_ptrs, mask=(offs_m[:, None] < dim) & (offs_n[None, :] < dstate), other=0.0) # this is our temp state
    A = tl.load(A_ptr).to(tl.float32)

    for s in tl.static_range(MASK_LEN):
        mask_indices_ptr_masklen = mask_indices_ptr + s * stride_mask_statelen
        m = tl.load(mask_indices_ptr_masklen, mask=(s < MASK_LEN), other=0.0).to(tl.int32) # (1,)

        x_cache_ptrs_masklen = x_cache_ptrs + m * stride_x_cache_seqlen
        B_cache_ptrs_masklen = B_cache_ptrs + m * stride_B_cache_seqlen
        dt_cache_ptrs_masklen = dt_cache_ptrs + m * stride_dt_cache_seqlen

        x_cache = tl.load(x_cache_ptrs_masklen, mask=(offs_m < dim) & (m != -1), other=0.0).to(tl.float32)
        B_cache = tl.load(B_cache_ptrs_masklen, mask=(offs_n < dstate) & (m != -1), other=0.0).to(tl.float32)
        # assumes tie hdim here
        # To prevent race condition on dt cache, we cloned the cache tensor to make sure each thread 
        # has a independent dt cache to work on, although what each thread did to it is the same
        dt_cache = tl.load(dt_cache_ptrs_masklen, mask=(m != -1), other=-1e4).to(tl.float32)
        if HAS_DT_BIAS:
            dt_cache += tl.load(dt_bias_ptr).to(tl.float32)[None, :]
        if DT_SOFTPLUS:
            dt_cache = softplus(dt_cache)

        # assumes tie hdim is true here
        dA_cache = tl.exp(A * dt_cache)  # (1,)
        dB_cache = B_cache * dt_cache  # vector of size (dstate,)
        state = state * dA_cache + dB_cache * x_cache[:, None]

    tl.store(state_cache_ptrs, state, mask=(offs_m[:, None] < dim) & (offs_n[None, :] < dstate))

    for s in tl.static_range(SEQLEN):
        x_ptrs_seqlen = x_ptrs + s * stride_x_seqlen
        dt_ptrs_seqlen = dt_ptrs + s * stride_dt_seqlen
        B_ptrs_seqlen = B_ptrs + s * stride_B_seqlen
        C_ptrs_seqlen = C_ptrs + s * stride_C_seqlen
        if HAS_Z:
            z_ptrs_seqlen = z_ptrs + s * stride_Z_seqlen
        out_ptrs_seqlen = out_ptrs + s * stride_out_seqlen
        x = tl.load(x_ptrs_seqlen, mask=offs_m < dim, other=0.0).to(tl.float32)
        # current implmentation assuming TIE_HDIM is true
        if not TIE_HDIM:
            dt = tl.load(dt_ptrs_seqlen, mask=offs_m < dim, other=0.0).to(tl.float32) # (dim_block_size,)
            if HAS_DT_BIAS:
                dt += tl.load(dt_bias_ptrs, mask=offs_m < dim, other=0.0)[:, None].to(tl.float32)
            if DT_SOFTPLUS:
                dt = softplus(dt)
            A = tl.load(A_ptrs, mask=(offs_m[:, None] < dim) & (offs_n[None, :] < dstate), other=0.0).to(tl.float32)
            dA = tl.exp(A * dt[:, None]) # (dim_block_size, dstate_block_size)
        else:
            dt = tl.load(dt_ptr + (s * stride_dt_seqlen)).to(tl.float32) # (1,)
            dt_cache_ptrs_seqlen = dt_cache_ptrs + s * stride_dt_cache_seqlen
            tl.store(dt_cache_ptrs_seqlen, dt)
            if HAS_DT_BIAS:
                dt += tl.load(dt_bias_ptr).to(tl.float32)[None, :]
            if DT_SOFTPLUS:
                dt = softplus(dt)
            # A = tl.load(A_ptr).to(tl.float32)
            dA = tl.exp(A * dt)  # (1,)

        B = tl.load(B_ptrs_seqlen, mask=offs_n < dstate, other=0.0).to(tl.float32)
        C = tl.load(C_ptrs_seqlen, mask=offs_n < dstate, other=0.0).to(tl.float32)
        if HAS_D:
            D = tl.load(D_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
        if HAS_Z:
            z = tl.load(z_ptrs_seqlen, mask=offs_m < dim, other=0.0).to(tl.float32)

        if not TIE_HDIM:
            dB = B[None, :] * dt[:, None] # (dim_block_size, dstate_block_size, seqlen)
        else:
            dB = B * dt  # vector of size (dstate,)

        state = state * dA + dB * x[:, None]
        out = tl.sum(state * C[None, :], axis=1)
        if HAS_D:
            out += x * D
        if HAS_Z:
            out *= z * tl.sigmoid(z)
        tl.store(out_ptrs_seqlen, out, mask=offs_m < dim)

        # storing x, dt, B in cache
        x_cache_ptrs_seqlen = x_cache_ptrs + s * stride_x_cache_seqlen
        B_cache_out_ptrs_seqlen = B_cache_out_ptrs + s * stride_B_cache_out_seqlen
        tl.store(x_cache_ptrs_seqlen, x, mask=offs_m < dim)
        # assume tie hdim here
        tl.store(B_cache_out_ptrs_seqlen, B, mask=offs_n < dstate)

    tl.store(state_ptrs, state, mask=(offs_m[:, None] < dim) & (offs_n[None, :] < dstate))

def selective_state_update_Nstep_arp(state, x, dt, A, B, C, state_cache, x_cache, dt_cache, B_cache, replay_mask, state_index, max_len, D=None, z=None, dt_bias=None, dt_softplus=False):
    """
    Argument:
        state: (batch, dim, dstate) or (batch, nheads, dim, dstate)
        x: (batch, seqlen, dim) or (batch, seqlen, nheads, dim)
        dt: (batch, seqlen, dim) or (batch, seqlen, nheads, dim)
        A: (dim, dstate) or (nheads, dim, dstate)
        B: (batch, seqlen, dstate) or (batch, seqlen, ngroups, dstate)
        C: (batch, seqlen, dstate) or (batch, seqlen, ngroups, dstate)
        D: (dim,) or (nheads, dim)
        z: (batch, seqlen, dim) or (batch, seqlen, nheads, dim)
        dt_bias: (dim,) or (nheads, dim)
    Return:
        out: (batch, seqlen, dim) or (batch, seqlen, nheads, dim)
    """
    has_heads = state.dim() > 3
    if state.dim() == 3:
        state = state.unsqueeze(1)
    if x.dim() == 3:
        x = x.unsqueeze(2)
    if dt.dim() == 3:
        dt = dt.unsqueeze(2)
    if A.dim() == 2:
        A = A.unsqueeze(0)
    if B.dim() == 3:
        B = B.unsqueeze(2)
    if C.dim() == 3:
        C = C.unsqueeze(2)
    if D is not None and D.dim() == 1:
        D = D.unsqueeze(0)
    if z is not None and z.dim() == 3:
        z = z.unsqueeze(2)
    if dt_bias is not None and dt_bias.dim() == 1:
        dt_bias = dt_bias.unsqueeze(0)
    batch, nheads, dim, dstate = state.shape
    SEQLEN = x.shape[1]
    assert x.shape == (batch, SEQLEN, nheads, dim)
    assert dt.shape == x.shape
    assert A.shape == (nheads, dim, dstate)
    ngroups = B.shape[2]
    assert nheads % ngroups == 0, "nheads must be divisible by ngroups"
    assert B.shape == (batch, SEQLEN, ngroups, dstate)
    assert C.shape == B.shape
    if D is not None:
        assert D.shape == (nheads, dim)
    if z is not None:
        assert z.shape == x.shape
    if dt_bias is not None:
        assert dt_bias.shape == (nheads, dim)
    out = torch.empty_like(x)
    grid = lambda META: (triton.cdiv(dim, META['BLOCK_SIZE_M']), batch, nheads)
    npad = batch // state_cache.shape[0]
    z_strides = ((z.stride(0), z.stride(1), z.stride(2), z.stride(3)) if z is not None else (0, 0, 0, 0))
    # We don't want autotune since it will overwrite the state
    # We instead tune by hand.
    BLOCK_SIZE_M, num_warps = ((32, 4) if dstate <= 16
                               else ((16, 4) if dstate <= 32 else
                                     ((8, 4) if dstate <= 64 else
                                      ((4, 4) if dstate <= 128 else
                                       ((4, 8))))))
    mask_indices = mask_index_gather(replay_mask, max_len).squeeze(0)
    # print(state_cache.shape, state_cache.stride(0), state_cache.stride(1), state_cache.stride(2), state_cache.stride(3))
    dt_cache_clone = dt_cache.clone() # stride(3) is not longer 0
    B_cache_out = torch.empty((batch,SEQLEN,nheads,dstate), dtype=B_cache.dtype, device=B_cache.device)
    # B_cache is of shape batch x seqlen x ngroups x dstate, where ngroups is 1 
    # multiple head is reading/writing to the same group, which cause race condition
    inplace_state_indexing(state_cache, state_index)
    inplace_state_indexing(B_cache, state_index)
    inplace_state_indexing(x_cache, state_index)
    inplace_state_indexing(dt_cache_clone, state_index)
    tie_hdim = A.stride(-1) == 0 and A.stride(-2) == 0 and dt.stride(-1) == 0 and dt_bias.stride(-1) == 0
    with torch.cuda.device(x.device.index):
        _selective_scan_update_Nstep_arp_kernel[grid](
            state, x, dt, dt_bias, A, B, C, 
            state_cache, x_cache, dt_cache_clone, B_cache, B_cache_out, mask_indices,
            D, z, out,
            batch, nheads, dim, dstate, nheads // ngroups,
            state.stride(0), state.stride(1), state.stride(2), state.stride(3),
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            dt.stride(0), dt.stride(1), dt.stride(2), dt.stride(3),
            *(dt_bias.stride(0), dt_bias.stride(1)) if dt_bias is not None else 0,
            A.stride(0), A.stride(1), A.stride(2),
            B.stride(0), B.stride(1), B.stride(2), B.stride(3),
            C.stride(0), C.stride(1), C.stride(2), C.stride(3),
            state_cache.stride(0), state_cache.stride(1), state_cache.stride(2), state_cache.stride(3),
            x_cache.stride(0), x_cache.stride(1), x_cache.stride(2), x_cache.stride(3),
            dt_cache_clone.stride(0), dt_cache_clone.stride(1), dt_cache_clone.stride(2), dt_cache_clone.stride(3),
            B_cache.stride(0), B_cache.stride(1), B_cache.stride(2), B_cache.stride(3),
            B_cache_out.stride(0), B_cache_out.stride(1), B_cache_out.stride(2), B_cache_out.stride(3),
            mask_indices.stride(0), mask_indices.stride(1),
            *(D.stride(0), D.stride(1)) if D is not None else 0,
            z_strides[0], z_strides[1], z_strides[2], z_strides[3],
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            dt_softplus,
            tie_hdim,
            BLOCK_SIZE_M,
            SEQLEN=SEQLEN,
            MASK_LEN=mask_indices.shape[1],
            num_warps=num_warps,
        )
        dt_cache[:,:,:,0].copy_(dt_cache_clone[:,:,:,0])
        B_cache.copy_(B_cache_out[:,:,[0],:])
    if not has_heads:
        out = out.squeeze(1)
    return out

def selective_state_update_tree_ref(state, x, dt, A, B, C, mask, D=None, z=None, dt_bias=None, dt_softplus=False):
    """
    Attempting to perform tree decoding with selective scan

    Argument:
        state: (batch, dim, dstate) or (batch, nheads, dim, dstate)
        x: (batch, seqlen, dim) or (batch, seqlen, nheads, dim)
        dt: (batch, seqlen, dim) or (batch, seqlen, nheads, dim)
        A: (dim, dstate) or (nheads, dim, dstate)
        B: (batch, seqlen, dstate) or (batch, seqlen, ngroups, dstate)
        C: (batch, seqlen, dstate) or (batch, seqlen, ngroups, dstate)
        mask: (batch, seqlen, seqlen): the attention mask indicating the tree structure
        D: (seqlen, dim) or (nheads, seqlen, dim)
        z: (batch, seqlen, dim) or (batch, seqlen, nheads, dim)
        dt_bias: (dim,) or (nheads, dim)
    Return:
        out: (batch, seqlen, dim) or (batch, seqlen, nheads, dim)
    """
    has_heads = state.dim() > 3
    if state.dim() == 3:
        state = state.unsqueeze(1)
    if x.dim() == 3:
        x = x.unsqueeze(2)
    if dt.dim() == 3:
        dt = dt.unsqueeze(2)
    if A.dim() == 2:
        A = A.unsqueeze(1)
    if B.dim() == 3:
        B = B.unsqueeze(2)
    if C.dim() == 3:
        C = C.unsqueeze(2)
    if D is not None and D.dim() == 1:
        D = D.unsqueeze(0)
    if z is not None and z.dim() == 3:
        z = z.unsqueeze(2)
    if dt_bias is not None and dt_bias.dim() == 1:
        dt_bias = dt_bias.unsqueeze(0)
    batch, nheads, dim, dstate = state.shape
    seqlen = x.shape[1]
    assert x.shape == (batch, seqlen, nheads, dim)
    assert dt.shape == x.shape
    assert A.shape == (nheads, dim, dstate)
    ngroups = B.shape[2]
    assert nheads % ngroups == 0, "nheads must be divisible by ngroups"
    assert B.shape == (batch, seqlen, ngroups, dstate)
    assert C.shape == B.shape
    if D is not None:
        assert D.shape == (nheads, dim)
    if z is not None:
        assert z.shape == x.shape
    if dt_bias is not None:
        assert dt_bias.shape == (nheads, dim)
        dt = dt + dt_bias
    dt = F.softplus(dt) if dt_softplus else dt # (batch, seqlen, nheads, dim)
    dA = torch.exp(rearrange(dt, "b l h d -> b l h d 1") * A)  # (batch, seqlen, nheads, dim, dstate)
    B = repeat(B, "b l g n -> b l (g h) n", h=nheads // ngroups)  # (batch, seqlen, nheads, dstate)
    C = repeat(C, "b l g n -> b l (g h) n", h=nheads // ngroups)  # (batch, seqlen, nheads, dstate)
    dB = rearrange(dt, "b l h d -> b l h d 1") * rearrange(B, "b l h n -> b l h 1 n")  # (batch, seqlen, nheads, dim, dstate)

    assert batch == 1 # TODO: assuming there is only 1 sequence
    parent_matrix = mask_index_gather(mask, mask.shape[1])
    parent_node_indices = torch.unique(parent_matrix[:, :, -2])
    all_node_indices = torch.arange(mask.shape[1], device=mask.device)
    leaf_node_indices = all_node_indices[~torch.isin(all_node_indices, parent_node_indices)]
    leaf_sequences = parent_matrix[:, leaf_node_indices, :]
    leaf_seqeunces_mask = leaf_sequences==-1
    leaf_sequences[leaf_seqeunces_mask] = 0
    out_inorder = torch.empty((batch, seqlen, nheads, dim), device=x.device, dtype=x.dtype)

    temp_state = repeat(state, "b ... -> (b r) ...", r=leaf_sequences.shape[1]).clone()
    out = []
    for i in range(leaf_sequences.shape[2]):
        dA_temp = torch.index_select(dA, dim=1, index=leaf_sequences[0, :, i]).squeeze()
        dA_temp[leaf_seqeunces_mask[0, :, i], ...] = 1
        dB_temp = torch.index_select(dB, dim=1, index=leaf_sequences[0, :, i]).squeeze()
        dB_temp[leaf_seqeunces_mask[0, :, i], ...] = 0

        x_temp = torch.index_select(x, dim=1, index=leaf_sequences[0, :, i]).squeeze()
        
        C_temp = torch.index_select(C, dim=1, index=leaf_sequences[0, :, i]).squeeze()
        temp_state.copy_(temp_state * dA_temp + dB_temp * rearrange(x_temp, "b h d -> b h d 1"))  
        out_temp = torch.einsum("bhdn,bhn->bhd", temp_state.to(C.dtype), C_temp)
        if D is not None:
            out_temp += (x_temp * D).to(out_temp.dtype)
        if z is not None:
            z_temp = torch.index_select(z, dim=1, index=leaf_sequences[0, :, i]).squeeze()
        out_temp = (out_temp * F.silu(z_temp)).to(x.dtype)
        out.append(out_temp)
    out = torch.stack(out, dim=1) 
    for j in range(leaf_sequences.shape[1]):
        for i in range(leaf_sequences.shape[2]):
            if not leaf_seqeunces_mask[0, j, i]:
                out_inorder[0, leaf_sequences[0, j, i], :, :] = out[j, i, :, :]
    if not has_heads:
        out_inorder = out_inorder.squeeze(1)
    return out_inorder

# This is a kernel that attempts to do tree decoding with selective scan. For each thread, we are going to process one sub sequence
@triton.heuristics({"HAS_DT_BIAS": lambda args: args["dt_bias_ptr"] is not None})
@triton.heuristics({"HAS_D": lambda args: args["D_ptr"] is not None})
@triton.heuristics({"HAS_Z": lambda args: args["z_ptr"] is not None})
@triton.heuristics({"BLOCK_SIZE_DSTATE": lambda args: triton.next_power_of_2(args["dstate"])})
@triton.jit
def _selective_scan_update_tree_kernel(
    # Pointers to matrices
    state_ptr, x_ptr, dt_ptr, dt_bias_ptr, A_ptr, B_ptr, C_ptr, D_ptr, z_ptr, out_ptr, leaf_seq_ptr,
    # Matrix dimensions
    batch, nheads, dim, dstate, nheads_ngroups_ratio,
    # Strides
    stride_state_batch, stride_state_head, stride_state_dim, stride_state_dstate,
    stride_x_batch, stride_x_seqlen, stride_x_head, stride_x_dim,
    stride_dt_batch, stride_dt_seqlen, stride_dt_head, stride_dt_dim,
    stride_dt_bias_head, stride_dt_bias_dim,
    stride_A_head, stride_A_dim, stride_A_dstate,
    stride_B_batch, stride_B_seqlen, stride_B_group, stride_B_dstate,
    stride_C_batch, stride_C_seqlen, stride_C_group, stride_C_dstate,
    stride_D_head, stride_D_dim,
    stride_z_batch, stride_Z_seqlen, stride_z_head, stride_z_dim,
    stride_out_batch, stride_out_seqlen, stride_out_head, stride_out_dim,
    stride_leaf_batch, stride_leaf_num_seq, stride_leaf_seqlen,
    # Meta-parameters
    DT_SOFTPLUS: tl.constexpr,
    TIE_HDIM: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    HAS_DT_BIAS: tl.constexpr,
    HAS_D: tl.constexpr,
    HAS_Z: tl.constexpr,
    BLOCK_SIZE_DSTATE: tl.constexpr,
    SEQLEN: tl.constexpr,
    NUM_SEQ: tl.constexpr
):
    pid_m = tl.program_id(axis=0)
    pid_b = tl.program_id(axis=1) // NUM_SEQ
    pid_l = tl.program_id(axis=1) % NUM_SEQ
    pid_h = tl.program_id(axis=2)
    state_ptr += pid_b * stride_state_batch + pid_h * stride_state_head
    x_ptr += pid_b * stride_x_batch + pid_h * stride_x_head
    dt_ptr += pid_b * stride_dt_batch  + pid_h * stride_dt_head
    if HAS_DT_BIAS:
        dt_bias_ptr += pid_h * stride_dt_bias_head
    A_ptr += pid_h * stride_A_head
    B_ptr += pid_b * stride_B_batch + (pid_h // nheads_ngroups_ratio) * stride_B_group
    C_ptr += pid_b * stride_C_batch + (pid_h // nheads_ngroups_ratio) * stride_C_group
    if HAS_Z:
        z_ptr += pid_b * stride_z_batch + pid_h * stride_z_head
    out_ptr += pid_b * stride_out_batch + pid_h * stride_out_head
    leaf_seq_ptr += pid_b * stride_leaf_batch + pid_l * stride_leaf_num_seq

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = tl.arange(0, BLOCK_SIZE_DSTATE)
    state_ptrs = state_ptr + (offs_m[:, None] * stride_state_dim + offs_n[None, :] * stride_state_dstate) # (dim_block_size, dstate_block_size)
    x_ptrs = x_ptr + (offs_m * stride_x_dim ) # (dim_block_size, )
    dt_ptrs = dt_ptr + (offs_m * stride_dt_dim ) # (dim_block_size, )
    if HAS_DT_BIAS:
        dt_bias_ptrs = dt_bias_ptr + offs_m * stride_dt_bias_dim
    if HAS_D:
        D_ptr += pid_h * stride_D_head
    A_ptrs = A_ptr + (offs_m[:, None] * stride_A_dim + offs_n[None, :] * stride_A_dstate) # (dim_block_size, dstate_block_size)
    B_ptrs = B_ptr + (offs_n * stride_B_dstate ) # (dstate_block_size, )
    C_ptrs = C_ptr + (offs_n * stride_C_dstate ) # (dstate_block_size, )
    if HAS_D:
        D_ptrs = D_ptr + offs_m * stride_D_dim # (dim_block_size,)
    if HAS_Z:
        z_ptrs = z_ptr + (offs_m * stride_z_dim )  # (dim_block_size, )
    out_ptrs = out_ptr + (offs_m * stride_out_dim )  # (dim_block_size, )

    state = tl.load(state_ptrs, mask=(offs_m[:, None] < dim) & (offs_n[None, :] < dstate), other=0.0)
    for s in tl.static_range(SEQLEN):
        leaf_ptr_seqlen = leaf_seq_ptr + s * stride_leaf_seqlen
        leaf_index = tl.load(leaf_ptr_seqlen).to(tl.int64)

        x_ptrs_seqlen = x_ptrs + leaf_index * stride_x_seqlen
        dt_ptrs_seqlen = dt_ptrs + leaf_index * stride_dt_seqlen
        B_ptrs_seqlen = B_ptrs + leaf_index * stride_B_seqlen
        C_ptrs_seqlen = C_ptrs + leaf_index * stride_C_seqlen
        if HAS_Z:
            z_ptrs_seqlen = z_ptrs + leaf_index * stride_Z_seqlen
        out_ptrs_seqlen = out_ptrs + leaf_index * stride_out_seqlen
        x = tl.load(x_ptrs_seqlen, mask=(offs_m < dim) & (leaf_index >= 0) , other=0.0).to(tl.float32)
        if not TIE_HDIM:
            dt = tl.load(dt_ptrs_seqlen, mask=(offs_m < dim) & (leaf_index >= 0), other=0.0).to(tl.float32) # (dim_block_size,)
            if HAS_DT_BIAS:
                dt += tl.load(dt_bias_ptrs, mask=offs_m < dim, other=0.0)[:, None].to(tl.float32)
            if DT_SOFTPLUS:
                dt = softplus(dt)
            A = tl.load(A_ptrs, mask=(offs_m[:, None] < dim) & (offs_n[None, :] < dstate), other=0.0).to(tl.float32)
            dA = tl.exp(A * dt[:, None]) # (dim_block_size, dstate_block_size)
            dA = tl.where(leaf_index < 0, 1, dA) # If padding position, we set dA to 1 to preserve state
        else:
            dt = tl.load(dt_ptr + (leaf_index * stride_dt_seqlen), mask=leaf_index >= 0, other=0.0).to(tl.float32) # (1,) 
            if HAS_DT_BIAS:
                dt += tl.load(dt_bias_ptr).to(tl.float32)[None, :]
            if DT_SOFTPLUS:
                dt = softplus(dt)
            A = tl.load(A_ptr).to(tl.float32)
            dA = tl.exp(A * dt)  # (1,)
            dA = tl.where(leaf_index < 0, 1, dA) # If padding position, we set dA to 1 to preserve state

        B = tl.load(B_ptrs_seqlen, mask=(offs_n < dstate) & (leaf_index >= 0), other=0.0).to(tl.float32)
        C = tl.load(C_ptrs_seqlen, mask=(offs_n < dstate) & (leaf_index >= 0), other=0.0).to(tl.float32)
        if HAS_D:
            D = tl.load(D_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
        if HAS_Z:
            z = tl.load(z_ptrs_seqlen, mask=(offs_m < dim) & (leaf_index >= 0), other=0.0).to(tl.float32)

        if not TIE_HDIM:
            dB = B[None, :] * dt[:, None] # (dim_block_size, dstate_block_size, seqlen)
        else:
            dB = B * dt  # vector of size (dstate,)

        state = state * dA + dB * x[:, None]
        out = tl.sum(state * C[None, :], axis=1)
        if HAS_D:
            out += x * D
        if HAS_Z:
            out *= z * tl.sigmoid(z)
        tl.store(out_ptrs_seqlen, out, mask=(offs_m < dim) & (leaf_index >= 0))

    # tl.store(state_ptrs, state, mask=(offs_m[:, None] < dim) & (offs_n[None, :] < dstate))

def selective_state_update_tree(state, x, dt, A, B, C, leaf_sequences, D=None, z=None, dt_bias=None, dt_softplus=False):
    """
    Argument:
        state: (batch, dim, dstate) or (batch, nheads, dim, dstate)
        x: (batch, seqlen, dim) or (batch, seqlen, nheads, dim)
        dt: (batch, seqlen, dim) or (batch, seqlen, nheads, dim)
        A: (dim, dstate) or (nheads, dim, dstate)
        B: (batch, seqlen, dstate) or (batch, seqlen, ngroups, dstate)
        C: (batch, seqlen, dstate) or (batch, seqlen, ngroups, dstate)
        leaf_sequences: (batch, NUM_SEQ, SEQLEN) indices of subsequences in the tree 
        D: (dim,) or (nheads, dim)
        z: (batch, seqlen, dim) or (batch, seqlen, nheads, dim)
        dt_bias: (dim,) or (nheads, dim)
    Return:
        out: (batch, seqlen, dim) or (batch, seqlen, nheads, dim)
    """
    has_heads = state.dim() > 3
    if state.dim() == 3:
        state = state.unsqueeze(1)
    if x.dim() == 3:
        x = x.unsqueeze(2)
    if dt.dim() == 3:
        dt = dt.unsqueeze(2)
    if A.dim() == 2:
        A = A.unsqueeze(0)
    if B.dim() == 3:
        B = B.unsqueeze(2)
    if C.dim() == 3:
        C = C.unsqueeze(2)
    if D is not None and D.dim() == 1:
        D = D.unsqueeze(0)
    if z is not None and z.dim() == 3:
        z = z.unsqueeze(2)
    if dt_bias is not None and dt_bias.dim() == 1:
        dt_bias = dt_bias.unsqueeze(0)
    assert leaf_sequences is not None
    batch, nheads, dim, dstate = state.shape
    assert batch == 1 # TODO: assuming that there is only 1 batch first
    SEQLEN_tree = x.shape[1]
    assert x.shape == (batch, SEQLEN_tree, nheads, dim)
    assert dt.shape == x.shape
    assert A.shape == (nheads, dim, dstate)
    ngroups = B.shape[2]
    assert nheads % ngroups == 0, "nheads must be divisible by ngroups"
    assert B.shape == (batch, SEQLEN_tree, ngroups, dstate)
    assert C.shape == B.shape
    if D is not None:
        assert D.shape == (nheads, dim)
    if z is not None:
        assert z.shape == x.shape
    if dt_bias is not None:
        assert dt_bias.shape == (nheads, dim)

    # from mamba_ssm.utils.profile import cuda_time
    # with cuda_time("mask gather"):
    #     parent_matrix = mask_index_gather(mask, max_seqlen)
    # with cuda_time("unique"):
    #     parent_node_indices = torch.unique(parent_matrix[:, :, -2])
    # with cuda_time("arange"):
    #     all_node_indices = torch.arange(mask.shape[2], device=mask.device)
    # with cuda_time("is in"):
    #     leaf_node_indices = all_node_indices[~torch.isin(all_node_indices, parent_node_indices)]
    # with cuda_time("processing"):
    #     leaf_sequences = parent_matrix[:, leaf_node_indices, :]
    #     SEQLEN = leaf_sequences.shape[2]
    #     NUM_SEQ = leaf_sequences.shape[1]
    SEQLEN = leaf_sequences.shape[2]
    NUM_SEQ = leaf_sequences.shape[1]

    out = torch.empty_like(x)
    grid = lambda META: (triton.cdiv(dim, META['BLOCK_SIZE_M']), batch * NUM_SEQ, nheads)
    z_strides = ((z.stride(0), z.stride(1), z.stride(2), z.stride(3)) if z is not None else (0, 0, 0, 0))
    # We don't want autotune since it will overwrite the state
    # We instead tune by hand.
    BLOCK_SIZE_M, num_warps = ((32, 4) if dstate <= 16
                               else ((16, 4) if dstate <= 32 else
                                     ((8, 4) if dstate <= 64 else
                                      ((4, 4) if dstate <= 128 else
                                       ((4, 8))))))
    tie_hdim = A.stride(-1) == 0 and A.stride(-2) == 0 and dt.stride(-1) == 0 and dt_bias.stride(-1) == 0
    with torch.cuda.device(x.device.index):
        _selective_scan_update_tree_kernel[grid](
            state, x, dt, dt_bias, A, B, C, D, z, out, leaf_sequences,
            batch, nheads, dim, dstate, nheads // ngroups,
            state.stride(0), state.stride(1), state.stride(2), state.stride(3),
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            dt.stride(0), dt.stride(1), dt.stride(2), dt.stride(3),
            *(dt_bias.stride(0), dt_bias.stride(1)) if dt_bias is not None else 0,
            A.stride(0), A.stride(1), A.stride(2),
            B.stride(0), B.stride(1), B.stride(2), B.stride(3),
            C.stride(0), C.stride(1), C.stride(2), C.stride(3),
            *(D.stride(0), D.stride(1)) if D is not None else 0,
            z_strides[0], z_strides[1], z_strides[2], z_strides[3],
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            leaf_sequences.stride(0), leaf_sequences.stride(1), leaf_sequences.stride(2),
            dt_softplus,
            tie_hdim,
            BLOCK_SIZE_M,
            SEQLEN=SEQLEN,
            num_warps=num_warps,
            NUM_SEQ=NUM_SEQ
        )
    if not has_heads:
        out = out.squeeze(1)
    return out