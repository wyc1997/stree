import torch
from torch import Tensor

import triton
import triton.language as tl
from einops import repeat

def inplace_state_indexing_ref(t:Tensor, index:Tensor):
    t.copy_(t.index_select(0, index))

def inplace_state_indexing_repeat_ref(t:Tensor, index:Tensor, r:int):
    t.copy_(repeat(t.index_select(0, index), 'b ... -> (n b) ...', n=r))


def inplace_state_indexing(t:Tensor, index:Tensor):
    '''
    Performs a inplace indexing for ssm state
    t: ssm state being indexed and stored into
    index: index of the tensor along the first dimension
    '''
    batch, nheads, dim, dstate = t.shape
    assert index.shape[0] == batch, 'For inplace indexing, we must have the same number of index and batch'
    BLOCK_SIZE_M = min(triton.next_power_of_2(dim), 64)
    BLOCK_SIZE_N = min(triton.next_power_of_2(dstate), 64)
    grid = (triton.cdiv(dim, BLOCK_SIZE_M), triton.cdiv(dstate, BLOCK_SIZE_N), nheads)
    repeated_last_dim = (t.stride(-1) == 0)
    with torch.cuda.device(t.device.index):
        _inplace_state_indexing_kernel[grid](
            t, index,
            batch, nheads, dim, dstate,
            t.stride(0), t.stride(1), t.stride(2), t.stride(3),
            index.stride(0),
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            REPEATED_LAST_DIM=repeated_last_dim
        )
    return t

@triton.heuristics({"BLOCK_SIZE_BATCH": lambda args: triton.next_power_of_2(args["batch"])})
@triton.jit
def _inplace_state_indexing_kernel(
    x_ptr, index_ptr,
    batch, nheads, dim, dstate,
    stride_x_batch, stride_x_nhead, stride_x_dim, stride_x_dstate,
    stride_index_batch,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_BATCH: tl.constexpr,
    REPEATED_LAST_DIM: tl.constexpr):

    pid_m = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_h = tl.program_id(2)

    x_ptr += pid_h * stride_x_nhead
    off_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    off_n = pid_s * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    off_b = tl.arange(0, BLOCK_SIZE_BATCH)
    index_ptrs = index_ptr + off_b * stride_index_batch
    indices = tl.load(index_ptrs, mask=off_b<batch, other=0.0).to(tl.int32)
    if REPEATED_LAST_DIM:
        x_ptrs = x_ptr + (indices[:, None] * stride_x_batch + off_m[None, :] * stride_x_dim)
        x = tl.load(x_ptrs, mask=(off_b[:, None] < batch) & (off_m[None, :] < dim), other=0.0)

        out_ptrs = x_ptr + (off_b[:, None] * stride_x_batch + off_m[None, :] * stride_x_dim)
        tl.store(out_ptrs, x, mask=(off_b[:, None] < batch) & (off_m[None, :] < dim))
    else:
        x_ptrs = x_ptr + (indices[:, None, None] * stride_x_batch + off_m[None, :, None] * stride_x_dim + off_n[None, None, :] * stride_x_dstate)
        x = tl.load(x_ptrs, mask=(off_b[:, None, None] < batch) & (off_m[None, :, None] < dim) & (off_n[None, None, :] < dstate), other=0.0)

        out_ptrs = x_ptr + (off_b[:, None, None] * stride_x_batch + off_m[None, :, None] * stride_x_dim + off_n[None, None, :] * stride_x_dstate)
        tl.store(out_ptrs, x, mask=(off_b[:, None, None] < batch) & (off_m[None, :, None] < dim) & (off_n[None, None, :] < dstate))

def inplace_state_indexing_repeat(t:Tensor, index:Tensor, r:int):
    '''
    Performs a inplace indexing for ssm state
    t: ssm state being indexed and stored into
    index: index of the tensor along the first dimension
    r: the number of times to repeat the selected elements, usually equal to t.shape[0] // index.shape[0]
    '''
    batch, nheads, dim, dstate = t.shape
    # assert index.shape[0] == batch, 'For inplace indexing, we must have the same number of index and batch'
    assert r == batch // index.shape[0], 'Assuming number of repeat is the difference between state batchsize and num of index'
    BLOCK_SIZE_M = min(triton.next_power_of_2(dim), 64)
    BLOCK_SIZE_N = min(triton.next_power_of_2(dstate), 64)
    REPEAT_SIZE = triton.next_power_of_2(r)
    grid = (triton.cdiv(dim, BLOCK_SIZE_M), triton.cdiv(dstate, BLOCK_SIZE_N), nheads*r)
    with torch.cuda.device(t.device.index):
        _inplace_state_indexing_repeat_kernel[grid](
            t, index,
            r, index.shape[0],
            batch, nheads, dim, dstate,
            t.stride(0), t.stride(1), t.stride(2), t.stride(3),
            index.stride(0),
            REPEAT_SIZE,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N
        )
    return t

@triton.heuristics({"BLOCK_SIZE_BATCH": lambda args: triton.next_power_of_2(args["index_count"])})
@triton.jit
def _inplace_state_indexing_repeat_kernel(
    x_ptr, index_ptr,
    repeat, index_count,
    batch, nheads, dim, dstate,
    stride_x_batch, stride_x_nhead, stride_x_dim, stride_x_dstate,
    stride_index_batch,
    REPEAT_SIZE: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_BATCH: tl.constexpr):

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_h = tl.program_id(2) % nheads #TODO: parallelizing repeat leads to race condition
    pid_r = tl.program_id(2) // nheads

    x_ptr += pid_h * stride_x_nhead
    # repeats = tl.zeros((REPEAT_SIZE,), tl.float16)
    # repeats_offset = tl.arange(0, REPEAT_SIZE).to(tl.int64)
    off_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    off_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    off_b = tl.arange(0, BLOCK_SIZE_BATCH)

    # off_b_repeated = tl.reshape(off_b[:,None] + repeats[None,:], (BLOCK_SIZE_BATCH*REPEAT_SIZE,),can_reorder=True)
    # off_b_repeated_mask = tl.reshape((off_b[:,None]<index_count) & (repeats_offset[None,:]<repeat), (BLOCK_SIZE_BATCH*REPEAT_SIZE,),can_reorder=True).to(tl.int1)
    # index_ptrs = (index_ptr + (off_b_repeated * stride_index_batch).to(tl.int64))
    # indices = tl.load(index_ptrs[:,None,None], mask=off_b_repeated_mask[:,None,None], other=0.0)
    index_ptrs = index_ptr + off_b * stride_index_batch
    print(off_b<batch)
    print(index_ptrs)
    indices = tl.load(index_ptrs, mask=off_b<batch, other=0.0).to(tl.int32)
    print(indices.dtype)
    # tl.device_print("idx", indices)

    x_ptrs = x_ptr + (indices[:,None,None] * stride_x_batch + off_m[None, :, None] * stride_x_dim + off_n[None, None, :] * stride_x_dstate)
    # print(x_ptr)
    print(indices[:,None,None] * stride_x_batch + off_m[None, :, None] * stride_x_dim + off_n[None, None, :] * stride_x_dstate, stride_x_batch)
    print(x_ptr, x_ptrs)
    print(x_ptrs.dtype)
    # tl.device_print("x_ptr", x_ptr)
    # tl.device_print("dim", (indices * stride_x_batch))
    # tl.device_print("expanded", (indices * stride_x_batch).expand_dims(1))
    # tl.device_print("x_ptrs",(indices[:, None] * stride_x_batch + off_m[None, :] * stride_x_dim))
    x = tl.load(x_ptrs, mask=(off_b[:, None, None] < batch) & (off_m[None, :, None] < dim) & (off_n[None, None, :] < dstate), other=0.0)

    off_out = off_b * repeat + pid_r
    # off_out = tl.reshape(off_out,  (BLOCK_SIZE_BATCH*REPEAT_SIZE,),can_reorder=True).to(tl.int64)
    out_ptrs = x_ptr + (off_out[:, None, None] * stride_x_batch + off_m[None, :, None] * stride_x_dim + off_n[None, None, :] * stride_x_dstate)
    print(off_out * stride_x_batch, pid_r, off_m* stride_x_dim, off_n* stride_x_dstate)
    print(off_out[:, None, None] * stride_x_batch + off_m[None, :, None] * stride_x_dim + off_n[None, None, :] * stride_x_dstate)
    print(x_ptr, out_ptrs)
    print(out_ptrs.dtype)
    # tl.device_print("stride_batch", stride_x_batch)
    # tl.device_print("stride_dim", stride_x_dim)
    # tl.device_print("stride_dstate", stride_x_dstate)
    # tl.device_print("out_ptrs", out_ptrs)
    print(x.dtype)
    tl.store(out_ptrs, x, mask=(off_out[:, None, None] < batch) & (off_m[None, :, None] < dim) & (off_n[None, None, :] < dstate))
