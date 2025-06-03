import torch
from torch import Tensor

import triton
import triton.language as tl
import pdb

# The original causal_conv1d_varlen_states from causal_conv1d, try to copy X into conv_state 
# The implementation assumes 0 padding when certain batch of X is not long enough for conv_state
@triton.jit
def _causal_conv1d_varlen_states(
    X,
    CU_SEQLENS,
    STATES,
    state_len,
    dim,
    stride_x_seqlen, stride_x_dim,
    stride_states_batch, stride_states_seqlen, stride_states_dim,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr
):
    batch_idx = tl.program_id(2)
    STATES += batch_idx * stride_states_batch
    end_idx = tl.load(CU_SEQLENS + batch_idx + 1)
    start_idx = tl.maximum(tl.load(CU_SEQLENS + batch_idx), end_idx - state_len)
    rows = end_idx - (tl.program_id(1) + 1) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    x = tl.load(X + rows[:, None] * stride_x_seqlen + cols[None, :] * stride_x_dim,
                mask=(rows[:, None] >= start_idx) & (cols[None, :] < dim),
                other=0)
    rows_states = state_len - (tl.program_id(1) + 1) * BLOCK_M + tl.arange(0, BLOCK_M)
    tl.store(STATES + rows_states[:, None] * stride_states_seqlen + cols[None, :] * stride_states_dim,
             x,
             mask=(rows_states[:, None] >= 0) & (cols[None, :] < dim))


def causal_conv1d_varlen_states(x: Tensor, cu_seqlens: Tensor, state_len: int) -> Tensor:
    """
    Forward pass only, does not support backward pass.
    Parameters:
        x: (total_tokens, dim)
        cu_seqlens: (batch + 1), must already be sorted. The cumulative sum of the sequence lengths, starting from 0.
        state_len: int. For each cu_seqlens, how many elements from x should be copied to the state.
            If some of those elements belong to a different sequence, the value of the states will be zero.
    Return:
        states: (batch, dim, state_len)
    """
    _, dim = x.shape
    batch = cu_seqlens.shape[0] - 1
    cu_seqlens = cu_seqlens.contiguous()
    states = torch.empty(batch, state_len, dim, dtype=x.dtype, device=x.device).transpose(1, 2)
    BLOCK_M = min(triton.next_power_of_2(state_len), 16)
    BLOCK_N = min(triton.next_power_of_2(dim), 256)
    grid = (triton.cdiv(dim, BLOCK_N), triton.cdiv(state_len, BLOCK_M), batch)
    with torch.cuda.device(x.device.index):
        _causal_conv1d_varlen_states[grid](
            x,
            cu_seqlens,
            states,
            state_len,
            dim,
            x.stride(0), x.stride(1),
            states.stride(0), states.stride(2), states.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N
        )
    return states


def causal_conv1d_varlen_states_ref(x: Tensor, cu_seqlens: Tensor, state_len: int) -> Tensor:
    """
    Forward pass only, does not support backward pass.
    Parameters:
        x: (total_tokens, dim)
        cu_seqlens: (batch + 1), must already be sorted. The cumulative sum of the sequence lengths, starting from 0.
        state_len: int. For each cu_seqlens, how many elements from x should be copied to the state.
            If some of those elements belong to a different sequence, the value of the states will be zero.
    Return:
        states: (batch, dim, state_len)
    """
    _, dim = x.shape
    batch = cu_seqlens.shape[0] - 1
    cu_seqlens = cu_seqlens.contiguous()
    states = torch.zeros(batch, state_len, dim, dtype=x.dtype, device=x.device).transpose(1, 2)
    for i in range(batch):
        end_idx = cu_seqlens[i + 1]
        start_idx = torch.maximum(cu_seqlens[i], end_idx - state_len)
        states[i, :, -(end_idx - start_idx):] = x[start_idx:end_idx].T
    return states


def causal_conv1d_varlen_states_update_ref(x: Tensor, cu_seqlens: Tensor, state: Tensor) -> Tensor:
    """
    Forward pass only, does not support backward pass.
    Parameters:
        x: (total_tokens, dim)
        cu_seqlens: (batch + 1), must already be sorted. The cumulative sum of the sequence lengths, starting from 0.
        state: (batch, dim, state_len) For each cu_seqlens, how many elements from x should be copied to the state.
            If some of those elements belong to a different sequence, the value of the states will be zero.
    Return:
        states: (batch, dim, state_len)
    """
    _, dim = x.shape
    _, _, state_len = state.shape
    batch = cu_seqlens.shape[0] - 1
    old_state = state.clone()
    cu_seqlens = cu_seqlens.contiguous()
    # states = torch.zeros(batch, state_len, dim, dtype=x.dtype, device=x.device).transpose(1, 2)
    for i in range(batch):
        end_idx = cu_seqlens[i + 1]
        start_idx = torch.maximum(cu_seqlens[i], end_idx - state_len)
        state[i, :, :-(end_idx - start_idx)] = old_state[i, :,(end_idx-start_idx):]
        state[i, :, -(end_idx - start_idx):] = x[start_idx:end_idx].T
    return state

# This is a modification of the original varlen_state_update, where if X is shorter and conv_state, 
# We will fill left side of the conv_state not by 0, but by old conv_state shifted by the the length of X
@triton.jit
def _causal_conv1d_varlen_states_update(
    X,
    CU_SEQLENS,
    STATES,
    OLD_STATES,
    state_len,
    dim,
    stride_x_seqlen, stride_x_dim,
    stride_states_batch, stride_states_seqlen, stride_states_dim,
    stride_old_states_batch, stride_old_states_seqlen, stride_old_states_dim,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr
):
    batch_idx = tl.program_id(2)
    STATES += batch_idx * stride_states_batch
    OLD_STATES += batch_idx * stride_old_states_batch
    end_idx = tl.load(CU_SEQLENS + batch_idx + 1)
    start_idx = tl.maximum(tl.load(CU_SEQLENS + batch_idx), end_idx - state_len)
    state_start_idx = end_idx - start_idx
    rows = end_idx - (tl.program_id(1) + 1) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    # loading previous state and storing to the right location
    rows_states = state_len - (tl.program_id(1) + 1) * BLOCK_M + tl.arange(0, BLOCK_M)
    shifted_rows = rows_states - state_start_idx
    previous_state = tl.load(OLD_STATES + rows_states[:, None] * stride_old_states_seqlen + cols[None, :] * stride_old_states_dim, 
                             mask=(rows_states[:, None] >= state_start_idx) & (cols[None, :] < dim),
                             other=0)
    tl.store(STATES + shifted_rows[:, None] * stride_states_seqlen + cols[None, :] * stride_states_dim,
             previous_state,
             mask=(shifted_rows[:, None] >= 0) & (shifted_rows[:, None] < (state_len - state_start_idx)) & (cols[None, :] < dim))

    x = tl.load(X + rows[:, None] * stride_x_seqlen + cols[None, :] * stride_x_dim,
                mask=(rows[:, None] >= start_idx) & (cols[None, :] < dim),
                other=0)

    tl.store(STATES + rows_states[:, None] * stride_states_seqlen + cols[None, :] * stride_states_dim,
             x,
             mask=(rows_states[:, None] >= 0) & (rows_states[:, None] >= (state_len - state_start_idx)) & (cols[None, :] < dim))
    
def causal_conv1d_varlen_states_update(x: Tensor, cu_seqlens: Tensor, state: Tensor) -> Tensor:
    """
    Forward pass only, does not support backward pass.
    Parameters:
        x: (total_tokens, dim)
        cu_seqlens: (batch + 1), must already be sorted. The cumulative sum of the sequence lengths, starting from 0.
        state: (batch, dim, state_len) int. For each cu_seqlens, how many elements from x should be copied to the state.
            If some of those elements belong to a different sequence, the value of the states will be zero.
    Return:
        state: (batch, dim, state_len)
    """
    _, dim = x.shape
    state_len = state.shape[2]
    batch = cu_seqlens.shape[0] - 1
    cu_seqlens = cu_seqlens.contiguous()
    old_state = state.clone()
    BLOCK_M = min(triton.next_power_of_2(state_len), 16)
    BLOCK_N = min(triton.next_power_of_2(dim), 256)
    grid = (triton.cdiv(dim, BLOCK_N), triton.cdiv(state_len, BLOCK_M), batch)
    with torch.cuda.device(x.device.index):
        _causal_conv1d_varlen_states_update[grid](
            x,
            cu_seqlens,
            state,
            old_state,
            state_len,
            dim,
            x.stride(0), x.stride(1),
            state.stride(0), state.stride(2), state.stride(1),
            old_state.stride(0), old_state.stride(2), old_state.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N
        )
    return state

def causal_conv1d_varlen_states_update_v2_ref(x: Tensor, mask: Tensor, state: Tensor) -> Tensor:
    """
    v2 turns x into a more friendly shape
    Forward pass only, does not support backward pass.
    Parameters:
        x: (batch, seqlen, dim)
        mask: (batch, seqlen) assume all sequence in the batch are left-padded if there is any padding
        state: (batch, dim, state_len)
    Return:
        state: (batch, dim, state_len+seqlen) 
    """
    batch, seqlen, dim = x.shape
    _, _, state_len = state.shape
    new_state = torch.zeros((batch, dim, state_len+seqlen), device=state.device, dtype=state.dtype)
    cu_seqlen = mask.sum(1)
    # states = torch.zeros(batch, state_len, dim, dtype=x.dtype, device=x.device).transpose(1, 2)
    for i in range(batch):
        start_idx = cu_seqlen[i]
        new_state[i, :, -(start_idx+state_len):-start_idx] = state[i, :, :]
        new_state[i, :, -start_idx:] = x[i,-start_idx:,:].T
    return new_state

# This is a version 2 of the varlen_state_update kernel. In this kernel, the shape of X no longer have to be 
# in the shape of (total_length, dim) (i.e. concatenating all the sequences together in the seq_len dimension)
# Here, the shape of X is (batch_size, seq_len, dim), which save some operation to turn X into the shape required by v1.
@triton.jit
def _causal_conv1d_varlen_states_update_v2(
    X,
    MASK,
    CU_SEQLENS,
    STATES,
    NEW_STATE,
    state_len,
    new_state_len,
    dim,
    seqlen,
    stride_x_batch, stride_x_seqlen, stride_x_dim,
    stride_states_batch, stride_states_seqlen, stride_states_dim,
    stride_new_states_batch, stride_new_states_seqlen, stride_new_states_dim,
    stride_mask_batch, stride_mask_seqlen,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr
):
    batch_idx = tl.program_id(2)
    STATES += batch_idx * stride_states_batch
    NEW_STATE += batch_idx * stride_new_states_batch
    X += batch_idx * stride_x_batch
    valid_len = tl.load(CU_SEQLENS + batch_idx)
    MASK += batch_idx * stride_mask_batch
    rows = seqlen - (tl.program_id(1) + 1) * BLOCK_M + tl.arange(0, BLOCK_M)
    mask = tl.load(MASK + rows[:, None] * stride_mask_seqlen, 
                   mask=rows[:, None] >= 0)
    mask = (mask == 1)
    cols = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    # loading previous state and storing to the right location
    rows_new_states = new_state_len - (tl.program_id(1) + 1) * BLOCK_M + tl.arange(0, BLOCK_M)
    rows_states = state_len - (tl.program_id(1) + 1) * BLOCK_M + tl.arange(0, BLOCK_M)

    shifted_rows = rows_new_states - valid_len
    previous_state = tl.load(STATES + rows_states[:, None] * stride_states_seqlen + cols[None, :] * stride_states_dim, 
                             mask=(rows_states[:, None] >= 0) & (cols[None, :] < dim),
                             other=0)
    tl.store(NEW_STATE + shifted_rows[:, None] * stride_new_states_seqlen + cols[None, :] * stride_new_states_dim,
             previous_state,
             mask=(shifted_rows[:, None] >= 0) & (shifted_rows[:, None] < (new_state_len - valid_len)) & (cols[None, :] < dim))

    x = tl.load(X + rows[:, None] * stride_x_seqlen + cols[None, :] * stride_x_dim,
                mask=mask & (cols[None, :] < dim),
                other=0)

    tl.store(NEW_STATE + rows_new_states[:, None] * stride_new_states_seqlen + cols[None, :] * stride_new_states_dim,
             x,
             mask=mask & (cols[None, :] < dim))
    

def causal_conv1d_varlen_states_update_v2(x: Tensor, mask: Tensor, state: Tensor) -> Tensor:
    """
    v2 turns x into a more friendly shape
    Forward pass only, does not support backward pass.
    Parameters:
        x: (batch, seqlen, dim)
        mask: (batch, seqlen) assume all sequence in the batch are left-padded if there is any padding
        state: (batch, dim, state_len)
    Return:
        state: (batch, dim, state_len+seqlen) 
    """
    batch, seqlen, dim = x.shape
    state_len = state.shape[2]
    cu_seqlen = mask.sum(1)
    new_state = torch.zeros((batch, dim, state_len+seqlen), device=state.device, dtype=state.dtype)
    BLOCK_M = min(triton.next_power_of_2(state_len+seqlen), 16)
    BLOCK_N = min(triton.next_power_of_2(dim), 256)
    grid = (triton.cdiv(dim, BLOCK_N), triton.cdiv(state_len+seqlen, BLOCK_M), batch)
    with torch.cuda.device(x.device.index):
        _causal_conv1d_varlen_states_update_v2[grid](
            x,
            mask,
            cu_seqlen,
            state,
            new_state,
            state_len,
            state_len+seqlen,
            dim,
            seqlen,
            x.stride(0), x.stride(1), x.stride(2),
            state.stride(0), state.stride(2), state.stride(1),
            new_state.stride(0), new_state.stride(2), new_state.stride(1),
            mask.stride(0), mask.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N
        )
    return new_state

def causal_conv1d_varlen_update_ref(x: Tensor, mask: Tensor, state: Tensor, weight: Tensor, bias: Tensor, activation: str) -> Tensor:
    """
    v2 turns x into a more friendly shape
    Forward pass only, does not support backward pass.
    Parameters:
        x: (batch, seqlen, dim)
        mask: (batch, seqlen) assume all sequence in the batch are left-padded if there is any padding
        state: (batch, dim, state_len)
        weight: (dim, state_len)
        bias: (dim,)
    Return:
        output: (batch, dim, state_len+seqlen) 
    """
    assert activation == "silu", "Only supporting silu activation for now"
    batch, seqlen, dim = x.shape
    _, _, state_len = state.shape
    new_state = torch.zeros((batch, dim, state_len+seqlen), device=state.device, dtype=state.dtype)
    cu_seqlen = mask.sum(1)
    # states = torch.zeros(batch, state_len, dim, dtype=x.dtype, device=x.device).transpose(1, 2)
    for i in range(batch):
        start_idx = cu_seqlen[i]
        new_state[i, :, -(start_idx+state_len):-start_idx] = state[i, :, :]
        new_state[i, :, -start_idx:] = x[i,-start_idx:,:].T
    output = torch.nn.functional.conv1d(new_state, weight, bias, padding=0, groups=dim)[:, :, -seqlen:]
    
    state.copy_(new_state[:, :, -state_len:])

    if activation == "silu":
        output = torch.nn.functional.silu(output)
    output = output * mask[:, None, :]

    return output

# This is performing causal_conv1 and state update in the same kernel to potentially fuse more operations
# Specifically, we are launching a thread for each batch and each sequence position
# Each sequence position is loading d_conv position from state and X (with proper masking to make sure either
# state or x is 0). Summing this two we get the convolution input. By doing a multiply and sum with the convolution weight,
# we compute the convolution for 1 sequence position and we put it back at the correct position. 
@triton.jit
def _causal_conv1d_varlen_update(
    X,
    MASK,
    CU_SEQLENS,
    STATES,
    NEW_STATE,
    WEIGHT,
    BIAS,
    OUTPUT,
    state_len,
    dim,
    seqlen,
    stride_x_batch, stride_x_seqlen, stride_x_dim,
    stride_states_batch, stride_states_seqlen, stride_states_dim,
    stride_new_states_batch, stride_new_states_seqlen, stride_new_states_dim,
    stride_mask_batch, stride_mask_seqlen,
    stride_weight_dim, stride_weight_len,
    stride_bias_dim,
    stride_output_batch, stride_output_seqlen, stride_output_dim,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    USE_SILU: tl.constexpr
):
    batch_idx = tl.program_id(2)
    seq_pos = tl.program_id(1) # this is the reverse sequence index (0 -> end of sequence)
    STATES_ptr = STATES + batch_idx * stride_states_batch
    OUTPUT_ptr = OUTPUT + batch_idx * stride_output_batch + (seqlen - seq_pos - 1) * stride_output_seqlen
    NEW_STATE_ptr = NEW_STATE + batch_idx * stride_new_states_batch 
    X_ptr = X + batch_idx * stride_x_batch
    valid_len = tl.load(CU_SEQLENS + batch_idx)
    MASK_ptr = MASK + batch_idx * stride_mask_batch
    state_idx = tl.arange(0, BLOCK_M)
    rows = seqlen - seq_pos - BLOCK_M + state_idx
    mask = tl.load(MASK_ptr + rows[:, None] * stride_mask_seqlen, 
                   mask=rows[:, None] >= 0)
    mask = (mask == 1)
    cols = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    # loading bias
    bias = tl.load(BIAS + cols * stride_bias_dim)
    # loading weight
    weight = tl.load(WEIGHT + state_idx[:, None] * stride_weight_len + cols[None, :] * stride_weight_dim,
                     mask=(cols[None, :] < dim),
                     other=0)

    # loading x, 
    x = tl.load(X_ptr + rows[:, None] * stride_x_seqlen + cols[None, :] * stride_x_dim,
                mask=mask & (cols[None, :] < dim),
                other=0)

    previous_state = tl.load(STATES_ptr + (state_idx[:, None] + valid_len - seq_pos) * stride_states_seqlen + cols[None, :] * stride_states_dim, 
                             mask=((state_idx[:, None] + valid_len - seq_pos) >= 0) & ((state_idx[:, None] + valid_len - seq_pos) < state_len) & (cols[None, :] < dim),
                             other=0)
    new_state = x + previous_state # (BLOCK_M, BLOCK_N)

    output = tl.sum(weight * new_state, axis=0) + bias # (BLOCK_N,)
    output = output.cast(tl.float32)
    if USE_SILU:
        output = output / (1 + tl.exp(-output))

    tl.store(OUTPUT_ptr + cols * stride_output_dim, 
             output, 
             mask=(seq_pos < valid_len) & (cols < dim))

    tl.store((NEW_STATE_ptr + ((state_idx - seq_pos)[:, None] * stride_new_states_seqlen).broadcast_to((BLOCK_M, 1)) + cols[None, :] * stride_new_states_dim), 
             new_state, 
             mask=((state_idx - seq_pos)[:, None] >= 0) & (cols[None, :] < dim)) # always only store the last one



def causal_conv1d_varlen_update(x: Tensor, mask: Tensor, state: Tensor, weight: Tensor, bias: Tensor, activation: str) -> Tensor:
    """
    v2 turns x into a more friendly shape
    Forward pass only, does not support backward pass.
    Parameters:
        x: (batch, dim, seqlen)
        mask: (batch, seqlen) assume all sequence in the batch are left-padded if there is any padding
        state: (batch, dim, state_len)
        weight: (dim, state_len)
        bias: (dim,)
        activation: str
    Return:
        output: (batch, dim, seqlen) 
    """
    batch, dim, seqlen = x.shape
    state_len = state.shape[2]
    cu_seqlen = mask.sum(1)
    new_state = torch.zeros_like(state)
    output = torch.zeros_like(x)
    BLOCK_M = min(triton.next_power_of_2(state_len), 16)
    BLOCK_N = min(triton.next_power_of_2(dim), 256)
    grid = (triton.cdiv(dim, BLOCK_N), seqlen, batch)
    use_silu = activation == "silu"
    with torch.cuda.device(x.device.index):
        _causal_conv1d_varlen_update[grid](
            x,
            mask,
            cu_seqlen,
            state,
            new_state,
            weight, 
            bias,
            output,
            state_len,
            dim,
            seqlen,
            x.stride(0), x.stride(2), x.stride(1),
            state.stride(0), state.stride(2), state.stride(1),
            new_state.stride(0), new_state.stride(2), new_state.stride(1),
            mask.stride(0), mask.stride(1),
            weight.stride(0), weight.stride(1),
            bias.stride(0),
            output.stride(0), output.stride(2), output.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, USE_SILU=use_silu
        )
    state.copy_(new_state)
    return output