# Copyright (c) 2023, Albert Gu, Tri Dao.
import gc
import time
from collections import namedtuple
from dataclasses import dataclass, field
from functools import partial
from typing import Callable, Optional, Sequence, Union

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import Tensor
from torch.profiler import ProfilerActivity, profile, record_function
from transformers.generation import GreedySearchDecoderOnlyOutput, SampleDecoderOnlyOutput, TextStreamer
from mamba_ssm.utils.profile import cuda_time


@dataclass
class InferenceParams:
    """Inference parameters that are passed to the main model in order
    to efficienly calculate and store the context during inference."""

    max_seqlen: int
    max_batch_size: int
    _seqlen_offset: int = 0
    batch_size_offset: int = 0
    key_value_memory_dict: dict = field(default_factory=dict)
    key_value_memory_dict_tf: dict = field(default_factory=dict) # to accommodate Zamba2
    verified_key_value_memory_dict: dict = field(default_factory=dict)
    value_cache: dict = field(default_factory=dict)
    lengths_per_sample: Optional[Tensor] = None
    layers: list[int] = None
    use_2step_kernel: bool = False
    jit_state_copy: bool = False
    use_Nstep_kernel: bool = False
    save_last_seq: bool = True
    activation_replay: bool = False
    npad: int = 0,
    ndraft: int = 1,
    first_iteration: bool = False
    num_input_seq: int = 1
    multi_seqlen_offset: list[int] = None
    use_tree_scan_kernel: bool = False
    unroll_tree: bool = False
    seqlen_offset_pt: torch.Tensor = None
    mask_type: str = "padding"

    def reset(self, max_seqlen, max_batch_size):
        self.max_seqlen = max_seqlen
        self.max_batch_size = max_batch_size
        self.seqlen_offset = 0
        self.multi_seqlen_offset = [0] * self.num_input_seq
        self.layers = None
        for l in self.key_value_memory_dict.keys():
            self.key_value_memory_dict[l][0].zero_()
            if self.key_value_memory_dict[l][1] is not None: # adding this to handle MHA doesn't have convolution layer
                self.key_value_memory_dict[l][1].zero_()
            if self.activation_replay:
                # Transformer blocks do not have conv_state and value cache
                if self.value_cache[l] is not None:
                    self.value_cache[l]['x'].zero_()
                    self.value_cache[l]['B'].zero_()
                    self.value_cache[l]['dt'].zero_()
                    self.value_cache[l]['xBC'].zero_()
                    self.verified_key_value_memory_dict['state'][l][1].zero_()
                self.verified_key_value_memory_dict['state'][l][0].zero_()
        if self.key_value_memory_dict_tf is not None: # zamba2 requires an additional cache
            for l in self.key_value_memory_dict_tf.keys():
                self.key_value_memory_dict_tf[l][0].zero_()
                self.key_value_memory_dict_tf[l][1].zero_()

        self.key_value_memory_dict
        if self.lengths_per_sample is not None:
            self.lengths_per_sample.zero_()
    
    @property
    def seqlen_offset(self):
        return self._seqlen_offset
    
    @seqlen_offset.setter
    def seqlen_offset(self, value):
        self._seqlen_offset = value
        if self.seqlen_offset_pt is None:
            self.seqlen_offset_pt = torch.tensor([value], dtype=torch.long, device="cuda")
        else:
            self.seqlen_offset_pt.copy_(value)


def modify_logits_for_min_p_filtering(logits, min_p):
    """Set the logits for none min_p values to -inf. Done in-place."""
    if min_p <= 0.0 or min_p >= 1.0:
        return
    indices_to_remove = logits < min_p
    logits.masked_fill_(indices_to_remove, float("-Inf"))
# https://github.com/NVIDIA/Megatron-LM/blob/0bb597b42c53355a567aba2a1357cc34b9d99ddd/megatron/text_generation/sampling.py
# https://github.com/huggingface/transformers/blob/a44985b41cfa2de48a5e1de7f1f93b7483da25d1/src/transformers/generation/logits_process.py#L231
def modify_logits_for_top_k_filtering(logits, top_k):
    """Set the logits for none top-k values to -inf. Done in-place."""
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits.masked_fill_(indices_to_remove, float("-Inf"))


# https://github.com/NVIDIA/Megatron-LM/blob/0bb597b42c53355a567aba2a1357cc34b9d99ddd/megatron/text_generation/sampling.py
# https://github.com/huggingface/transformers/blob/a44985b41cfa2de48a5e1de7f1f93b7483da25d1/src/transformers/generation/logits_process.py#L170
def modify_logits_for_top_p_filtering(logits, top_p):
    """Set the logits for none top-p values to -inf. Done in-place."""
    if top_p <= 0.0 or top_p >= 1.0:
        return
    # First sort and calculate cumulative sum of probabilities.
    sorted_logits, sorted_indices = torch.sort(logits, descending=False)
    cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
    # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
    sorted_indices_to_remove = cumulative_probs <= (1 - top_p)
    # scatter sorted tensors to original indexing
    indices_to_remove = sorted_indices_to_remove.scatter(
        -1, sorted_indices, sorted_indices_to_remove
    )
    logits.masked_fill_(indices_to_remove, float("-inf"))


def modify_logit_for_repetition_penalty(logits, prev_output_tokens, repetition_penalty=1.0):
    """Apply repetition penalty. See https://arxiv.org/abs/1909.05858
    logits: (batch_size, vocab_size) or (batch_size, vocab_size, s)
    prev_output_tokens: (batch_size, seq_len) or (batch_size, seq_len, s)
    """
    if repetition_penalty == 1.0:
        return logits
    score = torch.gather(logits, 1, prev_output_tokens)
    # if score < 0 then repetition penalty has to be multiplied to reduce the previous token probability
    score = torch.where(score < 0, score * repetition_penalty, score / repetition_penalty)
    logits.scatter_(1, prev_output_tokens, score)
    return logits


def sample(logits, num_samples=1, top_k=1, top_p=0.0, min_p=0.0, temperature=1.0, generator=None):
    """Sample from top-k logits.
    Arguments:
        logits: Tensor of shape (batch_size, vocab_size)
    """
    if top_k == 1:  # Short-circuit for greedy decoding
        if num_samples == 1:
            return logits.argmax(dim=-1)
        else:
            return torch.topk(logits, k=num_samples, dim=-1).indices
    else:
        if top_p > 0.0:
            assert top_p <= 1.0, "top-p should be in (0, 1]."
        if top_k > 0:
            top_k = min(top_k, logits.size(-1))  # Safety check
            logits_top, indices = torch.topk(logits, top_k, dim=-1)
            if temperature != 1.0:
                logits_top /= temperature
                logits_top = logits_top - torch.max(logits_top, dim=-1, keepdim=True).values
            modify_logits_for_top_p_filtering(logits_top, top_p)
            return indices[
                torch.arange(indices.shape[0], device=indices.device),
                torch.multinomial(torch.softmax(logits_top, dim=-1), num_samples=num_samples, generator=generator).squeeze(dim=-1),
            ]
        else:
            if min_p > 0.0:
                logits_top = logits.clone()
                max_prob = logits_top[..., 0].item()
                min_prob = max_prob * min_p
                modify_logits_for_min_p_filtering(logits_top, min_prob)
                if temperature != 1.0:
                    logits_top /= temperature
                    logits_top = logits_top - torch.max(logits_top, dim=-1, keepdim=True).values
                return torch.multinomial(torch.softmax(logits_top, dim=-1), num_samples=num_samples, generator=generator).squeeze(dim=-1)
            # Clone so that when we modify for top_p we don't change the original logits
            logits_top = logits / temperature if temperature != 1.0 else logits.clone()
            logits_top = logits_top - torch.max(logits_top, dim=-1, keepdim=True).values
            modify_logits_for_top_p_filtering(logits_top, top_p)
            return torch.multinomial(torch.softmax(logits_top, dim=-1), num_samples=num_samples, generator=generator).squeeze(
                dim=-1
            )


@torch.inference_mode()
def decode(
    input_ids,
    model,
    max_length,
    mask=None,
    top_k=1,
    top_p=0.0,
    min_p=0.0,
    temperature=1.0,
    repetition_penalty=1.0,
    eos_token_id=None,
    teacher_outputs=None,
    vocab_size=None,
    cg=False,
    enable_timing=False,
    streamer: Optional[TextStreamer] = None,
    mask_type="padding",
    layers = None
):
    """Decoding, either greedy or with top-k or top-p sampling.
    If top-k = 0, don't limit the number of candidates (pure sampling).
    Top-k and top-p can be used together. If top_k > 0 and top_p > 0, then top-k is applied first,
    then top-p.
    We assume that all sequences in the same batch have the same length.

    Arguments:
        input_ids: (batch, seq_len)
        max_length: int
        teacher_outputs (optional): (batch, seq_len). If provided, instead of sampling from the
            logits, the next token is taken from the teacher_outputs. Useful for testing.
    Returns: GreedySearchDecoderOnlyOutput or SampleDecoderOnlyOutput, with the following fields:
        sequences: (batch, max_length)
        scores: tuples of (batch, vocab_size)
    """
    if streamer is not None:
        streamer.put(input_ids.cpu())

    batch_size, seqlen_og = input_ids.shape
    teacher_output_len = teacher_outputs.shape[1] if teacher_outputs is not None else 0
    if cg:
        if not hasattr(model, "_decoding_cache"):
            model._decoding_cache = None
        model._decoding_cache = update_graph_cache(
            model,
            model._decoding_cache,
            batch_size,
            seqlen_og,
            max_length,
            ndraft=1,
            mask_type=mask_type
        )
        inference_params = model._decoding_cache.inference_params
        inference_params.reset(max_length, batch_size)
    else:
        inference_params = InferenceParams(max_seqlen=max_length, max_batch_size=batch_size)
        inference_params.ndraft = 1
        inference_params.reset(max_length, batch_size)
        inference_params.mask_type = mask_type

    if layers is not None:
        inference_params.layers = layers
    
    decode_time = []

    def get_logits(input_ids, inference_params):
        decoding = inference_params.seqlen_offset > 0
        if decoding:
            position_ids = torch.full(
                (batch_size, 1),
                inference_params.seqlen_offset,
                dtype=torch.long,
                device=input_ids.device,
            )
            if mask_type == "padding":
                input_mask = torch.ones_like(input_ids)
            elif mask_type == "attention":
                input_mask = torch.ones((input_ids.shape[0], input_ids.shape[1], input_ids.shape[1]), device=input_ids.device).tril(diagonal=0)
            else:
                raise NotImplementedError
        else:
            position_ids = repeat(torch.arange(input_ids.shape[1], device=input_ids.device), "l -> b l", b=batch_size)
            if mask is None:
                input_mask = torch.ones_like(input_ids)
            else:
                input_mask = mask
        if not cg or not decoding:
            logits = model(
                input_ids,
                position_ids=position_ids,
                inference_params=inference_params,
                mask=input_mask,
                num_last_tokens=1,
            ).logits.squeeze(dim=1)
        else:
            logits = model._decoding_cache.run(
                input_ids, position_ids, input_mask, inference_params.seqlen_offset
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
    decode_time = []
    # print("generation start", input_ids.shape)
    while not should_stop(sequences[-1], inference_params):
        scores.append(get_logits(sequences[-1], inference_params))
        # print(inference_params.key_value_memory_dict[0][1].shape, inference_params.key_value_memory_dict[0][1][0,0,0,0])
        # print(inference_params.key_value_memory_dict_tf[1][0].shape, inference_params.key_value_memory_dict_tf[1][0][:,0,0,0])
        # print(inference_params.seqlen_offset)
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
        if streamer is not None:
            streamer.put(sampled_tokens.cpu())
    if streamer is not None:
        streamer.end()
    if enable_timing:
        end.record()
        torch.cuda.synchronize()
        print(f"Prompt processing + decoding time: {(start.elapsed_time(end)):.0f}ms")
    # print('avg forward time: ', sum(decode_time) / len(decode_time))
    output_cls = GreedySearchDecoderOnlyOutput if top_k == 1 else SampleDecoderOnlyOutput
    return output_cls(sequences=torch.cat(sequences, dim=1), scores=tuple(scores))

@dataclass
class DecodingCGCache:
    max_batch_size: int = 0
    max_seqlen: int = 0
    device = None
    dtype = None
    callables: dict = field(default_factory=dict)
    mempool = None
    inference_params: Optional[InferenceParams] = None
    run: Optional[Callable] = None


@torch.inference_mode()
def update_graph_cache(
    model,
    cache,
    batch_size,
    seqlen_og,
    max_seqlen,
    decoding_seqlens=(1,),
    dtype=None,
    n_warmups=2,
    use_2step_kernel=False,
    use_Nstep_kernel=False,
    save_last_seq=False,
    npad=0,
    ndraft=1,
    jit_state_copy=False,
    activation_replay=False,
    first_iteration=False,
    num_input_seq=1,
    use_tree_scan_kernel=False,
    unroll_tree=False,
    mask_type="padding"
):
    if cache is None:
        cache = DecodingCGCache()
    param_example = next(iter(model.parameters()))
    device = param_example.device
    if dtype is None:
        dtype = param_example.dtype
    if (
        (device, dtype) != (cache.device, cache.dtype)
        or batch_size > cache.max_batch_size
        or max_seqlen > cache.max_seqlen
    ):  # Invalidate the cache
        cache.callables = {}
        cache.mempool = None
        cache.inference_params = None
        gc.collect()
        cache.device, cache.dtype = device, dtype
        cache.max_batch_size, cache.max_seqlen = batch_size, max_seqlen
        assert hasattr(model, "allocate_inference_cache"), "CUDA graph decoding requires that the model has a method allocate_inference_cache"
        inf_cache = model.allocate_inference_cache(batch_size, max_seqlen, dtype)
        if isinstance(inf_cache, tuple):
            tf_inf_cache = inf_cache[1]
            inf_cache = inf_cache[0]
        else:
            inf_cache = inf_cache
            tf_inf_cache = None
        verified_cache = {"indices": torch.zeros((num_input_seq,), dtype=torch.long, device=device),
                          "mask": torch.zeros((num_input_seq, npad+1), dtype=torch.bool, device=device)}
        value_cache = None
        if activation_replay:
            verified_cache["state"] = model.allocate_inference_cache(num_input_seq, max_seqlen, dtype)
            value_cache = model.allocate_value_cache(batch_size, npad+1, dtype)
        lengths_per_sample = torch.full((batch_size,), seqlen_og, dtype=torch.int32, device=device)
        cache.inference_params = InferenceParams(
            max_seqlen=max_seqlen,
            max_batch_size=batch_size,
            _seqlen_offset=seqlen_og,
            seqlen_offset_pt=torch.zeros(1, dtype=torch.long, device=device),
            key_value_memory_dict=inf_cache,
            key_value_memory_dict_tf=tf_inf_cache,
            verified_key_value_memory_dict=verified_cache,
            value_cache=value_cache,
            lengths_per_sample=lengths_per_sample,
            use_2step_kernel=use_2step_kernel,
            use_Nstep_kernel=use_Nstep_kernel,
            save_last_seq=save_last_seq,
            npad=npad,
            ndraft=ndraft,
            jit_state_copy=jit_state_copy,
            activation_replay=activation_replay,
            first_iteration=first_iteration,
            num_input_seq=num_input_seq,
            multi_seqlen_offset=[0] * num_input_seq,
            use_tree_scan_kernel=use_tree_scan_kernel,
            unroll_tree=unroll_tree,
            mask_type=mask_type
        )
        cache.mempool = torch.cuda.graphs.graph_pool_handle()
    for decoding_seqlen in decoding_seqlens:
        if (batch_size, decoding_seqlen) not in cache.callables:
            cache.callables[batch_size, decoding_seqlen] = capture_graph(
                model,
                cache.inference_params,
                batch_size,
                max_seqlen,
                decoding_seqlen=decoding_seqlen,
                mempool=cache.mempool,
                n_warmups=n_warmups,
                mask_type=mask_type
            )

    def dispatch(input_ids, position_ids, mask, seqlen):
        batch_size, decoding_seqlen = input_ids.shape[:2]
        return cache.callables[batch_size, decoding_seqlen](input_ids, position_ids, mask, seqlen)
    cache.run = dispatch
    cache.inference_params.seqlen_offset = 0  # Reset so it's not confusing
    return cache


def capture_graph(
    model, inference_params, batch_size, max_seqlen, decoding_seqlen=1, mempool=None, n_warmups=2, mask_type="padding"
):
    device = next(iter(model.parameters())).device
    input_ids = torch.full((batch_size, decoding_seqlen), 0, dtype=torch.long, device=device)
    position_ids = torch.full((batch_size, decoding_seqlen), 0, dtype=torch.long, device=device)
    if inference_params.use_tree_scan_kernel:
        mask = torch.full((batch_size, decoding_seqlen, decoding_seqlen), 1, dtype=torch.long, device=device)
    else:
        if mask_type == "padding":
            mask = torch.ones_like(input_ids) # for mamba2 
        elif mask_type == "attention":
            mask = torch.full((batch_size, decoding_seqlen, decoding_seqlen), 1, dtype=torch.long, device=device) # for zamaba2 and mambainLlama
        else:
            raise NotImplementedError
    seqlen_offset_og = inference_params.seqlen_offset
    inference_params.seqlen_offset = max_seqlen - decoding_seqlen
    inference_params.lengths_per_sample[:] = inference_params.seqlen_offset

    # Warmup before capture
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(n_warmups):
            logits = model(
                input_ids,
                position_ids=position_ids,
                inference_params=inference_params,
                num_last_tokens=decoding_seqlen,
                mask=mask
            ).logits
        s.synchronize()
        # This might be needed for correctness if we run with NCCL_GRAPH_MIXING_SUPPORT=0,
        # which requires that graph launch and non-captured launch to not overlap (I think,
        # that's how I interpret the documentation). I'm not sure if this is required.
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
    torch.cuda.current_stream().wait_stream(s)
    # Captures the graph
    # To allow capture, automatically sets a side stream as the current stream in the context
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, pool=mempool):
        logits = model(
            input_ids,
            position_ids=position_ids,
            inference_params=inference_params,
            num_last_tokens=decoding_seqlen,
            mask=mask
        ).logits

    def run(new_input_ids, new_position_ids, new_mask, seqlen):
        inference_params.lengths_per_sample[:] = seqlen
        input_ids.copy_(new_input_ids)
        position_ids.copy_(new_position_ids)
        mask.copy_(new_mask)
        graph.replay()
        return logits.clone()

    inference_params.seqlen_offset = seqlen_offset_og
    return run


def capture_graph_huggingface(
    model, kv_cache, batch_size, max_seqlen, decoding_seqlen=1, mempool=None, n_warmups=2
):
    device = next(iter(model.parameters())).device
    dtype = next(iter(model.parameters())).dtype
    input_ids = torch.full((batch_size, decoding_seqlen), 0, dtype=torch.long, device=device)
    position_ids = torch.full((batch_size, decoding_seqlen), 0, dtype=torch.long, device=device)
    mask = torch.ones((batch_size, 1, decoding_seqlen, max_seqlen), device=device, dtype=dtype)
    seqlen_offset_pt = torch.zeros(1, device=device, dtype=torch.long)

    # Warmup before capture
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(n_warmups):
            logits = model.forward(
                input_ids=input_ids,
                attention_mask=mask,
                position_ids=position_ids,
                use_cache=True,
                return_dict=True,
                past_key_values=kv_cache,
                cache_position=seqlen_offset_pt+torch.arange(decoding_seqlen, device='cuda')
            ).logits
        s.synchronize()
        # This might be needed for correctness if we run with NCCL_GRAPH_MIXING_SUPPORT=0,
        # which requires that graph launch and non-captured launch to not overlap (I think,
        # that's how I interpret the documentation). I'm not sure if this is required.
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
    torch.cuda.current_stream().wait_stream(s)
    # Captures the graph
    # To allow capture, automatically sets a side stream as the current stream in the context
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, pool=mempool):
        logits = model.forward(
            input_ids=input_ids,
            attention_mask=mask,
            position_ids=position_ids,
            use_cache=True,
            return_dict=True,
            past_key_values=kv_cache,
            cache_position=seqlen_offset_pt+torch.arange(decoding_seqlen, device='cuda')
        ).logits

    def run(new_input_ids, new_position_ids, new_mask, seqlen):
        seqlen_offset_pt.copy_(seqlen)
        input_ids.copy_(new_input_ids)
        position_ids.copy_(new_position_ids)
        mask.copy_(new_mask)
        graph.replay()
        return logits.clone()

    return run
