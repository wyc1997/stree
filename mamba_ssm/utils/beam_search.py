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
from mamba_ssm.utils.generation_utils import InferenceParams, update_graph_cache, modify_logit_for_repetition_penalty, sample

@torch.inference_mode()
def beam_search_decode(
    input_ids,
    model,
    max_length,
    mask=None,
    top_k=1,
    top_p=0.0,
    min_p=0.0,
    temperature=1.0,
    repetition_penalty=1.0,
    num_beam=1,
    eos_token_id=None,
    teacher_outputs=None,
    vocab_size=None,
    cg=False,
    enable_timing=False,
    streamer: Optional[TextStreamer] = None,
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
    # TODO: assuming batch_size is 1 for now
    assert batch_size == 1, "assuming batch_size is 1 for now"
    beam_size = batch_size * num_beam
    teacher_output_len = teacher_outputs.shape[1] if teacher_outputs is not None else 0
    if cg:
        if not hasattr(model, "_decoding_cache"):
            model._decoding_cache = None
        model._decoding_cache = update_graph_cache(
            model,
            model._decoding_cache,
            beam_size,
            seqlen_og,
            max_length,
            ndraft=1,
            num_input_seq=num_beam,
            jit_state_copy=True,
        )
        inference_params = model._decoding_cache.inference_params
        inference_params.reset(max_length, batch_size)
    else:
        inference_params = InferenceParams(max_seqlen=max_length, max_batch_size=num_beam)
        eg_param = next(iter(model.parameters()))
        dtype = eg_param.dtype
        device = eg_param.device
        inference_params.key_value_memory_dict = model.allocate_inference_cache(batch_size=num_beam, max_seqlen=max_length, dtype=dtype)
        # verified dict saves two indices to the verified state in states. In this case we always take 0
        inference_params.verified_key_value_memory_dict = {"indices":torch.zeros((num_beam,), dtype=torch.long, device=device)}
        inference_params.ndraft = 1
        inference_params.jit_state_copy = True
        inference_params.num_input_seq = num_beam

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
            input_mask = torch.ones_like(input_ids)
        else:
            position_ids = None
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
            token = sample(logits, top_k=num_beam, temperature=temperature)
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
    log_probability = torch.zeros((batch_size * num_beam, 1), device=input_ids.device, dtype=torch.float32)
    decode_time = []
    # setting up first iteration in beam search 
    scores.append(get_logits(sequences_cat, inference_params))
    inference_params.seqlen_offset += sequences_cat.shape[1]
    if repetition_penalty == 1.0:
        curr_prob = torch.softmax(scores[-1], dim=-1) # (batchsize * num_beam, vocabsize)
        sampled_output = curr_prob.topk(num_beam, dim=-1)
    else:
        logits = modify_logit_for_repetition_penalty(
            scores[-1].clone(), sequences_cat, repetition_penalty
        )
        curr_prob = torch.softmax(logits, dim=-1)
        sampled_output = curr_prob.topk(num_beam, dim=-1)
    vocab_size = curr_prob.shape[-1] if vocab_size is None else vocab_size
    sampled_prob = sampled_output.values
    sampled_tokens = sampled_output.indices

    log_probability += torch.log(sampled_prob.view((batch_size * num_beam, 1)))
    sequences_cat = repeat(sequences_cat, 'b l -> (n b) l', n=num_beam)
    sequences_cat = torch.cat([sequences_cat, sampled_tokens.view(batch_size * num_beam, 1)], dim=1)

    inference_params.verified_key_value_memory_dict['indices'].copy_(0)

    # Iterating through beam search iterations. 
    while not should_stop(sequences_cat[:, [-1]], inference_params):
        scores.append(get_logits(sequences_cat[:, [-1]], inference_params))
        inference_params.seqlen_offset += sequences_cat[:, [-1]].shape[1]
        if repetition_penalty == 1.0:
            curr_prob = torch.softmax(scores[-1], dim=-1) # (batchsize * num_beam, vocabsize)
        else:
            logits = modify_logit_for_repetition_penalty(
                scores[-1].clone(), sequences_cat, repetition_penalty
            )
            curr_prob = torch.softmax(logits, dim=-1) # (batchsize * num_beam, vocabsize)

        # Trying to do it in one topk
        curr_prob = torch.log(curr_prob)
        curr_prob = (curr_prob + log_probability).view(batch_size, num_beam * curr_prob.shape[-1]) # (1, num_beam * vocabsize)
        top_beam_output = curr_prob.topk(num_beam, dim=-1) # (batchsize, num_beam)
        state_indices = (top_beam_output.indices / vocab_size).to(torch.int).squeeze(0)
        log_probability.copy_(top_beam_output.values.view(batch_size * num_beam, 1))
        
        sequences_cat = sequences_cat[state_indices, :]

        tokens = (top_beam_output.indices % vocab_size).view((batch_size * num_beam, 1))

        sequences_cat = torch.cat([sequences_cat, tokens], dim=1)
        

        inference_params.verified_key_value_memory_dict['indices'].copy_(state_indices)

        if streamer is not None:
            streamer.put(sampled_tokens.cpu())
    if streamer is not None:
        streamer.end()
    if enable_timing:
        end.record()
        torch.cuda.synchronize()
        print(f"Prompt processing + decoding time: {(start.elapsed_time(end)):.0f}ms")

    output_cls = GreedySearchDecoderOnlyOutput if top_k == 1 else SampleDecoderOnlyOutput

    # returning the sequence with the highest probability, 
    out_sequence_idx = torch.argmax(log_probability, dim=0)
    out_sequence = sequences_cat[[out_sequence_idx], :]
    return output_cls(sequences=out_sequence, scores=tuple(scores), past_key_values=log_probability)
