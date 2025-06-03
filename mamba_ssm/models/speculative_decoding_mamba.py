
import math
from functools import partial
import json
import os
import copy

from collections import namedtuple
from typing import Optional
from transformers.generation import GreedySearchDecoderOnlyOutput, SampleDecoderOnlyOutput, TextStreamer

import torch
from einops import repeat

from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.utils.generation_utils import InferenceParams, update_graph_cache, modify_logit_for_repetition_penalty, sample
from mamba_ssm.utils.speculative_sampling import speculative_sampling, sampling_verification
from mamba_ssm.utils.tree_verification import verify_beam_search_tree, \
    sampling_verification_tree, unroll_tree, compress_tree, TokenTree

from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.models.speculative_decoding_strategies import STRAT_DICT
from mamba_ssm.utils.profile import cuda_time

try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


def maxcontig(input_block, output_block, arr):
    """
    Returns i,j indicies of maximum left contigous
    common array

    input_block (pad block of model inputs)
    output block (input_block predictions from model)
    arr (arange of size input_block)
    """

    pos_unchanged = output_block[:, :-1] == input_block
    left_contig = torch.cumprod(pos_unchanged, dim=-1)
    mask = torch.where(left_contig == 1, 0, -10 ** 9)
    contigs = torch.argmax(arr + mask, dim=-1)
    jstar, istar = contigs.max(dim=-1)
    return istar.item(), jstar.item()

def maxcontig_multiple(input_block, output_block, arr, num_input_seq):
    """
    Returns i,j indicies of maximum left contigous
    common array

    input_block (pad block of model inputs)
    output block (input_block predictions from model)
    arr (arange of size input_block)
    """
    Ndraft = input_block.shape[0] // num_input_seq
    pos_unchanged = output_block[:, :-1] == input_block
    left_contig = torch.cumprod(pos_unchanged, dim=-1)
    mask = torch.where(left_contig == 1, 0, -10 ** 9)
    mask = mask.view(num_input_seq, Ndraft, input_block.shape[1])
    contigs = torch.argmax(arr + mask, dim=-1)
    jstar, istar = contigs.max(dim=-1)
    return istar, jstar

class MambaLMHeadSpecDecModel(MambaLMHeadModel):
    def __init__(self, config: MambaConfig, initializer_cfg=None, device=None, dtype=None, strategy="png", npad=4, ndraft=5, num_beam=1, draft_num_beam=0, use_2step_kernel=False, jit_state_copy=True, save_last_seq=True, use_Nstep_kernel=False, use_tree_decoding=False, activation_replay=False, sample_target_only=False, unroll_tree=False) -> None:
        super().__init__(config, initializer_cfg, device, dtype)
        self.strat = STRAT_DICT[strategy]("mamba_ssm/strategies/model2gram/mamba2-2.7b/float16/extended_2gram_rankings.pth", Ndraft=ndraft, Npad=npad, base_num_beam=num_beam, draft_num_beam=draft_num_beam, max_length=10000)
        self.save_last_seq = save_last_seq
        self.use_Nstep_kernel = use_Nstep_kernel
        self.use_2step_kernel = use_2step_kernel
        self.jit_state_copy = jit_state_copy
        self.use_Nstep_kernel = use_Nstep_kernel
        self.use_tree_decoding = use_tree_decoding
        self.activation_replay = activation_replay
        self.npad = npad
        self.ndraft = ndraft
        self.strategy = strategy
        self.sample_target_only = sample_target_only
        self.unroll_tree = unroll_tree
        assert not (npad != 1 and use_2step_kernel), "npad has to be 1 when use_2step_kernel is true"
        assert (not (use_Nstep_kernel and use_2step_kernel)), "use_Nstep_kernel and use_2step_kernel cannot both be true"
        assert (not (use_2step_kernel and save_last_seq)), "incompatible options: use_2step_kernel, save_last_seq cannot be both true"
        assert not (npad != 1 and not save_last_seq and not activation_replay), "save last word with no activation replay only work for npad=1"

    @torch.inference_mode()
    def generate(self, input_ids, num_beam=1, draft_num_beam=0, *args, **kwargs):
        if self.save_last_seq:
            return self.generate_save_last_seq(input_ids, *args, **kwargs)
        else:
            if input_ids.shape[0] > 1:
                return self.generate_multi_sequence(input_ids, *args, **kwargs)
            elif self.unroll_tree:
                return self.generate_unroll_tree(input_ids, num_beam=num_beam, draft_num_beam=draft_num_beam, *args, **kwargs)
            elif self.use_tree_decoding:
                return self.generate_tree_decoding(input_ids, num_beam=num_beam, draft_num_beam=draft_num_beam, *args, **kwargs)
            else:   
                return self.generate_save_last_word(input_ids, *args, **kwargs)

    def generate_save_last_seq(self, input_ids, max_length, top_k=1, top_p=0, min_p=0, temperature=1, repetition_penalty=1, return_dict_in_generate=False, output_scores=False, eos_token_id=None, cg=False, **kwargs):
        '''
        generate_save_last_seq is implementing a joint attainment approach, where there is always 1 sequence prepended to the drafted input.
        The sequence is a padded sequence with tokens we verified to be correct in the last iteration. In this case, the states for the 1st sequence in the batch is 
        always gauranteed to be correct, because there will never be speculated tokens in the 1st sequence. During state copying, we always copy the 1st state to other position
        in the batch because we know for sure it is correct.
        Since we might get multiple tokens correct, we need to save the sequence of tokens we get correct to advance the verified state. Thus the name generate_save_last_seq.
        '''
        ndraft = self.ndraft
        npad = self.npad
        self.generation_hist = []
        # setting up inference parameters that hold ssm_states and conv_states and other values. 
        if cg:
            if not hasattr(self, "_decoding_cache"):
                self._decoding_cache = None
            self._decoding_cache = update_graph_cache(
                self,
                self._decoding_cache,
                ndraft+1,
                input_ids.shape[1],
                max_length,
                decoding_seqlens=list(range(npad+1, 2*npad+1+1)), # initializing CUDA graph of a list of npad+1 sequence lengths 
                use_2step_kernel=self.use_2step_kernel,
                jit_state_copy=self.jit_state_copy,
                save_last_seq=self.save_last_seq,
                use_Nstep_kernel=self.use_Nstep_kernel,
                npad=npad,
                ndraft=ndraft+1,
                first_iteration=False
            )
            inference_params = self._decoding_cache.inference_params
            inference_params.reset(max_length, ndraft+1)
        else:
            inference_params = InferenceParams(max_seqlen=max_length, max_batch_size=ndraft+1)
            eg_param = next(iter(self.parameters()))
            dtype = eg_param.dtype
            device = eg_param.device
            inference_params.key_value_memory_dict = self.allocate_inference_cache(batch_size=ndraft+1, max_seqlen=npad+1, dtype=dtype)
            # verified dict saves two indices to the verified state in states. In this case we always take 0
            inference_params.verified_key_value_memory_dict = {"indices":torch.zeros((1,), dtype=torch.long, device=device),
                                                               "length":torch.tensor(0, dtype=torch.long, device=device)}
            inference_params.use_2step_kernel = self.use_2step_kernel
            inference_params.save_last_seq = self.save_last_seq
            inference_params.use_Nstep_kernel = self.use_Nstep_kernel
            inference_params.npad = npad
            inference_params.ndraft = ndraft+1
            inference_params.jit_state_copy = self.jit_state_copy
            inference_params.first_iteration = False

        def should_stop(current_token, inference_params):
            if inference_params.seqlen_offset == 0:
                return False
            if eos_token_id is not None and (current_token == eos_token_id).all():
                return True
            if inference_params.seqlen_offset >= max_length - 1:
                return True
            return False
    
        sequences = [input_ids[:, :-1], input_ids[:, [-1]]]

        # prefill the ssm_states with prompts
        prefill_ids = input_ids[:, :-1]
        self.forward(prefill_ids, inference_params=inference_params)
        inference_params.seqlen_offset += prefill_ids.shape[1]
        # Saving a copy of the states 
        # Verified_key_value_memory_dict always holds a state that is at the end of the second last element in sequences
        # Need to replicate the states so that the batch of state matches batch of input later
        if self.strategy == "jacobi":
            inference_params.first_iteration = True
        elif self.strategy == "mamba":
            self.strat.initialize(input_ids[:, :-1], ndraft, npad, max_length)
        if self.jit_state_copy:
            inference_params.verified_key_value_memory_dict['indices'].copy_(0) 

        istar, jstar = -1, -1
        last_output = None
        decode_time = []
        drafting_time = []

        while not should_stop(current_token=sequences[-1], inference_params=inference_params):
            # drafted_block is is (Ndraft, len(in_seq)+npad)
            last_seq_len = sequences[-1].shape[1]
            drafted_block = -torch.ones((ndraft, last_seq_len + npad), device=input_ids.device).long()
            output_block = -torch.ones((ndraft, last_seq_len + npad + 1), device=input_ids.device).long()
            arr = torch.arange(drafted_block.size(1)).to(input_ids.device)
            # for the drafted block it starts with last decoded token and follows by speculated tokens
            step_block, draft_logit = self.strat.update(drafted_block,
                                                        output_block,
                                                        sequences[-1],
                                                        output_ids=torch.cat(sequences, dim=-1),
                                                        jstar=jstar, 
                                                        last_output=last_output,
                                                        sampling_params={"top_k":top_k, "top_p":top_p, "min_p": min_p,
                                                                            "temperature":temperature, 'repetition_penalty':repetition_penalty})
            step_Ndraft, step_seq_len = (step_block.size(0), step_block.size(1))
            # input_block is (ndraft+1, len(in_seq)+npad)
            # Adding a padded input sequence to retrieve the last verified state
            input_block = torch.cat([torch.nn.functional.pad(sequences[-1], (npad, 0), value=0), drafted_block], dim=0)
            input_mask = torch.ones_like(input_block)
            input_mask[0, :npad] = 0

            if not cg: 
                outputs = self.forward(
                    input_block,
                    inference_params=inference_params,
                    num_last_tokens=step_seq_len,
                    mask=input_mask
                )
                # We don't need the output of the first batch since we are only using it to retrieve states
                logits = outputs.logits

            else: 
                position_ids = torch.zeros_like(input_block) # place_holder to enable cuda graph, not used anywhere
                logits = self._decoding_cache.run(
                    input_block, position_ids, input_mask, inference_params.seqlen_offset
                ).squeeze(dim=1)

            logits = logits[1:, ...]
            if repetition_penalty != 1.0:
                logits = logits.transpose(1,2)
                logits = modify_logit_for_repetition_penalty(logits, 
                                                                repeat(torch.cat(sequences, dim=-1), 'a b -> a b r', r = logits.shape[-1]), 
                                                                repetition_penalty).transpose(1,2)

            if top_k == 1: # Greedy 
                preds = logits.argmax(dim=-1)
                output_block[:step_Ndraft, last_seq_len:] = preds[:, last_seq_len-1:]
                istar, jstar = maxcontig(input_block=step_block, output_block=output_block[:step_Ndraft, :step_seq_len + 1], arr=arr[:step_seq_len])
            else: # performing speculative sampling
                assert ndraft == 1 # TODO: Speculative sampling only support drafting 1 seqeunce now
                logits = logits[:,:,:self.config.vocab_size]
                sampled_output, _ = speculative_sampling(draft_logit[:, last_seq_len:, :], drafted_block[:, last_seq_len:], logits[:, last_seq_len-1:, :], temperature=temperature, top_k=top_k, top_p=top_p, min_p=min_p)
                # print(sampled_output, drafted_block, last_seq_len)
                istar = 0 
                jstar = last_seq_len + sampled_output.shape[1] - 2
                output_block[:step_Ndraft, last_seq_len:last_seq_len+sampled_output.shape[1]] = sampled_output 

            nverified = jstar + 1
            # if there is nverified token that means the 
            accepted_tokens = output_block[[istar], sequences[-1].shape[1]:nverified+1]
            inference_params.seqlen_offset += accepted_tokens.shape[1]
            sequences.append(accepted_tokens)

            if self.strategy == "jacobi":
                inference_params.first_iteration = False

            # when not doing jit state copy, we copy all the state at the end of the iteration. 
            if not self.jit_state_copy:
                for l in inference_params.key_value_memory_dict.keys():
                    inference_params.key_value_memory_dict[l][0].copy_(repeat(inference_params.key_value_memory_dict[l][0][[0], ...], 'b ... -> (r b) ...', r=input_block.shape[0])) 
                    inference_params.key_value_memory_dict[l][1].copy_(repeat(inference_params.key_value_memory_dict[l][1][[0], ...], 'b ... -> (r b) ...', r=input_block.shape[0]))

            self.generation_hist.append(accepted_tokens)
            last_output = output_block.clone()
        # print("avg forward pass time: ", sum(decode_time[3:]) / len(decode_time[3:]))
        # print("avg drafting time: ", sum(drafting_time[3:]) / len(drafting_time[3:]))
        return GreedySearchDecoderOnlyOutput(sequences=torch.cat(sequences, dim=1))
    
    def generate_save_last_word(self, input_ids, max_length, top_k=1, top_p=0, min_p=0, temperature=1, repetition_penalty=1, return_dict_in_generate=False, output_scores=False, eos_token_id=None, cg=False, **kwargs):
        '''
        generate_save_last_word implements multiple approach depending on the flags. 
        Without --activation_replay, it is implementing a joint attainment approach (same as generate_save_last_seq), but can only work for npad=1 (i.e. 
        speculating a maximum of 1 token into the future). With this approach, the key difference with generate_save_last_seq is that we might not always take 
        the 1st state. The 1 speculated token can either be correct or incorrect. When it is correct, we can take the state after that specific speculated 
        sequence. When it is incorrect, we can always fall back to the 1st verification sequence. In this case, we can only save the last word and have a fixed
        size input because we advance the state more when we accept more tokens (as compared to generate_save_last_seq, where we advance the state by the 
        verification sequence length always).
        With --activation_replay, it is implementing a activation checkpointing approach, where activations necessary for state computation is saved in the previous 
        iteration. Then we verify our speculation and take note how far we got correct. Before state is needed in the next iteration, we use how far we got correct
        to recompute the verified state. With this option, we do not need an additional verification sequence in the batch and would save compute as necessary 
        activations are saved.
        '''
        generator = torch.Generator(device='cuda').manual_seed(1)
        self.generation_hist = []
        ndraft = self.ndraft
        npad = self.npad

        if self.activation_replay:
            batchsize = ndraft
        else:
            batchsize = ndraft + 1
        if cg:
            if not hasattr(self, "_decoding_cache"):
                self._decoding_cache = None
            self._decoding_cache = update_graph_cache(
                self,
                self._decoding_cache,
                batchsize,
                input_ids.shape[1],
                max_length,
                decoding_seqlens=(npad+1,),
                use_2step_kernel=self.use_2step_kernel,
                jit_state_copy=self.jit_state_copy,
                save_last_seq=self.save_last_seq,
                use_Nstep_kernel=self.use_Nstep_kernel,
                activation_replay=self.activation_replay,
                npad=npad,
                ndraft=batchsize,
                first_iteration=False
            )
            inference_params = self._decoding_cache.inference_params
            inference_params.reset(max_length, batchsize)
        else:
            inference_params = InferenceParams(max_seqlen=max_length, max_batch_size=batchsize)
            eg_param = next(iter(self.parameters()))
            dtype = eg_param.dtype
            device = eg_param.device
            inference_params.key_value_memory_dict = self.allocate_inference_cache(batch_size=batchsize, max_seqlen=npad+1, dtype=dtype)
            # verified dict saves two indices to the verified state in states
            inference_params.verified_key_value_memory_dict = {"indices":torch.zeros((1,), dtype=torch.long, device=device),
                                                               "mask":torch.zeros((1, npad+1), dtype=torch.bool, device=device),
                                                               "state":self.allocate_inference_cache(batch_size=1, max_seqlen=npad+1, dtype=dtype)}
            inference_params.value_cache = self.allocate_value_cache(batch_size=batchsize, max_seqlen=npad+1, dtype=dtype)

            inference_params.use_2step_kernel = self.use_2step_kernel
            inference_params.save_last_seq = self.save_last_seq
            inference_params.use_Nstep_kernel = self.use_Nstep_kernel
            inference_params.npad = npad
            inference_params.ndraft = batchsize
            inference_params.jit_state_copy = self.jit_state_copy
            inference_params.activation_replay = self.activation_replay
            inference_params.first_iteration = False

        def should_stop(current_token, inference_params):
            if inference_params.seqlen_offset == 0:
                return False
            if eos_token_id is not None and (current_token == eos_token_id).all():
                return True
            if inference_params.seqlen_offset >= max_length - 1:
                return True
            return False
        
        sequences = [input_ids[:, :-1], input_ids[:, [-1]]]
        # prefill the ssm_states with prompts
        prefill_ids = input_ids[:, :-1]
        self.forward(prefill_ids, inference_params=inference_params)
        inference_params.seqlen_offset += prefill_ids.shape[1]

        if self.strategy == "mamba":
            self.strat.initialize(input_ids[:, :-1], Ndraft=ndraft, Npad=npad, max_length=max_length)
        # Saving a copy of the states 
        # Verified_key_value_memory_dict always holds a state that is at the end of the second last element in sequences
        # Need to replicate the states so that the batch of state matches batch of input later
        if self.jit_state_copy:
            if self.activation_replay:
                inference_params.verified_key_value_memory_dict['indices'].copy_(-1)
                inference_params.verified_key_value_memory_dict["mask"].copy_(torch.zeros((1, npad+1)))
                inference_params.first_iteration = True
            else:
                inference_params.verified_key_value_memory_dict['indices'].copy_(0)
        draft_time = []
        forward_time = []
        verfication_time = []

        draft_smapling_params = {"top_k":top_k, "top_p":top_p, "min_p": min_p, 
                                "temperature":temperature, 'repetition_penalty':repetition_penalty}
        if self.sample_target_only:
            draft_smapling_params = {"top_k":1, "top_p":top_p, "min_p": min_p, 
                                "temperature":temperature, 'repetition_penalty':repetition_penalty}

        while not should_stop(current_token=sequences[-1], inference_params=inference_params):
            # drafted_block is is (Ndraft, len(in_seq)+npad)
            last_seq_len = sequences[-1].shape[1]
            drafted_block = -torch.ones((ndraft, last_seq_len + npad), device=input_ids.device).long()
            output_block = -torch.ones((ndraft, last_seq_len + npad + 1), device=input_ids.device).long()
            arr = torch.arange(drafted_block.size(1)).to(input_ids.device)
            # for the drafted block it starts with last decoded token and follows by speculated tokens
            step_block, _, draft_logit = self.strat.update(drafted_block, 
                                                        output_block, 
                                                        sequences[-1], 
                                                        output_ids=torch.cat(sequences, dim=-1),
                                                        sampling_params=draft_smapling_params)
            step_Ndraft, step_seq_len = (step_block.size(0), step_block.size(1))
            # input_block is (ndraft+1, len(in_seq)+npad)
            # Adding a padded input sequence to retrieve the last verified state
            if not self.activation_replay:
                input_block = torch.cat([torch.nn.functional.pad(sequences[-1], (npad, 0), value=0), drafted_block], dim=0)
                input_mask = torch.ones_like(input_block)
                input_mask[0, :npad] = 0
            else:
                input_block = drafted_block
                input_mask = torch.ones_like(input_block)

            if not cg or inference_params.first_iteration:
                outputs = self.forward(
                    input_block,
                    inference_params=inference_params,
                    num_last_tokens=step_seq_len,
                    mask=input_mask
                )
                # We don't need the output of the first batch since we are only using it to retrieve states
                logits = outputs.logits

            else:
                position_ids = torch.zeros_like(input_block) # place_holder to enable cuda graph, not used anywhere
                logits = self._decoding_cache.run(
                    input_block, position_ids, input_mask, inference_params.seqlen_offset
                ).squeeze(dim=1)

            if not self.activation_replay:
                logits = logits[1:, ...]
            
            if repetition_penalty != 1.0:
                logits = logits.transpose(1,2)
                logits = modify_logit_for_repetition_penalty(logits, 
                                                                repeat(torch.cat(sequences, dim=-1), 'a b -> a b r', r = logits.shape[-1]), 
                                                                repetition_penalty).transpose(1,2)
                    
            if top_k == 1: # Greedy
                preds = logits.argmax(dim=-1)    
                output_block[:step_Ndraft, last_seq_len:] = preds[:, last_seq_len-1:]
                istar, jstar = maxcontig(input_block=step_block, output_block=output_block[:step_Ndraft, :step_seq_len + 1], arr=arr[:step_seq_len])
            elif self.sample_target_only:
                logits = logits[:,:,:self.config.vocab_size]
                sampled_output = sampling_verification(drafted_block[:, last_seq_len:],
                                                       logits[:, last_seq_len-1:, :],
                                                       temperature=temperature, 
                                                       top_k=top_k, 
                                                       top_p=top_p, 
                                                       min_p=min_p,
                                                       generator=generator)
                istar = 0
                jstar = last_seq_len + sampled_output.shape[1] - 2
                output_block[:step_Ndraft, last_seq_len:last_seq_len+sampled_output.shape[1]] = sampled_output
            else: # performing speculative sampling
                if ndraft == 1:
                    assert ndraft == 1 # TODO: Speculative sampling only support drafting 1 seqeunce now
                    logits = logits[:,:,:self.config.vocab_size]
                    sampled_output, _ = speculative_sampling(draft_logit[:, last_seq_len:, :], 
                                                        drafted_block[:, last_seq_len:], 
                                                        logits[:, last_seq_len-1:, :], 
                                                        temperature=temperature, 
                                                        top_k=top_k, 
                                                        top_p=top_p, 
                                                        min_p=min_p,
                                                        generator=generator)
                    # print(sampled_output, drafted_block, last_seq_len)
                    istar = 0 
                    jstar = last_seq_len + sampled_output.shape[1] - 2
                    output_block[:step_Ndraft, last_seq_len:last_seq_len+sampled_output.shape[1]] = sampled_output 
                else:
                    logits = logits[:,:,:self.config.vocab_size]
                    tree = TokenTree.from_independent_sequences(input_block, input_mask, draft_logit, logits)
                    sequences_cat, state_indices_base, log_probability, verified_step, verified_mask = tree.multi_step_speculative_sampling(
                        sequence_cat=torch.cat(sequences, dim=1),
                        log_probability=0,
                        top_k=top_k,
                        temperature=temperature,
                        top_p=top_p,
                        min_p=min_p,
                        generator=generator
                    )
                    istar = tree.get_token_sequence(verified_mask) # this should not be state_indices_base
                    jstar = verified_step + last_seq_len - 2
                    output_block[[istar], last_seq_len:last_seq_len+verified_step] = sequences_cat[:, -verified_step:]

            nverified = jstar + 1
            accepted_tokens = output_block[[istar], sequences[-1].shape[1]:nverified+1]
            inference_params.seqlen_offset += accepted_tokens.shape[1]
            for i in range(accepted_tokens.shape[1]):
                sequences.append(accepted_tokens[:, [i]])

            if self.jit_state_copy:
                if self.activation_replay:
                    inference_params.verified_key_value_memory_dict['indices'].copy_(istar)
                    verified_mask = torch.index_fill(torch.zeros((1, npad+1), device=input_ids.device, dtype=torch.bool), 
                                                    dim=1, 
                                                    index=torch.arange(nverified, device=input_ids.device), 
                                                    value=1)
                    inference_params.verified_key_value_memory_dict["mask"].copy_(verified_mask)
                    inference_params.first_iteration = False
                else:
                    if accepted_tokens.shape[1] > 1:
                        inference_params.verified_key_value_memory_dict['indices'].copy_(istar+1)
                    else:
                        inference_params.verified_key_value_memory_dict['indices'].copy_(0)
            else:
                for l in inference_params.key_value_memory_dict.keys():
                    if accepted_tokens.shape[1] > 1:
                        inference_params.key_value_memory_dict[l][0].copy_(repeat(inference_params.key_value_memory_dict[l][0][[istar+1], ...], 'b ... -> (r b) ...', r=input_block.shape[0])) 
                        inference_params.key_value_memory_dict[l][1].copy_(repeat(inference_params.key_value_memory_dict[l][1][[istar+1], ...], 'b ... -> (r b) ...', r=input_block.shape[0]))
                    else:
                        inference_params.key_value_memory_dict[l][0].copy_(repeat(inference_params.key_value_memory_dict[l][0][[0], ...], 'b ... -> (r b) ...', r=input_block.shape[0])) 
                        inference_params.key_value_memory_dict[l][1].copy_(repeat(inference_params.key_value_memory_dict[l][1][[0], ...], 'b ... -> (r b) ...', r=input_block.shape[0]))

            # for the purpose of computing acceptance rate and average length
            self.generation_hist.append(accepted_tokens)
        # print("avg drafting time: ", sum(draft_time) / len(draft_time))
        # print("avg forward time: ", sum(forward_time[3:]) / len(forward_time[3:]))
        # print("avg verification time: ", sum(verfication_time[3:]) / len(verfication_time[3:]))
        return GreedySearchDecoderOnlyOutput(sequences=torch.cat(sequences, dim=1))
    
    def generate_multi_sequence(self, input_ids, max_length, mask=None, input_matop_k=1, top_p=0, min_p=0, temperature=1, repetition_penalty=1, return_dict_in_generate=False, output_scores=False, eos_token_id=None, cg=False, **kwargs):
        '''
        generate_multi_sequence implements activation replay approach but has additional features to keep track of several generated sequences.
        With --activation_replay, it is implementing a activation checkpointing approach, where activations necessary for state computation is saved in the previous 
        iteration. Then we verify our speculation and take note how far we got correct. Before state is needed in the next iteration, we use how far we got correct
        to recompute the verified state. With this option, we do not need an additional verification sequence in the batch and would save compute as necessary 
        activations are saved.
        '''
        ndraft = self.ndraft # how many to draft for each sequence
        npad = self.npad
        assert self.activation_replay and self.jit_state_copy # requires activation_replay to be true 
        num_input_seq = input_ids.shape[0]
        self.generation_hist = [[] for _ in range(num_input_seq)]
        prompt_len = mask.sum(1)
        batchsize = ndraft * num_input_seq

        
        # Initializing inference params,
        if cg:
            if not hasattr(self, "_decoding_cache"):
                self._decoding_cache = None
            self._decoding_cache = update_graph_cache(
                self,
                self._decoding_cache,
                batchsize,
                input_ids.shape[1],
                max_length,
                decoding_seqlens=(npad+1,),
                use_2step_kernel=self.use_2step_kernel,
                jit_state_copy=self.jit_state_copy,
                save_last_seq=self.save_last_seq,
                use_Nstep_kernel=self.use_Nstep_kernel,
                activation_replay=self.activation_replay,
                npad=npad,
                ndraft=ndraft,
                first_iteration=False,
                num_input_seq=num_input_seq
            )
            inference_params = self._decoding_cache.inference_params
            inference_params.reset(max_length, batchsize)
        else:
            inference_params = InferenceParams(max_seqlen=max_length, max_batch_size=batchsize)
            eg_param = next(iter(self.parameters()))
            dtype = eg_param.dtype
            device = eg_param.device
            inference_params.key_value_memory_dict = self.allocate_inference_cache(batch_size=batchsize, max_seqlen=npad+1, dtype=dtype)
            # verified dict saves two indices to the verified state in states
            inference_params.verified_key_value_memory_dict = {"indices":torch.zeros((num_input_seq,), dtype=torch.long, device=device),
                                                               "mask":torch.zeros((num_input_seq, npad+1), dtype=torch.bool, device=device),
                                                               "state":self.allocate_inference_cache(batch_size=num_input_seq, max_seqlen=npad+1, dtype=dtype)}
            inference_params.value_cache = self.allocate_value_cache(batch_size=batchsize, max_seqlen=npad+1, dtype=dtype)

            inference_params.use_2step_kernel = self.use_2step_kernel
            inference_params.save_last_seq = self.save_last_seq
            inference_params.use_Nstep_kernel = self.use_Nstep_kernel
            inference_params.npad = npad
            inference_params.ndraft = ndraft
            inference_params.jit_state_copy = self.jit_state_copy
            inference_params.activation_replay = self.activation_replay
            inference_params.first_iteration = False
            inference_params.num_input_seq = num_input_seq
            inference_params.multi_seqlen_offset = [0] * num_input_seq
        
        # The stopping criteria is set to all input sequences generate at least max_length - input_ids.shape length (genlen)
        def should_stop(current_token, inference_params): 
            if inference_params.seqlen_offset == 0:
                return False
            if eos_token_id is not None and (current_token == eos_token_id).all():
                return True
            if (torch.tensor(inference_params.multi_seqlen_offset, device=prompt_len.device) - prompt_len).min() >= max_length - input_ids.shape[1]:
                return True
            return False
        
        sequences = [[] for _ in range(num_input_seq)]
        for i in range(num_input_seq):
            sequences[i].append(input_ids[[i], -mask[i, :].sum(0):-1])
            sequences[i].append(input_ids[[i], -1:])
        # prefill the ssm_states with prompts
        # mask is needed here as the prompts are of different length and we need to mask off the pad tokens introduced.
        prefill_ids = input_ids[:, :-1]
        self.forward(prefill_ids, inference_params=inference_params, mask=mask[:, :-1])
        inference_params.seqlen_offset += prefill_ids.shape[1]
        # multi_seqlen_offset is a list of seqlen_offset to keep track of multiple sequence generation.
        # Here when accumulating the seqlen_offset, we want to skip the pad tokens introduced for aligning input length
        for i in range(num_input_seq):
            inference_params.multi_seqlen_offset[i] += mask[i, :-1].sum(0)
        
        # We don't need to do anything replaying in the first decoding step so these are dummy values
        # First iteration is set to True to indicate to the model that we do not need to replay 
        inference_params.verified_key_value_memory_dict['indices'].copy_(-1)
        inference_params.verified_key_value_memory_dict["mask"].copy_(0)
        inference_params.first_iteration = True

        while not should_stop(current_token=sequences[-1], inference_params=inference_params):
            # drafted_block is is (Ndraft, len(in_seq)+npad)
            last_words = torch.cat([x[-1] for x in sequences], dim=0)
            drafted_block = -torch.ones((num_input_seq * ndraft, 1 + npad), device=input_ids.device).long()
            output_block = -torch.ones((num_input_seq * ndraft, 1 + npad + 1), device=input_ids.device).long()
            arr = torch.arange(drafted_block.size(1)).to(input_ids.device)
            # for the drafted block it starts with last decoded token and follows by speculated tokens
            step_block = self.strat.update_multiple(drafted_block, 
                                                    output_block, 
                                                    last_words, 
                                                    output_ids=[torch.cat(sequences[i], dim=-1) for i in range(num_input_seq)])
            step_batchsize, step_seq_len = (step_block.size(0), step_block.size(1))

            input_block = drafted_block
            input_mask = torch.ones_like(input_block)

            if not cg or inference_params.first_iteration:
                outputs = self.forward(
                    input_block,
                    inference_params=inference_params,
                    num_last_tokens=step_seq_len,
                    mask=input_mask
                )
                preds = outputs.logits.argmax(dim=-1)
            else:
                position_ids = torch.zeros_like(input_block) # place_holder to enable cuda graph, not used anywhere
                logits = self._decoding_cache.run(
                    input_block, position_ids, input_mask, inference_params.seqlen_offset
                ).squeeze(dim=1)
                preds = logits.argmax(dim=-1)
            output_block[:step_batchsize, 1:] = preds[:, 0:]
            # This returns multiple istar, jstar depending on the num_input_seq
            istar, jstar = maxcontig_multiple(input_block=step_block, output_block=output_block[:step_batchsize, :step_seq_len + 1], arr=arr[:step_seq_len], num_input_seq=num_input_seq)

            nverified = jstar + 1
            output_block = output_block.view(num_input_seq, ndraft, output_block.shape[1])
            # Picking out the verified sequence for each input sequences
            accepted_sequences = output_block[torch.arange(num_input_seq), istar]
            inference_params.seqlen_offset += (nverified[0]) 
            # For each sequence, we want to append the generated output to generation hist and increment multi_seqlen_offset accordingly
            for i in range(num_input_seq):
                accepted_tokens = accepted_sequences[[i],1:nverified[i]+1]
                self.generation_hist[i].append(accepted_tokens)
                if prompt_len[i] == 0:
                    inference_params.multi_seqlen_offset[i] += (npad+1)
                    for j in range(npad+1):
                        sequences[i].append(torch.zeros_like(accepted_tokens[:,[0]]))
                else:
                    inference_params.multi_seqlen_offset[i] += accepted_tokens.shape[1]
                    for j in range(accepted_tokens.shape[1]):
                        sequences[i].append(accepted_tokens[:, [j]])

            # The istar we get is wr
            adjusted_istar = istar + torch.arange(istar.shape[0], device=istar.device, dtype=istar.dtype) * ndraft
            inference_params.verified_key_value_memory_dict['indices'].copy_(adjusted_istar)
            verified_mask = torch.zeros((num_input_seq, npad+1), device=input_ids.device, dtype=torch.bool)
            for i in range(num_input_seq):
                verified_mask[i] = torch.index_fill(torch.zeros((1, npad+1), device=input_ids.device, dtype=torch.bool), 
                                                dim=1, 
                                                index=torch.arange(nverified[i], device=input_ids.device), 
                                                value=1)
            inference_params.verified_key_value_memory_dict["mask"].copy_(verified_mask)
            inference_params.first_iteration = False

        # concatenating the generation output into sequences and pad them to the same length accordingly
        sequences = [torch.cat(x, dim=1) for x in sequences]
        max_gen_len = max(inference_params.multi_seqlen_offset) 
        sequences = [torch.nn.functional.pad(sequences[x], (0, max_gen_len - sequences[x].shape[1])) for x in range(num_input_seq)]
        # Pass multiple sequence out with batches
        return GreedySearchDecoderOnlyOutput(sequences=torch.cat(sequences, dim=0))
    


    def generate_tree_decoding(self, input_ids, max_length, top_k=1, top_p=0, min_p=0, temperature=1, repetition_penalty=1, num_beam=1, draft_num_beam=1, return_dict_in_generate=False, output_scores=False, eos_token_id=None, cg=False, **kwargs):
        '''
        generate_beam_search implements several settings, including speculative beam search,
        speculative decoding with a tree (drafting a tree and verify 1 sequence)
        Speculative execution (naive sampling instead of speculative sampling) 
        '''
        assert self.jit_state_copy and self.activation_replay, "speculative beam search only implemented for these two options"
        self.generation_hist = []
        ndraft = self.ndraft
        npad = self.npad
        generator = torch.Generator(device="cuda").manual_seed(1)

        batchsize = 1
        if draft_num_beam > 0 or num_beam > 1: # doing beam search related
            decoding_seqlen_min = 1 + npad * math.ceil(draft_num_beam / num_beam)
            decoding_seqlen_max = 2 + npad * draft_num_beam
            decoding_seqlen_step = 1
            graph_cache_npad = npad * draft_num_beam
        else: # general 
            decoding_seqlen_min = 1 + npad
            decoding_seqlen_max = 2 + npad * ndraft # 2 + since range is exclsive at the end
            decoding_seqlen_step = npad # we pad to the next whole Ndraft to reduce the number of 
            graph_cache_npad = npad * ndraft

        if cg:
            if not hasattr(self, "_decoding_cache"):
                self._decoding_cache = None
            self._decoding_cache = update_graph_cache(
                self,
                self._decoding_cache,
                num_beam,
                input_ids.shape[1],
                max_length,
                decoding_seqlens=list(range(decoding_seqlen_min, decoding_seqlen_max, decoding_seqlen_step)),
                use_2step_kernel=False,
                jit_state_copy=self.jit_state_copy,
                save_last_seq=self.save_last_seq,
                use_Nstep_kernel=False,
                activation_replay=self.activation_replay,
                npad=graph_cache_npad,
                ndraft=1,
                first_iteration=False,
                use_tree_scan_kernel=True,
                num_input_seq=num_beam
            )
            inference_params = self._decoding_cache.inference_params
            inference_params.reset(max_length, num_beam)
        else:
            inference_params = InferenceParams(max_seqlen=max_length, max_batch_size=num_beam)
            eg_param = next(iter(self.parameters()))
            dtype = eg_param.dtype
            device = eg_param.device
            inference_params.key_value_memory_dict = self.allocate_inference_cache(batch_size=num_beam, max_seqlen=npad+1, dtype=dtype)
            # verified dict saves two indices to the verified state in states
            inference_params.verified_key_value_memory_dict = {"indices":torch.zeros((num_beam,), dtype=torch.long, device=device),
                                                               "mask":torch.zeros((num_beam, graph_cache_npad+1), dtype=torch.bool, device=device),
                                                               "state":self.allocate_inference_cache(batch_size=num_beam, max_seqlen=npad+1, dtype=dtype)}
            inference_params.value_cache = self.allocate_value_cache(batch_size=num_beam, max_seqlen=graph_cache_npad+1, dtype=dtype)

            inference_params.use_2step_kernel = False
            inference_params.save_last_seq = self.save_last_seq
            inference_params.use_Nstep_kernel = False
            inference_params.use_tree_scan_kernel = True
            inference_params.num_input_seq = num_beam
            inference_params.npad = graph_cache_npad
            inference_params.ndraft = 1
            inference_params.jit_state_copy = self.jit_state_copy
            inference_params.activation_replay = self.activation_replay
            inference_params.first_iteration = False

        def should_stop(current_token, inference_params):
            if inference_params.seqlen_offset == 0:
                return False
            if eos_token_id is not None and (current_token == eos_token_id).all():
                return True
            if inference_params.seqlen_offset >= max_length - 1:
                return True
            return False
        
        sequences = [input_ids[:, :-1], input_ids[:, [-1]]]
        sequences_cat = input_ids
        log_probability = torch.zeros((batchsize * num_beam, 1), device=input_ids.device, dtype=torch.float32)
        vocab_size = self.config.vocab_size
        # prefill the ssm_states with prompts and get the initial 3 beams
        if self.strategy == "mamba-bs":
            prefill_ids = input_ids
            mask = torch.ones_like(prefill_ids) # we don't use a causal mask here since the prefill step is still using a chunkscan
            output = self.forward(prefill_ids, mask=mask, num_last_tokens=1, inference_params=inference_params)
            inference_params.seqlen_offset += prefill_ids.shape[1]
            output_logits = output.logits[:,:,:vocab_size]

            if self.sample_target_only:
                sampled_tokens = sample(output_logits[:, 0, :], top_k=top_k, top_p=top_p, min_p=min_p, temperature=temperature)
                sequences_cat = torch.cat([sequences_cat, sampled_tokens.unsqueeze(0)], dim=1)
                inference_params.first_iteration = True
            else:
                curr_prob = torch.softmax(output_logits, dim=-1) # (batchsize * num_beam, vocabsize)
                sampled_output = curr_prob.topk(num_beam, dim=-1)
                sampled_prob = sampled_output.values
                sampled_tokens = sampled_output.indices

                log_probability += torch.log(sampled_prob.view((batchsize * num_beam, 1)))
                sequences_cat = repeat(sequences_cat, 'b l -> (n b) l', n=num_beam)
                sequences_cat = torch.cat([sequences_cat, sampled_tokens.view(batchsize * num_beam, 1)], dim=1)
                inference_params.first_iteration = True
        else:
            prefill_ids = input_ids[:, :-1]
            self.forward(prefill_ids, inference_params=inference_params)
            inference_params.seqlen_offset += prefill_ids.shape[1]
            inference_params.first_iteration = True
        # print(log_probability)
        

        # Only implemented for mamba model drafting.
        # assert self.strategy == "mamba-bs"
        available_strat = ["mamba-bs", "mamba", "png"]
        assert self.strategy in available_strat
        if self.strategy == "mamba-bs":
            self.strat.initialize(input_ids, Npad=npad, max_length=max_length, draft_num_beam=draft_num_beam, base_num_beam=num_beam, cg=cg)
        elif self.strategy == "mamba":
            self.strat.initialize(input_ids[:, :-1], Ndraft=ndraft, Npad=npad, max_length=max_length)
        # Saving a copy of the states 
        # Verified_key_value_memory_dict always holds a state that is at the end of the second last element in sequences
        # Need to replicate the states so that the batch of state matches batch of input later
        
        state_indices_base = torch.arange(num_beam, device=input_ids.device)
        inference_params.verified_key_value_memory_dict['indices'].copy_(torch.arange(num_beam, device=input_ids.device))
        temp_mask = torch.zeros((num_beam, graph_cache_npad+1), device=input_ids.device)
        temp_mask[:,0] = 1
        inference_params.verified_key_value_memory_dict["mask"].copy_(temp_mask) # TODO: figure out what to do with the mask.

        draft_time = []
        forward_time = []
        verification_time = []
        draft_smapling_params = {"top_k":top_k, "top_p":top_p, "min_p": min_p, 
                                "temperature":temperature, 'repetition_penalty':repetition_penalty}
        if self.sample_target_only:
            draft_smapling_params = {"top_k":1, "top_p":top_p, "min_p": min_p, 
                                "temperature":temperature, 'repetition_penalty':repetition_penalty}

        while not should_stop(current_token=sequences[-1], inference_params=inference_params):
            # drafted_block is is (Ndraft, len(in_seq)+npad)
            last_seq_len = sequences[-1].shape[1]
            # These are just place holders here.
            drafted_block = -torch.ones((ndraft, last_seq_len + npad), device=input_ids.device).long()
            output_block = -torch.ones((ndraft, last_seq_len + npad + 1), device=input_ids.device).long()
            arr = torch.arange(drafted_block.size(1)).to(input_ids.device)
            # for the drafted block it starts with last decoded token and follows by speculated tokens
            # with cuda_time("drafting"):
            out_tokens, out_masks, draft_logits = self.strat.update(drafted_block, 
                                                        output_block, 
                                                        sequences_cat[:, [-1]], 
                                                        output_ids=sequences_cat,
                                                        state_indices_base=state_indices_base,
                                                        sampling_params=draft_smapling_params)

            input_block = torch.where(out_tokens==-1, 0, out_tokens)
            input_mask = out_masks
            if self.strategy == "mamba" or self.strategy == "png":
                # with cuda_time("tree forming", l=draft_time):
                tree = TokenTree.from_independent_sequences(input_block, input_mask, draft_logits)
                input_block = tree.input_token
                input_mask = tree.input_mask
                # pad to the next multiple of decoding seqlen
                original_seqlen = input_block.shape[1]
                to_pad =  (math.ceil((input_block.shape[1]-1) / npad) * npad + 1) - input_block.shape[1]
                input_block = torch.nn.functional.pad(input_block, (0, to_pad), value=0)
                input_mask = torch.nn.functional.pad(input_mask, (0, to_pad, 0, to_pad), value=0)

            position_ids = torch.zeros_like(input_block) # place_holder to enable cuda graph, not used anywhere

            # with cuda_time("forward pass"):
            if not cg or inference_params.first_iteration:
                logits = self.forward(
                    input_ids=input_block, 
                    position_ids=position_ids, 
                    inference_params=inference_params,
                    num_last_tokens=-1,
                    mask=input_mask
                ).logits[:,:,:vocab_size]
            else:
                logits = self._decoding_cache.run(
                    input_block, position_ids, input_mask, inference_params.seqlen_offset
                ).squeeze(dim=1)[:,:,:vocab_size]

            # with cuda_time("verification"):
            if top_k == 1:
                # with cuda_time("verification", l=verification_time):
                sequences_cat, state_indices_base, log_probability, verified_step, verified_mask = verify_beam_search_tree(
                    input_tokens=input_block, 
                    input_mask=input_mask,
                    output_logits=logits,
                    log_probability=log_probability,
                    sequence_cat=sequences_cat,
                    base_num_beams=num_beam,
                    draft_num_beams=draft_num_beam
                )
            elif self.sample_target_only:
                sequences_cat, state_indices_base, log_probability, verified_step, verified_mask = sampling_verification_tree(
                    input_tokens=input_block, 
                    input_mask=input_mask,
                    output_logits=logits,
                    log_probability=log_probability,
                    sequence_cat=sequences_cat,
                    top_k=top_k,
                    temperature=temperature,
                    top_p=top_p,
                    min_p=min_p,
                    generator=generator
                )
            else:
                # the MSS sampling 
                if self.strategy == "mamba" or self.strategy == "png":
                    tree.target_logits = logits[:, :original_seqlen, :]
                    sequences_cat, state_indices_base, log_probability, verified_step, verified_mask = tree.multi_step_speculative_sampling(
                        sequence_cat=sequences_cat,
                        log_probability=log_probability,
                        top_k=top_k,
                        temperature=temperature,
                        top_p=top_p,
                        min_p=min_p,
                        generator=generator
                    )
                else:
                    tree = TokenTree(input_token=input_block,
                                    input_mask=input_mask,
                                    target_logits=logits, 
                                    draft_logits=draft_logits)
                    sequences_cat, state_indices_base, log_probability, verified_step, verified_mask = tree.multi_step_speculative_sampling(
                        sequence_cat=sequences_cat,
                        log_probability=log_probability,
                        top_k=top_k,
                        temperature=temperature,
                        top_p=top_p,
                        min_p=min_p,
                        generator=generator
                    )
            
            inference_params.seqlen_offset += verified_step

            inference_params.verified_key_value_memory_dict['indices'].copy_(state_indices_base)
            verified_mask = torch.nn.functional.pad(verified_mask, (0, graph_cache_npad+1-verified_mask.shape[1]))

            inference_params.verified_key_value_memory_dict["mask"].copy_(verified_mask)
            inference_params.first_iteration = False

            self.generation_hist.append(verified_step)

        # print("draft:", sum(draft_time)/len(draft_time))
        # print("forward:", sum(forward_time)/len(forward_time))
        # print("verification:", sum(verification_time)/len(verification_time))
        return GreedySearchDecoderOnlyOutput(sequences=sequences_cat, scores=log_probability)
    
    def generate_unroll_tree(self, input_ids, max_length, top_k=1, top_p=0, min_p=0, temperature=1, repetition_penalty=1, num_beam=1, draft_num_beam=1, return_dict_in_generate=False, output_scores=False, eos_token_id=None, cg=False, **kwargs):
        '''
        generate_unroll_tree implements a baseline where the drafted tree is unrolled into multiple sequence 
        and use normal Nstep kernel to compute the output
        It is a combination of generate_beam_search, where the drafting part comes from generate beam search
        and generate_save_last_word, where we want to do the forward pass using Nstep kernel.
        It only supports verifying 1 sequence at the moment
        '''
        assert self.jit_state_copy and self.activation_replay, "speculative beam search only implemented for these two options"
        self.generation_hist = []
        ndraft = self.ndraft
        npad = self.npad
        generator = torch.Generator(device="cuda").manual_seed(1)

        batchsize = 1
        decoding_seqlen_min = math.ceil(draft_num_beam / num_beam)
        max_batch_size = (npad - 1) * (draft_num_beam - 1) + draft_num_beam
        
        if cg:
            if not hasattr(self, "_decoding_cache"):
                self._decoding_cache = None
            for b in range(max_batch_size, draft_num_beam-1, -1):
                self._decoding_cache = update_graph_cache(
                    self,
                    self._decoding_cache,
                    b,
                    input_ids.shape[1],
                    max_length,
                    decoding_seqlens=[npad+1, ],
                    use_2step_kernel=False,
                    jit_state_copy=self.jit_state_copy,
                    save_last_seq=self.save_last_seq,
                    use_Nstep_kernel=self.use_Nstep_kernel,
                    activation_replay=self.activation_replay,
                    npad=npad,
                    ndraft=1,
                    first_iteration=False,
                    use_tree_scan_kernel=False,
                    num_input_seq=1,
                    unroll_tree=self.unroll_tree
                )
            inference_params = self._decoding_cache.inference_params
            inference_params.reset(max_length, 1)
        else:
            inference_params = InferenceParams(max_seqlen=max_length, max_batch_size=max_batch_size)
            eg_param = next(iter(self.parameters()))
            dtype = eg_param.dtype
            device = eg_param.device
            inference_params.key_value_memory_dict = self.allocate_inference_cache(batch_size=max_batch_size, max_seqlen=npad+1, dtype=dtype)
            # verified dict saves two indices to the verified state in states
            inference_params.verified_key_value_memory_dict = {"indices":torch.zeros((1,), dtype=torch.long, device=device),
                                                               "mask":torch.zeros((1, npad+1), dtype=torch.bool, device=device),
                                                               "state":self.allocate_inference_cache(batch_size=1, max_seqlen=npad+1, dtype=dtype)}
            inference_params.value_cache = self.allocate_value_cache(batch_size=max_batch_size, max_seqlen=npad+1, dtype=dtype)

            inference_params.use_2step_kernel = False
            inference_params.save_last_seq = self.save_last_seq
            inference_params.use_Nstep_kernel = self.use_Nstep_kernel
            inference_params.use_tree_scan_kernel = False
            inference_params.num_input_seq = num_beam
            inference_params.npad = npad
            inference_params.ndraft = 1
            inference_params.jit_state_copy = self.jit_state_copy
            inference_params.activation_replay = self.activation_replay
            inference_params.first_iteration = False
            inference_params.unroll_tree = self.unroll_tree

        def should_stop(current_token, inference_params):
            if inference_params.seqlen_offset == 0:
                return False
            if eos_token_id is not None and (current_token == eos_token_id).all():
                return True
            if inference_params.seqlen_offset >= max_length - 1:
                return True
            return False
        
        sequences = [input_ids[:, :-1], input_ids[:, [-1]]]
        sequences_cat = input_ids
        log_probability = torch.zeros((batchsize * num_beam, 1), device=input_ids.device, dtype=torch.float32)
        vocab_size = self.config.vocab_size
        # prefill the ssm_states with prompts and get the initial 3 beams
        prefill_ids = input_ids
        mask = torch.ones_like(prefill_ids) # we don't use a causal mask here since the prefill step is still using a chunkscan
        output = self.forward(prefill_ids, mask=mask, num_last_tokens=1, inference_params=inference_params)
        inference_params.seqlen_offset += prefill_ids.shape[1]
        output_logits = output.logits[:,:,:vocab_size]

        if self.sample_target_only:
            sampled_tokens = sample(output_logits[:, 0, :], top_k=top_k, top_p=top_p, min_p=min_p, temperature=temperature)
            sequences_cat = torch.cat([sequences_cat, sampled_tokens.unsqueeze(0)], dim=1)
            inference_params.first_iteration = True
        else:
            curr_prob = torch.softmax(output_logits, dim=-1) # (batchsize * num_beam, vocabsize)
            sampled_output = curr_prob.topk(num_beam, dim=-1)
            sampled_prob = sampled_output.values
            sampled_tokens = sampled_output.indices

            log_probability += torch.log(sampled_prob.view((batchsize * num_beam, 1)))
            sequences_cat = repeat(sequences_cat, 'b l -> (n b) l', n=num_beam)
            sequences_cat = torch.cat([sequences_cat, sampled_tokens.view(batchsize * num_beam, 1)], dim=1)
            inference_params.first_iteration = True
        # print(log_probability)
        

        # Only implemented for mamba model drafting.
        assert self.strategy == "mamba-bs"
        self.strat.initialize(input_ids, npad, max_length, draft_num_beam, num_beam, cg=cg)
        # Saving a copy of the states 
        # Verified_key_value_memory_dict always holds a state that is at the end of the second last element in sequences
        # Need to replicate the states so that the batch of state matches batch of input later
        
        state_indices_base = torch.arange(num_beam, device=input_ids.device)
        inference_params.verified_key_value_memory_dict['indices'].copy_(torch.arange(num_beam, device=input_ids.device))
        temp_mask = torch.zeros((num_beam, npad+1), device=input_ids.device)
        temp_mask[:,0] = 1
        inference_params.verified_key_value_memory_dict["mask"].copy_(temp_mask) # TODO: figure out what to do with the mask.

        draft_time = []
        forward_time = []
        verification_time = []

        while not should_stop(current_token=sequences[-1], inference_params=inference_params):
            # drafted_block is is (Ndraft, len(in_seq)+npad)
            last_seq_len = sequences[-1].shape[1]
            # These are just place holders here.
            drafted_block = -torch.ones((ndraft, last_seq_len + npad), device=input_ids.device).long()
            output_block = -torch.ones((ndraft, last_seq_len + npad + 1), device=input_ids.device).long()
            arr = torch.arange(drafted_block.size(1)).to(input_ids.device)
            # for the drafted block it starts with last decoded token and follows by speculated tokens
            # with cuda_time("drafting"):
            out_tokens, out_masks, draft_logits = self.strat.update(drafted_block, 
                                                        output_block, 
                                                        sequences[-1], 
                                                        output_ids=sequences_cat,
                                                        state_indices_base=state_indices_base)

            unroll_input, unroll_mask, compress_matrix = unroll_tree(out_tokens, out_masks)
            input_block = torch.where(unroll_input==-1, 0, unroll_input)
            input_mask = unroll_mask

            position_ids = torch.zeros_like(input_block) # place_holder to enable cuda graph, not used anywhere
            # with cuda_time("forward pass"):
            if not cg or inference_params.first_iteration:
                logits = self.forward(
                    input_ids=input_block, 
                    position_ids=position_ids, 
                    inference_params=inference_params,
                    num_last_tokens=-1,
                    mask=input_mask
                ).logits[:,:,:vocab_size]
            else:
                logits = self._decoding_cache.run(
                    input_block, position_ids, input_mask, inference_params.seqlen_offset
                ).squeeze(dim=1)[:,:,:vocab_size]

            # need to put the output back into a tree?
            logits = compress_tree(logits, compress_matrix)
            # with cuda_time("verification"):
            if top_k == 1:
                sequences_cat, state_indices_base, log_probability, verified_step, verified_mask = verify_beam_search_tree(
                    input_tokens=out_tokens, 
                    input_mask=out_masks,
                    output_logits=logits,
                    log_probability=log_probability,
                    sequence_cat=sequences_cat,
                    base_num_beams=num_beam,
                    draft_num_beams=draft_num_beam
                )
            elif self.sample_target_only:
                sequences_cat, state_indices_base, log_probability, verified_step, verified_mask = sampling_verification_tree(
                    input_tokens=out_tokens, 
                    input_mask=out_masks,
                    output_logits=logits,
                    log_probability=log_probability,
                    sequence_cat=sequences_cat,
                    top_k=top_k,
                    temperature=temperature,
                    top_p=top_p,
                    min_p=min_p,
                    generator=generator
                )
            else:
                # the MSS sampling 
                tree = TokenTree(input_token=out_tokens,
                                 input_mask=out_masks,
                                 draft_logits=draft_logits,
                                 target_logits=logits)
                sequences_cat, state_indices_base, log_probability, verified_step, verified_mask = tree.multi_step_speculative_sampling(
                    sequence_cat=sequences_cat,
                    log_probability=log_probability,
                    top_k=top_k,
                    temperature=temperature,
                    top_p=top_p,
                    min_p=min_p,
                    generator=generator
                )
            
            inference_params.seqlen_offset += verified_step
            # we need to convert the indices and mask into the unrolled format
            last_correct_token_pos = torch.max(torch.arange(verified_mask.shape[1], device=verified_mask.device) * \
                verified_mask)
            correct_token_indices = torch.nonzero(compress_matrix==last_correct_token_pos)[0]
            # print(last_correct_token_pos)
            # print(compress_matrix)
            # print(correct_token_indices)
            state_indices = correct_token_indices[1]
            left_contiguous_mask = torch.arange(input_block.shape[1], device=verified_mask.device) <= correct_token_indices[2]
            left_contiguous_mask = torch.where(input_mask[state_indices]==1, left_contiguous_mask, 0)
            # print(state_indices)
            # print(left_contiguous_mask)
            inference_params.verified_key_value_memory_dict['indices'].copy_(state_indices)
            verified_mask = torch.nn.functional.pad(left_contiguous_mask, (0, npad+1-left_contiguous_mask.shape[0]))
            # print(verified_mask)

            inference_params.verified_key_value_memory_dict["mask"].copy_(verified_mask)
            inference_params.first_iteration = False

            self.generation_hist.append(verified_step)

        # print("draft:", sum(draft_time)/len(draft_time))
        # print("forward:", sum(forward_time)/len(forward_time))
        # print("verification:", sum(verification_time)/len(verification_time))
        return GreedySearchDecoderOnlyOutput(sequences=sequences_cat, scores=log_probability)