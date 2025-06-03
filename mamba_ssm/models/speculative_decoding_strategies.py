import torch
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
import copy
from mamba_ssm.utils.generation_utils import update_graph_cache, sample, modify_logit_for_repetition_penalty, \
    InferenceParams, capture_graph_huggingface
from mamba_ssm.utils.profile import cuda_time
from mamba_ssm.utils.speculative_sampling import adjust_logits
from einops import repeat
from transformers import AutoModelForCausalLM, AutoConfig, StaticCache, AutoTokenizer


class ModelBigram:
    def __init__(self, model_path, **kwargs):
        with torch.inference_mode():
            self.rankings = torch.load(model_path)
        
    # change last word to be last seq
    def update(self, input_block, output_block, last_seq, **kwargs):
        Ndraft= input_block.size(0)
        input_block[:, :last_seq.shape[1]] = last_seq
        output_block[:, :last_seq.shape[1]] = last_seq

        last_word = last_seq[0, -1]
        bg = self.rankings[last_word][:Ndraft].reshape(Ndraft, 1)
        input_block[:, last_seq.shape[1]:] = bg.to(input_block.device)
        # also return a logit
        draft_logits = -torch.inf * torch.ones((*input_block.shape, self.rankings.shape[0]), device=input_block.device, dtype=torch.float16)
        draft_logits.scatter_(2, input_block[:,:,None], 1)
        return input_block, draft_logits

    def update_(self, input_block, output_block, last_word, **kwargs):
        Ndraft= input_block.size(0)
        input_block[:, 0] = last_word
        output_block[:, 0] = last_word

        bg = self.rankings[last_word][:Ndraft].reshape(Ndraft, 1)
        input_block[:, 1:] = bg.to(input_block.device)
        if not hasattr(self, 'draft_ids'):
            self.draft_ids = ['ModelBigram' for _ in range(Ndraft)]
        return input_block

    def get_strat_keys_(self):
        return ['ModelBigram']

class ModelBigramModelExt:
    def __init__(self, model_path, **kwargs):
        with torch.inference_mode():
            self.rankings = torch.load(model_path)
        
    
    def update(self, input_block, output_block, last_seq, **kwargs):
        Ndraft= input_block.size(0)
        Npad = input_block.size(1) - last_seq.shape[1]
        input_block[:, :last_seq.shape[1]] = last_seq
        output_block[:, :last_seq.shape[1]] = last_seq
        last_word = last_seq[0, -1]

        bg = self.rankings[last_word][:Ndraft, :Npad].reshape(Ndraft, Npad)
        input_block[:, last_seq.shape[1]:] = bg.to(input_block.device)
        # also return a logit
        draft_logits = -torch.inf * torch.ones((*input_block.shape, self.rankings.shape[0]), device=input_block.device, dtype=torch.float16)
        draft_logits.scatter_(2, input_block[:,:,None], 1)
        return input_block, draft_logits

    def update_multiple(self, input_block, output_block, last_seq, **kwargs):
        Npad = input_block.size(1) - last_seq.shape[1]
        Ndraft= input_block.size(0) // last_seq.shape[0]
        repeated_last_seq = torch.repeat_interleave(last_seq, repeats=Ndraft, dim=0)
        input_block[:, :1] = repeated_last_seq
        output_block[:, :1] = repeated_last_seq

        bg = self.rankings[last_seq.squeeze(1), :Ndraft, :Npad]
        # print(input_block.shape, bg.shape, last_seq.shape)
        bg = bg.reshape(last_seq.shape[0] * Ndraft, Npad)
        input_block[:, 1:] = bg.to(input_block.device)
        return input_block

    def update_(self, input_block, output_block, last_word, **kwargs):
        Ndraft= input_block.size(0)
        Npad = input_block.size(1) - 1
        input_block[:, 0] = last_word
        output_block[:, 0] = last_word

        bg = self.rankings[last_word][:Ndraft, :Npad].reshape(Ndraft, Npad)
        input_block[:, 1:] = bg.to(input_block.device)
        if not hasattr(self, 'draft_ids'):
            self.draft_ids = ['ModelBigramModelExt' for _ in range(Ndraft)]
        return input_block

    def get_strat_keys_(self):
        return ['ModelBigramModelExt']

@torch.inference_mode
def forward_ngram_matcher(input_ids, id, Ndraft=1, N=2):
    # use unfold to obtain all N grams
    if input_ids.shape[1] < N:
        return None
    grams = input_ids.flatten().unfold(0, N, 1)
    # extract mask of matching ngrams
    mask = grams[:, 0] == id
    if torch.any(mask):
        matching_grams = grams[mask]
        # obtain counts of all ngrams
        matches, counts = torch.unique(matching_grams, dim=0, return_counts=True)
        Nfound = counts.size(0)
        Ntake = min(Ndraft, Nfound)
        # take up to top Ndraft occuring Ngrams
        most_freq_ids = counts.topk(Ntake).indices
        return matches[most_freq_ids]
    else:
        return None

class PNG_ModelBigramModelExt:
    '''
    Prompt lookup and pad everything else with unigram model
    '''
    def __init__(self, model_path, **kwargs):
        with torch.inference_mode():
            self.ext_bigram = torch.load(model_path)


    def update(self, input_block, output_block, last_seq, output_ids, **kwargs):

        input_block[:, :last_seq.shape[1]] = last_seq
        output_block[:, :last_seq.shape[1]] = last_seq
        Ndraft= input_block.size(0)
        Npad = input_block.size(1) - last_seq.shape[1]

        last_word = last_seq[0, -1]
        # obtain the matching grams
        matches = forward_ngram_matcher(output_ids, id=last_word, Ndraft=Ndraft, N=Npad+1)
        if matches is None:
            ng, rs = (0, Ndraft)
        else:
            ng = matches.size(0)
            # residual space left
            rs = Ndraft - ng
            input_block[:ng, last_seq.shape[1]:] = matches[:, 1:]
        if rs > 0:
            bi_ext = self.ext_bigram[last_word, :rs, :Npad].to(input_block.device)
            input_block[ng:, last_seq.shape[1]:]  = bi_ext

        # also return a logit
        draft_logits = -torch.inf * torch.ones((*input_block.shape, self.ext_bigram.shape[0]), device=input_block.device, dtype=torch.float16)
        draft_logits.scatter_(2, input_block[:,:,None], 1)
        input_mask = torch.ones_like(input_block)
        return input_block, input_mask, draft_logits
    
    def update_multiple(self, input_block, output_block, last_seq, output_ids, **kwargs):
        Npad = input_block.size(1) - last_seq.shape[1]
        Ndraft= input_block.size(0) // last_seq.shape[0]
        repeated_last_seq = torch.repeat_interleave(last_seq, repeats=Ndraft, dim=0)
        input_block[:, :1] = repeated_last_seq
        output_block[:, :1] = repeated_last_seq

        for i in range(last_seq.shape[0]):
            last_word = last_seq[i, -1]
            matches = forward_ngram_matcher(output_ids[i], id=last_word, Ndraft=Ndraft, N=Npad+1)
            if matches is None:
                ng, rs = (0, Ndraft)
            else:
                ng = matches.size(0)
                # residual space left
                rs = Ndraft - ng
                input_block[i*Ndraft:i*Ndraft+ng, last_seq.shape[1]:] = matches[:, 1:]
            if rs > 0:
                bi_ext = self.ext_bigram[last_word, :rs, :Npad].to(input_block.device)
                input_block[i*Ndraft+ng:(i+1)*Ndraft, last_seq.shape[1]:]  = bi_ext

        return input_block
    def update_(self, input_block, output_block, last_word, output_ids, **kwargs):

        input_block[:, 0] = last_word
        output_block[:, 0] = last_word
        Ndraft= input_block.size(0)
        Npad = input_block.size(1) - 1

        # obtain the matching grams
        matches = forward_ngram_matcher(output_ids, id=last_word, Ndraft=Ndraft, N=input_block.size(-1))
        draft_ids = []
        if matches is None:
            ng, rs = (0, Ndraft)
        else:
            ng = matches.size(0)
            rs = Ndraft - ng
            # residual space left
            input_block[:ng, :] = matches
            draft_ids += ['Prompt' for _ in range(ng)]
        if rs > 0:
            bi_ext = self.ext_bigram[last_word, :rs, :Npad].to(input_block.device)
            input_block[ng:, 1:]  = bi_ext
            draft_ids += ['ModelBigramModelExt' for _ in range(rs)]

        self.draft_ids = draft_ids
    
        return input_block
 

    def get_strat_keys_(self):
        return ['Prompt', 'ModelBigramModelExt']
    
class JacobiDecoding:
    '''
    Implementing Jacobi decoding: 
    If output block is None: 
        initialize input block randomly/or by Ngram
    else:
        check how many tokens (x) in last_seq. This is how many we get correct (at least 1)
        new_input <- take out[x-1:] and fill rest with random/Ngram 
        
    '''
    def __init__(self, model_path):
        with torch.inference_mode():
            self.rankings = torch.load(model_path)

    def update(self, input_block, output_block, last_seq, output_ids, jstar, last_output, **kwargs):
        # first iteration, we don't have a output yet, fill with bigram
        if jstar == -1:
            input_block[:, :last_seq.shape[1]] = last_seq
            output_block[:, :last_seq.shape[1]] = last_seq
            Npad = input_block.size(1) - last_seq.shape[1]
            last_word = last_seq[0, -1]

            bg = self.rankings[last_word][:1, :Npad].reshape(1, Npad)
            input_block[:, last_seq.shape[1]:] = bg.to(input_block.device)
        # we have some output, we want to use part of output as input
        else:
            input_block[:, :last_seq.shape[1]] = last_seq
            output_block[:, :last_seq.shape[1]] = last_seq
            input_block[:, last_seq.shape[1]:(last_seq.shape[1]+last_output.shape[1]-(jstar+2))] = last_output[:, jstar+2:]
            rs = input_block.shape[1] - (last_seq.shape[1]+last_output.shape[1]-(jstar+2))
            if rs > 0:
                bg = self.rankings[last_output[0, -1]][:1, :rs].reshape(1, rs)
                input_block[:, -rs:] = bg 
        # also return a logit
        draft_logits = -torch.inf * torch.ones((*input_block.shape, self.rankings.shape[0]), device=input_block.device, dtype=torch.float16)
        draft_logits.scatter_(2, input_block[:,:,None], 1)
        return input_block, draft_logits
    
class MambaModel:
    @torch.inference_mode()
    def __init__(self, model_name, Ndraft, Npad, max_length, **kwargs):
        self.model = MambaLMHeadModel.from_pretrained("state-spaces/mamba2-130m", device='cuda', dtype=torch.float16)
        self.generator = torch.Generator(device="cuda")
        # moved decoding cache 
        if not hasattr(self.model, "_decoding_cache"):
            self.model._decoding_cache_verified = None
            self.model._decoding_cache_draft = None
        self.model._decoding_cache_verified = update_graph_cache(
            self.model,
            self.model._decoding_cache_verified,
            1,
            Npad+1,
            max_length,
            ndraft=1,
            decoding_seqlens=list(range(1, Npad+2)),
            use_Nstep_kernel=True,
            use_2step_kernel=False,
            jit_state_copy=False,
            num_input_seq=1
        )
        self.model._decoding_cache_draft = update_graph_cache(
            self.model,
            self.model._decoding_cache_draft,
            Ndraft,
            1,
            max_length,
            ndraft=Ndraft,
            jit_state_copy=False
        )
        
        self.inference_params_verified = self.model._decoding_cache_verified.inference_params
        self.inference_params_draft = self.model._decoding_cache_draft.inference_params

    @torch.inference_mode()
    def initialize(self, prompt, Ndraft, Npad, max_length, **kwargs):
        self.inference_params_verified.reset(1, Ndraft)
        self.inference_params_draft.reset(1, Ndraft)
        self.generator.manual_seed(0) # for reproducibility 
        
        logits = self.get_logits(prompt, torch.ones_like(prompt), self.inference_params_verified)
        self.inference_params_verified.seqlen_offset += prompt.shape[1]
        self.inference_params_draft.seqlen_offset += prompt.shape[1]
        # self.state_cache = self.model.allocate_inference_cache(Ndraft, Npad+1, torch.float16)
        for l in self.inference_params_draft.key_value_memory_dict.keys():
            self.inference_params_draft.key_value_memory_dict[l][0].copy_(self.inference_params_verified.key_value_memory_dict[l][0][[0]]) 
            self.inference_params_draft.key_value_memory_dict[l][1].copy_(self.inference_params_verified.key_value_memory_dict[l][1][[0]])
        self.gen_hist = prompt.clone()

    @torch.inference_mode()
    def update(self, input_block, output_block, last_seq, output_ids, sampling_params, **kwargs):
        input_block[:, :last_seq.shape[1]] = last_seq
        output_block[:, :last_seq.shape[1]] = last_seq
        Ndraft= input_block.size(0) # TODO: right now assumes Ndraft==1, what if we want to draft more sequences?
        Npad = input_block.size(1) - last_seq.shape[1]
        
        # self.inference_params.use_Nstep_kernel = True
        # self.inference_params.use_2step_kernel = False
        last_verified = output_ids[:,self.gen_hist.shape[1]:]
        logit = self.model._decoding_cache_verified.run(
            last_verified, torch.zeros_like(last_verified), torch.ones_like(last_verified), self.inference_params_verified.seqlen_offset
        )
        logit = logit[:, -1:, :self.model.config.vocab_size]
        self.gen_hist = torch.cat([self.gen_hist, last_verified], dim=1)
        if sampling_params['repetition_penalty'] != 1.0:
            logit = logit.transpose(1,2)
            logit = modify_logit_for_repetition_penalty(logit, 
                                                            repeat(output_ids, 'a b -> a b r', r = logit.shape[-1]), 
                                                            sampling_params['repetition_penalty']).transpose(1,2)
        # want to sample muliplt (Ndraft) token here
        token = sample(repeat(logit.squeeze(1), 'b ... -> (b r) ...', r=Ndraft), top_k=sampling_params['top_k'], 
                top_p=sampling_params['top_p'], 
                min_p=sampling_params['min_p'], 
                temperature=sampling_params['temperature'], 
                generator=self.generator).unsqueeze(1)
        self.inference_params_verified.seqlen_offset += last_seq.shape[1]

        for l in self.inference_params_verified.key_value_memory_dict.keys():
            self.inference_params_draft.key_value_memory_dict[l][0].copy_(self.inference_params_verified.key_value_memory_dict[l][0][[0]]) 
            self.inference_params_draft.key_value_memory_dict[l][1].copy_(self.inference_params_verified.key_value_memory_dict[l][1][[0]])
        
        # self.inference_params.use_Nstep_kernel = False
        # self.inference_params.use_2step_kernel = False
        last_word = token
        
        out_logits = [repeat(logit, 'b ... -> (b r) ...', r=Ndraft)]
        out_token = [token]
        for i in range(Npad-1):
            logit = self.get_logits(last_word, torch.ones_like(last_word), self.inference_params_draft, draft_cache=True)
            logit = logit[:, :, :self.model.config.vocab_size]
            if sampling_params['repetition_penalty'] != 1.0:
                logit = logit.transpose(1,2)
                logit = modify_logit_for_repetition_penalty(logit, 
                                                                repeat(output_ids, 'a b -> a b r', r = logit.shape[-1]), 
                                                                sampling_params['repetition_penalty']).transpose(1,2)
            token = sample(logit.squeeze(1), top_k=sampling_params['top_k'], 
                        top_p=sampling_params['top_p'], 
                        min_p=sampling_params['min_p'], 
                        temperature=sampling_params['temperature'],
                        generator=self.generator).unsqueeze(1)

            out_logits.append(logit)
            out_token.append(token)
            last_word = token
            # print(logit.shape)

        # for l in self.inference_params.key_value_memory_dict.keys():
        #     self.inference_params.key_value_memory_dict[l][0].copy_(self.state_cache[l][0]) 
        #     self.inference_params.key_value_memory_dict[l][1].copy_(self.state_cache[l][1])

        # out = self.model.generate(
        #     input_ids=output_ids,
        #     max_length=Npad+output_ids.shape[1],
        #     mask=torch.ones_like(output_ids),
        #     cg=True,
        #     return_dict_in_generate=True,
        #     output_scores=False,
        #     enable_timing=False,
        #     temperature=0,
        #     top_k=1,
        # )
        drafted_ids = torch.cat(out_token, dim=1)
        out_logits = torch.cat(out_logits, dim=1)
        out_logits = torch.nn.functional.pad(out_logits, (0,0,last_seq.shape[1],0), value=0)
        input_block[:, last_seq.shape[1]:] = drafted_ids
        input_mask = torch.ones_like(input_block)

        return input_block, input_mask, out_logits

    def get_logits(self, input_ids, mask, inference_params, draft_cache=False):
        decoding = inference_params.seqlen_offset > 0
        if decoding:
            position_ids = torch.full(
                (input_ids.shape[0], 1),
                inference_params.seqlen_offset,
                dtype=torch.long,
                device=input_ids.device,
            )
            input_mask = torch.ones_like(input_ids)
        else:
            position_ids = repeat(torch.arange(input_ids.shape[1],device=input_ids.device), 'l -> b l', b=input_ids.shape[0])
            if mask is None:
                input_mask = torch.ones_like(input_ids)
            else:
                input_mask = mask
        if not decoding:
            logits = self.model(
                input_ids,
                position_ids=position_ids,
                inference_params=inference_params,
                mask=input_mask,
                num_last_tokens=1,
            ).logits
        else:
            if draft_cache:
                logits = self.model._decoding_cache_draft.run(
                    input_ids, position_ids, input_mask, inference_params.seqlen_offset
                )
            else:
                logits = self.model._decoding_cache_verified.run(
                    input_ids, position_ids, input_mask, inference_params.seqlen_offset
                )                
        return logits


class MambaBeamSearchModel:
    '''
    Performs beam search with Mamba2 model as a draft model, which requires it to be able to advance the state to where ever the 
    verifcation model ends. 
    Every update should return a tensor Num_beam x max_length input and a Num_beam x max_length x max_length mask, where each batch contains a tree that stems from previous input beam token
    Note the tree can be of different length which means that there needs to be padding tokens, or we can use the mask to ignore everything behind
    
    '''
    def __init__(self, model_name, Npad, base_num_beam, draft_num_beam, max_length, **kwargs):
        self.device = 'cuda'
        self.dtype = torch.float16
        self.model = MambaLMHeadModel.from_pretrained("state-spaces/mamba2-130m", device=self.device, dtype=self.dtype)
        if not hasattr(self.model, "_decoding_cache_base") or not hasattr(self.model, "_decoding_cache_draft"):
            self.model._decoding_cache_base = None
            self.model._decoding_cache_draft = None
        self.model._decoding_cache_base = update_graph_cache(
            self.model,
            self.model._decoding_cache_base,
            base_num_beam,
            Npad+1,
            max_length,
            ndraft=1,
            decoding_seqlens=list(range(1, Npad+2)),
            num_input_seq=base_num_beam,
            use_Nstep_kernel=True,
            use_2step_kernel=False,
            jit_state_copy=True
        )
        # The drafting CUDA graph can only perform single step update
        # The batchsize (number of states) correspond to number of beams for the draft model
        self.model._decoding_cache_draft = update_graph_cache(
            self.model,
            self.model._decoding_cache_draft,
            draft_num_beam,
            1,
            max_length,
            ndraft=1,
            jit_state_copy=True,
            num_input_seq=draft_num_beam
        )
        self.inference_params_base = self.model._decoding_cache_base.inference_params
        self.inference_params_draft = self.model._decoding_cache_draft.inference_params

    @torch.inference_mode()
    def initialize(self, prompt, Npad, max_length, draft_num_beam, base_num_beam, cg=True, **kwargs):
        self.cg = cg
        batch_size = prompt.shape[0]
        assert batch_size == 1, "Assuming batch size is 1 first."
        # The base CUDA graph can perform multi-step update
        # Used to advance correct state once we have the verified tokens from the base model
        # also used as a state cache that stores state for the already verified sequence 
        if self.cg: 
            self.draft_num_beam = draft_num_beam
            self.base_num_beam = base_num_beam
            self.npad = Npad
            self.inference_params_base.reset(1, batch_size)
            self.inference_params_draft.reset(1, batch_size)
        else:
            self.inference_params_base = InferenceParams(max_seqlen=max_length, max_batch_size=batch_size*base_num_beam)
            self.inference_params_base.ndraft = 1
            self.inference_params_base.num_input_seq = base_num_beam
            self.inference_params_base.use_Nstep_kernel = True
            self.inference_params_base.use_2step_kernel = False
            self.inference_params_base.use_tree_scan_kernel = False
            self.inference_params_base.jit_state_copy = True
            self.inference_params_base.key_value_memory_dict = self.model.allocate_inference_cache(batch_size=base_num_beam, max_seqlen=max_length, dtype=self.dtype)
            self.inference_params_base.verified_key_value_memory_dict = {"indices":torch.zeros((base_num_beam,), dtype=torch.long, device=self.device),
                                                    "mask":torch.zeros((base_num_beam, Npad*base_num_beam+1), dtype=torch.bool, device=self.device),
                                                    "state":self.model.allocate_inference_cache(batch_size=base_num_beam, max_seqlen=max_length, dtype=self.dtype)}
            self.inference_params_draft = InferenceParams(max_seqlen=max_length, max_batch_size=batch_size*draft_num_beam)
            self.inference_params_draft.ndraft = 1
            self.inference_params_draft.num_input_seq = draft_num_beam 
            self.inference_params_draft.jit_state_copy = True
            self.inference_params_draft.key_value_memory_dict = self.model.allocate_inference_cache(batch_size=draft_num_beam, max_seqlen=max_length, dtype=self.dtype)
            self.inference_params_draft.verified_key_value_memory_dict = {"indices":torch.zeros((draft_num_beam,), dtype=torch.long, device=self.device),
                                        "mask":torch.zeros((draft_num_beam, Npad*draft_num_beam+1), dtype=torch.bool, device=self.device),
                                        "state":self.model.allocate_inference_cache(batch_size=draft_num_beam, max_seqlen=max_length, dtype=self.dtype)}
            self.draft_num_beam = draft_num_beam
            self.base_num_beam = base_num_beam
            self.npad = Npad


        self.sequence_cat = prompt
        # this is the persistent probability that will only hold joint probability to the 
        self.verified_log_probability = torch.zeros((batch_size * base_num_beam, 1), device=prompt.device, dtype=torch.float32)
        # this is the probability used during auto-regressive drafting
        self.draft_log_probability = torch.zeros((batch_size * draft_num_beam, 1), device=prompt.device, dtype=torch.float32)
        self.vocab_size = self.model.config.vocab_size
        
        logits = self.get_logits(prompt, torch.ones_like(prompt), self.inference_params_base, draft_cache=False)
        logits = logits[:,:,:self.vocab_size]
        self.inference_params_base.seqlen_offset += prompt.shape[1]
        self.inference_params_draft.seqlen_offset += prompt.shape[1]
        curr_prob = torch.softmax(logits, dim=-1) # (batchsize, vocabsize)
        sampled_output = curr_prob.topk(draft_num_beam, dim=-1)
        sampled_prob = sampled_output.values
        sampled_tokens = sampled_output.indices

        self.verified_last_tokens_logits = repeat(logits, 'b l d-> (n b) l d', n=base_num_beam)
        # self.verified_probability *= sampled_prob.view((batch_size * draft_num_beam, 1))
        self.sequence_cat = repeat(self.sequence_cat, 'b l -> (n b) l', n=base_num_beam)
        # self.sequence_cat = torch.cat([self.sequence_cat, sampled_tokens.view(batch_size * num_beam, 1)], dim=1)

        # We don't need a state cache anymore since the inference_param_base stores the state cache
        # self.state_cache = self.model.allocate_inference_cache(num_beam, Npad+1, torch.float16)
        # Copy from inference_param_base to inference_param_draft
        for l in self.inference_params_draft.key_value_memory_dict.keys():
            self.inference_params_draft.key_value_memory_dict[l][0].copy_(self.inference_params_base.key_value_memory_dict[l][0][[0]]) 
            self.inference_params_draft.key_value_memory_dict[l][1].copy_(self.inference_params_base.key_value_memory_dict[l][1][[0]])

    @torch.inference_mode()
    def update(self, input_block, output_block, last_seq, output_ids, state_indices_base, **kwargs):
        '''
        state_indices: The index of original beams that the base model would take
        '''
        # input_block[:, :last_seq.shape[1]] = last_seq
        # output_block[:, :last_seq.shape[1]] = last_seq
        # Ndraft= input_block.size(0) # TODO: right now assumes Ndraft==1, what if we want to draft more sequences?
        # Npad = input_block.size(1) - last_seq.shape[1]
        self.inference_params_base.verified_key_value_memory_dict['indices'].copy_(state_indices_base)
        last_verified = output_ids[:,self.sequence_cat.shape[1]:]
        # Advancing the generated beams to the current position by selecting beams and appending
        self.sequence_cat = torch.cat([self.sequence_cat[state_indices_base, :], last_verified], dim=1)
        curr_sequence_cat = self.sequence_cat.clone()
        out_logits = torch.zeros((self.base_num_beam, 1, self.vocab_size), device=state_indices_base.device)

        if self.cg:
            logit = self.model._decoding_cache_base.run(
                last_verified, torch.zeros_like(last_verified), torch.ones_like(last_verified), self.inference_params_base.seqlen_offset
            )
            logit = logit[:, :, :self.vocab_size] # (base_num_beam, last_verified_length, vocabsize)
        else:
            logit = self.model.forward(
                input_ids=last_verified, 
                position_ids=torch.zeros_like(last_verified), 
                inference_params=self.inference_params_base,
                num_last_tokens=-1,
                mask=torch.ones_like(last_verified)
            ).logits[:, :, :self.vocab_size]

        # shifting the probability by 1 and adding the probability from last iteration
        last_verified_logits = torch.cat([self.verified_last_tokens_logits[state_indices_base], logit[:, :-1, :]], dim=1)
        last_verified_prob = torch.softmax(last_verified_logits, dim=-1) 
        # selecing the tokens generated by the verification model
        last_verified_prob = last_verified_prob.view((-1, self.vocab_size))
        last_verified_prob = last_verified_prob[torch.arange(last_verified_prob.shape[0]), last_verified.reshape((-1))].view(last_verified.shape)
        # cumulative product to compute the final conditional prob
        last_verified_prob = torch.sum(torch.log(last_verified_prob), dim=-1, keepdim=True)
        # updating the probability to be the joint prob after the last verified token
        self.verified_log_probability = (self.verified_log_probability[state_indices_base] + last_verified_prob)
        # Saving the output for the last token to next iteration
        out_logits[:,0,:] = self.verified_last_tokens_logits[:, 0, :]
        self.verified_last_tokens_logits = logit[:, -1:, :]
        
        # Getting the last probability 
        curr_log_prob = torch.log(torch.softmax(logit[:, -1, :], dim=-1))
        # Generating the next N beams  
        curr_log_prob = (curr_log_prob + self.verified_log_probability).view(1, self.base_num_beam * curr_log_prob.shape[-1]) # (1, num_beam * vocabsize)
        top_beam_output = curr_log_prob.topk(self.draft_num_beam, dim=-1) # (1, num_beam)
        state_indices = (top_beam_output.indices / self.vocab_size).to(torch.int).squeeze(0)
        self.draft_log_probability.copy_(top_beam_output.values.view(1 * self.draft_num_beam, 1))
        tokens = (top_beam_output.indices % self.vocab_size).view((1 * self.draft_num_beam, 1))
        self.inference_params_base.seqlen_offset += last_verified.shape[1]

        for l in self.inference_params_draft.key_value_memory_dict.keys():
            self.inference_params_draft.key_value_memory_dict[l][0].copy_(self.inference_params_base.key_value_memory_dict[l][0][state_indices,...]) 
            self.inference_params_draft.key_value_memory_dict[l][1].copy_(self.inference_params_base.key_value_memory_dict[l][1][state_indices,...])
        
        # persistent variables
        curr_sequence_cat = curr_sequence_cat[state_indices, :]
        curr_sequence_cat = torch.cat([curr_sequence_cat, tokens], dim=1)
        root_parent = state_indices.clone() # determines which root it starts from, value N: 0 < N < base_num_beam
        beam_parent = torch.zeros((self.draft_num_beam), device=state_indices.device) # 
        temp_parent = torch.zeros_like(beam_parent)
        out_tokens = torch.zeros((self.base_num_beam, 1), device=state_indices.device, dtype=torch.long)
        parent_index = -torch.ones((self.base_num_beam, 1), device=state_indices.device, dtype=torch.long)
        # roots of the new trees
        out_tokens[:,0] = last_verified[:,-1]

        # first iteration 
        values, counts = torch.unique(state_indices, dim=0, return_counts=True)
        # how much should the input batch be expanded by here 
        additional_length = torch.max(counts)

        new_tokens = -torch.ones((self.base_num_beam, additional_length), device=state_indices.device, dtype=torch.long)
        new_parents = -torch.ones((self.base_num_beam, additional_length), device=state_indices.device, dtype=torch.long)
        new_logits = -torch.ones((self.base_num_beam, additional_length, self.vocab_size), device=state_indices.device)
        acc = torch.zeros(self.base_num_beam, device=state_indices.device, dtype=torch.long)
        # figure out a more efficient way to do this maybe?
        for i in range(self.draft_num_beam):
            new_tokens[root_parent[i], acc[state_indices[i]]] = tokens[i, 0]
            new_logits[root_parent[i], acc[state_indices[i]], :] = logit[state_indices[i], -1, :]
            new_parents[root_parent[i], acc[state_indices[i]]] = beam_parent[i]
            temp_parent[i] = out_tokens.shape[1] + acc[state_indices[i]]
            acc[state_indices[i]] += 1
        
        beam_parent = temp_parent
        out_tokens = torch.cat([out_tokens, new_tokens], dim=1)
        out_logits = torch.cat([out_logits, new_logits], dim=1)
        parent_index = torch.cat([parent_index, new_parents], dim=1)
        # print("in drafting", parent_index, out_tokens, self.draft_log_probability,  torch.log(prob[:, -1, :])[state_indices, tokens])
        # Already updated so no need to change   
        self.inference_params_draft.verified_key_value_memory_dict['indices'].copy_(torch.arange(state_indices.shape[0], device=state_indices.device))
        # taking multiple steps.
        for i in range(self.npad-1):
            last_word = tokens
            logit = self.get_logits(last_word, torch.ones_like(last_word), self.inference_params_draft, draft_cache=True)
            logit = logit[:, -1, :self.model.config.vocab_size]   
            curr_prob = torch.softmax(logit, dim=-1)
            curr_log_prob = torch.log(curr_prob)

            curr_log_prob = (curr_log_prob + self.draft_log_probability).view(1, self.draft_num_beam * curr_log_prob.shape[-1]) # (1, num_beam * vocabsize)
            top_beam_output = curr_log_prob.topk(self.draft_num_beam, dim=-1) # (batchsize, num_beam)
            state_indices = (top_beam_output.indices / self.vocab_size).to(torch.int).squeeze(0)
            self.draft_log_probability.copy_(top_beam_output.values.view(1 * self.draft_num_beam, 1))

            curr_sequence_cat = curr_sequence_cat[state_indices, :]
            beam_parent = beam_parent[state_indices]
            root_parent = root_parent[state_indices]

            tokens = (top_beam_output.indices % self.vocab_size).view((1 * self.draft_num_beam, 1))
            curr_sequence_cat = torch.cat([curr_sequence_cat, tokens], dim=1)

            # first iteration 
            values, counts = torch.unique(root_parent, dim=0, return_counts=True)
            # how much should the input batch be expanded by here 
            additional_length = torch.max(counts)
            temp_parent = torch.zeros_like(beam_parent)
            new_tokens = -torch.ones((self.base_num_beam, additional_length), device=state_indices.device, dtype=torch.long)
            new_parents = -torch.ones((self.base_num_beam, additional_length), device=state_indices.device, dtype=torch.long)
            new_logits = -torch.ones((self.base_num_beam, additional_length, self.vocab_size), device=state_indices.device)
            acc = torch.zeros(self.base_num_beam, device=state_indices.device, dtype=torch.long)
            # TODO: figure out a more efficient way to do this maybe?
            for i in range(self.draft_num_beam):
                new_tokens[root_parent[i], acc[root_parent[i]]] = tokens[i, 0]
                new_logits[root_parent[i], acc[root_parent[i]], :] = logit[state_indices[i], :]
                new_parents[root_parent[i], acc[root_parent[i]]] = beam_parent[i]
                temp_parent[i] = out_tokens.shape[1] + acc[root_parent[i]]
                acc[root_parent[i]] += 1
            
            beam_parent = temp_parent
            out_tokens = torch.cat([out_tokens, new_tokens], dim=1)
            out_logits = torch.cat([out_logits, new_logits], dim=1)
            parent_index = torch.cat([parent_index, new_parents], dim=1)
            # Update the states
            self.inference_params_draft.verified_key_value_memory_dict['indices'].copy_(state_indices)
            # print("in drafting", parent_index, out_tokens, self.draft_log_probability, torch.log(curr_prob)[state_indices, tokens])
        
        # We don't need this as the state will be restored in the next update call
        # for l in self.inference_params.key_value_memory_dict.keys():
        #     self.inference_params.key_value_memory_dict[l][0].copy_(self.state_cache[l][0]) 
        #     self.inference_params.key_value_memory_dict[l][1].copy_(self.state_cache[l][1])

        # out = self.model.generate(
        #     input_ids=output_ids,
        #     max_length=Npad+output_ids.shape[1],
        #     mask=torch.ones_like(output_ids),
        #     cg=True,
        #     return_dict_in_generate=True,
        #     output_scores=False,
        #     enable_timing=False,
        #     temperature=0,
        #     top_k=1,
        # )
        # construct a mask based on parent index
        out_mask = repeat(torch.eye(out_tokens.shape[1], device=state_indices.device), 'n m -> b n m', b=out_tokens.shape[0]).clone()
        for i in range(parent_index.shape[0]):
            for j in range(parent_index.shape[1]):
                if parent_index[i, j] == -1 and j != 0:
                    out_mask[i, j] = 0
                elif j != 0:
                    out_mask[i, j] = out_mask[i, parent_index[i, j]] + out_mask[i, j]

        return out_tokens, out_mask, out_logits

    def get_logits(self, input_ids, mask, inference_params, draft_cache=False):
        decoding = inference_params.seqlen_offset > 0 and self.cg
        if decoding:
            position_ids = torch.full(
                (input_ids.shape[0], 1),
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
        if not decoding:
            logits = self.model(
                input_ids,
                position_ids=position_ids,
                inference_params=inference_params,
                mask=input_mask,
                num_last_tokens=1,
            ).logits
        else:
            if draft_cache:
                logits = self.model._decoding_cache_draft.run(
                    input_ids, position_ids, input_mask, inference_params.seqlen_offset
                )
            else:
                logits = self.model._decoding_cache_base.run(
                    input_ids, position_ids, input_mask, inference_params.seqlen_offset
                )
        return logits


class LlamaModel:
    @torch.inference_mode()
    def __init__(self, model_name, Ndraft, Npad, max_length, **kwargs):
        self.model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B", attn_implementation='sdpa', torch_dtype=torch.float16).to('cuda')
        self.config = AutoConfig.from_pretrained("meta-llama/Llama-3.2-1B")
        self.generator = torch.Generator(device="cuda")
        self.kv_cache = StaticCache(config=self.config, max_batch_size=1, max_cache_len=max_length, device="cuda", dtype=torch.float16)
        self.max_length = max_length
        self.seqlen_offset_pt = torch.zeros(1, dtype=torch.long, device="cuda")
        self.verified_seqlen_offset_pt = torch.zeros(1, dtype=torch.long, device="cuda")
        self.callables = {}
        # moved decoding cache 
        # if not hasattr(self.model, "_decoding_cache"):
        #     self.model._decoding_cache_verified = None
        #     self.model._decoding_cache_draft = None
        self.model._decoding_cache = self.update_graph_cache(
            batch_size=1, 
            decoding_seqlens=list(range(1, Npad+2)))
        # self.model._decoding_cache_draft = update_graph_cache(
        #     self.model,
        #     self.model._decoding_cache_draft,
        #     Ndraft,
        #     1,
        #     max_length,
        #     ndraft=Ndraft,
        #     jit_state_copy=False
        # )
        
        # self.inference_params_verified = self.model._decoding_cache_verified.inference_params
        # self.inference_params_draft = self.model._decoding_cache_draft.inference_params

    def update_graph_cache(self, batch_size, decoding_seqlens):
        for decoding_seqlen in decoding_seqlens:
            if (batch_size, decoding_seqlen) not in self.callables:
                self.callables[batch_size, decoding_seqlen] = capture_graph_huggingface(
                    self.model,
                    self.kv_cache, 
                    batch_size=batch_size, 
                    max_seqlen=self.max_length, 
                    decoding_seqlen=decoding_seqlen)
    
    def decoding_cache_run(self, input_ids, position_ids, mask, seqlen_offset_pt):
        batch_size, decoding_seqlen = input_ids.shape
        return self.callables[batch_size, decoding_seqlen](input_ids, position_ids, mask, seqlen_offset_pt)

    @torch.inference_mode()
    def initialize(self, prompt, Ndraft, Npad, max_length, **kwargs):
        # self.inference_params_verified.reset(1, Ndraft)
        # self.inference_params_draft.reset(1, Ndraft)
        for i, kcache in enumerate(self.kv_cache.key_cache):
            self.kv_cache.key_cache[i].zero_()
            self.kv_cache.value_cache[i].zero_()
        self.seqlen_offset_pt.copy_(0)
        self.verified_seqlen_offset_pt.copy_(0)
        self.generator.manual_seed(0) # for reproducibility 
        
        # position_ids and mask are handled by default
        logits = self.get_logits(prompt, position_ids=None, mask=None, seqlen_offset_pt=self.seqlen_offset_pt)
        self.seqlen_offset_pt += prompt.shape[1]
        self.verified_seqlen_offset_pt += prompt.shape[1]
        # self.state_cache = self.model.allocate_inference_cache(Ndraft, Npad+1, torch.float16)
        # for l in self.inference_params_draft.key_value_memory_dict.keys():
        #     self.inference_params_draft.key_value_memory_dict[l][0].copy_(self.inference_params_verified.key_value_memory_dict[l][0][[0]]) 
        #     self.inference_params_draft.key_value_memory_dict[l][1].copy_(self.inference_params_verified.key_value_memory_dict[l][1][[0]])
        self.gen_hist = prompt.clone()

    @torch.inference_mode()
    def update(self, input_block, output_block, last_seq, output_ids, sampling_params, **kwargs):
        input_block[:, :last_seq.shape[1]] = last_seq
        output_block[:, :last_seq.shape[1]] = last_seq
        Ndraft= input_block.size(0) # TODO: right now assumes Ndraft==1, what if we want to draft more sequences?
        Npad = input_block.size(1) - last_seq.shape[1]
        
        # self.inference_params.use_Nstep_kernel = True
        # self.inference_params.use_2step_kernel = False
        last_verified = output_ids[:,self.gen_hist.shape[1]:]
        logit = self.get_logits(
            last_verified, position_ids=None, mask=None, seqlen_offset_pt=self.verified_seqlen_offset_pt
        )
        logit = logit[:, -1:, :self.model.config.vocab_size]
        self.gen_hist = torch.cat([self.gen_hist, last_verified], dim=1)
        self.verified_seqlen_offset_pt += last_verified.shape[1]
        if sampling_params['repetition_penalty'] != 1.0:
            logit = logit.transpose(1,2)
            logit = modify_logit_for_repetition_penalty(logit, 
                                                            repeat(output_ids, 'a b -> a b r', r = logit.shape[-1]), 
                                                            sampling_params['repetition_penalty']).transpose(1,2)
        # want to sample muliplt (Ndraft) token here
        token = sample(repeat(logit.squeeze(1), 'b ... -> (b r) ...', r=Ndraft), top_k=sampling_params['top_k'], 
                top_p=sampling_params['top_p'], 
                min_p=sampling_params['min_p'], 
                temperature=sampling_params['temperature'], 
                generator=self.generator).unsqueeze(1)
        self.seqlen_offset_pt.copy_(self.verified_seqlen_offset_pt)
        # self.inference_params_verified.seqlen_offset += last_seq.shape[1]

        # for l in self.inference_params_verified.key_value_memory_dict.keys():
        #     self.inference_params_draft.key_value_memory_dict[l][0].copy_(self.inference_params_verified.key_value_memory_dict[l][0][[0]]) 
        #     self.inference_params_draft.key_value_memory_dict[l][1].copy_(self.inference_params_verified.key_value_memory_dict[l][1][[0]])
        
        # self.inference_params.use_Nstep_kernel = False
        # self.inference_params.use_2step_kernel = False
        last_word = token
        
        out_logits = [repeat(logit, 'b ... -> (b r) ...', r=Ndraft)]
        out_token = [token]
        for i in range(Npad-1):
            logit = self.get_logits(last_word, position_ids=None, mask=None, seqlen_offset_pt=self.seqlen_offset_pt)
            logit = logit[:, :, :self.model.config.vocab_size]
            if sampling_params['repetition_penalty'] != 1.0:
                logit = logit.transpose(1,2)
                logit = modify_logit_for_repetition_penalty(logit, 
                                                                repeat(output_ids, 'a b -> a b r', r = logit.shape[-1]), 
                                                                sampling_params['repetition_penalty']).transpose(1,2)
            token = sample(logit.squeeze(1), top_k=sampling_params['top_k'], 
                        top_p=sampling_params['top_p'], 
                        min_p=sampling_params['min_p'], 
                        temperature=sampling_params['temperature'],
                        generator=self.generator).unsqueeze(1)

            out_logits.append(logit)
            out_token.append(token)
            last_word = token
            self.seqlen_offset_pt += token.shape[1]
            # print(logit.shape)

        # for l in self.inference_params.key_value_memory_dict.keys():
        #     self.inference_params.key_value_memory_dict[l][0].copy_(self.state_cache[l][0]) 
        #     self.inference_params.key_value_memory_dict[l][1].copy_(self.state_cache[l][1])

        # out = self.model.generate(
        #     input_ids=output_ids,
        #     max_length=Npad+output_ids.shape[1],
        #     mask=torch.ones_like(output_ids),
        #     cg=True,
        #     return_dict_in_generate=True,
        #     output_scores=False,
        #     enable_timing=False,
        #     temperature=0,
        #     top_k=1,
        # )
        drafted_ids = torch.cat(out_token, dim=1)
        out_logits = torch.cat(out_logits, dim=1)
        out_logits = torch.nn.functional.pad(out_logits, (0,0,last_seq.shape[1],0), value=0)
        input_block[:, last_seq.shape[1]:] = drafted_ids
        input_mask = torch.ones_like(input_block)

        return input_block, input_mask, out_logits

    def get_logits(self, input_ids, position_ids, mask, seqlen_offset_pt):
        decoding = seqlen_offset_pt > 0
        batch_size, decoding_seqlen = input_ids.shape
        if position_ids is None:
            position_ids = (seqlen_offset_pt + torch.arange(input_ids.shape[1], device=input_ids.device)).unsqueeze(0)
            # position_ids = torch.full(
            #     (input_ids.shape[0], 1),
            #     inference_params.seqlen_offset,
            #     dtype=torch.long,
            #     device=input_ids.device,
            # )
            # input_mask = torch.ones_like(input_ids)
        else:
            position_ids = None

        if mask is None:
            input_mask = torch.ones((batch_size,1,decoding_seqlen,self.max_length), dtype=torch.float16, device='cuda').tril(seqlen_offset_pt.item()-1)
        else:
            input_mask = mask
        if not decoding:
            # prefilling
            logits = self.model(
                input_ids=input_ids,
                attention_mask=input_mask,
                use_cache=True,
                return_dict=True,
                past_key_values=self.kv_cache,
                cache_position=seqlen_offset_pt+torch.arange(input_ids.shape[1], device='cuda')
            ).logits
        else:
            # 
            logits = self.decoding_cache_run(
                input_ids, position_ids, input_mask, seqlen_offset_pt
            )
        return logits

class MambaStaticTreeModel:
    '''
    Build a token tree with a predefined structure with Mamba2 model as a draft model, which requires it to be able to advance the state to where ever the 
    verifcation model ends. 

    '''
    def __init__(self, model_name, Npad, base_num_beam, draft_num_beam, max_length, **kwargs):
        self.device = 'cuda'
        self.dtype = torch.float16
        self.model = MambaLMHeadModel.from_pretrained("state-spaces/mamba2-130m", device=self.device, dtype=self.dtype)
        self.generator = torch.Generator(device="cuda")
        assert base_num_beam==1, "allowing only 1 base beams only for now" #TODO
        if not hasattr(self.model, "_decoding_cache_base") or not hasattr(self.model, "_decoding_cache_draft"):
            self.model._decoding_cache_base = None
            self.model._decoding_cache_draft = None
        self.model._decoding_cache_base = update_graph_cache(
            self.model,
            self.model._decoding_cache_base,
            base_num_beam,
            Npad+1,
            max_length,
            ndraft=1,
            decoding_seqlens=list(range(1, Npad+2)),
            num_input_seq=base_num_beam,
            use_Nstep_kernel=True,
            use_2step_kernel=False,
            jit_state_copy=True
        )
        # The drafting CUDA graph can only perform single step update
        # The batchsize (number of states) correspond to number of beams for the draft model
        self.model._decoding_cache_draft = update_graph_cache(
            self.model,
            self.model._decoding_cache_draft,
            draft_num_beam,
            1,
            max_length,
            ndraft=1,
            jit_state_copy=True,
            num_input_seq=draft_num_beam
        )
        self.inference_params_base = self.model._decoding_cache_base.inference_params
        self.inference_params_draft = self.model._decoding_cache_draft.inference_params
        self.static_state_indices = [[0,0,0,0,0,-1,-1,-1],
                                     [0,0,1,2,-1,-1,-1,-1],
                                     [0,0,1,2,-1,-1,-1,-1],
                                     [0,0,1,1,-1,-1,-1,-1]]
        self.static_state_indices = torch.tensor(self.static_state_indices, device="cuda", dtype=torch.int)
        self.static_branch = [[5,0,0,0,0,0,0,0],
                              [2,1,1,0,0,0,0,0],
                              [2,1,1,0,0,0,0,0],
                              [2,2,0,0,0,0,0,0]]
        self.static_branch = torch.tensor(self.static_branch, device="cuda", dtype=torch.int)
        # checking if static tree can be generated 
        assert Npad == self.static_state_indices.shape[0] and Npad == self.static_branch.shape[0]
        assert draft_num_beam == self.static_state_indices.shape[1]
        assert (draft_num_beam >= torch.sum(self.static_branch, dim=1)).all()
        # deriving attention mask 
        total_len = torch.sum(self.static_state_indices!=-1)
        self.attention_mask = torch.zeros((1, total_len+1, total_len+1), device="cuda", dtype=torch.int)
        self.attention_mask[0,0,0] = 1
        token_count = 1
        index_map = [0]
        for i in range(Npad):
            for j in range(draft_num_beam):
                if self.static_state_indices[i, j] == -1:
                    index_map.append(-1)
                    continue
                if i == 0: 
                    parent_index = 0
                    self.attention_mask[:, token_count, :] = self.attention_mask[:, parent_index, :]
                    self.attention_mask[:, token_count, token_count] = 1
                    index_map.append(token_count)
                    token_count += 1
                else:
                    parent_index = index_map[1 + (i-1) * draft_num_beam + self.static_state_indices[i, j]]
                    self.attention_mask[:, token_count, :] = self.attention_mask[:, parent_index, :]
                    self.attention_mask[:, token_count, token_count] = 1
                    index_map.append(token_count)
                    token_count += 1
        assert token_count == total_len + 1 
        self.total_len = total_len.item()
            


    @torch.inference_mode()
    def initialize(self, prompt, Npad, max_length, draft_num_beam, base_num_beam, cg=True, **kwargs):
        self.cg = cg
        batch_size = prompt.shape[0]
        self.generator.manual_seed(42)
        assert batch_size == 1, "Assuming batch size is 1 first."
        # The base CUDA graph can perform multi-step update
        # Used to advance correct state once we have the verified tokens from the base model
        # also used as a state cache that stores state for the already verified sequence 
        if self.cg: 
            self.draft_num_beam = draft_num_beam
            self.base_num_beam = base_num_beam
            self.npad = Npad
            self.inference_params_base.reset(1, batch_size)
            self.inference_params_draft.reset(1, batch_size)
        else:
            self.inference_params_base = InferenceParams(max_seqlen=max_length, max_batch_size=batch_size*base_num_beam)
            self.inference_params_base.ndraft = 1
            self.inference_params_base.num_input_seq = base_num_beam
            self.inference_params_base.use_Nstep_kernel = True
            self.inference_params_base.use_2step_kernel = False
            self.inference_params_base.use_tree_scan_kernel = False
            self.inference_params_base.jit_state_copy = True
            self.inference_params_base.key_value_memory_dict = self.model.allocate_inference_cache(batch_size=base_num_beam, max_seqlen=max_length, dtype=self.dtype)
            self.inference_params_base.verified_key_value_memory_dict = {"indices":torch.zeros((base_num_beam,), dtype=torch.long, device=self.device),
                                                    "mask":torch.zeros((base_num_beam, Npad*base_num_beam+1), dtype=torch.bool, device=self.device),
                                                    "state":self.model.allocate_inference_cache(batch_size=base_num_beam, max_seqlen=max_length, dtype=self.dtype)}
            self.inference_params_draft = InferenceParams(max_seqlen=max_length, max_batch_size=batch_size*draft_num_beam)
            self.inference_params_draft.ndraft = 1
            self.inference_params_draft.num_input_seq = draft_num_beam 
            self.inference_params_draft.jit_state_copy = True
            self.inference_params_draft.key_value_memory_dict = self.model.allocate_inference_cache(batch_size=draft_num_beam, max_seqlen=max_length, dtype=self.dtype)
            self.inference_params_draft.verified_key_value_memory_dict = {"indices":torch.zeros((draft_num_beam,), dtype=torch.long, device=self.device),
                                        "mask":torch.zeros((draft_num_beam, Npad*draft_num_beam+1), dtype=torch.bool, device=self.device),
                                        "state":self.model.allocate_inference_cache(batch_size=draft_num_beam, max_seqlen=max_length, dtype=self.dtype)}
            self.draft_num_beam = draft_num_beam
            self.base_num_beam = base_num_beam
            self.npad = Npad


        self.sequence_cat = prompt
        # this is the persistent probability that will only hold joint probability to the 
        self.verified_log_probability = torch.zeros((batch_size * base_num_beam, 1), device=prompt.device, dtype=torch.float32)
        # this is the probability used during auto-regressive drafting
        self.draft_log_probability = torch.zeros((batch_size * draft_num_beam, 1), device=prompt.device, dtype=torch.float32)
        self.vocab_size = self.model.config.vocab_size
        
        logits = self.get_logits(prompt, torch.ones_like(prompt), self.inference_params_base, draft_cache=False)
        logits = logits[:,:,:self.vocab_size]
        self.inference_params_base.seqlen_offset += prompt.shape[1]
        self.inference_params_draft.seqlen_offset += prompt.shape[1]
        curr_prob = torch.softmax(logits, dim=-1) # (batchsize, vocabsize)
        sampled_output = curr_prob.topk(draft_num_beam, dim=-1)
        sampled_prob = sampled_output.values
        sampled_tokens = sampled_output.indices

        self.verified_last_tokens_logits = repeat(logits, 'b l d-> (n b) l d', n=base_num_beam)
        # self.verified_probability *= sampled_prob.view((batch_size * draft_num_beam, 1))
        self.sequence_cat = repeat(self.sequence_cat, 'b l -> (n b) l', n=base_num_beam)
        # self.sequence_cat = torch.cat([self.sequence_cat, sampled_tokens.view(batch_size * num_beam, 1)], dim=1)

        # We don't need a state cache anymore since the inference_param_base stores the state cache
        # self.state_cache = self.model.allocate_inference_cache(num_beam, Npad+1, torch.float16)
        # Copy from inference_param_base to inference_param_draft
        for l in self.inference_params_draft.key_value_memory_dict.keys():
            self.inference_params_draft.key_value_memory_dict[l][0].copy_(self.inference_params_base.key_value_memory_dict[l][0][[0]]) 
            self.inference_params_draft.key_value_memory_dict[l][1].copy_(self.inference_params_base.key_value_memory_dict[l][1][[0]])

    @torch.inference_mode()
    def update(self, input_block, output_block, last_seq, output_ids, state_indices_base, sampling_params, **kwargs):
        '''
        state_indices: The index of original beams that the base model would take
        '''
        # input_block[:, :last_seq.shape[1]] = last_seq
        # output_block[:, :last_seq.shape[1]] = last_seq
        # Ndraft= input_block.size(0) # TODO: right now assumes Ndraft==1, what if we want to draft more sequences?
        # Npad = input_block.size(1) - last_seq.shape[1]
        # self.inference_params_base.verified_key_value_memory_dict['indices'].copy_(state_indices_base)
        last_verified = output_ids[:,self.sequence_cat.shape[1]:]
        # Advancing the generated beams to the current position by selecting beams and appending
        self.sequence_cat = torch.cat([self.sequence_cat, last_verified], dim=1)
        curr_sequence_cat = self.sequence_cat.clone()
        out_logits = torch.zeros((self.base_num_beam, self.attention_mask.shape[1], self.vocab_size), device=state_indices_base.device)
        out_len = 0

        if self.cg:
            logit = self.model._decoding_cache_base.run(
                last_verified, torch.zeros_like(last_verified), torch.ones_like(last_verified), self.inference_params_base.seqlen_offset
            )
            logit = logit[:, :, :self.vocab_size] # (base_num_beam, last_verified_length, vocabsize)
        else:
            logit = self.model.forward(
                input_ids=last_verified, 
                position_ids=torch.zeros_like(last_verified), 
                inference_params=self.inference_params_base,
                num_last_tokens=-1,
                mask=torch.ones_like(last_verified)
            ).logits[:, :, :self.vocab_size]

        # shifting the probability by 1 and adding the probability from last iteration
        last_verified_logits = torch.cat([self.verified_last_tokens_logits, logit[:, :-1, :]], dim=1)
        # last_verified_prob = torch.softmax(last_verified_logits, dim=-1) 
        # selecing the tokens generated by the verification model
        # last_verified_prob = last_verified_prob.view((-1, self.vocab_size))
        # last_verified_prob = last_verified_prob[torch.arange(last_verified_prob.shape[0]), last_verified.reshape((-1))].view(last_verified.shape)
        # cumulative product to compute the final conditional prob
        # last_verified_prob = torch.sum(torch.log(last_verified_prob), dim=-1, keepdim=True)
        # updating the probability to be the joint prob after the last verified token
        # self.verified_log_probability = (self.verified_log_probability[state_indices_base] + last_verified_prob)
        # Saving the output for the last token to next iteration
        out_logits[:,0,:] = last_verified_logits[:, -1, :]
        self.verified_last_tokens_logits = logit[:, -1:, :]
        
        # Getting the last probability 
        # curr_log_prob = torch.log(torch.softmax(logit[:, -1, :], dim=-1))
        # Generating the next N beams  
        # curr_log_prob = (curr_log_prob + self.verified_log_probability).view(1, self.base_num_beam * curr_log_prob.shape[-1]) # (1, num_beam * vocabsize)
        tokens = sample(logit[:, -1, :], num_samples=self.static_branch[0,0].item(), 
                        top_k=sampling_params['top_k'], # using greedy 
                        top_p=sampling_params['top_p'], 
                        min_p=sampling_params['min_p'], 
                        temperature=sampling_params['temperature'],
                        generator=self.generator)
        # top_beam_output = curr_log_prob.topk(self.draft_num_beam, dim=-1) # (1, num_beam)
        # state_indices = (top_beam_output.indices / self.vocab_size).to(torch.int).squeeze(0)
        # self.draft_log_probability.copy_(top_beam_output.values.view(1 * self.draft_num_beam, 1))
        # tokens = (top_beam_output.indices % self.vocab_size).view((1 * self.draft_num_beam, 1))
        # reordering tokens to keep the higher probility ones at top
        if sampling_params['top_k'] != 1:
            token_logit = logit[:, -1, tokens.squeeze()]
            order = torch.argsort(token_logit.squeeze(), dim=-1, descending=True)
            tokens = tokens[:, order]
        self.inference_params_base.seqlen_offset += last_verified.shape[1]
        state_indices = torch.zeros((self.draft_num_beam), dtype=torch.long, device="cuda")
        for l in self.inference_params_draft.key_value_memory_dict.keys():
            self.inference_params_draft.key_value_memory_dict[l][0].copy_(self.inference_params_base.key_value_memory_dict[l][0]) 
            self.inference_params_draft.key_value_memory_dict[l][1].copy_(self.inference_params_base.key_value_memory_dict[l][1])
        
        # persistent variables
        # curr_sequence_cat = curr_sequence_cat[state_indices, :]
        # curr_sequence_cat = torch.cat([curr_sequence_cat, tokens], dim=1)
        # root_parent = state_indices.clone() # determines which root it starts from, value N: 0 < N < base_num_beam
        # beam_parent = torch.zeros((self.draft_num_beam), device=state_indices.device) # 
        # temp_parent = torch.zeros_like(beam_parent)
        # parent_index = -torch.ones((self.base_num_beam, 1), device=state_indices.device, dtype=torch.long)
        out_tokens = torch.zeros((self.base_num_beam, self.attention_mask.shape[1]), device=logit.device, dtype=torch.long)
        # roots of the new trees
        out_tokens[:,0] = last_verified[:,-1]
        out_tokens[:,1:(1+tokens.shape[1])] = tokens.to(torch.long)
        out_logits[:,1:(1+tokens.shape[1]), :] = logit[:, [-1], :]
        out_len += (1+tokens.shape[1])

        # first iteration 
        # values, counts = torch.unique(state_indices, dim=0, return_counts=True)
        # how much should the input batch be expanded by here 
        # additional_length = torch.max(counts)

        # new_tokens = -torch.ones((self.base_num_beam, additional_length), device=state_indices.device, dtype=torch.long)
        # new_parents = -torch.ones((self.base_num_beam, additional_length), device=state_indices.device, dtype=torch.long)
        # new_logits = -torch.ones((self.base_num_beam, additional_length, self.vocab_size), device=state_indices.device)
        # acc = torch.zeros(self.base_num_beam, device=state_indices.device, dtype=torch.long)
        # figure out a more efficient way to do this maybe?
        # for i in range(self.draft_num_beam):
        #     new_tokens[root_parent[i], acc[state_indices[i]]] = tokens[i, 0]
        #     new_logits[root_parent[i], acc[state_indices[i]], :] = logit[state_indices[i], -1, :]
        #     new_parents[root_parent[i], acc[state_indices[i]]] = beam_parent[i]
        #     temp_parent[i] = out_tokens.shape[1] + acc[state_indices[i]]
        #     acc[state_indices[i]] += 1
        
        # beam_parent = temp_parent
        # out_tokens = torch.cat([out_tokens, new_tokens], dim=1)
        # out_logits = torch.cat([out_logits, new_logits], dim=1)
        # parent_index = torch.cat([parent_index, new_parents], dim=1)
        # print("in drafting", parent_index, out_tokens, self.draft_log_probability,  torch.log(prob[:, -1, :])[state_indices, tokens])
        # Already updated so no need to change   
        # taking multiple steps.
        for i in range(1, self.npad):
            # Update the states
            valid_count = torch.sum(self.static_state_indices[i-1]!=-1)
            state_indices = torch.where(self.static_state_indices[i-1]!=-1, self.static_state_indices[i-1], 0)
            self.inference_params_draft.verified_key_value_memory_dict['indices'].copy_(state_indices)
            last_word = torch.zeros((self.draft_num_beam, 1), device=state_indices.device, dtype=torch.long)
            # print(valid_count, tokens)
            last_word[:valid_count, 0] = tokens[0,:] # if number does match errors will be thrown
            logit = self.get_logits(last_word, torch.ones_like(last_word), self.inference_params_draft, draft_cache=True)
            logit = logit[:, -1, :self.model.config.vocab_size]   

            # curr_prob = torch.softmax(logit, dim=-1)
            # curr_log_prob = torch.log(curr_prob)

            # curr_log_prob = (curr_log_prob + self.draft_log_probability).view(1, self.draft_num_beam * curr_log_prob.shape[-1]) # (1, num_beam * vocabsize)
            # top_beam_output = curr_log_prob.topk(self.draft_num_beam, dim=-1) # (batchsize, num_beam)
            # state_indices = (top_beam_output.indices / self.vocab_size).to(torch.int).squeeze(0)
            # self.draft_log_probability.copy_(top_beam_output.values.view(1 * self.draft_num_beam, 1))

            # curr_sequence_cat = curr_sequence_cat[state_indices, :]
            # beam_parent = beam_parent[state_indices]
            # root_parent = root_parent[state_indices]
            tokens = torch.zeros((self.base_num_beam, self.draft_num_beam), device=logit.device)
            new_logits = torch.zeros((self.base_num_beam, self.draft_num_beam, self.vocab_size), device=logit.device)
            acc = 0
            for j in range(self.draft_num_beam):
                if self.static_branch[i,j] == 0:
                    continue
                token = sample(logit[[j], :], num_samples=self.static_branch[i,j],
                        top_k=sampling_params['top_k'], # using greedy or not
                        top_p=sampling_params['top_p'], 
                        min_p=sampling_params['min_p'], 
                        temperature=sampling_params['temperature'],
                        generator=self.generator)
                if len(token.shape) == 1:
                    token = token.unsqueeze(0)
                if sampling_params['top_k'] != 1:
                    token_logit = logit[[j], token.squeeze()]
                    order = torch.argsort(token_logit, dim=-1, descending=True)
                    token = token[:, order]
                tokens[:, acc:acc+token.shape[1]] = token
                new_logits[0, acc:acc+token.shape[1], :] = logit[[j], :]
                acc += token.shape[1]
            tokens = tokens[:, :acc]
            new_logits = new_logits[:, :acc, :]
            # tokens = (top_beam_output.indices % self.vocab_size).view((1 * self.draft_num_beam, 1))
            # curr_sequence_cat = torch.cat([curr_sequence_cat, tokens], dim=1)

            # first iteration 
            # values, counts = torch.unique(root_parent, dim=0, return_counts=True)
            # how much should the input batch be expanded by here 
            # additional_length = torch.max(counts)
            # temp_parent = torch.zeros_like(beam_parent)
            # new_tokens = -torch.ones((self.base_num_beam, additional_length), device=state_indices.device, dtype=torch.long)
            # new_parents = -torch.ones((self.base_num_beam, additional_length), device=state_indices.device, dtype=torch.long)
            # new_logits = -torch.ones((self.base_num_beam, additional_length, self.vocab_size), device=state_indices.device)
            # acc = torch.zeros(self.base_num_beam, device=state_indices.device, dtype=torch.long)
            # TODO: figure out a more efficient way to do this maybe?
            # for i in range(self.draft_num_beam):
            #     new_tokens[root_parent[i], acc[root_parent[i]]] = tokens[i, 0]
            #     new_logits[root_parent[i], acc[root_parent[i]], :] = logit[state_indices[i], :]
            #     new_parents[root_parent[i], acc[root_parent[i]]] = beam_parent[i]
            #     temp_parent[i] = out_tokens.shape[1] + acc[root_parent[i]]
            #     acc[root_parent[i]] += 1
            
            out_tokens[:, out_len:(out_len+tokens.shape[1])] = tokens.to(torch.long)
            out_logits[:, out_len:(out_len+tokens.shape[1]), :] = new_logits
            out_len += tokens.shape[1]
            # parent_index = torch.cat([parent_index, new_parents], dim=1)

            # self.inference_params_draft.verified_key_value_memory_dict['indices'].copy_(self.static_state_indices[])
            # print("in drafting", parent_index, out_tokens, self.draft_log_probability, torch.log(curr_prob)[state_indices, tokens])
        
        # We don't need this as the state will be restored in the next update call
        # for l in self.inference_params.key_value_memory_dict.keys():
        #     self.inference_params.key_value_memory_dict[l][0].copy_(self.state_cache[l][0]) 
        #     self.inference_params.key_value_memory_dict[l][1].copy_(self.state_cache[l][1])

        # out = self.model.generate(
        #     input_ids=output_ids,
        #     max_length=Npad+output_ids.shape[1],
        #     mask=torch.ones_like(output_ids),
        #     cg=True,
        #     return_dict_in_generate=True,
        #     output_scores=False,
        #     enable_timing=False,
        #     temperature=0,
        #     top_k=1,
        # )
        # construct a mask based on parent index
        # out_mask = repeat(torch.eye(out_tokens.shape[1], device=state_indices.device), 'n m -> b n m', b=out_tokens.shape[0]).clone()
        # for i in range(parent_index.shape[0]):
        #     for j in range(parent_index.shape[1]):
        #         if parent_index[i, j] == -1 and j != 0:
        #             out_mask[i, j] = 0
        #         elif j != 0:
        #             out_mask[i, j] = out_mask[i, parent_index[i, j]] + out_mask[i, j]
        out_mask = self.attention_mask

        return out_tokens, out_mask, out_logits

    def get_logits(self, input_ids, mask, inference_params, draft_cache=False):
        decoding = inference_params.seqlen_offset > 0 and self.cg
        if decoding:
            position_ids = torch.full(
                (input_ids.shape[0], 1),
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
        if not decoding:
            logits = self.model(
                input_ids,
                position_ids=position_ids,
                inference_params=inference_params,
                mask=input_mask,
                num_last_tokens=1,
            ).logits
        else:
            if draft_cache:
                logits = self.model._decoding_cache_draft.run(
                    input_ids, position_ids, input_mask, inference_params.seqlen_offset
                )
            else:
                logits = self.model._decoding_cache_base.run(
                    input_ids, position_ids, input_mask, inference_params.seqlen_offset
                )
        return logits


STRAT_DICT = {
    "png": PNG_ModelBigramModelExt,
    "jacobi": JacobiDecoding,
    "bigram": ModelBigramModelExt,
    "mamba": MambaModel,
    "mamba-bs": MambaBeamSearchModel,
    "llama": LlamaModel,
    "mamba-st": MambaStaticTreeModel
}