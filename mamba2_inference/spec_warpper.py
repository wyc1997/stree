
# Copyright (c) 2023, Albert Gu, Tri Dao.
import os
import json, math

import torch
import torch.nn as nn

from dataclasses import dataclass, field
from mamba_ssm.utils.generation import GenerationMixin

from transformers import AutoModelForCausalLM
from transformers.generation import GreedySearchDecoderOnlyOutput
from transformers.utils.hub import cached_file

from mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf
from mamba_ssm.models.speculative_decoding_strategies import MambaModel, MambaBeamSearchModel, MambaStaticTreeModel, STRAT_DICT
from mamba_ssm.utils.generation_utils import InferenceParams, update_graph_cache, modify_logit_for_repetition_penalty, sample
from mamba_ssm.utils.speculative_sampling import speculative_sampling, sampling_verification
from mamba_ssm.utils.tree_verification import verify_beam_search_tree, \
    sampling_verification_tree, unroll_tree, compress_tree, TokenTree
from mamba_ssm.utils.profile import cuda_time

from mamba2.hybrid_mamba_config import MambaConfig
from mamba2_inference.hybrid_spec_mamba import SpecMambaDecoderLayer, MHADecoderLayer
from mamba2_inference.hybrid_wrapper import MambaTransformerHybridModelWrapper

from util import load_safetensors_to_dict
from collections import namedtuple
from einops import repeat

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

def merge_projections_for_layers(checkpoint, layer_indices):
    for layer_idx in layer_indices:
        # Get the weights for q_proj, k_proj, and v_proj
        q_proj_key = f"model.layers.{layer_idx}.self_attn.q_proj.weight"
        k_proj_key = f"model.layers.{layer_idx}.self_attn.k_proj.weight"
        v_proj_key = f"model.layers.{layer_idx}.self_attn.v_proj.weight"
        o_proj_key = f"model.layers.{layer_idx}.self_attn.o_proj.weight"

        # Check if the keys exist in the checkpoint
        if q_proj_key in checkpoint and k_proj_key in checkpoint and v_proj_key in checkpoint:
            # Assuming all the projections have the same shape, otherwise adjust accordingly
            q_proj_weight = checkpoint[q_proj_key]
            k_proj_weight = checkpoint[k_proj_key]
            v_proj_weight = checkpoint[v_proj_key]

            # Concatenate the weights along the first dimension (often dimension 0)
            in_proj_weight = torch.cat([q_proj_weight, k_proj_weight, v_proj_weight], dim=0)

            # Assign the new weight to the corresponding in_proj key
            in_proj_key = f"model.layers.{layer_idx}.mha.in_proj.weight"
            checkpoint[in_proj_key] = in_proj_weight

            # Optionally, remove the old keys to clean up the checkpoint
            del checkpoint[q_proj_key]
            del checkpoint[k_proj_key]
            del checkpoint[v_proj_key]

        if o_proj_key in checkpoint:
            out_proj_key = f"model.layers.{layer_idx}.mha.out_proj.weight"
            checkpoint[out_proj_key] = checkpoint[o_proj_key]
            del checkpoint[o_proj_key]

    return checkpoint

MAMBA_CONFIG_NAME = "mamba_config.json"

class MambaInLlamaDistilledModel(MambaModel):
    @torch.inference_mode()
    def __init__(self, model_name, Ndraft, Npad, max_length, **kwargs):
        self.model = MambaTransformerHybridModelWrapper.from_pretrained("checkpoint/mamba2-distilled-small", torch_dtype=torch.bfloat16)
        self.model.to("cuda")
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


class MambaInLlamaDistilledBeamSearchModel(MambaBeamSearchModel):
    '''
    Performs beam search with Mamba2 model as a draft model, which requires it to be able to advance the state to where ever the 
    verifcation model ends. 
    Every update should return a tensor Num_beam x max_length input and a Num_beam x max_length x max_length mask, where each batch contains a tree that stems from previous input beam token
    Note the tree can be of different length which means that there needs to be padding tokens, or we can use the mask to ignore everything behind
    
    '''
    def __init__(self, model_name, Npad, base_num_beam, draft_num_beam, max_length, **kwargs):
        self.device = 'cuda'
        self.model = MambaTransformerHybridModelWrapper.from_pretrained("checkpoint/mamba2-distilled-small", torch_dtype=torch.bfloat16)
        self.model.to("cuda")
        self.generator = torch.Generator(device="cuda")
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


class MambaInLlamaDistilledStaticTreeModel(MambaStaticTreeModel):
    '''
    Build a token tree with a predefined structure with Mamba2 model as a draft model, which requires it to be able to advance the state to where ever the 
    verifcation model ends. 

    '''
    def __init__(self, model_name, Npad, base_num_beam, draft_num_beam, max_length, **kwargs):
        self.device = 'cuda'
        self.dtype = torch.bfloat16
        self.model = MambaTransformerHybridModelWrapper.from_pretrained("checkpoint/mamba2-distilled-small", torch_dtype=torch.bfloat16)
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
        self.static_state_indices = [[0,0,0],
                                    [0,0,1],
                                    [0,0,1],
                                    [0,0,0]]
        self.static_state_indices = torch.tensor(self.static_state_indices, device="cuda", dtype=torch.int)
        self.static_branch =[[3,0,0],
                             [2,1,0],
                             [2,1,0],
                             [3,0,0]]
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
            

class SpecMambaTransformerHybridModelWrapper(nn.Module, GenerationMixin):

    def __init__(self, checkpoint_path, transformer_model, mamba_config, attn_layers, dtype, load_from_hub=False, strategy="png", npad=4, ndraft=5, num_beam=1, draft_num_beam=0, use_2step_kernel=False, jit_state_copy=True, save_last_seq=True, use_Nstep_kernel=False, use_tree_decoding=False, activation_replay=False, sample_target_only=False, unroll_tree=False, **kwargs):
        super(SpecMambaTransformerHybridModelWrapper, self).__init__()
        self.mamba_config = mamba_config
        self.attn_layers = attn_layers
        self.model = transformer_model
        self.config = self.model.config
        
        if strategy == "MIL":
            self.strat = MambaInLlamaDistilledModel("../AIFRMambaSpeculativeDecoding/mamba_ssm/strategies/model2gram/Llama3.2-Mamba2-3B-distill/float16/extended_2gram_rankings.pth", Ndraft=ndraft, Npad=npad, base_num_beam=num_beam, draft_num_beam=draft_num_beam, max_length=10000)
        elif strategy == "MIL-bs":
            self.strat = MambaInLlamaDistilledBeamSearchModel("../AIFRMambaSpeculativeDecoding/mamba_ssm/strategies/model2gram/Llama3.2-Mamba2-3B-distill/float16/extended_2gram_rankings.pth", Ndraft=ndraft, Npad=npad, base_num_beam=num_beam, draft_num_beam=draft_num_beam, max_length=10000)
        elif strategy == "MIL-st":
            self.strat = MambaInLlamaDistilledStaticTreeModel("../AIFRMambaSpeculativeDecoding/mamba_ssm/strategies/model2gram/Llama3.2-Mamba2-3B-distill/float16/extended_2gram_rankings.pth", Ndraft=ndraft, Npad=npad, base_num_beam=num_beam, draft_num_beam=draft_num_beam, max_length=10000)
        else:
            self.strat = STRAT_DICT[strategy]("../AIFRMambaSpeculativeDecoding/mamba_ssm/strategies/model2gram/Llama3.2-Mamba2-3B-distill/float16/extended_2gram_rankings.pth", Ndraft=ndraft, Npad=npad, base_num_beam=num_beam, draft_num_beam=draft_num_beam, max_length=10000)
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
        
        for layer_idx in range(mamba_config.n_layer):
            if layer_idx in attn_layers:
                layer_encoder = MHADecoderLayer(
                    self.config,
                    layer_idx,
                    device="cuda",
                    dtype=dtype,
                )
            else:
                layer_encoder = SpecMambaDecoderLayer(
                    mamba_config,
                    layer_idx,
                    device="cuda",
                    dtype=dtype,
                )
            self.model.model.layers[layer_idx] = layer_encoder
            
        # print("self.model:", self.model)      
           
        if checkpoint_path is not None:
            if load_from_hub:
                # load from a huggingface hub
                ckpt = load_state_dict_hf(checkpoint_path, device=torch.device("cpu"), dtype=dtype)
            else:
                # load from a local directory
                if os.path.exists(f"{checkpoint_path}/pytorch_model.bin"):
                    # support save from bin file
                    ckpt = torch.load(f"{checkpoint_path}/pytorch_model.bin", map_location=torch.device("cpu"))
                else:
                    # support save from safetensors
                    ckpt = load_safetensors_to_dict(checkpoint_path)
        
            merge_projections_for_layers(ckpt, self.attn_layers)
            self.model.load_state_dict(ckpt)
        self.model = self.model.to(dtype).cuda()
        self.device = self.model.device
        self.can_generate = self.model.can_generate
        self.generation_config = self.model.generation_config

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.model.model.layers)
        }

    def allocate_value_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {i: layer.allocate_value_cache(batch_size, max_seqlen, dtype=dtype, **kwargs) 
                for i, layer in enumerate(self.model.model.layers)}

    def forward(self, input_ids, position_ids=None, mask=None, inference_params=None, num_last_tokens=0, **mixer_kwargs):
        """
        "position_ids" is just to be compatible with Transformer generation. We don't use it.
        num_last_tokens: if > 0, only return the logits for the last n tokens
        """
        hidden_states = self.model.model.embed_tokens(input_ids)
        for decoder_layer in self.model.model.layers:
            hidden_states = decoder_layer(hidden_states, position_ids=position_ids, inference_params=inference_params, mask=mask, **mixer_kwargs)
        hidden_states = self.model.model.norm(hidden_states)
        if num_last_tokens > 0:
            hidden_states = hidden_states[:, -num_last_tokens:]
        lm_logits = self.model.lm_head(hidden_states)
        CausalLMOutput = namedtuple("CausalLMOutput", ["logits"])
        return CausalLMOutput(logits=lm_logits)

    @staticmethod
    def init_distillation(
        checkpoint_path,
        tranformer_name,
        mamba_config,
        attn_layers,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        init_with_kqvo=True,
        **kwargs,
    ):
        transformer_model = AutoModelForCausalLM.from_pretrained(tranformer_name, torch_dtype=dtype, attn_implementation=attn_implementation)
        return SpecMambaTransformerHybridModelWrapper(checkpoint_path, transformer_model, mamba_config, attn_layers, dtype, init_with_kqvo)

    @staticmethod
    def from_pretrained_local(pretrained_model_name, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", **kwargs):
        config_data = load_config_hf(pretrained_model_name)
        transformer_model = AutoModelForCausalLM.from_pretrained(config_data["_name_or_path"], torch_dtype=torch_dtype, attn_implementation=attn_implementation)
        with open(f'{pretrained_model_name}/{MAMBA_CONFIG_NAME}', 'r') as json_file:
            config_dict = json.load(json_file)
        mamba_config = MambaConfig(**config_dict)
        return SpecMambaTransformerHybridModelWrapper(pretrained_model_name, transformer_model, mamba_config, mamba_config.attn_layers, torch_dtype, init_with_kqvo=False, **kwargs) 

    @staticmethod
    def from_pretrained_hub(pretrained_model_name, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", **kwargs):
        config_data = load_config_hf(pretrained_model_name)
        transformer_model = AutoModelForCausalLM.from_pretrained(config_data["_name_or_path"], torch_dtype=torch_dtype, attn_implementation=attn_implementation)
        resolved_archive_file = cached_file(pretrained_model_name, MAMBA_CONFIG_NAME, _raise_exceptions_for_missing_entries=False)
        config_dict = json.load(open(resolved_archive_file))
        mamba_config = MambaConfig(**config_dict)
        return SpecMambaTransformerHybridModelWrapper(pretrained_model_name, transformer_model, mamba_config, mamba_config.attn_layers, torch_dtype, init_with_kqvo=False, load_from_hub=True, **kwargs) 

    @staticmethod
    def from_pretrained(pretrained_model_name, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", **kwargs):
        if os.path.exists(pretrained_model_name):
            return SpecMambaTransformerHybridModelWrapper.from_pretrained_local(pretrained_model_name, torch_dtype, attn_implementation, **kwargs)
        else:
            return SpecMambaTransformerHybridModelWrapper.from_pretrained_hub(pretrained_model_name, torch_dtype, attn_implementation, **kwargs)

    def save_config(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)
        config_path = os.path.join(save_directory, 'mamba_config.json')
        with open(config_path, 'w') as f:
            json.dump(self.mamba_config.__dict__, f, indent=4)

    def get_memory_footprint(self):
        return self.model.get_memory_footprint()

    @torch.inference_mode()
    def generate(
        self,
        input_ids,
        max_length=1024,
        mask=None,
        top_k=1,
        top_p=0.0,
        min_p=0.0,
        temperature=1.0,
        return_dict_in_generate=False,
        output_scores=False,
        eos_token_id=None,
        **kwargs,
    ):

        if kwargs is not None:
            max_new_tokens = kwargs.pop('max_new_tokens', None)
            if max_new_tokens is not None:
                max_length = max_new_tokens + input_ids.shape[1]
            do_sample = kwargs.pop('do_sample', True)
            if not do_sample:
                top_k, top_p, min_p = 1, 0.0, 0.0
            cg = kwargs.pop('cg', True)

            # eos_token_id = kwargs.pop('eos_token_id', None)
            # if eos_token_id is None:
            #     eos_token_id = self.config.eos_token_id
            #     eos_token_id = torch.tensor(eos_token_id, device=self.model.device, dtype=torch.long)

            attention_mask = kwargs.pop('attention_mask', None)
            pad_token_id = kwargs.pop('pad_token_id', None)
            no_repeat_ngram_size = kwargs.pop('no_repeat_ngram_size', None)
            length_penalty = kwargs.pop('length_penalty', None)
            num_return_sequences = kwargs.pop('num_return_sequences', None)
            num_beams = kwargs.pop('num_beams', None)
            low_memory = kwargs.pop('low_memory', None)
            stopping_criteria = kwargs.pop('stopping_criteria', None)
        
        if self.use_tree_decoding:
            return self.generate_tree_decoding(
                input_ids=input_ids,
                max_length=max_length,
                cg=cg,
                mask=mask,
                top_k=top_k,
                top_p=top_p,
                min_p=min_p,
                temperature=temperature,
                return_dict_in_generate=return_dict_in_generate,
                output_scores=output_scores,
                eos_token_id=eos_token_id,
                **kwargs)
        else:
            return self.generate_save_last_word(
                input_ids=input_ids,
                max_length=max_length,
                cg=cg,
                mask=mask,
                top_k=top_k,
                top_p=top_p,
                min_p=min_p,
                temperature=temperature,
                return_dict_in_generate=return_dict_in_generate,
                output_scores=output_scores,
                eos_token_id=eos_token_id,
                **kwargs,
            )
    

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
                max_length+npad+1,
                decoding_seqlens=(npad+1,),
                use_2step_kernel=self.use_2step_kernel,
                jit_state_copy=self.jit_state_copy,
                save_last_seq=self.save_last_seq,
                use_Nstep_kernel=self.use_Nstep_kernel,
                activation_replay=self.activation_replay,
                npad=npad,
                ndraft=batchsize,
                first_iteration=False,
                mask_type="attention"
            )
            inference_params = self._decoding_cache.inference_params
            inference_params.reset(max_length, batchsize)
        else:
            inference_params = InferenceParams(max_seqlen=max_length, max_batch_size=batchsize)
            eg_param = next(iter(self.parameters()))
            dtype = eg_param.dtype
            device = eg_param.device
            # we need the max_seqlen to be a bit longer so that the kvcache is able to fit the last iteration
            inference_params.key_value_memory_dict = self.allocate_inference_cache(batch_size=batchsize, max_seqlen=max_length+npad+1, dtype=dtype)
            # verified dict saves two indices to the verified state in states
            inference_params.verified_key_value_memory_dict = {"indices":torch.zeros((1,), dtype=torch.long, device=device),
                                                               "mask":torch.zeros((1, npad+1), dtype=torch.bool, device=device),
                                                               "state":self.allocate_inference_cache(batch_size=1, max_seqlen=max_length+npad+1, dtype=dtype)}
            inference_params.value_cache = self.allocate_value_cache(batch_size=batchsize, max_seqlen=npad+1, dtype=dtype)

            inference_params.use_2step_kernel = self.use_2step_kernel
            inference_params.save_last_seq = self.save_last_seq
            inference_params.use_Nstep_kernel = self.use_Nstep_kernel
            inference_params.npad = npad
            inference_params.ndraft = batchsize
            inference_params.jit_state_copy = self.jit_state_copy
            inference_params.activation_replay = self.activation_replay
            inference_params.first_iteration = False
            inference_params.mask_type = "attention"
            inference_params.reset(max_length, batchsize)

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
        position_ids = repeat(torch.arange(prefill_ids.shape[1], device=prefill_ids.device), 'l -> b l', b=prefill_ids.shape[0])
        mask = torch.ones((prefill_ids.shape[0], prefill_ids.shape[1], prefill_ids.shape[1]), dtype=torch.long, device=prefill_ids.device).tril()
        self.forward(prefill_ids, position_ids=position_ids, mask=mask, inference_params=inference_params)
        inference_params.seqlen_offset += prefill_ids.shape[1]

        if self.strategy in ["mamba", "llama", "MIL"]:
            self.strat.initialize(input_ids[:, :-1], Ndraft=ndraft, Npad=npad, max_length=max_length+npad+1)
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
            # with cuda_time("", l=draft_time):
            step_block, _, draft_logit = self.strat.update(drafted_block, 
                                                        output_block, 
                                                        sequences[-1], 
                                                        output_ids=torch.cat(sequences, dim=-1),
                                                        sampling_params=draft_smapling_params)
            step_Ndraft, step_seq_len = (step_block.size(0), step_block.size(1))
            # input_block is (ndraft+1, len(in_seq)+npad)
            # Adding a padded input sequence to retrieve the last verified state
            # if not self.activation_replay:
            #     input_block = torch.cat([torch.nn.functional.pad(sequences[-1], (npad, 0), value=0), drafted_block], dim=0)
            #     input_mask = torch.ones_like(input_block)
            #     input_mask[0, :npad] = 0
            # else:
            input_block = drafted_block
            input_mask = torch.ones((input_block.shape[0], input_block.shape[1], input_block.shape[1]), dtype=torch.long, device=input_block.device).tril()
            # with cuda_time("", l=forward_time):
            position_ids = repeat(torch.arange(inference_params.seqlen_offset, inference_params.seqlen_offset+input_block.shape[1], device=input_mask.device), \
                                   'l -> b l', b=input_block.shape[0])
            if not cg or inference_params.first_iteration:
                outputs = self.forward(
                    input_block,
                    position_ids=position_ids,
                    inference_params=inference_params,
                    num_last_tokens=step_seq_len,
                    mask=input_mask
                )
                logits = outputs.logits

            else:
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
            # print(inference_params.key_value_memory_dict[27][0][0,:,0,0,0])
            # print(inference_params.verified_key_value_memory_dict["state"][26][1][0,0,0,:10])
            # print(inference_params.key_value_memory_dict[26][1][0,0,0,:10])
            # print(input_block)
            # print(torch.cat(sequences, dim=1))
            # print(nverified)
            # print(logits[0,0,[4948, 9420]])
            # print("kvcache", inference_params.key_value_memory_dict[1][0][0,:,0,0,0])
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
        
        if self.strategy == "MIL-st":
            decoding_seqlen_min = 1 + self.strat.total_len
            decoding_seqlen_max = 2 + self.strat.total_len
            decoding_seqlen_step = 1
            graph_cache_npad = self.strat.total_len
        elif draft_num_beam > 0 or num_beam > 1: # doing beam search related
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
                max_length+decoding_seqlen_max+1,
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
                num_input_seq=num_beam,
                mask_type="attention"
            )
            inference_params = self._decoding_cache.inference_params
            inference_params.reset(max_length, num_beam)
        else:
            inference_params = InferenceParams(max_seqlen=max_length, max_batch_size=num_beam)
            eg_param = next(iter(self.parameters()))
            dtype = eg_param.dtype
            device = eg_param.device
            inference_params.key_value_memory_dict = self.allocate_inference_cache(batch_size=num_beam, max_seqlen=max_length+decoding_seqlen_max, dtype=dtype)
            # verified dict saves two indices to the verified state in states
            inference_params.verified_key_value_memory_dict = {"indices":torch.zeros((num_beam,), dtype=torch.long, device=device),
                                                               "mask":torch.zeros((num_beam, graph_cache_npad+1), dtype=torch.bool, device=device),
                                                               "state":self.allocate_inference_cache(batch_size=num_beam, max_seqlen=max_length+decoding_seqlen_max, dtype=dtype)}
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
            inference_params.mask_type = "attention"
            inference_params.reset(max_length+decoding_seqlen_max, num_beam)

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
        if num_beam > 1:
            prefill_ids = input_ids
            # mask = torch.ones_like(prefill_ids) # we don't use a causal mask here since the prefill step is still using a chunkscan
            mask = torch.ones((prefill_ids.shape[0], prefill_ids.shape[1], prefill_ids.shape[1]), device=prefill_ids.device).tril()
            position_ids = repeat(torch.arange(prefill_ids.shape[1], device=prefill_ids.device), 'l -> b l', b=prefill_ids.shape[0])
            output = self.forward(prefill_ids, position_ids=position_ids, mask=mask, num_last_tokens=1, inference_params=inference_params)
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
        
        # print("after prefill",sequences_cat)
        # Only implemented for mamba model drafting.
        # assert self.strategy == "mamba-bs"
        available_strat = ["mamba-bs", "mamba", "png", "MIL-bs", "MIL-st"]
        assert self.strategy in available_strat
        if num_beam > 1:
            self.strat.initialize(input_ids, Npad=npad, max_length=max_length, draft_num_beam=draft_num_beam, base_num_beam=num_beam, cg=cg)
        else:
            self.strat.initialize(input_ids[:, :-1], Ndraft=ndraft, Npad=npad, max_length=max_length, draft_num_beam=draft_num_beam, base_num_beam=num_beam, cg=cg)
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
            # with cuda_time("drafting", l=draft_time):
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

            position_ids = inference_params.seqlen_offset + input_mask.sum(dim=2) - 1 
            # with cuda_time("forward pass", l=forward_time):
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

            # with cuda_time("verification", l=verification_time):
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
            # print(position_ids)
            # print(inference_params.key_value_memory_dict[27][0][0,:,0,0,0])
            # print(inference_params.verified_key_value_memory_dict["state"][26][1][0,0,0,:10])
            # print(inference_params.key_value_memory_dict[26][1][0,0,0,:10])
            # print(input_block)
            # print(sequences_cat)
            # print(verified_mask)
            # print(input_block)
            # print(sequences_cat)
            # print(verified_mask)
            inference_params.seqlen_offset += verified_step

            inference_params.verified_key_value_memory_dict['indices'].copy_(state_indices_base)
            verified_mask = torch.nn.functional.pad(verified_mask, (0, graph_cache_npad+1-verified_mask.shape[1]))

            inference_params.verified_key_value_memory_dict["mask"].copy_(verified_mask)
            inference_params.first_iteration = False

            self.generation_hist.append(verified_step)

        # print("draft:", sum(draft_time)/len(draft_time))
        # print(draft_time)
        # print("forward:", sum(forward_time)/len(forward_time))
        # print("verification:", sum(verification_time)/len(verification_time))
        # print(verification_time)
        return GreedySearchDecoderOnlyOutput(sequences=sequences_cat, scores=log_probability)