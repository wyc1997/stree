
import argparse
import time
import json

import torch
import torch.nn.functional as F

from einops import rearrange

from transformers import AutoTokenizer, AutoModelForCausalLM

import os, sys
sys.path.insert(0, os.getcwd())
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel, MambaSelfDraftLMHeadModel
from mamba_ssm.models.speculative_decoding_mamba import MambaLMHeadSpecDecModel
from mamba_ssm.utils.generation_utils import InferenceParams


parser = argparse.ArgumentParser(description="Generation benchmarking")
parser.add_argument("--model-name", type=str, default="state-spaces/mamba-130m")
parser.add_argument("--prompt", type=str, nargs="+", default=None)
parser.add_argument("--promptlen", type=int, default=100)
parser.add_argument("--genlen", type=int, default=100)
parser.add_argument("--temperature", type=float, default=1.0)
parser.add_argument("--num_beam", type=int, default=1)
parser.add_argument("--draft_num_beam", type=int, default=0)
parser.add_argument("--topk", type=int, default=1)
parser.add_argument("--topp", type=float, default=1.0)
parser.add_argument("--minp", type=float, default=0.0)
parser.add_argument("--repetition-penalty", type=float, default=1.0)
parser.add_argument("--batch", type=int, default=1)
parser.add_argument("--model_layer_split", type=int, default=None)
parser.add_argument("--spec_Ngram", action='store_true')
parser.add_argument("--cg", action="store_true")
parser.add_argument("--use_Nstep_kernel", action="store_true")
parser.add_argument("--use_tree_decoding", action="store_true")
parser.add_argument("--save_last_seq", action="store_true")
parser.add_argument("--jit_state_copy", action="store_true")
parser.add_argument("--activation_replay", action="store_true")
parser.add_argument("--npad", type=int)
parser.add_argument("--ndraft", type=int, default=5)
parser.add_argument("--strategy", type=str, default="png")
parser.add_argument("--sample_target_only", action="store_true")
parser.add_argument("--unroll_tree", action="store_true")
args = parser.parse_args()

if __name__ == "__main__":
    repeats = 3
    device = "cuda"
    dtype = torch.bfloat16

    print(f"Loading model {args.model_name}")
    is_mamba = args.model_name.startswith("state-spaces/mamba") or args.model_name.startswith("state-spaces/transformerpp")
    is_MIL = (args.model_name.startswith("JunxiongWang/"))
    if is_mamba:
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        if args.model_layer_split is not None:
            model = MambaSelfDraftLMHeadModel.from_pretrained(args.model_name, device=device, dtype=dtype, self_draft_layer=args.model_layer_split, drafting_topk=10)
        elif args.spec_Ngram:
            model = MambaLMHeadSpecDecModel.from_pretrained(args.model_name, device=device, dtype=dtype,
                                                            jit_state_copy=args.jit_state_copy,
                                                            use_Nstep_kernel=args.use_Nstep_kernel,
                                                            use_tree_decoding=args.use_tree_decoding,
                                                            save_last_seq=args.save_last_seq,
                                                            activation_replay=args.activation_replay, 
                                                            npad=args.npad,
                                                            ndraft=args.ndraft,
                                                            num_beam=args.num_beam,
                                                            draft_num_beam=args.draft_num_beam,
                                                            strategy=args.strategy,
                                                            sample_target_only=args.sample_target_only,
                                                            unroll_tree=args.unroll_tree)
        else:
            model = MambaLMHeadModel.from_pretrained(args.model_name, device=device, dtype=dtype)
            # import json
            # from mamba_ssm.models.config_mamba import MambaConfig
            # with open('mamba2-7b_config.json', 'r') as f:
            #     model_config = json.load(f)
            #     print(model_config)
            # model = MambaLMHeadModel(config=MambaConfig(**model_config), device=device, dtype=dtype)
    elif is_MIL:
        import sys, os
        sys.path.insert(0, os.getcwd())
        if args.spec_Ngram:
            from mamba2_inference.spec_warpper import SpecMambaTransformerHybridModelWrapper
            tokenizer = AutoTokenizer.from_pretrained(args.model_name)
            model = SpecMambaTransformerHybridModelWrapper.from_pretrained(args.model_name, torch_dtype=dtype,
                                                                            jit_state_copy=args.jit_state_copy,
                                                                            use_Nstep_kernel=args.use_Nstep_kernel,
                                                                            use_tree_decoding=args.use_tree_decoding,
                                                                            save_last_seq=False,
                                                                            activation_replay=args.activation_replay, 
                                                                            npad=args.npad,
                                                                            ndraft=args.ndraft,
                                                                            num_beam=args.num_beam,
                                                                            draft_num_beam=args.draft_num_beam,
                                                                            strategy=args.strategy,
                                                                            sample_target_only=args.sample_target_only,
                                                                            unroll_tree=args.unroll_tree)
        else:
            from mamba2_inference.hybrid_wrapper import MambaTransformerHybridModelWrapper
            tokenizer = AutoTokenizer.from_pretrained(args.model_name)
            model = MambaTransformerHybridModelWrapper.from_pretrained(args.model_name, torch_dtype=dtype)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map={"": device}, torch_dtype=dtype)
    model.eval()
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    torch.random.manual_seed(0)
    if args.prompt is None:
        input_ids = torch.randint(1, 1000, (args.batch, args.promptlen), dtype=torch.long, device="cuda")
        attn_mask = torch.ones_like(input_ids, dtype=torch.long, device="cuda")
    else:
        input_id_list = []
        mask_list = []
        length_list = []
        for prompt in args.prompt:
            tokens = tokenizer(prompt, return_tensors="pt")
            input_id_list.append(tokens.input_ids.to(device=device))
            mask_list.append(tokens.attention_mask.to(device=device))
            length_list.append(tokens.input_ids.shape[1])

        input_seqlen = max(length_list)
        input_id_list = [F.pad(x, (input_seqlen - x.shape[1], 0)) for x in input_id_list]
        mask_list = [F.pad(x, (input_seqlen - x.shape[1], 0)) for x in mask_list]
        causal_attn_mask_list = [torch.ones((1, s, s), device=device).tril(diagonal=0) for s in length_list]
        causal_attn_mask_list = [F.pad(x, (0, input_seqlen-x.shape[1], 0, input_seqlen-x.shape[1])) for x in causal_attn_mask_list]
        # tokens = tokenizer(args.prompt, return_tensors="pt")
        input_ids = torch.cat(input_id_list, dim=0)
        # attn_mask = torch.cat(mask_list, dim=0)
        attn_mask = torch.cat(causal_attn_mask_list, dim=0)
    max_length = input_ids.shape[1] + args.genlen
    if is_mamba or is_MIL:
        fn = lambda: model.generate(
            input_ids=input_ids,
            max_length=max_length,
            mask=attn_mask,
            cg=args.cg,
            return_dict_in_generate=True,
            output_scores=True,
            enable_timing=False,
            temperature=args.temperature,
            num_beam=args.num_beam,
            draft_num_beam=args.draft_num_beam,
            top_k=args.topk,
            top_p=args.topp,
            min_p=args.minp,
            repetition_penalty=args.repetition_penalty,
            mask_type="attention"
        )
    else:
        fn = lambda: model.generate(
            input_ids=input_ids,
            attention_mask=attn_mask,
            max_length=max_length,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=args.temperature,
            top_k=args.topk,
            top_p=args.topp,
            repetition_penalty=args.repetition_penalty,
        )
    out = fn()
    if args.prompt is not None:
        for i in range(out.sequences.shape[0]):
            print(tokenizer.batch_decode(out.sequences[[i],:].tolist()))

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(repeats):
        torch.random.manual_seed(0)
        fn()
    torch.cuda.synchronize()
    end = time.time()
    if args.spec_Ngram:
        if input_ids.shape[0] == 1:
            if not args.use_tree_decoding and not args.unroll_tree:
                sequences = model.generation_hist
                correct_count = 0
                correct_len = []
                for seq in sequences:
                    correct_len.append(seq.shape[1])
                    if seq.shape[1] > 1:
                        correct_count += 1
                print("Draft correct rate: ", correct_count / len(sequences))
                print("Average generated tokens per step: ", sum(correct_len) / len(correct_len))
            else:
                gen_len = model.generation_hist
                correct_count = 0
                for x in gen_len:
                    if x > 1:
                        correct_count += 1
                print("Draft correct rate: ", correct_count / len(gen_len))
                print("Average generated tokens per step: ", sum(gen_len) / len(gen_len))
        else:
            total_tokens = []
            draft_correct_rate = []
            tokens_per_step = []
            for i in range(input_ids.shape[0]):
                sequences = model.generation_hist[i]
                correct_count = 0
                correct_len = []
                for seq in sequences:
                    correct_len.append(seq.shape[1])
                    if seq.shape[1] > 1:
                        correct_count += 1
                total_tokens.append(torch.cat(sequences, dim=1).shape[1])
                draft_correct_rate.append(correct_count / len(sequences))
                tokens_per_step.append(sum(correct_len) / len(correct_len))
                print("Sequence {} generation length: ".format(i), torch.cat(sequences, dim=1).shape[1])
                print("Sequence {} draft correct rate: ".format(i), correct_count / len(sequences))
                print("Sequence {} average generated tokens per step: ".format(i), sum(correct_len) / len(correct_len))
            print("total number of tokens generated: ", sum(total_tokens))
            print("Overall draft correct rate: ", sum(draft_correct_rate) / len(draft_correct_rate))
            print("Overall average token per step: ", sum(tokens_per_step) / len(tokens_per_step))
            
    print(f"Prompt length: {len(input_ids[0])}, generation length: {len(out.sequences[0]) - len(input_ids[0])}")
    print(f"{args.model_name} prompt processing + decoding time: {(end - start) / repeats * 1000:.0f}ms")

    if input_ids.shape[0] == 1:
        # if args.topk != 1: # doing speculative sampling, measure the perplexity of the generated sequence.
        out_seq = out.sequences.clone().detach()
        with torch.inference_mode():
            args.npad = 0 if args.npad is None else args.npad
            inference_params = InferenceParams(max_seqlen=max_length+args.npad+1, max_batch_size=1)
            inference_params.ndraft = 1
            inference_params.mask_type = "attention"
            inference_params.reset(max_length+args.npad+1, 1)
            logits = model(
                out_seq,
                position_ids=torch.arange(out_seq.shape[1], device=out_seq.device).unsqueeze(0),
                inference_params=inference_params,
                mask=torch.ones((1, out_seq.shape[1], out_seq.shape[1]), device=device).tril(diagonal=0),
                num_last_tokens=0,
            ).logits.squeeze(dim=1)
            probs = torch.softmax(logits, dim=-1)
            # probs = probs[:, torch.arange(probs.shape[1]), out.sequences[0,:]]
            logits = logits[:, len(input_ids[0])-1:-1, :]
            output = out.sequences[:, len(input_ids[0]):]
            ANLL = torch.nn.functional.cross_entropy(logits.transpose(1,2), output, reduction='none').mean(dim=1)
            perplexity = torch.exp(ANLL)
            if args.num_beam == 1:
                p = perplexity.item()
            else:
                p = " ".join([str(x) for x in perplexity.cpu().tolist()])
            print("Perplexity of generated sentence in original mamba2: {}".format(p))