import argparse
import json
import os
import random
import time
from typing import Optional
from transformers import AutoTokenizer
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.models.speculative_decoding_mamba import MambaLMHeadSpecDecModel
import torch.nn.functional as F

import shortuuid
import torch
from tqdm import tqdm

from fastchat.model import load_model, get_conversation_template
from fastchat.conversation import get_conv_template
from fastchat.utils import str_to_torch_dtype

temperature_config = {
    "writing": 0.7,
    "roleplay": 0.7,
    "extraction": 0.0,
    "math": 0.0,
    "coding": 0.0,
    "reasoning": 0.0,
    "stem": 0.1,
    "humanities": 0.1,
    "arena-hard-200": 0.0,
}

@torch.inference_mode()
def get_model_answers_gsm8k(
    model, 
    tokenizer,
    model_path,
    model_id,
    questions,
    answer_file,
    max_new_token,
    num_choices,
    dtype,
    spec_Ngram,
    cg,
    num_beam,
    draft_num_beam,
    temperature=0.0,
    top_k=1,
    top_p=0.0,
    min_p=0.0,
    **kwargs
):
    max_length = 0
    max_length_index = 0
    for i, question in enumerate(questions):
        qs = question["prompt"]
        # pr = 'Solve the following math problem: ' + qs
        pr = qs
        conv = get_conv_template("raw")
        conv.append_message(conv.roles[0], pr)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer([prompt]).input_ids
        if len(input_ids[0]) > max_length:
            max_length_index = i
            max_length = len(input_ids[0])
    question = questions[max_length_index]
    print(max_length)

    # warmup
    for _ in tqdm(range(3), desc='warmup'):

        chat = []

        qs = question["prompt"]
        # pr = 'Solve the following math problem: ' + qs
        pr = qs
        conv = get_conv_template("raw")
        conv.append_message(conv.roles[0], pr)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer([prompt]).input_ids
        mask = torch.ones((1, len(input_ids[0]), len(input_ids[0]))).cuda().tril()

        out = model.generate(
            input_ids=torch.as_tensor(input_ids).cuda(),
            max_length=len(input_ids[0])+max_new_token,
            mask=mask,
            cg=cg,
            return_dict_in_generate=True,
            output_scores=True,
            enable_timing=False,
            temperature=temperature,
            top_k=top_k, 
            top_p=top_p,
            min_p=min_p,
            num_beam=num_beam,
            draft_num_beam=draft_num_beam,
            mask_type="attention"
        )

        output_ids = out[len(input_ids[0]) :]
        
        output = tokenizer.decode(
            output_ids,
            spaces_between_special_tokens=False,
        )

        chat.append({'role' : 'assistant', 'content' : output})

    print('Warmup done')
    NQues = len(questions)
    all_chats = {}
    for qi in tqdm(range(NQues), desc='questions'):

        choices = []

        torch.manual_seed(0)
        wall_time = []
        new_tokens = []

        chat = []
        question = questions[qi]

        example_id = qi

        qs = question["prompt"]
        # pr = 'Solve the following math problem: ' + qs
        pr = qs
        conv = get_conv_template("raw")
        conv.append_message(conv.roles[0], pr)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer([prompt]).input_ids


        # some models may error out when generating long outputs
        try:
            torch.cuda.synchronize()
            start_time = time.time()
            mask = torch.ones((1, len(input_ids[0]), len(input_ids[0]))).cuda().tril()
            out = model.generate(
                input_ids=torch.as_tensor(input_ids).cuda(),
                max_length=len(input_ids[0])+max_new_token,
                mask=mask,
                cg=cg,
                return_dict_in_generate=True,
                output_scores=True,
                enable_timing=False,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                min_p=min_p,
                num_beam=num_beam,
                draft_num_beam=draft_num_beam,
                mask_type="attention"
            )
            torch.cuda.synchronize()
            total_time = time.time() - start_time

            output_ids = out.sequences[0][len(input_ids[0]) :]
            new_token = len(out.sequences[0]) - len(input_ids[0])

            # be consistent with the template's stop_token_ids
            if conv.stop_token_ids:
                stop_token_ids_index = [
                    i
                    for i, id in enumerate(output_ids)
                    if id in conv.stop_token_ids
                ]
                if len(stop_token_ids_index) > 0:
                    output_ids = output_ids[: stop_token_ids_index[0]]

            output = tokenizer.decode(
                output_ids,
                spaces_between_special_tokens=False,
            )
            if conv.stop_str and isinstance(conv.stop_str, list):
                stop_str_indices = sorted(
                    [
                        output.find(stop_str)
                        for stop_str in conv.stop_str
                        if output.find(stop_str) > 0
                    ]
                )
                if len(stop_str_indices) > 0:
                    output = output[: stop_str_indices[0]]
            elif conv.stop_str and output.find(conv.stop_str) > 0:
                output = output[: output.find(conv.stop_str)]

            for special_token in tokenizer.special_tokens_map.values():
                if isinstance(special_token, list):
                    for special_tok in special_token:
                        output = output.replace(special_tok, "")
                else:
                    output = output.replace(special_token, "")
            output = output.strip()

            if conv.name == "xgen" and output.startswith("Assistant:"):
                output = output.replace("Assistant:", "", 1).strip()
        except RuntimeError as e:
            print("ERROR question ID: ", question["question_id"])
            output = "ERROR"

        conv.update_last_message(output)
        all_chats[qi] = chat
        wall_time.append(total_time)
        new_tokens.append(new_token)

        choices.append({"index": 0, "chats": [output], "wall_time": wall_time, "new_tokens": new_tokens})

        # Dump answers
        os.makedirs(os.path.dirname(answer_file), exist_ok=True)
        with open(os.path.expanduser(answer_file), "a") as fout:
            ans_json = {
                "question_id": example_id,
                "answer_id": shortuuid.uuid(),
                "model_id": model_id,
                "choices": choices,
                "tstamp": time.time(),
            }
            if spec_Ngram:
                if not model.use_tree_decoding:
                    sequences = model.generation_hist
                    correct_count = 0
                    correct_len = []
                    for seq in sequences:
                        correct_len.append(seq.shape[1])
                        if seq.shape[1] > 1:
                            correct_count += 1
                    ans_json["draft_correct_rate"] = correct_count / len(sequences)
                    ans_json["avg_token_per_step"] = sum(correct_len) / len(correct_len)
                else:
                    gen_len = model.generation_hist
                    correct_count = 0
                    for x in gen_len:
                        if x > 1:
                            correct_count += 1
                    ans_json["draft_correct_rate"] = correct_count / len(gen_len)
                    ans_json["avg_token_per_step"] = sum(gen_len) / len(gen_len)
            fout.write(json.dumps(ans_json) + "\n")


@torch.inference_mode()
def get_model_answers_gsm8k_multi(
    model, 
    tokenizer,
    model_path,
    model_id,
    questions,
    answer_file,
    max_new_token,
    num_choices,
    dtype,
    spec_Ngram,
    cg,
    num_input_seq
):

    question_batch = questions[0:0+num_input_seq]

    # warmup
    for _ in tqdm(range(3), desc='warmup'):

        chat = [] 
        convs = [get_conv_template("zero_shot") for _ in range(num_input_seq)]

        new_token = 0
        input_id_list = []
        mask_list = []
        length_list = []
        for k in range(num_input_seq):
            example_id = k
            qs = question_batch[k]["prompt"]
            pr = qs
            convs[k].append_message(convs[k].roles[0], pr)
            convs[k].append_message(convs[k].roles[1], None)
            prompt = convs[k].get_prompt()
            tokens = tokenizer([prompt], return_tensors='pt')
            input_id_list.append(tokens.input_ids)
            mask_list.append(tokens.attention_mask)
            length_list.append(tokens.input_ids.shape[1])
        
        input_seqlen = max(length_list)
        input_id_list = [F.pad(x, (input_seqlen - x.shape[1], 0)) for x in input_id_list]
        mask_list = [F.pad(x, (input_seqlen - x.shape[1], 0)) for x in mask_list]
        # tokens = tokenizer(args.prompt, return_tensors="pt")
        input_ids = torch.cat(input_id_list, dim=0)
        attn_mask = torch.cat(mask_list, dim=0)


        out = model.generate(
            input_ids=torch.as_tensor(input_ids).cuda(),
            max_length=len(input_ids[0])+max_new_token,
            mask=attn_mask.cuda(),
            cg=cg,
            return_dict_in_generate=True,
            output_scores=True,
            enable_timing=False,
            temperature=0.0,
            top_k=1, # greedy decoding
        )
        
        output = [tokenizer.batch_decode(
            out.sequences[[k], length_list[k]:],
            spaces_between_special_tokens=False,
        ) for k in range(num_input_seq)]

        chat.append({'role' : 'assistant', 'content' : output})

    print('Warmup done')

    for idx in tqdm(range(0, len(questions), num_input_seq)):
        question_batch = [questions[a] for a in range(idx, min(len(questions), idx + num_input_seq))]
        Nseq = len(question_batch)

        convs = [get_conv_template("zero_shot") for _ in range(Nseq)]
        turns = [[] for _  in range(Nseq)]
        wall_time = []
        new_tokens = [[] for _ in range(Nseq)]

        new_token = []
        input_id_list = []
        mask_list = []
        length_list = []
        id_list = []

        for k in range(Nseq):
            example_id = idx + k
            qs = question_batch[k]["prompt"]
            pr = qs
            convs[k].append_message(convs[k].roles[0], pr)
            convs[k].append_message(convs[k].roles[1], None)
            prompt = convs[k].get_prompt()
            tokens = tokenizer([prompt], return_tensors='pt')
            input_id_list.append(tokens.input_ids)
            mask_list.append(tokens.attention_mask)
            length_list.append(tokens.input_ids.shape[1])
            id_list.append(example_id)
        
        input_seqlen = max(length_list)
        input_id_list = [F.pad(x, (input_seqlen - x.shape[1], 0)) for x in input_id_list]
        mask_list = [F.pad(x, (input_seqlen - x.shape[1], 0)) for x in mask_list]
        # tokens = tokenizer(args.prompt, return_tensors="pt")
        input_ids = torch.cat(input_id_list, dim=0)
        attn_mask = torch.cat(mask_list, dim=0)

        # remainders when dataset not divisible by num_input_seq
        if input_ids.shape[0] < num_input_seq:
            input_ids = torch.cat([input_ids, 
                                    torch.zeros((num_input_seq - input_ids.shape[0], input_ids.shape[1]), device=input_ids.device, dtype=input_ids.dtype)], dim=0)
            attn_mask = torch.cat([attn_mask, 
                                    torch.zeros((num_input_seq - attn_mask.shape[0], attn_mask.shape[1]), device=attn_mask.device, dtype=attn_mask.dtype)], dim=0)

        output_list = []
        new_token_list = []

        # some models may error out when generating long outputs
        try:
            torch.cuda.synchronize()
            start_time = time.time()
            out = model.generate(
                input_ids=torch.as_tensor(input_ids).cuda(),
                max_length=len(input_ids[0])+max_new_token,
                mask=attn_mask.cuda(),
                cg=cg,
                return_dict_in_generate=True,
                output_scores=True,
                enable_timing=False,
                temperature=0.0,
                top_k=1, # greedy decoding
            )
            torch.cuda.synchronize()
            total_time = time.time() - start_time
            for k in range(Nseq):
                if spec_Ngram:
                    output_ids = out.sequences[k, (length_list[k]):]
                    new_token = torch.cat(model.generation_hist[k], dim=-1).shape[1]
                    new_tokens[k].append(new_token)
                else:
                    output_ids = out.sequences[k, input_ids.shape[1]:]
                    new_tokens[k].append(out.sequences.shape[1] - input_ids.shape[1])

                # be consistent with the template's stop_token_ids
                if convs[k].stop_token_ids:
                    stop_token_ids_index = [
                        i
                        for i, id in enumerate(output_ids)
                        if id in convs[k].stop_token_ids
                    ]
                    if len(stop_token_ids_index) > 0:
                        output_ids = output_ids[: stop_token_ids_index[0]]
                output = tokenizer.decode(
                    output_ids,
                    spaces_between_special_tokens=False,
                )
                if convs[k].stop_str and isinstance(convs[k].stop_str, list):
                    stop_str_indices = sorted(
                        [
                            output.find(stop_str)
                            for stop_str in convs[k].stop_str
                            if output.find(stop_str) > 0
                        ]
                    )
                    if len(stop_str_indices) > 0:
                        output = output[: stop_str_indices[0]]
                elif convs[k].stop_str and output.find(convs[k].stop_str) > 0:
                    output = output[: output.find(convs[k].stop_str)]

                for special_token in tokenizer.special_tokens_map.values():
                    if isinstance(special_token, list):
                        for special_tok in special_token:
                            output = output.replace(special_tok, "")
                    else:
                        output = output.replace(special_token, "")
                output = output.strip()

                if convs[k].name == "xgen" and output.startswith("Assistant:"):
                    output = output.replace("Assistant:", "", 1).strip()
                output_list.append(output)
        except RuntimeError as e:
            raise e
            print("ERROR question ID: ", question_batch[0]["question_id"])
            output = "ERROR"
        

        wall_time.append(total_time)
        for k in range(Nseq):
            convs[k].update_last_message(output_list[k])
            turns[k].append(output_list[k])

        # Dump answers
        for k in range(Nseq):
            os.makedirs(os.path.dirname(answer_file), exist_ok=True)
            with open(os.path.expanduser(answer_file), "a") as fout:
                ans_json = {
                    "question_id": id_list[k],
                    "answer_id": shortuuid.uuid(),
                    "model_id": model_id,
                    "choices": [{"index": 0, "turns": turns[k], "wall_time": wall_time, "new_tokens": new_tokens[k]}],
                    "tstamp": time.time(),
                    "generation_batch":idx, 
                }
                if spec_Ngram:
                    sequences = model.generation_hist[k]
                    correct_count = 0
                    correct_len = []
                    for seq in sequences:
                        correct_len.append(seq.shape[1])
                        if seq.shape[1] > 1:
                            correct_count += 1
                    ans_json["draft_correct_rate"] = correct_count / len(sequences)
                    ans_json["avg_token_per_step"] = sum(correct_len) / len(correct_len)
                fout.write(json.dumps(ans_json) + "\n")