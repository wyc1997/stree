"""Generate answers with local models.

Usage:
python3 gen_model_answer.py --model-path lmsys/fastchat-t5-3b-v1.0 --model-id fastchat-t5-3b-v1.0
"""
import argparse
import json
import os, sys
import random
import time
from typing import Optional
from transformers import AutoTokenizer
sys.path.insert(0, os.getcwd())
from mamba2_inference.hybrid_wrapper import MambaTransformerHybridModelWrapper
from mamba2_inference.spec_warpper import SpecMambaTransformerHybridModelWrapper
from llm_judge.mt_bench import get_model_answers_mt_bench, get_model_answers_mt_bench_multi
from llm_judge.gsm import get_model_answers_gsm8k, get_model_answers_gsm8k_multi
from llm_judge.human_eval import get_model_answers_human_eval, get_model_answers_human_eval_multi

import shortuuid
import torch
from tqdm import tqdm

from fastchat.model import load_model, get_conversation_template
from fastchat.utils import str_to_torch_dtype

# Sampling temperature configs for
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

def load_questions(question_file: str, begin: Optional[int], end: Optional[int]):
    """Load questions from a file."""
    questions = []
    with open(question_file, "r") as ques_file:
        for line in ques_file:
            if line:
                questions.append(json.loads(line))
    questions = questions[begin:end]
    return questions

def run_eval(
    bench_name,
    model_path,
    model_id,
    question_file,
    question_begin,
    question_end,
    answer_file,
    max_new_token,
    num_choices,
    num_gpus_per_model,
    num_gpus_total,
    max_gpu_memory,
    spec_Ngram,
    cg, 
    use_2step_kernel,
    use_Nstep_kernel,
    use_tree_decoding,
    save_last_seq, 
    jit_state_copy,
    activation_replay,
    npad,
    ndraft,
    strategy,
    dtype,
    revision,
    num_input_seq,
    num_beam,
    draft_num_beam,
    unroll_tree,
    sample_target_only,
    temperature, 
    top_k,
    top_p,
    min_p,
):
    questions = load_questions(question_file, question_begin, question_end)
    # random shuffle the questions to balance the loading
    # random.shuffle(questions)

    # tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    if spec_Ngram:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = SpecMambaTransformerHybridModelWrapper.from_pretrained(model_path, torch_dtype=dtype,
                                                                        use_2step_kernel=use_2step_kernel, 
                                                                        jit_state_copy=jit_state_copy,
                                                                        use_Nstep_kernel=use_Nstep_kernel,
                                                                        use_tree_decoding=use_tree_decoding,
                                                                        save_last_seq=save_last_seq,
                                                                        activation_replay=activation_replay, 
                                                                        npad=npad,
                                                                        ndraft=ndraft,
                                                                        num_beam=num_beam,
                                                                        draft_num_beam=draft_num_beam,
                                                                        strategy=strategy,
                                                                        unroll_tree=unroll_tree,
                                                                        sample_target_only=sample_target_only)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = MambaTransformerHybridModelWrapper.from_pretrained(model_path, torch_dtype=dtype)

    # Split the question file into `num_gpus` files
    assert num_gpus_total % num_gpus_per_model == 0

    if bench_name == "mt_bench":
        get_answers_func = get_model_answers_mt_bench
        if num_input_seq > 1:
            get_answers_func = get_model_answers_mt_bench_multi
    elif bench_name == "gsm8k":
        get_answers_func = get_model_answers_gsm8k
        if num_input_seq > 1:
            get_answers_func = get_model_answers_gsm8k_multi
    elif bench_name == "human_eval":
        get_answers_func = get_model_answers_human_eval
        if num_input_seq > 1:
            get_answers_func = get_model_answers_human_eval_multi
    else:
        raise NotImplementedError

    chunk_size = len(questions) // (num_gpus_total // num_gpus_per_model)
    ans_handles = []
    for i in range(0, len(questions), chunk_size):
        ans_handles.append(
            get_answers_func(
                model,
                tokenizer,
                model_path,
                model_id,
                questions[i : i + chunk_size],
                answer_file,
                max_new_token,
                num_choices,
                dtype=dtype,
                spec_Ngram=spec_Ngram,
                cg=cg,
                num_input_seq=num_input_seq,
                num_beam=num_beam,
                draft_num_beam=draft_num_beam,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                min_p=min_p
            )
        )

def reorg_answer_file(answer_file):
    """Sort by question id and de-duplication"""
    answers = {}
    with open(answer_file, "r") as fin:
        for l in fin:
            qid = json.loads(l)["question_id"]
            answers[qid] = l

    qids = sorted(list(answers.keys()))
    with open(answer_file, "w") as fout:
        for qid in qids:
            fout.write(answers[qid])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
    )
    parser.add_argument(
        "--model-id", type=str, required=True, help="A custom name for the model."
    )
    parser.add_argument(
        "--bench-name",
        type=str,
        default="mt_bench",
        help="The name of the benchmark question set.",
    )
    parser.add_argument(
        "--question-begin",
        type=int,
        help="A debug option. The begin index of questions.",
    )
    parser.add_argument(
        "--question-end", type=int, help="A debug option. The end index of questions."
    )
    parser.add_argument("--answer-file", type=str, help="The output answer file.")
    parser.add_argument(
        "--max-new-token",
        type=int,
        default=1024,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--num-choices",
        type=int,
        default=1,
        help="How many completion choices to generate.",
    )
    parser.add_argument(
        "--num-gpus-per-model",
        type=int,
        default=1,
        help="The number of GPUs per model.",
    )
    parser.add_argument(
        "--num-gpus-total", type=int, default=1, help="The total number of GPUs."
    )
    parser.add_argument(
        "--max-gpu-memory",
        type=str,
        help="Maxmum GPU memory used for model weights per GPU.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "float16", "bfloat16"],
        help="Override the default dtype. If not set, it will use float16 on GPU and float32 on CPU.",
        default="bfloat16",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="main",
        help="The model revision to load.",
    )
    parser.add_argument("--spec_Ngram", action="store_true")
    parser.add_argument("--cg", action="store_true")
    parser.add_argument("--use_2step_kernel", action="store_true")
    parser.add_argument("--jit_state_copy", action="store_true")
    parser.add_argument("--use_Nstep_kernel", action="store_true")
    parser.add_argument("--use_tree_decoding", action="store_true")
    parser.add_argument("--save_last_seq", action="store_true")
    parser.add_argument("--activation_replay", action="store_true")
    parser.add_argument("--npad", type=int, default=1, required=False)
    parser.add_argument("--ndraft", type=int, default=5, required=False)
    parser.add_argument("--strategy", type=str, default="png", required=False)
    parser.add_argument("--num_input_seq", type=int, default=1)
    parser.add_argument("--num_beam", type=int, default=1)
    parser.add_argument("--draft_num_beam", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=1)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--min_p", type=float, default=0.0)
    parser.add_argument("--unroll_tree", action="store_true")
    parser.add_argument("--sample_target_only", action="store_true")

    args = parser.parse_args()

    question_file = f"llm_judge/data/{args.bench_name}/question.jsonl"
    if args.answer_file:
        answer_file = args.answer_file
    else:
        answer_file = f"llm_judge/data/{args.bench_name}/model_answer/{args.model_id}.jsonl"

    print(f"Output to {answer_file}")

    run_eval(
        bench_name=args.bench_name,
        model_path=args.model_path,
        model_id=args.model_id,
        question_file=question_file,
        question_begin=args.question_begin,
        question_end=args.question_end,
        answer_file=answer_file,
        max_new_token=args.max_new_token,
        num_choices=args.num_choices,
        num_gpus_per_model=args.num_gpus_per_model,
        num_gpus_total=args.num_gpus_total,
        max_gpu_memory=args.max_gpu_memory,
        dtype=str_to_torch_dtype(args.dtype),
        spec_Ngram=args.spec_Ngram,
        cg=args.cg,
        use_2step_kernel=args.use_2step_kernel,
        use_Nstep_kernel=args.use_Nstep_kernel, 
        use_tree_decoding=args.use_tree_decoding,
        save_last_seq=args.save_last_seq,
        jit_state_copy=args.jit_state_copy,
        activation_replay=args.activation_replay,
        npad=args.npad,
        ndraft=args.ndraft, 
        strategy=args.strategy,
        revision=args.revision,
        num_input_seq=args.num_input_seq,
        num_beam=args.num_beam,
        draft_num_beam=args.draft_num_beam,
        unroll_tree=args.unroll_tree,
        sample_target_only=args.sample_target_only,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        min_p=args.min_p,
    )

    reorg_answer_file(answer_file)
