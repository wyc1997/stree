import argparse
import json
import os, glob
import random
import time
from typing import Optional

import shortuuid
import torch
from tqdm import tqdm


def load_model_answers(answer_dir: str):
    """Load model answers.

    The return value is a python dict of type:
    Dict[model_name: str -> Dict[question_id: int -> answer: dict]]
    """
    if answer_dir.endswith('.jsonl'):
        filenames = [answer_dir]
    else:
        filenames = glob.glob(os.path.join(answer_dir, "*.jsonl"))
        filenames.sort()
    model_answers = {}

    for filename in filenames:
        model_name = os.path.basename(filename)[:-6]
        answer = {}
        with open(filename) as fin:
            for line in fin:
                line = json.loads(line)
                answer[line["question_id"]] = line
        model_answers[model_name] = answer

    return model_answers


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bench-name",
        type=str,
        default="mt_bench",
        help="The name of the benchmark question set.",
    )
    parser.add_argument(
        "--model-id",
        type=str, 
        required=True, 
        help="A custom name for the model."
    )
    args = parser.parse_args()

    answer_file = "llm_judge/data/{}/model_answer/{}.jsonl".format(args.bench_name, args.model_id)
    model_answer = load_model_answers(answer_file)
    
    avg_token_per_call = []
    avg_token_per_second = []
    perplexities = []
    if "generation_batch" in list(model_answer[args.model_id].values())[0]:
        token_per_second = []

        result_batch = []
        curr_batch = 0
        for question_id, ans in model_answer[args.model_id].items():
            if "avg_token_per_step" in ans:
                avg_token_per_call.append(ans['avg_token_per_step'])
            if ans['generation_batch'] == curr_batch:
                result_batch.append(ans)
                continue
            
            wall_time = result_batch[0]['choices'][0]['wall_time']
            new_tokens = [0 for _ in wall_time]
            
            for a in result_batch:
                for i in range(len(new_tokens)):
                    new_tokens[i] += a['choices'][0]["new_tokens"][i]
            for i in range(len(new_tokens)):
                token_per_second.append(new_tokens[i] / wall_time[i])
            avg_token_per_second.append(sum(token_per_second) / len(token_per_second))

            result_batch = [ans]
            curr_batch = ans['generation_batch']
    else:
        for question_id, ans in model_answer[args.model_id].items():
            token_per_second = []

            for j in range(len(ans["choices"])):
                new_tokens = ans["choices"][j]["new_tokens"]
                wall_time = ans["choices"][j]["wall_time"]
                if "perplexity" in ans["choices"][j]:
                    perplexity = ans["choices"][j]["perplexity"]
                    perplexities.append(perplexity)

                
                for i, new_token in enumerate(new_tokens):
                    if new_token != 0: # Error occured
                        token_per_second.append(new_token / wall_time[i])
            
            if len(token_per_second) != 0:
                if "avg_token_per_step" in ans:
                    avg_token_per_call.append(ans['avg_token_per_step'])

                avg_token_per_second.append(sum(token_per_second) / len(token_per_second))
    
    print("Number of answers: {}".format(len(avg_token_per_second)))
    print("Average token per second: {}".format(sum(avg_token_per_second) / len(avg_token_per_second)))
    if len(avg_token_per_call) > 0:
        print("Average token per call: {}".format(sum(avg_token_per_call) / len(avg_token_per_call)))
    if len(perplexities) != 0:
        print("Average perplexity: {}".format(sum(perplexities)/len(perplexities)))
        

