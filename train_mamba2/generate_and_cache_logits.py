#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Supervised fine-tuning script for decoder language models.
"""

import logging
import random
import sys, os, math
from tqdm.auto import tqdm
os.environ['HF_HOME'] = '/media/data1/ycwu/cache/'

import datasets
import torch
from torch.utils.data import DataLoader
import transformers
from transformers import AutoConfig, AutoModelForCausalLM, set_seed, BitsAndBytesConfig, AutoTokenizer

from alignment import (
    DataArguments,
    H4ArgumentParser,
    ModelArguments,
    apply_chat_template,
    decontaminate_humaneval,
    get_checkpoint,
    get_datasets,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
    get_tokenizer,
)
from trl import setup_chat_format

sys.path.insert(0, os.getcwd())
from trainer.kd_trainer import KDTrainer

# from mamba2.hybrid_wrapper import MambaTransformerHybridModelWrapper
from mamba2_inference.hybrid_wrapper import MambaTransformerHybridModelWrapper
from mamba2.hybrid_mamba_config import MambaConfig
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
# from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.utils.hf import load_state_dict_hf
import json
from accelerate import PartialState

from train_configs import SFTDistillConfig
from util import construct_layer_dict

import torch.distributed as dist
from datetime import timedelta

logger = logging.getLogger(__name__)
use_flash_attention_2 = True

def main():
    parser = H4ArgumentParser((ModelArguments, DataArguments, SFTDistillConfig))
    model_args, data_args, training_args = parser.parse()

    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = get_checkpoint(training_args)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    ###############
    # Load datasets
    ###############
    raw_datasets = get_datasets(
        data_args,
        splits=data_args.dataset_splits,
        configs=data_args.dataset_configs,
        columns_to_keep=["messages", "chosen", "rejected", "prompt", "completion", "label"],
    )

    logger.info(
        f"Training on the following datasets and their proportions: {[split + ' : ' + str(dset.num_rows) for split, dset in raw_datasets.items()]}"
    )
    column_names = list(raw_datasets["train"].features)

    ################
    # Load tokenizer
    ################
    tokenizer = get_tokenizer(model_args, data_args)
    logger.info("Tokenizer max length: {}".format(tokenizer.model_max_length))

    #######################
    # Load pretrained model
    #######################
    logger.info("*** Load pretrained model ***")
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)

    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        use_flash_attention_2=use_flash_attention_2,
        torch_dtype=torch_dtype,
        use_cache=False,
        # use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

    model = model_args.model_name_or_path
    # For ChatML we need to add special tokens and resize the embedding layer
    if "<|im_start|>" in tokenizer.chat_template and "gemma-tokenizer-chatml" not in tokenizer.name_or_path:
        model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, **model_kwargs)
        model, tokenizer = setup_chat_format(model, tokenizer)
        model_kwargs = None

    #####################
    # Apply chat template
    #####################
    raw_datasets = raw_datasets.map(
        apply_chat_template,
        fn_kwargs={
            "tokenizer": tokenizer,
            "task": "sft",
            "auto_insert_empty_system_msg": data_args.auto_insert_empty_system_msg,
        },
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        desc="Applying chat template",
    )

    ##########################
    # Decontaminate benchmarks
    ##########################
    if training_args.decontaminate:
        num_raw_train_samples = len(raw_datasets["train"])
        raw_datasets = raw_datasets.filter(decontaminate_humaneval, batched=True, batch_size=10_000, num_proc=data_args.preprocessing_num_workers)
        num_filtered_train_samples = num_raw_train_samples - len(raw_datasets["train"])
        logger.info(
            f"Decontaminated {num_filtered_train_samples} ({num_filtered_train_samples/num_raw_train_samples * 100:.2f}%) samples from the training set."
        )

    train_dataset = raw_datasets["train"]
    # print(tokenizer(train_dataset[[1,2]]['text'], return_tensors='pt', padding=True))
    total_len = len(train_dataset)
    logger.info("global number of training data: {}".format(len(train_dataset)))

    eval_dataset = raw_datasets["test"]

    with training_args.main_process_first(desc="Log a few random samples from the processed training set"):
        for index in random.sample(range(len(raw_datasets["train"])), 3):
            logger.info(f"Sample {index} of the processed training set:\n\n{raw_datasets['train'][index]['text']}")

    # loading model
    teacher_model = MambaTransformerHybridModelWrapper.from_pretrained(training_args.teacher_model_name_or_path, 
                                                                    torch_dtype=torch_dtype)
    # teacher_model.to(distributed_state.device)
    teacher_model.eval()

    if not os.path.exists(training_args.output_dir):
        os.mkdir(training_args.output_dir)

    rank = training_args.rank
    indices = range(rank * (total_len // 4), (rank+1) * (total_len // 4))
    if training_args.overwrite_output_dir:
        logger.info("Overwriting output dir {}".format(training_args.output_dir))
    logger.info("Running rank {}, with indices {} - {}".format(rank, indices[0], indices[-1]))
    with torch.inference_mode():
        local_len = len(indices)
        batch_size = training_args.per_device_eval_batch_size
        num_batches = math.ceil(local_len / batch_size)
        for i in tqdm(range(num_batches), desc="rank {}".format(training_args.rank)):
            start = i * batch_size
            length = min(local_len - start, batch_size)
            index_to_get = indices[start:start+length]
            # if not training_args.overwrite_output_dir:
            #     skip_batch = True
            #     for j in range(length):
            #         if not os.path.exists(os.path.join(training_args.output_dir, "{:0>8}.pt".format(index_to_get[j]))):
            #             skip_batch = False
            #             break
            #     if skip_batch:
            #         logger.debug("Skipping batch {} as all items has been processed".format(i))
            #         continue
            samples = train_dataset[index_to_get]
            model_input = tokenizer(samples['text'], return_tensors='pt', padding=True)
            model_input = model_input.to("cuda")
            out = teacher_model(**model_input, 
                                return_dict=True,
                                output_hidden_states=True,
                                logits_to_keep=0)

            for j in range(length):
                seq_len = torch.sum(model_input["attention_mask"][j])
                seq_logit = out.logits[[j], :seq_len, :]
                seq_hidden_state = out.hidden_states[[j], :seq_len, :]
                id = index_to_get[j]
                print(seq_len, model_input["input_ids"][j].shape, seq_hidden_state.shape, id, samples["text"][j])
                # save_file = os.path.join(training_args.output_dir, "{:0>8}.pt".format(id))
                # torch.save(seq_hidden_state.detach().to("cpu"), save_file)

    ###############
    # Training loop
    ###############

    

    # logger.info("*** Train ***")
    # checkpoint = None
    # if training_args.resume_from_checkpoint is not None:
    #     checkpoint = training_args.resume_from_checkpoint
    # elif last_checkpoint is not None:
    #     checkpoint = last_checkpoint
    # train_result = trainer.train(resume_from_checkpoint=checkpoint)
    # metrics = train_result.metrics
    # metrics["train_samples"] = len(train_dataset)
    # trainer.log_metrics("train", metrics)
    # trainer.save_metrics("train", metrics)
    # trainer.save_state()

    # ##################################
    # # Save model and create model card
    # ##################################
    # logger.info("*** Save model ***")
    # trainer.save_model(training_args.output_dir)
    # logger.info(f"Model saved to {training_args.output_dir}")

    # # Save everything else on main process
    # kwargs = {
    #     "model_name": model_args.model_name_or_path,
    #     # "dataset": list(data_args.dataset_mixer.keys()),
    #     # "dataset_tags": list(data_args.dataset_mixer.keys()),
    #     "tags": ["alignment-handbook"],
    # }
    # if trainer.accelerator.is_main_process:
    #     # trainer.create_model_card(**kwargs)
    #     super(KDTrainer, trainer).create_model_card(**kwargs)
    #     # Restore k,v cache for fast inference
    #     trainer.model.config.use_cache = True
    #     trainer.model.config.save_pretrained(training_args.output_dir)

    # ##########
    # # Evaluate
    # ##########
    # if training_args.do_eval:
    #     logger.info("*** Evaluate ***")
    #     metrics = trainer.evaluate()
    #     metrics["eval_samples"] = len(eval_dataset)
    #     trainer.log_metrics("eval", metrics)
    #     trainer.save_metrics("eval", metrics)

    # if training_args.push_to_hub is True:
    #     logger.info("Pushing to hub...")
    #     trainer.push_to_hub(**kwargs)

    # logger.info("*** Training complete ***")


if __name__ == "__main__":
    main()
