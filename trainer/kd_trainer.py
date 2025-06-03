import os
from dataclasses import dataclass
import warnings
from copy import deepcopy
from typing import Any, Callable, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from accelerate.utils import is_deepspeed_available
from transformers import AutoModelForCausalLM, PreTrainedModel, is_wandb_available
from transformers.data.data_collator import DataCollatorMixin

from trl.models import PreTrainedModelWrapper
from trl.trainer.sft_trainer import SFTTrainer
from trl.trainer.utils import pad

from accelerate import AutocastKwargs

from train_configs import SFTDistillConfig

if is_deepspeed_available():
    import deepspeed

if is_wandb_available():
    import wandb

@dataclass
class DataCollatorForLanguageModelingWithHiddenStates(DataCollatorMixin):
    """
    Data collator used for language modeling data. Inputs are dynamically padded to the maximum length of a batch if
    they are not all of the same length.

    Args:
        pad_token_id (`int`):
            Token ID to use for padding.
        return_tensors (`str`, *optional*, defaults to `"pt"`):
            Type of Tensor to return. Only `"pt"` is currently supported.

    Examples:
    ```python
    >>> from trl import DataCollatorForLanguageModeling
    >>> collator = DataCollatorForLanguageModeling(pad_token_id=0)
    >>> examples = [
    ...     {"input_ids": [1, 2, 3]},
    ...     {"input_ids": [4, 5]}
    ... ]
    >>> collator(examples)
    {'input_ids': tensor([[   1,   2,   3],
                          [   4,   5,   0]]),
     'attention_mask': tensor([[  1,   1,   1],
                               [  1,   1,   0]]),
     'labels': tensor([[   1,    2,    3],
                       [   4,    5, -100]])}
    ```
    """

    pad_token_id: int
    return_tensors: str = "pt"

    def torch_call(self, examples: list[Union[list[int], Any, dict[str, Any]]]) -> dict[str, Any]:
        # Convert to tensor
        input_ids = [torch.tensor(example["input_ids"]) for example in examples]
        attention_mask = [torch.ones_like(input_ids) for input_ids in input_ids]
        labels = [torch.tensor(example["input_ids"]) for example in examples]
        hidden_states = [example["hidden_states"] for example in examples]

        # Pad
        output = {}
        output["input_ids"] = pad(input_ids, padding_value=self.pad_token_id, padding_side="right")
        output["attention_mask"] = pad(attention_mask, padding_value=0, padding_side="right")
        output["labels"] = pad(labels, padding_value=-100, padding_side="right")
        output["hidden_states"] = pad(hidden_states, padding_value=0, padding_side="right")

        return output

class KDTrainer(SFTTrainer):
    _tag_names = ["trl", "kd"]

    def __init__(
        self,
        teacher_model: Union[PreTrainedModel, nn.Module, str],
        args: Optional[SFTDistillConfig] = None,
        *sft_args,
        **kwargs,
    ):

        super().__init__(*sft_args, args=args, **kwargs)

        if args.teacher_model_init_kwargs is None:
            teacher_model_init_kwargs = {}
        elif not isinstance(teacher_model, str):
            raise ValueError(
                "You passed teacher_model_init_kwargs to the KDConfig, but your teacher_model is already instantiated."
            )
        else:
            teacher_model_init_kwargs = args.teacher_model_init_kwargs
            # teacher_model_init_kwargs["torch_dtype"] = (
            #     teacher_model_init_kwargs["torch_dtype"]
            #     if teacher_model_init_kwargs["torch_dtype"] in ["auto", None]
            #     else getattr(torch, teacher_model_init_kwargs["torch_dtype"])
            # )

        if isinstance(teacher_model, str):
            warnings.warn(
                "You passed a teacher model_id to the KDTrainer. This will automatically create an "
                "`AutoModelForCausalLM`"
            )
            print("teacher_model_init_kwargs:", teacher_model_init_kwargs)
            teacher_model = AutoModelForCausalLM.from_pretrained(teacher_model, **teacher_model_init_kwargs)

        if self.is_deepspeed_enabled:
            self.teacher_model = self._prepare_deepspeed(teacher_model)
        else:
            self.teacher_model = self.accelerator.prepare_model(teacher_model, evaluation_mode=True)

        self.kl_weight = args.kl_weight
        self.ce_weight = args.ce_weight

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # compute teacher output in eval mode
        self.teacher_model.eval()
        with torch.no_grad():
            outputs_teacher = self.teacher_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                labels=inputs["labels"],
            )
        
        # compute student output
        teacher_logits = outputs_teacher.logits.detach()
        if self.ce_weight == 0:
            outputs_student = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"]
            )
            cross_entropy_loss = 0.0
        else:
            # compute student output
            outputs_student = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                labels=inputs["labels"],
            )
            cross_entropy_loss = outputs_student.loss
        # compute cross entropy loss
        student_logits = outputs_student.logits
        kl_loss = F.kl_div(F.log_softmax(student_logits, dim=-1), F.softmax(teacher_logits, dim=-1), reduction='batchmean')
        loss = self.kl_weight * kl_loss + self.ce_weight * cross_entropy_loss
        # Return loss
        return (loss, outputs_student) if return_outputs else loss

    def _prepare_deepspeed(self, model: PreTrainedModelWrapper):
        # Adapted from accelerate: https://github.com/huggingface/accelerate/blob/739b135f8367becb67ffaada12fe76e3aa60fefd/src/accelerate/accelerator.py#L1473
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        config_kwargs = deepcopy(deepspeed_plugin.deepspeed_config)

        if model is not None:
            if hasattr(model, "config"):
                hidden_size = (
                    max(model.config.hidden_sizes)
                    if getattr(model.config, "hidden_sizes", None)
                    else getattr(model.config, "hidden_size", None)
                )
                if hidden_size is not None and config_kwargs["zero_optimization"]["stage"] == 3:
                    # Note that `stage3_prefetch_bucket_size` can produce DeepSpeed messages like: `Invalidate trace cache @ step 0: expected module 1, but got module 0`
                    # This is expected and is not an error, see: https://github.com/microsoft/DeepSpeed/discussions/4081
                    config_kwargs.update(
                        {
                            "zero_optimization.reduce_bucket_size": hidden_size * hidden_size,
                            "zero_optimization.stage3_param_persistence_threshold": 10 * hidden_size,
                            "zero_optimization.stage3_prefetch_bucket_size": 0.9 * hidden_size * hidden_size,
                        }
                    )

        # If ZeRO-3 is used, we shard both the active and reference model.
        # Otherwise, we assume the reference model fits in memory and is initialized on each device with ZeRO disabled (stage 0)
        if config_kwargs["zero_optimization"]["stage"] != 3:
            config_kwargs["zero_optimization"]["stage"] = 0
        
        # if you get an OOM error because of a super large teacher model, consider to try this
        # config_kwargs['zero_optimization']['offload_param'] = {
        #     'device': 'cpu',  # Offload parameters to CPU
        #     'pin_memory': True,  # Enable pinned memory for faster CPU-GPU transfers
        #     'nvme_path': None  # No NVMe offloading
        # }
        # config_kwargs["optimizer"] = {"type": None}
        # print("config_kwargs:", config_kwargs)

        model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
        model.eval()
        return model


class KDTrainerOffline(SFTTrainer):
    _tag_names = ["trl", "kd"]

    def __init__(
        self,
        teacher_model: Union[PreTrainedModel, nn.Module, str],
        data_collator=None,
        args: Optional[SFTDistillConfig] = None,
        *sft_args,
        **kwargs,
    ):

        super().__init__(*sft_args, data_collator=data_collator, args=args, **kwargs)

        if args.teacher_model_init_kwargs is None:
            teacher_model_init_kwargs = {}
        elif not isinstance(teacher_model, str):
            raise ValueError(
                "You passed teacher_model_init_kwargs to the KDConfig, but your teacher_model is already instantiated."
            )
        else:
            teacher_model_init_kwargs = args.teacher_model_init_kwargs
            # teacher_model_init_kwargs["torch_dtype"] = (
            #     teacher_model_init_kwargs["torch_dtype"]
            #     if teacher_model_init_kwargs["torch_dtype"] in ["auto", None]
            #     else getattr(torch, teacher_model_init_kwargs["torch_dtype"])
            # )
        
        if isinstance(teacher_model, str):
            warnings.warn(
                "You passed a teacher model_id to the KDTrainer. This will automatically create an "
                "`AutoModelForCausalLM`"
            )
            print("teacher_model_init_kwargs:", teacher_model_init_kwargs)
            teacher_model = AutoModelForCausalLM.from_pretrained(teacher_model, **teacher_model_init_kwargs)

        if self.is_deepspeed_enabled:
            self.teacher_model = self._prepare_deepspeed(teacher_model)
        else:
            self.teacher_model = self.accelerator.prepare_model(teacher_model, evaluation_mode=True)

        self.kl_weight = args.kl_weight
        self.ce_weight = args.ce_weight
    
    def _prepare_dataset(
        self,
        dataset,
        processing_class,
        args,
        packing,
        formatting_func,
        dataset_name,
    ):
        dataset = super()._prepare_dataset(dataset, processing_class, args, packing, formatting_func, dataset_name)
        # adding transform func to not be cached 
        # Apply set transform to fetch the hidden states lazily
        def fetch_hidden_state(example):
            hidden_states = []
            for idx in example['idx']:
                hidden_state_path = "output/MIL-3B-hs-5percent/{:0>8}.pt".format(idx)
                hidden_state = torch.load(hidden_state_path)
                if args.max_length is not None:
                    if hidden_state.shape[1] > args.max_length:
                        hidden_state = hidden_state[:, :args.max_length, :]
                hidden_states.append(hidden_state)
            example['hidden_states'] = hidden_states
            return example
        dataset.set_transform(fetch_hidden_state)
        return dataset

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # compute teacher output in eval mode
        self.teacher_model.eval()
        # with deepspeed.zero.GatheredParameters([self.teacher_model.model.lm_head.weight]), \
        with deepspeed.zero.GatheredParameters([self.teacher_model.weight]), \
            torch.autocast(device_type="cuda", dtype=torch.bfloat16), torch.no_grad():
            # print("weight", self.teacher_model.model.lm_head.weight.data)
            # print(inputs["hidden_states"])
            # teacher_logits = self.teacher_model.model.lm_head(
            #     inputs["hidden_states"]
            # ).detach()
            teacher_logits = self.teacher_model(
                inputs["hidden_states"]
            ).detach()
            # print(teacher_logits)
        # compute student output
        # teacher_logits = outputs_teacher.logits.detach()

        if self.ce_weight == 0:
            outputs_student = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"]
            )
            cross_entropy_loss = 0.0
        else:
            # compute student output
            outputs_student = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                labels=inputs["labels"],
            )
            cross_entropy_loss = outputs_student.loss
        # compute cross entropy loss
        student_logits = outputs_student.logits
        kl_loss = F.kl_div(F.log_softmax(student_logits, dim=-1), F.softmax(teacher_logits, dim=-1), reduction='batchmean')
        loss = self.kl_weight * kl_loss + self.ce_weight * cross_entropy_loss
        # Return loss
        return (loss, outputs_student) if return_outputs else loss

    def _prepare_deepspeed(self, model: PreTrainedModelWrapper):
        # Adapted from accelerate: https://github.com/huggingface/accelerate/blob/739b135f8367becb67ffaada12fe76e3aa60fefd/src/accelerate/accelerator.py#L1473
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        config_kwargs = deepcopy(deepspeed_plugin.deepspeed_config)

        if model is not None:
            if hasattr(model, "config"):
                hidden_size = (
                    max(model.config.hidden_sizes)
                    if getattr(model.config, "hidden_sizes", None)
                    else getattr(model.config, "hidden_size", None)
                )
                if hidden_size is not None and config_kwargs["zero_optimization"]["stage"] == 3:
                    # Note that `stage3_prefetch_bucket_size` can produce DeepSpeed messages like: `Invalidate trace cache @ step 0: expected module 1, but got module 0`
                    # This is expected and is not an error, see: https://github.com/microsoft/DeepSpeed/discussions/4081
                    config_kwargs.update(
                        {
                            "zero_optimization.reduce_bucket_size": hidden_size * hidden_size,
                            "zero_optimization.stage3_param_persistence_threshold": 10 * hidden_size,
                            "zero_optimization.stage3_prefetch_bucket_size": 0.9 * hidden_size * hidden_size,
                        }
                    )

        # If ZeRO-3 is used, we shard both the active and reference model.
        # Otherwise, we assume the reference model fits in memory and is initialized on each device with ZeRO disabled (stage 0)
        if config_kwargs["zero_optimization"]["stage"] != 3:
            config_kwargs["zero_optimization"]["stage"] = 0
        
        # if you get an OOM error because of a super large teacher model, consider to try this
        # config_kwargs['zero_optimization']['offload_param'] = {
        #     'device': 'cpu',  # Offload parameters to CPU
        #     'pin_memory': True,  # Enable pinned memory for faster CPU-GPU transfers
        #     'nvme_path': None  # No NVMe offloading
        # }
        # config_kwargs["optimizer"] = {"type": None}
        # print("config_kwargs:", config_kwargs)

        model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
        model.eval()
        return model
