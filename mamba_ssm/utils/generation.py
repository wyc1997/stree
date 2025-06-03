# Copyright (c) 2023, Albert Gu, Tri Dao.
from mamba_ssm.utils.beam_search import beam_search_decode
from mamba_ssm.utils.generation_utils import decode

class GenerationMixin:
    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        raise NotImplementedError

    def generate(
        self,
        input_ids,
        max_length,
        mask=None,
        top_k=1,
        top_p=0.0,
        min_p=0.0,
        num_beam=1, 
        temperature=1.0,
        return_dict_in_generate=False,
        output_scores=False,
        draft_num_beam=0,
        mask_type="padding",
        **kwargs,
    ):
        if num_beam != 1:
            output = beam_search_decode(
                input_ids, self, max_length, mask=mask, top_k=top_k, top_p=top_p, min_p=min_p, num_beam=num_beam, temperature=temperature, **kwargs
            )
        else:
            output = decode(
                input_ids, self, max_length, mask=mask, top_k=top_k, top_p=top_p, min_p = min_p, temperature=temperature, mask_type=mask_type, **kwargs
            )
        if not output_scores:
            output.scores = None
        return output if return_dict_in_generate else output.sequences
