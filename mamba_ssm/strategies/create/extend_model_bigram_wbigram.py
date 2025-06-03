from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.utils.generation_utils import InferenceParams
import pickle
import torch
import tqdm


if __name__ == '__main__':

    import argparse
    from transformers import logging
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='mistralai/Mistral-7B-Instruct-v0.2', type=str, help='hf model card path')
    parser.add_argument('--device', default=0, help='device_map for model')
    parser.add_argument('--torch_dtype', default='bfloat16', help='dtype attribute as a str')
    parser.add_argument('--sos', default=True, help='use start of string before tokens')
    parser.add_argument('--L', default=5, type=int, help='(1, L) gram extension')
    parser.add_argument('--Ndraft', default=32, type=int, help='(1, L) gram extension')
    logging.set_verbosity_warning()

    args = parser.parse_args()

    with torch.inference_mode():

        DEVICE = args.device
        dtype = torch.float16 if args.torch_dtype == "float16" else torch.bfloat16

        torch_dtype = getattr(torch, args.torch_dtype)
        if args.model_path == "JunxiongWang/Llama3.2-Mamba2-3B-distill":
            import sys, os
            sys.path.insert(0, "../MambaInLlama/")
            from mamba2_inference.hybrid_wrapper import MambaTransformerHybridModelWrapper
            tokenizer = AutoTokenizer.from_pretrained(args.model_path)
            model = MambaTransformerHybridModelWrapper.from_pretrained(args.model_path, torch_dtype=dtype)
        else:
            model = MambaLMHeadModel.from_pretrained(args.model_path, device=DEVICE, dtype=dtype)
            model.device = DEVICE
            tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        model_id = args.model_path.split('/')[-1]
        save_path = Path('mamba_ssm/strategies').joinpath('model2gram').joinpath(model_id).joinpath(args.torch_dtype)
        V = model.config.vocab_size

        tpath = save_path.joinpath('2gram_rankings.pth')
        assert tpath.exists(), 'need 2 gram ranking to exist'

        bigram = torch.load(tpath)

        continuations = -torch.ones((V, args.Ndraft, args.L)).long()
        continuations[:, :, 0] = torch.arange(V).reshape(-1, 1)
        continuations[:, :, 1] = bigram[:, :args.Ndraft]

        for i in tqdm.tqdm(range(V), desc='vocab'):
            for j in range(args.L - 2):
                nw_ids = bigram[continuations[i, :, j + 1]][:, 0]
                continuations[i, :, j + 2] = nw_ids
        torch.save(continuations[:, :, 1:], save_path.joinpath('extended_2gram_rankings.pth'))
    print('generation finished')

        # for i in tqdm.tqdm(range(300), desc='word id'):
        #     for j in range(args.L - 1):
        #         continuations[]
        # #     input_ids = continuations[i, :, :2].to(model.device)
        # #     greedy_preds = model.generate(input_ids=input_ids, do_sample=False, max_new_tokens=args.L-1)
        # #     continuations[i, :, 2:] = greedy_preds[:, 2:].cpu()
        # # import ipdb; ipdb.set_trace()

        
        