from pathlib import Path
# from src.model.modelling_mistral_kv import MistralForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.utils.generation_utils import InferenceParams
import pickle
import torch
import tqdm


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='mistralai/Mistral-7B-Instruct-v0.2', type=str, help='hf model card path')
    parser.add_argument('--device', default=0,  type=int, help='device_map for model')
    parser.add_argument('--torch_dtype', default='bfloat16', help='dtype attribute as a str')
    parser.add_argument('--topk', default=None, help="top k token of the respective token is stored")
    parser.add_argument('--sos', default=True, help='use start of string before tokens')

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

        if save_path.joinpath('2gram_rankings.pth').exists():
            print('2 gram exists')
            exit()

        if not save_path.is_dir():
            save_path.mkdir(parents=True)
        else:
            print("save path exists")

        V = model.config.vocab_size
        if args.topk == None:
            args.topk = V
        if args.sos:
            x = torch.ones((V, 2)).long()
            x[:, 1] = torch.arange(V)
        else:
            x = torch.arange(V).reshape(V, -1)
        rankings =  - torch.ones((V, args.topk), dtype=torch.int64)

        BS = 1
        assert V % BS == 0

        Ncalls = V // BS

        base_ids = torch.arange(BS).to(model.device).reshape(BS,  1)

        arr = torch.arange(BS).to(model.device)

        inference_params = InferenceParams(10,1)
        inference_params.ndraft = 1
        for i in tqdm.tqdm(range(Ncalls), desc='vocab batch'):
            input_ids = x[i * BS: (i + 1) * BS, :].to(model.device)
            outs = model(input_ids=input_ids, inference_params=inference_params, use_cache=False)
            ranks = outs.logits[:, -1].topk(100).indices
            rankings[i * BS : (i + 1) * BS] = ranks.cpu()

        torch.save(rankings, save_path.joinpath('2gram_rankings.pth'))