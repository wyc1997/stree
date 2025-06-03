from pathlib import Path
from src.model.modelling_mistral_kv import MistralForCausalLM
from transformers import AutoTokenizer
import pickle
import torch
import tqdm


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='mistralai/Mistral-7B-Instruct-v0.2', type=str, help='hf model card path')
    parser.add_argument('--device', default=0, help='device_map for model')
    parser.add_argument('--torch_dtype', default='bfloat16', help='dtype attribute as a str')
    parser.add_argument('--sos', default=True, help='use start of string before tokens')

    args = parser.parse_args()

    with torch.inference_mode():

        DEVICE = args.device

        torch_dtype = getattr(torch, args.torch_dtype)
        model = MistralForCausalLM.from_pretrained(args.model_path, device_map=DEVICE, torch_dtype=torch_dtype)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        model_id = args.model_path.split('/')[-1]
        save_path = Path('src/strategies').joinpath('model2gram').joinpath(model_id).joinpath(args.torch_dtype)

        if save_path.joinpath('2gram_heatmap.pth').exists():
            print('2 gram heatmap exists')
            exit()

        if not save_path.is_dir():
            save_path.mkdir(parents=True)

        V = model.config.vocab_size
        if args.sos:
            x = torch.ones((V, 2)).long()
            x[:, 1] = torch.arange(V)
        else:
            x = torch.arange(V).reshape(V, -1)
        heatmap =  - torch.ones((V, V), dtype=torch.float32)

        BS = 320

        assert V % BS == 0

        Ncalls = V // BS

        base_ids = torch.arange(BS).to(model.device).reshape(BS,  1)

        arr = torch.arange(BS).to(model.device)

        for i in tqdm.tqdm(range(Ncalls), desc='vocab batch'):
            input_ids = x[i * BS: (i + 1) * BS, :].to(model.device)
            outs = model(input_ids=input_ids, use_cache=False)
            
            heatmap[i * BS : (i + 1) * BS] = outs.logits[:, -1].cpu()

        torch.save(heatmap, save_path.joinpath('2gram_heatmap.pth'))