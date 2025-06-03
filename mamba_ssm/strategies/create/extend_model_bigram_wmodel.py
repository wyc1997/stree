from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import pickle
import torch
import tqdm


if __name__ == '__main__':

    import argparse
    from transformers import logging
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='mistralai/Mistral-7B-Instruct-v0.2', type=str, help='hf model card path')
    parser.add_argument('--device', default=0, type=int, help='device_map for model')
    parser.add_argument('--torch_dtype', default='bfloat16', help='dtype attribute as a str')
    parser.add_argument('--sos', default=True, help='use start of string before tokens')
    parser.add_argument('--L', default=5, type=int, help='(1, L) gram extension')
    parser.add_argument('--Ndraft', default=30, type=int, help='(1, L) gram extension')
    logging.set_verbosity_warning()

    args = parser.parse_args()

    with torch.inference_mode():

        DEVICE = args.device

        torch_dtype = getattr(torch, args.torch_dtype)
        model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map=DEVICE, torch_dtype=torch_dtype, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        if args.model_path == 'lmsys/vicuna-13b-v1.3':
            chat_template = open('vicuna.jinja').read()
            chat_template = chat_template.replace('    ', '').replace('\n', '')
            tokenizer.chat_template = chat_template

        model_id = args.model_path.split('/')[-1]
        save_path = Path('src/strategies').joinpath('model2gram').joinpath(model_id).joinpath(args.torch_dtype)
        V = model.config.vocab_size

        tpath = save_path.joinpath('2gram_rankings.pth')
        assert tpath.exists(), 'need 2 gram ranking to exist'

        bigram = torch.load(tpath)

        continuations = -torch.ones((V, args.Ndraft, args.L + 1)).long()
        continuations[:, :, 0] = torch.arange(V).reshape(-1, 1)
        continuations[:, :, 1] = bigram[:, :args.Ndraft]

        # PRE_PROMPT = '[INST] Complete the following text: [/INST]'
        chat = []
        chat.append({'role' : 'user', 'content' : 'Complete the following text:'})
        pre_ids = tokenizer.apply_chat_template(chat, return_tensors='pt').to(model.device)
        # pre_ids = tokenizer(PRE_PROMPT, return_tensors='pt')['input_ids'].to(model.device)
        prompt_len = pre_ids.size(-1) + 1
        pre_ids = pre_ids.expand(args.Ndraft, -1)


        # for phi3 we saw that 19251 was problematic
        problem_ids = [] 
        for i in tqdm.tqdm(range(V), desc='vocab'):

            input_ids = torch.cat((pre_ids, continuations[i, :, :2].to(model.device)), dim=-1)
            outs = model.generate(input_ids=input_ids, do_sample=False, max_new_tokens=args.L - 1, early_stopping=False, min_length=args.L - 1)
            extension = outs[:, prompt_len + 1:].to(continuations.device)
            try:
                continuations[i, :, 2:] = extension
            except:
                # for some reason on phi we get bugs on ids, and then when debugging they no longer occur.
                es = extension.size(-1)
                continuations[i, :, 2: 2 + es]  = extension
                continuations[i, : , 2 + es:] = continuations[i, :, 1:2]
                problem_ids.append(i)

            
            # continuations[i, :, 2:extension.size(-1)] = extension
            # for j in range(args.L - 2):
            #     nw_ids = bigram[continuations[i, :, j + 1]][:, 0]
            #     continuations[i, :, j + 2] = nw_ids
        try:
            torch.save(continuations[:, :, 1:], save_path.joinpath('model_extended_2gram_rankings.pth'))
        except:
            print('save problem')
            import ipdb; ipdb.set_trace()
    print('generation finished')

        # for i in tqdm.tqdm(range(300), desc='word id'):
        #     for j in range(args.L - 1):
        #         continuations[]
        # #     input_ids = continuations[i, :, :2].to(model.device)
        # #     greedy_preds = model.generate(input_ids=input_ids, do_sample=False, max_new_tokens=args.L-1)
        # #     continuations[i, :, 2:] = greedy_preds[:, 2:].cpu()
        # # import ipdb; ipdb.set_trace()

        
        