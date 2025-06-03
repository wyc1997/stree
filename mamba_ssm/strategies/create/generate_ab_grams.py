from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import pickle
import torch
import tqdm
from src.strategies.zerolayerunigram import ZeroLayerUnigram



def fuse_dicts(main_dict, sub_dict):
    for (k, v) in sub_dict.items():
        main_dict[k] = main_dict.get(k, []) + v
    return main_dict

def create_ABgram_dict(gen_data, a, L):
    '''
    Given a block of text, extracts the A matches
    with continuations of length L 
    '''
    sub_dict = {}
    for row in gen_data:
        key = tuple(row[:a].tolist())
        value = row[a:a + L].tolist()
        sub_dict[key] = sub_dict.get(key, []) + [value]
    return sub_dict

def sort_unique_by_count(x):
    if isinstance(x, list):
        x = torch.tensor(x)
    x = torch.unique() 


def post_process(main_dict):
    '''
    concatenates all values to torch tensors of size n_i x L
    '''
    return {k : torch.tensor(v) for (k, v) in main_dict.items()}





if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='mistralai/Mistral-7B-Instruct-v0.2', type=str, help='hf model card path')
    parser.add_argument('--device', default=0, help='device_map for model')
    parser.add_argument('--torch_dtype', default='bfloat16', help='dtype attribute as a str')
    parser.add_argument('--sos', default=True, help='use start of string before tokens')
    parser.add_argument('--L', default=5, type=int, help='length of gram continuation')
    parser.add_argument('--Nimp', default=None, type=int, help='restrict vocab size')
    parser.add_argument('--num_beam_groups', default=32, type=int, help='beam group for diversity, if equal to num_beams every nw continuation will be different')
    parser.add_argument('--num_beams', default=32, type=int, help='Number of grams to create for each word')
    parser.add_argument('--amax', default=2, type=int, help='maximum size to match with')
    parser.add_argument('--diversity_penalty', default=1e9, type=float, help='maximum size to match with')
    parser.add_argument('--BS', default=16, type=int, help='batch size for running Exp')
    parser.add_argument('--save_name', default=None,  help='name to save dictionaries with')


    args = parser.parse_args()
    if args.num_beam_groups != args.num_beams:
        assert False, 'Code not suited for this scenario as repeat tokens may cause loss of nw diversity so need to rearrange'


    dict_collection = {i : {} for i  in range(1, args.amax + 1)}

    with torch.inference_mode():

        DEVICE = args.device

        torch_dtype = getattr(torch, args.torch_dtype)
        model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map=DEVICE, torch_dtype=torch_dtype)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        model_id = args.model_path.split('/')[-1]
        strat = ZeroLayerUnigram(model)
        if args.Nimp is None:
            important_vocab = strat.unigram_head_ranking
        else:
            important_vocab = strat.unigram_head_ranking[:args.Nimp]

        BS = args.BS
        NB = len(important_vocab) // BS
        input_ids = torch.ones((BS, 2)).long().to(model.device) * tokenizer.bos_token_id
        max_length = args.amax + args.L

        PRE_PROMPT = '[INST] Complete the following text: [/INST]'
        pre_ids = tokenizer(PRE_PROMPT, return_tensors='pt')['input_ids'].to(model.device)
        prompt_len = pre_ids.size(-1)

        input_ids = torch.zeros((BS, prompt_len + 1)).long().to(model.device)
        input_ids[:, :prompt_len] = pre_ids.expand(BS, -1)

        for i in tqdm.tqdm(range(NB)):

            input_ids[:, -1] = important_vocab[i * BS : (i  + 1) * BS]
            outputs = model.generate(input_ids=input_ids, max_new_tokens=args.L + args.amax - 1,
                                     num_return_sequences=args.num_beams,
                                     num_beams=args.num_beams, num_beam_groups=args.num_beam_groups, early_stopping=False,
                                     diversity_penalty=args.diversity_penalty, do_sample=False, output_scores=True, return_dict_in_generate=True)

            # split into grams for each word (each word has args.num_beamgroups continuations) 
            for (grams, scores) in zip(outputs.sequences.split(args.num_beam_groups),
                                        outputs.sequences_scores.split(args.num_beam_groups)):
                # sort by rank 
                rank_perm = scores.argsort()
                # reorder by rank
                grams = grams[:, prompt_len:]
                grams = grams[rank_perm]

                # store a - > L lookups ordered by rank.
                for a, adict in dict_collection.items():
                    sub_dict = create_ABgram_dict(grams, a, args.L)
                    adict = fuse_dicts(main_dict=adict, sub_dict=sub_dict) 
    

        path = Path('src/strategies/model_ab_gram').joinpath(args.model_path.split('/')[1], str(torch_dtype))
        if args.save_name is not None:
            path = path.joinpath(args.save_name)
        if not path.is_dir():
            path.mkdir(parents=True)
        for k in dict_collection.keys():
            dict_collection[k] = {key : torch.tensor(value) for (key, value) in dict_collection[k].items()}
            suffix = f'{k}_{args.L}gram.pickle'

            with open(path.joinpath(suffix), 'wb') as f:
                pickle.dump(dict_collection[k], f)
