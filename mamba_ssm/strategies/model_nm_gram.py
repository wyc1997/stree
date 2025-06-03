from .basestrat import PaddingStrat
import torch
from pathlib import Path
import pickle

# class Gram_1_5_ZeroLayerUnigram(PaddingStrat):
#     def __init__(self, model=None, device=None, **kwargs):
#         with torch.inference_mode():
#             model_name = model.config._name_or_path.split("/")[-1]
#             dtype_str = str(model.config.torch_dtype).split('.')[-1]
#             path = Path(f'src/strategies/model_ab_gram/{model_name}/1_5gram.pickle')
#             assert path.is_file(), 'file doesnt exist'
#             with open(path, 'rb') as f:
#                 self.lookup = pickle.load(f)
            
#         if device is None:
#             device=model.device
#         if hasattr(self,'unigram_head_ranking'):
#             return self.unigram_head_ranking
#         else:
#             Wenc = model.get_input_embeddings().weight.detach().to(device)
#             covV = Wenc.T @ Wenc / Wenc.shape[0]
#             Wdec = model.lm_head.weight.detach().to(device)

#             # look at the distance from the mean embedding in the decoder space.
#             # distance is defined by a Kernel.
#             mu = Wdec.mean(dim=0, keepdim=True)
#             dists = mu @ covV @ Wdec.T
#             dists = dists.squeeze()
#             ranks = torch.topk(-dists, k=dists.size(0)).indices
#             self.unigram_head_ranking = ranks
    
#     def update(self, input_block, output_block, last_word, **kwargs):
#         Ndraft= input_block.size(0)
#         input_block[:, 0] = last_word
#         output_block[:, 0] = last_word

#         bg = self.rankings[last_word][:Ndraft].reshape(Ndraft, 1)
#         input_block[:, 1:] = bg.to(input_block.device)


class OneLGram_ModelBigramModelExt(PaddingStrat):
    def __init__(self, model=None, **kwargs):
        with torch.inference_mode():
            model_name = model.config._name_or_path.split("/")[-1]
            dtype_str = str(model.config.torch_dtype).split('.')[-1]
            self.rankings = torch.load(
                Path('src/strategies/model2gram').joinpath(model_name, dtype_str, 'model_extended_2gram_rankings.pth')
            )
            gram_path = Path('src/strategies/model_ab_gram').joinpath(model_name, dtype_str, '6400NI', '1_10gram.pickle')

            with open(gram_path, 'rb') as f:
                self.lookup = pickle.load(f)
        
    
    def update(self, input_block, output_block, last_word,  **kwargs):
        key = (last_word.item(),)
        matches = self.lookup.get(key, None)

        Ndraft= input_block.size(0)
        input_block[:, 0] = last_word
        output_block[:, 0] = last_word
        Npad = input_block.size(1) - 1

        if matches is not None:
            Nfound = matches.size(0)
            Nfound = min(Ndraft, Nfound)
            input_block[:Nfound, 1:] = matches[:Nfound, :Npad].to(input_block.device)
            rs = Ndraft - Nfound

            if rs > 0:
                bg = self.rankings[last_word][:rs, :Npad].reshape(rs, Npad)

                # bg = self.rankings[last_word][:rs].reshape(rs, 1)
                input_block[Nfound: , 1:] = bg
        
        else:
            bg = self.rankings[last_word][:Ndraft, :Npad].reshape(Ndraft, Npad)
            input_block[:, 1:] = bg.to(input_block.device)
        

    def update_(self, input_block, output_block, last_word, **kwargs):
        key = (last_word.item(), )
        matches = self.lookup.get(key, None)

        Ndraft= input_block.size(0)
        input_block[:, 0] = last_word
        output_block[:, 0] = last_word
        Npad = input_block.size(1) - 1

        if matches is not None:
            Nfound = matches.size(0)
            Nfound = min(Ndraft, Nfound)
            input_block[:Nfound, 1:] = matches[:Nfound, :Npad].to(input_block.device)
            rs = Ndraft - Nfound
            self.draft_ids = ['1LGram_6400NI' for _ in range(Nfound)]

            if rs > 0:
                bg = self.rankings[last_word][:rs, :Npad].reshape(rs, Npad)

                # bg = self.rankings[last_word][:rs].reshape(rs, 1)
                input_block[Nfound: , 1:] = bg
                self.draft_ids += ['ModelBigramModelExt' for _ in range(rs)]
        
        else:
            bg = self.rankings[last_word][:Ndraft, :Npad].reshape(Ndraft, Npad)
            input_block[:, 1:] = bg.to(input_block.device)
            self.draft_ids = ['ModelBigramModelExt' for _ in range(Ndraft)]
        

    def get_strat_keys_(self):
        return ['ModelBigramModelExt', '1LGram_6400NI']




class TwoLGram_ModelBigramModelExt(PaddingStrat):
    def __init__(self, model=None, **kwargs):
        with torch.inference_mode():
            model_name = model.config._name_or_path.split("/")[-1]
            dtype_str = str(model.config.torch_dtype).split('.')[-1]
            self.rankings = torch.load(
                Path('src/strategies/model2gram').joinpath(model_name, dtype_str, 'model_extended_2gram_rankings.pth')
            )
            gram_path = Path('src/strategies/model_ab_gram').joinpath(model_name, dtype_str, '6400NI', '2_10gram.pickle')

            with open(gram_path, 'rb') as f:
                self.lookup = pickle.load(f)
        
    
    def update(self, input_block, output_block, last_word, output_ids, **kwargs):
    
        key = (output_ids[-1].item(), last_word.item())
        matches = self.lookup.get(key, None)

        Ndraft= input_block.size(0)
        input_block[:, 0] = last_word
        output_block[:, 0] = last_word
        Npad = input_block.size(1) - 1

        if matches is not None:

            Nfound = matches.size(0)
            Nfound = min(Ndraft, Nfound)
            input_block[:Nfound, 1:] = matches[:Nfound, :Npad].to(input_block.device)
            rs = Ndraft - Nfound

            if rs > 0:
                bg = self.rankings[last_word][:rs, :Npad].reshape(rs, Npad)

                # bg = self.rankings[last_word][:rs].reshape(rs, 1)
                input_block[Nfound: , 1:] = bg
        
        else:
            bg = self.rankings[last_word][:Ndraft, :Npad].reshape(Ndraft, Npad)
            input_block[:, 1:] = bg.to(input_block.device)
        

    def update_(self, input_block, output_block, last_word, output_ids, **kwargs):
        key = (output_ids[-1].item(), last_word.item())
        matches = self.lookup.get(key, None)

        Ndraft= input_block.size(0)
        input_block[:, 0] = last_word
        output_block[:, 0] = last_word
        Npad = input_block.size(1) - 1

        if matches is not None:
            Nfound = matches.size(0)
            Nfound = min(Ndraft, Nfound)
            input_block[:Nfound, 1:] = matches[:Nfound, :Npad].to(input_block.device)
            rs = Ndraft - Nfound
            self.draft_ids = ['2LGram_6400NI' for _ in range(Nfound)]

            if rs > 0:
                bg = self.rankings[last_word][:rs, :Npad].reshape(rs, Npad)

                # bg = self.rankings[last_word][:rs].reshape(rs, 1)
                input_block[Nfound: , 1:] = bg
                self.draft_ids += ['ModelBigramModelExt' for _ in range(rs)]
        
        else:
            bg = self.rankings[last_word][:Ndraft, :Npad].reshape(Ndraft, Npad)
            input_block[:, 1:] = bg.to(input_block.device)
            self.draft_ids = ['ModelBigramModelExt' for _ in range(Ndraft)]
        

    def get_strat_keys_(self):
        return ['ModelBigramModelExt', '2LGram_6400NI']



class Gram_2_5_ModelBigram(PaddingStrat):
    def __init__(self, model=None, device=None, **kwargs):
        with torch.inference_mode():
            model_name = model.config._name_or_path.split("/")[-1]
            dtype_str = str(model.config.torch_dtype).split('.')[-1]
            path = Path(f'src/strategies/model_ab_gram/{model_name}/2_5gram.pickle')
            assert path.is_file(), 'file doesnt exist'
            with open(path, 'rb') as f:
                self.lookup = pickle.load(f)
            
            self.rankings = torch.load(
                Path('src/strategies/model2gram').joinpath(model_name, dtype_str, '2gram_rankings.pth')
            )


    # WARNING. CAN GET REPEATS
    def update(self, input_block, output_block, last_word, output_ids, **kwargs):
        key = (output_ids[-1].item(), last_word.item())

        matches = self.lookup.get(key, None)
        Ndraft= input_block.size(0)
        input_block[:, 0] = last_word
        output_block[:, 0] = last_word
        Npad = input_block.size(1) - 1

        if matches is not None:
            Nfound = matches.size(0)
            Nfound = min(Ndraft, Nfound)
            input_block[:Nfound, 1:] = matches[:Nfound, :Npad].to(input_block.device)
            rs = Ndraft - Nfound

            # NOT OPTIMIZED : TO DO MAKE RECURSIVE
            if rs > 0:
                bg = self.rankings[last_word][:rs].reshape(rs, 1)
                input_block[Nfound: , 1:] = bg
        
        else:
            bg = self.rankings[last_word][:Ndraft].reshape(Ndraft, 1)
            input_block[:, 1:] = bg.to(input_block.device)
        

