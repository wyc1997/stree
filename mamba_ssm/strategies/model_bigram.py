from .basestrat import PaddingStrat
import torch
from pathlib import Path

class ModelBigram(PaddingStrat):
    def __init__(self, model=None, **kwargs):
        with torch.inference_mode():
            model_name = model.config._name_or_path.split("/")[-1]
            dtype_str = str(model.dtype).split('.')[-1]
            self.rankings = torch.load(
                Path('src/strategies/model2gram').joinpath(model_name, dtype_str, '2gram_rankings.pth')
            )
        
    
    def update(self, input_block, output_block, last_word, **kwargs):
        Ndraft= input_block.size(0)
        input_block[:, 0] = last_word
        output_block[:, 0] = last_word

        bg = self.rankings[last_word][:Ndraft].reshape(Ndraft, 1)
        input_block[:, 1:] = bg.to(input_block.device)
        return input_block

    def update_(self, input_block, output_block, last_word, **kwargs):
        Ndraft= input_block.size(0)
        input_block[:, 0] = last_word
        output_block[:, 0] = last_word

        bg = self.rankings[last_word][:Ndraft].reshape(Ndraft, 1)
        input_block[:, 1:] = bg.to(input_block.device)
        if not hasattr(self, 'draft_ids'):
            self.draft_ids = ['ModelBigram' for _ in range(Ndraft)]
        return input_block

    def get_strat_keys_(self):
        return ['ModelBigram']



class ModelBigramModelExt(PaddingStrat):
    def __init__(self, model=None, **kwargs):
        with torch.inference_mode():
            model_name = model.config._name_or_path.split("/")[-1]
            # dtype_str = str(model.config.torch_dtype).split('.')[-1]
            dtype_str = str(model.dtype).split('.')[-1]
            self.rankings = torch.load(
                Path('src/strategies/model2gram').joinpath(model_name, dtype_str, 'model_extended_2gram_rankings.pth')
            )
        
    
    def update(self, input_block, output_block, last_word, **kwargs):
        Ndraft= input_block.size(0)
        Npad = input_block.size(1) - 1
        input_block[:, 0] = last_word
        output_block[:, 0] = last_word

        bg = self.rankings[last_word][:Ndraft, :Npad].reshape(Ndraft, Npad)
        input_block[:, 1:] = bg.to(input_block.device)
        return input_block

    def update_(self, input_block, output_block, last_word, **kwargs):
        Ndraft= input_block.size(0)
        Npad = input_block.size(1) - 1
        input_block[:, 0] = last_word
        output_block[:, 0] = last_word

        bg = self.rankings[last_word][:Ndraft, :Npad].reshape(Ndraft, Npad)
        input_block[:, 1:] = bg.to(input_block.device)
        if not hasattr(self, 'draft_ids'):
            self.draft_ids = ['ModelBigramModelExt' for _ in range(Ndraft)]
        return input_block

    def get_strat_keys_(self):
        return ['ModelBigramModelExt']