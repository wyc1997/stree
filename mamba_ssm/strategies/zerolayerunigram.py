from .basestrat import PaddingStrat
import torch

class ZeroLayerUnigram(PaddingStrat):
    def __init__(self, model=None, device=None, **kwargs):
        if device is None:
            device=model.device
        if hasattr(self,'unigram_head_ranking'):
            return self.unigram_head_ranking
        else:
            Wenc = model.get_input_embeddings().weight.detach().to(device)
            covV = Wenc.T @ Wenc / Wenc.shape[0]
            Wdec = model.lm_head.weight.detach().to(device)

            # look at the distance from the mean embedding in the decoder space.
            # distance is defined by a Kernel.
            mu = Wdec.mean(dim=0, keepdim=True)
            dists = mu @ covV @ Wdec.T
            dists = dists.squeeze()
            ranks = torch.topk(-dists, k=dists.size(0)).indices
            self.unigram_head_ranking = ranks.cpu()

    def update(self, input_block, output_block, last_word, **kwargs):
        input_block[:, 0] = last_word
        output_block[:, 0] = last_word
        Ndraft= input_block.size(0)
        input_block[:, 1:] = self.unigram_head_ranking[:Ndraft].reshape(-1, 1)
        return input_block


    def update_(self, input_block, output_block, last_word, **kwargs):
        input_block[:, 0] = last_word
        output_block[:, 0] = last_word
        Ndraft= input_block.size(0)
        input_block[:, 1:] = self.unigram_head_ranking[:Ndraft].reshape(-1, 1)
        if not hasattr(self, 'draft_ids'):
            self.draft_ids = ['ZeroLayerUnigram' for _ in range(Ndraft)]
        return input_block

    def get_strat_keys_(self):
        return ['ZeroLayerUnigram']



