import torch
from pathlib import Path


@torch.inference_mode
def forward_ngram_matcher(input_ids, id, Ndraft=1, N=2):
    # use unfold to obtain all N grams
    grams = input_ids.flatten().unfold(0, N, 1)
    # extract mask of matching ngrams
    mask = grams[:, 0] == id
    if torch.any(mask):
        matching_grams = grams[mask]
        # obtain counts of all ngrams
        matches, counts = torch.unique(matching_grams, dim=0, return_counts=True)
        Nfound = counts.size(0)
        Ntake = min(Ndraft, Nfound)
        # take up to top Ndraft occuring Ngrams
        most_freq_ids = counts.topk(Ntake).indices
        return matches[most_freq_ids]
    else:
        return None



    
# PNG = prompt N gram
class Rampdown:
    '''
    Prompt lookup and pad everything else with unigram model
    '''
    def __init__(self, start=25, finish=5, nsteps=256, model=None, device=None, **kwargs):
     
        model_name = model.config._name_or_path.split("/")[-1]
        # dtype_str = str(model.config.torch_dtype).split('.')[-1]
        dtype_str = str(model.dtype).split('.')[-1]
        self.ext_bigram = torch.load(
            Path('src/strategies/model2gram').joinpath(model_name, dtype_str, 'model_extended_2gram_rankings.pth')
        )
        self.nsteps = nsteps
        self.start=start
        self.finish=finish 
        self.count = 0

    def get_bs(self):
        if self.count < 0 or self.count > self.nsteps:
            raise ValueError("count cannot be negative or larger than finish")
        decrement_per_step = (self.start - self.finish) / self.nsteps

        Ndraft = self.start - (decrement_per_step * self.count)
        
        # Return the integer value, ensuring it decreases linearly
        return int(round(Ndraft))



    def update(self, input_block, output_block, last_word, output_ids, **kwargs):

        input_block[:, 0] = last_word
        output_block[:, 0] = last_word
        Ndraft= self.get_bs()
        Npad = input_block.size(1) - 1

        step_block = torch.narrow(input_block, dim=0, start=0, length=Ndraft)

        # obtain the matching grams
        matches = forward_ngram_matcher(output_ids, id=last_word, Ndraft=Ndraft, N=input_block.size(-1))
        if matches is None:
            ng, rs = (0, Ndraft)
        else:
            ng = matches.size(0)
            # residual space left
            rs = Ndraft - ng
            step_block[:ng, :] = matches
        if rs > 0:
            bi_ext = self.ext_bigram[last_word, :rs, :Npad].to(input_block.device)
            step_block[ng:, 1:]  = bi_ext
        self.count += 1
            
        return step_block

    # def update_(self, input_block, output_block, last_word, output_ids, **kwargs):

    #     input_block[:, 0] = last_word
    #     output_block[:, 0] = last_word
    #     Ndraft= input_block.size(0)
    #     Npad = input_block.size(1) - 1

    #     # obtain the matching grams
    #     matches = forward_ngram_matcher(output_ids, id=last_word, Ndraft=Ndraft, N=input_block.size(-1))
    #     draft_ids = []
    #     if matches is None:
    #         ng, rs = (0, Ndraft)
    #     else:
    #         ng = matches.size(0)
    #         rs = Ndraft - ng
    #         # residual space left
    #         input_block[:ng, :] = matches
    #         draft_ids += ['Prompt' for _ in range(ng)]
    #     if rs > 0:
    #         bi_ext = self.ext_bigram[last_word, :rs, :Npad].to(input_block.device)
    #         input_block[ng:, 1:]  = bi_ext
    #         draft_ids += ['ModelBigramModelExt' for _ in range(rs)]

    #     self.draft_ids = draft_ids
    #     self.count +=1
    
    #     return input_block
 

    def get_strat_keys_(self):
        return ['Prompt', 'ModelBigramModelExt']
