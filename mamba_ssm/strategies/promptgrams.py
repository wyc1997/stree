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


@torch.inference_mode
def context_ngram_matcher(context, query, Ndraft=1, Npad=2):
    '''
    matches query of any length Q to grams of size Q + Npad

    query : tensor (query to match with)
    Ndraft : int (number of drafts)
    Npad : int (number of tokens to speculate with)
    '''
    # obtain length of query
    Q = query.size(-1)
    # use unfold to obtain all N grams
    grams = context.flatten().unfold(0, Q + Npad, 1)

    # extract mask of matching ngrams
    mask = torch.all(grams[:, :Q] == query, dim=-1)
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


class Debugger:
    '''
    Context n gram matching as priority,
    defaulting back to model ext bigram for remaining drafts
    '''
    def __init__(self, model=None, device=None, query_size=1, **kwargs):
        model_name = model.config._name_or_path.split("/")[-1]
        # dtype_str = str(model.config.torch_dtype).split('.')[-1]
        dtype_str = str(model.dtype).split('.')[-1]
        self.ext_bigram = torch.load(
            Path('src/strategies/model2gram').joinpath(model_name, dtype_str, 'model_extended_2gram_rankings.pth')
        )
        self.query_size=query_size


    def update(self, input_block, output_block, last_word, context, **kwargs):

        input_block[:, 0] = last_word
        output_block[:, 0] = last_word
        Ndraft= input_block.size(0)
        Npad = input_block.size(1) - 1

        # obtain the matching grams
        matches = context_ngram_matcher(context=context, query=context[-self.query_size:], Ndraft=Ndraft, Npad=Npad)
        matches_ = forward_ngram_matcher(input_ids=context, id=last_word, Ndraft=Ndraft, N=input_block.size(1))

        input_block_ = torch.clone(input_block)

        if matches is not None:
            if not torch.all(matches_ == matches):
                print('mismatch')
                import ipdb; ipdb.set_trace()
        
        if matches is None:
            ng, rs = (0, Ndraft)
        else:
            ng = matches.size(0)
            rs = Ndraft - ng
            # residual space left
            input_block[:ng, 1:] = matches[:, self.query_size:]
        if rs > 0:
            bi_ext = self.ext_bigram[last_word, :rs, :Npad].to(input_block.device)
            input_block[ng:, 1:]  = bi_ext


        # -------------------------------- check against old update ----------------------- 
        if matches_ is None:
            ng, rs = (0, Ndraft)
        else:
            ng = matches_.size(0)
            rs = Ndraft - ng
            # residual space left
            input_block_[:ng, :] = matches_
        if rs > 0:
            bi_ext = self.ext_bigram[last_word, :rs, :Npad].to(input_block.device)
            input_block_[ng:, 1:]  = bi_ext

        if not torch.all(input_block == input_block_):
            import ipdb; ipdb.set_trace()

        return input_block
    
    # def update(self, input_block, output_block, last_word, output_ids, **kwargs):

    #     input_block[:, 0] = last_word
    #     output_block[:, 0] = last_word
    #     Ndraft= input_block.size(0)
    #     Npad = input_block.size(1) - 1

    #     # obtain the matching grams
    #     matches = forward_ngram_matcher(output_ids, id=last_word, Ndraft=Ndraft, N=input_block.size(-1))
    #     if matches is None:
    #         ng, rs = (0, Ndraft)
    #     else:
    #         ng = matches.size(0)
    #         rs = Ndraft - ng
    #         # residual space left
    #         input_block[:ng, :] = matches
    #     if rs > 0:
    #         bi_ext = self.ext_bigram[last_word, :rs, :Npad].to(input_block.device)
    #         input_block[ng:, 1:]  = bi_ext
            
    #     return input_block
            
    def update_(self, input_block, output_block, last_word, context, **kwargs):

        input_block[:, 0] = last_word
        output_block[:, 0] = last_word
        Ndraft= input_block.size(0)
        Npad = input_block.size(1) - 1

        # obtain the matching grams

        matches = context_ngram_matcher(context=context, query=context[-self.query_size:], Ndraft=Ndraft, Npad=Npad)
        draft_ids = []
        if matches is None:
            ng, rs = (0, Ndraft)
        else:
            ng = matches.size(0)
            rs = Ndraft - ng
            # residual space left
            input_block[:ng, 1:] = matches[:, self.query_size:]
            draft_ids += [f'Context_q{self.query_size}' for _ in range(ng)]
        if rs > 0:
            bi_ext = self.ext_bigram[last_word, :rs, :Npad].to(input_block.device)
            input_block[ng:, 1:]  = bi_ext
            draft_ids += ['ModelBigramModelExt' for _ in range(rs)]

        self.draft_ids = draft_ids
        return input_block
 

    def get_strat_keys_(self):
        return [f'Context_q{self.query_size}' , 'ModelBigramModelExt']




    
class StaticContextBiModelExt:
    '''
    Context n gram matching as priority,
    defaulting back to model ext bigram for remaining drafts
    '''
    def __init__(self, model=None, device=None, query_size=2, **kwargs):
        model_name = model.config._name_or_path.split("/")[-1]
        # dtype_str = str(model.config.torch_dtype).split('.')[-1]
        dtype_str = str(model.dtype).split('.')[-1]
        self.ext_bigram = torch.load(
            Path('src/strategies/model2gram').joinpath(model_name, dtype_str, 'model_extended_2gram_rankings.pth')
        )
        self.query_size=query_size


    def update(self, input_block, output_block, last_word, context, **kwargs):

        input_block[:, 0] = last_word
        output_block[:, 0] = last_word
        Ndraft= input_block.size(0)
        Npad = input_block.size(1) - 1

        # obtain the matching grams
        matches = context_ngram_matcher(context=context, query=context[-self.query_size:], Ndraft=Ndraft, Npad=Npad)
        if matches is None:
            ng, rs = (0, Ndraft)
        else:
            ng = matches.size(0)
            rs = Ndraft - ng
            # residual space left
            input_block[:ng, 1:] = matches[:, self.query_size:]
        if rs > 0:
            bi_ext = self.ext_bigram[last_word, :rs, :Npad].to(input_block.device)
            input_block[ng:, 1:]  = bi_ext

        return input_block
            
    def update_(self, input_block, output_block, last_word, context, **kwargs):

        input_block[:, 0] = last_word
        output_block[:, 0] = last_word
        Ndraft= input_block.size(0)
        Npad = input_block.size(1) - 1

        # obtain the matching grams

        matches = context_ngram_matcher(context=context, query=context[-self.query_size:], Ndraft=Ndraft, Npad=Npad)
        draft_ids = []
        if matches is None:
            ng, rs = (0, Ndraft)
        else:
            ng = matches.size(0)
            rs = Ndraft - ng
            # residual space left
            input_block[:ng, 1:] = matches[:, self.query_size:]
            draft_ids += [f'Context_q{self.query_size}' for _ in range(ng)]
        if rs > 0:
            bi_ext = self.ext_bigram[last_word, :rs, :Npad].to(input_block.device)
            input_block[ng:, 1:]  = bi_ext
            draft_ids += ['ModelBigramModelExt' for _ in range(rs)]

        self.draft_ids = draft_ids
        return input_block
 

    def get_strat_keys_(self):
        return [f'Context_q{self.query_size}' , 'ModelBigramModelExt']





# ------------------------------------------------------------ LEGACY BELOW -------------------------------------------------------------------------------







# # import statistics
# # @torch.inference_mode
# # def time_length(L, N):
# #     input_ids = torch.randint(0, 100, (5, 1, L)).to(0)
# #     times = []
# #     for _ in range(5):
# #         start = torch.cuda.Event(enable_timing=True)
# #         end = torch.cuda.Event(enable_timing=True)
# #         start.record()
# #         grams = input_ids[i].flatten().unfold(0,N, 1)
# #         end.record()
# #         torch.cuda.synchronize()
# #         times.append(start.elapsed_time(end))
# #     return  statistics.mean(times)


# # PNG = prompt N gram
# class PNG_ZeroLayerUnigram:
#     '''
#     Prompt lookup and pad everything else with unigram model
#     '''
#     def __init__(self, model=None, device=None, **kwargs):
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


#     def update(self, input_block, output_block, last_word, output_ids, **kwargs):

#         input_block[:, 0] = last_word
#         output_block[:, 0] = last_word
#         Ndraft= input_block.size(0)

#         # obtain the matching grams
#         matches = forward_ngram_matcher(output_ids, id=last_word, Ndraft=Ndraft, N=input_block.size(-1))
#         if matches is None:
#             ng, rs = (0, Ndraft)
#         else:
#             ng = matches.size(0)
#             rs = Ndraft - ng
#             # residual space left
#             input_block[:ng, :] = matches
#         if rs > 0:
#             uni = self.unigram_head_ranking[:rs].reshape(-1, 1)
#             input_block[ng:, 1:] = uni


#     def update_(self, input_block, output_block, last_word, output_ids, **kwargs):

#         input_block[:, 0] = last_word
#         output_block[:, 0] = last_word
#         Ndraft= input_block.size(0)

#         # obtain the matching grams
#         matches = forward_ngram_matcher(output_ids, id=last_word, Ndraft=Ndraft, N=input_block.size(-1))
#         draft_ids = []
#         if matches is None:
#             ng, rs = (0, Ndraft)
#         else:
#             ng = matches.size(0)
#             rs = Ndraft - ng
#             # residual space left
#             input_block[:ng, :] = matches
#             draft_ids += ['Prompt' for _ in range(ng)]
#         if rs > 0:
#             uni = self.unigram_head_ranking[:rs].reshape(-1, 1)
#             input_block[ng:, 1:] = uni
#             draft_ids += ['ZeroLayerUnigram' for _ in range(rs)]
#         self.draft_ids = draft_ids


#     def get_strat_keys_(self):
#         return ['Prompt', 'ZeroLayerUnigram']



    
# # PNG = prompt N gram
# class PNG_ModelBigram_UniExt:
#     '''
#     Prompt lookup and pad everything else with unigram model
#     '''
#     def __init__(self, model=None, device=None, **kwargs):
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

#             model_name = model.config._name_or_path.split("/")[-1]
#             dtype_str = str(model.config.torch_dtype).split('.')[-1]
#             self.bigram = torch.load(
#                 Path('src/strategies/model2gram').joinpath(model_name, dtype_str, '2gram_rankings.pth')
#             )


#     def update(self, input_block, output_block, last_word, output_ids, **kwargs):

#         input_block[:, 0] = last_word
#         output_block[:, 0] = last_word
#         Ndraft= input_block.size(0)

#         # obtain the matching grams
#         matches = forward_ngram_matcher(output_ids, id=last_word, Ndraft=Ndraft, N=input_block.size(-1))
#         if matches is None:
#             ng, rs = (0, Ndraft)
#         else:
#             ng = matches.size(0)
#             rs = Ndraft - ng
#             # residual space left
#             input_block[:ng, :] = matches
#         if rs > 0:
#             bi = self.bigram[last_word, :rs].reshape(-1, 1).to(input_block.device)
#             input_block[ng:, 1:2] = bi
#             if input_block.size(1) > 2:
#                 uni = self.unigram_head_ranking[:rs].reshape(-1, 1)
#                 input_block[ng:, 2:] = uni


#     def update_(self, input_block, output_block, last_word, output_ids, **kwargs):

#         input_block[:, 0] = last_word
#         output_block[:, 0] = last_word
#         Ndraft= input_block.size(0)

#         # obtain the matching grams
#         matches = forward_ngram_matcher(output_ids, id=last_word, Ndraft=Ndraft, N=input_block.size(-1))
#         draft_ids = []
#         if matches is None:
#             ng, rs = (0, Ndraft)
#         else:
#             ng = matches.size(0)
#             rs = Ndraft - ng
#             # residual space left
#             input_block[:ng, :] = matches
#             draft_ids += ['Prompt' for _ in range(ng)]
#         if rs > 0:
#             bi = self.bigram[last_word, :rs].reshape(-1, 1).to(input_block.device)
#             input_block[ng:, 1:2] = bi
#             if input_block.size(1) > 2:
#                 uni = self.unigram_head_ranking[:rs].reshape(-1, 1)
#                 input_block[ng:, 2:] = uni
#             draft_ids += ['ModelBigram_UniExt' for _ in range(rs)]
#         self.draft_ids = draft_ids

#     def get_strat_keys_(self):
#         return ['Prompt', 'ModelBigram_UniExt']



    
# # PNG = prompt N gram
# class PNG_ModelBigramExt:
#     '''
#     Prompt lookup and pad everything else with unigram model
#     '''
#     def __init__(self, model=None, device=None, **kwargs):
#             model_name = model.config._name_or_path.split("/")[-1]
#             dtype_str = str(model.config.torch_dtype).split('.')[-1]
#             self.ext_bigram = torch.load(
#                 Path('src/strategies/model2gram').joinpath(model_name, dtype_str, 'extended_2gram_rankings.pth')
#             )


#     def update(self, input_block, output_block, last_word, output_ids, **kwargs):

#         input_block[:, 0] = last_word
#         output_block[:, 0] = last_word
#         Ndraft= input_block.size(0)
#         Npad = input_block.size(1) - 1

#         # obtain the matching grams
#         matches = forward_ngram_matcher(output_ids, id=last_word, Ndraft=Ndraft, N=input_block.size(-1))
#         if matches is None:
#             ng, rs = (0, Ndraft)
#         else:
#             ng = matches.size(0)
#             rs = Ndraft - ng
#             # residual space left
#             input_block[:ng, :] = matches
#         if rs > 0:
#             bi_ext = self.ext_bigram[last_word, :rs, :Npad].to(input_block.device)
#             input_block[ng:, 1:]  = bi_ext
            
#     def update_(self, input_block, output_block, last_word, output_ids, **kwargs):

#         input_block[:, 0] = last_word
#         output_block[:, 0] = last_word
#         Ndraft= input_block.size(0)
#         Npad = input_block.size(1) - 1

#         # obtain the matching grams
#         matches = forward_ngram_matcher(output_ids, id=last_word, Ndraft=Ndraft, N=input_block.size(-1))
#         draft_ids = []
#         if matches is None:
#             ng, rs = (0, Ndraft)
#         else:
#             ng = matches.size(0)
#             rs = Ndraft - ng
#             # residual space left
#             input_block[:ng, :] = matches
#             draft_ids += ['Prompt' for _ in range(ng)]
#         if rs > 0:
#             bi_ext = self.ext_bigram[last_word, :rs, :Npad].to(input_block.device)
#             input_block[ng:, 1:]  = bi_ext
#             draft_ids += ['ModelBigramExt' for _ in range(rs)]

#         self.draft_ids = draft_ids
 

#     def get_strat_keys_(self):
#         return ['Prompt', 'ModelBigramExt']



    
# PNG = prompt N gram
class PNG_ModelBigramModelExt:
    '''
    Prompt lookup and pad everything else with unigram model
    '''
    def __init__(self, model=None, device=None, **kwargs):
            model_name = model.config._name_or_path.split("/")[-1]
            # dtype_str = str(model.config.torch_dtype).split('.')[-1]
            dtype_str = str(model.dtype).split('.')[-1]
            self.ext_bigram = torch.load(
                Path('src/strategies/model2gram').joinpath(model_name, dtype_str, 'model_extended_2gram_rankings.pth')
            )


    def update(self, input_block, output_block, last_word, output_ids, **kwargs):

        input_block[:, 0] = last_word
        output_block[:, 0] = last_word
        Ndraft= input_block.size(0)
        Npad = input_block.size(1) - 1

        # obtain the matching grams
        matches = forward_ngram_matcher(output_ids, id=last_word, Ndraft=Ndraft, N=input_block.size(-1))
        if matches is None:
            ng, rs = (0, Ndraft)
        else:
            ng = matches.size(0)
            # residual space left
            rs = Ndraft - ng
            input_block[:ng, :] = matches
        if rs > 0:
            bi_ext = self.ext_bigram[last_word, :rs, :Npad].to(input_block.device)
            input_block[ng:, 1:]  = bi_ext
            
        return input_block
    def update_(self, input_block, output_block, last_word, output_ids, **kwargs):

        input_block[:, 0] = last_word
        output_block[:, 0] = last_word
        Ndraft= input_block.size(0)
        Npad = input_block.size(1) - 1

        # obtain the matching grams
        matches = forward_ngram_matcher(output_ids, id=last_word, Ndraft=Ndraft, N=input_block.size(-1))
        draft_ids = []
        if matches is None:
            ng, rs = (0, Ndraft)
        else:
            ng = matches.size(0)
            rs = Ndraft - ng
            # residual space left
            input_block[:ng, :] = matches
            draft_ids += ['Prompt' for _ in range(ng)]
        if rs > 0:
            bi_ext = self.ext_bigram[last_word, :rs, :Npad].to(input_block.device)
            input_block[ng:, 1:]  = bi_ext
            draft_ids += ['ModelBigramModelExt' for _ in range(rs)]

        self.draft_ids = draft_ids
    
        return input_block
 

    def get_strat_keys_(self):
        return ['Prompt', 'ModelBigramModelExt']
