import torch
from pathlib import Path
import numpy as np
import pickle


@torch.inference_mode
def context_ngram_matcher(context, query, Ndraft=1, Npad=2):
    '''
    matches query of any length Q to grams of size Q + Npad
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
class FullMom:
    '''
    Prompt lookup and pad everything else with unigram model
    '''
    def __init__(self, model=None, device=None, init_Npad=2, mom_Npad=2, mom_Ndraft=2, **kwargs):
            model_name = model.config._name_or_path.split("/")[-1]
            dtype_str = str(model.config.torch_dtype).split('.')[-1]
            self.ext_bigram = torch.load(
                Path('src/strategies/model2gram').joinpath(model_name, dtype_str, 'model_extended_2gram_rankings.pth')
            )
            self.init_Npad = init_Npad
            self.mom_Npad = mom_Npad
            self.mom_Ndraft = mom_Ndraft
            self.cur_Npad = init_Npad
            self.cur_Ndraft = None


    def update(self, input_block, output_block, last_word, output_ids, prev_nverified, **kwargs):
        input_block[:, 0] = last_word
        output_block[:, 0] = last_word
        Ndraft= input_block.size(0)
        # Npad = input_block.size(1) - 1

        if prev_nverified is None:
            self.cur_Ndraft = Ndraft
        elif prev_nverified == self.cur_Npad + 1:
            self.cur_Npad += self.mom_Npad
            self.cur_Npad = min(input_block.size(-1) - 1, self.cur_Npad)

            self.cur_Ndraft -= self.mom_Ndraft
            self.cur_Ndraft = max(2, self.cur_Ndraft)

        else:
            self.cur_Npad = self.init_Npad
            self.cur_Ndraft = Ndraft


        # obtain the matching grams
        matches = forward_ngram_matcher(output_ids, id=last_word, Ndraft=self.cur_Ndraft, N=self.cur_Npad + 1)
        step_block = torch.narrow(input_block, dim=0, start=0, length=self.cur_Ndraft)
        step_block = torch.narrow(step_block, dim=1, start=0, length=self.cur_Npad + 1)

        if matches is None:
            ng, rs = (0, self.cur_Ndraft)
        else:
            ng = matches.size(0)
            # residual space left
            rs = self.cur_Ndraft - ng
            step_block[:ng, :] = matches
        if rs > 0:
            bi_ext = self.ext_bigram[last_word, :rs, :self.cur_Npad].to(input_block.device)
            step_block[ng:, 1:]  = bi_ext
            
        return step_block

    def update_(self, input_block, output_block, last_word, output_ids, **kwargs):
        raise ValueError


    def get_strat_keys_(self):
        return ['Prompt', 'ModelBigramModelExt']






# PNG = prompt N gram
class NpadMom:
    '''
    Prompt lookup and pad everything else with unigram model
    '''
    def __init__(self, model=None, device=None, init_Npad=2, mom_Npad=2, **kwargs):
            model_name = model.config._name_or_path.split("/")[-1]
            dtype_str = str(model.config.torch_dtype).split('.')[-1]
            self.ext_bigram = torch.load(
                Path('src/strategies/model2gram').joinpath(model_name, dtype_str, 'model_extended_2gram_rankings.pth')
            )
            self.init_Npad = init_Npad
            self.mom_Npad = mom_Npad
            self.cur_Npad = init_Npad


    def update(self, input_block, output_block, last_word, output_ids, prev_nverified, **kwargs):
        input_block[:, 0] = last_word
        output_block[:, 0] = last_word
        Ndraft= input_block.size(0)
        # Npad = input_block.size(1) - 1

        if prev_nverified is None:
            pass
        elif prev_nverified == self.cur_Npad + 1:
            self.cur_Npad += self.mom_Npad
            self.cur_Npad = min(input_block.size(-1) - 1, self.cur_Npad)
        else:
            self.cur_Npad = self.init_Npad


        # obtain the matching grams
        matches = forward_ngram_matcher(output_ids, id=last_word, Ndraft=Ndraft, N=self.cur_Npad + 1)
        step_block = torch.narrow(input_block, dim=1, start=0, length=self.cur_Npad + 1)

        if matches is None:
            ng, rs = (0, Ndraft)
        else:
            ng = matches.size(0)
            # residual space left
            rs = Ndraft - ng
            step_block[:ng, :] = matches
        if rs > 0:
            bi_ext = self.ext_bigram[last_word, :rs, :self.cur_Npad].to(input_block.device)
            step_block[ng:, 1:]  = bi_ext
            
        return step_block

    def update_(self, input_block, output_block, last_word, output_ids, **kwargs):
        raise ValueError


    def get_strat_keys_(self):
        return ['Prompt', 'ModelBigramModelExt']



# class MomTest:
#     def __init__(self,
#                  model=None,
#                  query_size=1,
#                  max_Npad=14,
#                  mom_Npad=2,
#                  min_Npad=2,
#                  **kwargs,
#     ):

#         model_name = model.config._name_or_path.split("/")[-1]
#         dtype_str = str(model.config.torch_dtype).split('.')[-1]
#         self.ext_bigram = torch.load(
#             Path('src/strategies/model2gram').joinpath(model_name, dtype_str, 'model_extended_2gram_rankings.pth')
#         )
#         self.query_size=query_size
#         self.min_Npad=min_Npad
#         self.min_Ndraft=1

#         self.mom_Npad=mom_Npad

#         self.max_Npad=max_Npad

#         self.cur_Npad = min_Npad
    

#     def update(self, input_block, output_block, last_word, context, prev_nverified, **kwargs):

#         input_block[:, 0] = last_word
#         output_block[:, 0] = last_word
#         Ndraft = input_block.size(0)

#         # at initialization leave rectangle alone
#         if prev_nverified is None:
#             pass
#         # if all of strategy matched then depthen rectangle
#         elif prev_nverified == self.cur_Npad + 1:
#             self.cur_Npad += self.mom_Npad
#             self.cur_Npad = min(self.cur_Npad, self.max_Npad)
#         # if not all of strategy matched then reset
#         else:
#             self.cur_Npad -= self.mom_Npad
#             self.cur_Npad = max(self.cur_Npad, self.min_Npad)

#         # print(f'Ndraft, Npad = ({self.cur_Ndraft}, {self.cur_Npad})')

#         step_block = torch.narrow(input_block, 0, 0, Ndraft)
#         step_block = torch.narrow(step_block, 1, 0, self.cur_Npad + 1)

        
#         # obtain the matching grams
#         matches = context_ngram_matcher(context=context, query=context[-self.query_size:], Ndraft=Ndraft, Npad=self.cur_Npad)

#         if matches is None:
#             ng, rs = (0, Ndraft)
#         else:
#             ng = matches.size(0)
#             rs = self.cur_Ndraft - ng
#             # residual space left
#             step_block[:ng, 1:] = matches[:, self.query_size:]
#         if rs > 0:
#             bi_ext = self.ext_bigram[last_word, :rs, :self.cur_Npad].to(input_block.device)
#             step_block[ng:, 1:]  = bi_ext

#         return step_block


#     def update_(**kwargs):
#         raise ValueError('Not yet implemented')
    





# class MomContextBiModelExt:
#     def __init__(self,
#                  model=None,
#                  query_size=1,
#                  max_Ndraft=16,
#                  max_Npad=14,
#                  mom_Ndraft=2,
#                  mom_Npad=2,
#                  min_Npad=2,
#                  **kwargs,
#     ):

#         model_name = model.config._name_or_path.split("/")[-1]
#         dtype_str = str(model.config.torch_dtype).split('.')[-1]
#         self.ext_bigram = torch.load(
#             Path('src/strategies/model2gram').joinpath(model_name, dtype_str, 'model_extended_2gram_rankings.pth')
#         )
#         self.query_size=query_size
#         self.min_Npad=min_Npad
#         self.min_Ndraft=1

#         self.mom_Npad=mom_Npad
#         self.mom_Ndraft=mom_Ndraft

#         self.max_Ndraft=max_Ndraft
#         self.max_Npad=max_Npad

#         self.cur_Ndraft = max_Ndraft
#         self.cur_Npad = min_Npad
    

#     def update(self, input_block, output_block, last_word, context, prev_nverified, **kwargs):

#         input_block[:, 0] = last_word
#         output_block[:, 0] = last_word

#         # at initialization leave rectangle alone
#         if prev_nverified is None:
#             pass
#         # if all of strategy matched then depthen rectangle
#         elif prev_nverified == self.cur_Npad:
#             self.cur_Npad += self.mom_Npad
#             self.cur_Npad = min(self.cur_Npad, self.max_Npad)

#             self.cur_Ndraft -= self.mom_Ndraft
#             self.cur_Ndraft = max(self.cur_Ndraft, 2)
#         # if not all of strategy matched then reset
#         else:
#             self.cur_Npad = self.min_Npad
#             self.cur_Ndraft = self.max_Ndraft

#         # print(f'Ndraft, Npad = ({self.cur_Ndraft}, {self.cur_Npad})')

#         step_block = torch.narrow(input_block, 0, 0, self.cur_Ndraft)
#         step_block = torch.narrow(step_block, 1, 0, self.cur_Npad + 1)

        
#         # obtain the matching grams
#         matches = context_ngram_matcher(context=context, query=context[-self.query_size:], Ndraft=self.cur_Ndraft, Npad=self.cur_Npad)

#         if matches is None:
#             ng, rs = (0, self.cur_Ndraft)
#         else:
#             ng = matches.size(0)
#             rs = self.cur_Ndraft - ng
#             # residual space left
#             step_block[:ng, 1:] = matches[:, self.query_size:]
#         if rs > 0:
#             bi_ext = self.ext_bigram[last_word, :rs, :self.cur_Npad].to(input_block.device)
#             step_block[ng:, 1:]  = bi_ext

#         return step_block


#     def update_(**kwargs):
#         raise ValueError('Not yet implemented')
    


# class ParetoSetDyn:
#     def __init__(self, model, slowdown=5,):
#         model_name = model.config._name_or_path.split("/")[-1]
#         dtype_str = str(model.config.torch_dtype).split('.')[-1]

#         path = Path('gputimings/results').joinpath(model_name, dtype_str, model.config._attn_implementation, 'pareto_fronts.pkl')
#         with open(path, 'rb') as f:
#             pfs = pickle.load(f)
#         assert slowdown in pfs.keys(), 'Please choose a slow down that is a multiple of 5%'
#         self.pareto_fn = {}
#         for (k, v) in pfs[slowdown].items():
#             self.pareto_fn[k] = self.create_pareto_dict_from_list_of_tuples(v)

#     def create_pareto_dict_from_list_of_tuples(self, lot):
#         pdict = {}
#         for (Ndraft, Npad) in lot:
#             pdict[Npad] = max(Ndraft, 1)
#         return pdict
            
#     def get_optimal_bs(self, kv_len, Npad):
#         '''
#         Mapping Npad (not seq_len) to maximum batch
#         '''
#         kv_len_keys = list(self.pareto_fn.keys())
#         bin_id = np.digitize(kv_len, kv_len_keys)
#         kv_ = kv_len_keys[bin_id]

#         return self.pareto_fn[kv_][Npad]



    
# class DynContextBiModelExt:
#     '''
#     Context n gram matching as priority,
#     defaulting back to model ext bigram for remaining drafts
#     Fixed Npad, with dynamic draft
#     '''
#     def __init__(self, 
#             model=None,
#             query_size=2,
#             slowdown=5,
#             **kwargs):
#         model_name = model.config._name_or_path.split("/")[-1]
#         dtype_str = str(model.config.torch_dtype).split('.')[-1]
#         self.ext_bigram = torch.load(
#             Path('src/strategies/model2gram').joinpath(model_name, dtype_str, 'model_extended_2gram_rankings.pth')
#         )
#         self.query_size=query_size
#         self.pareto = ParetoSetDyn(model=model, slowdown=slowdown)
           


#     def update(self, input_block, output_block, last_word, context, **kwargs):

#         input_block[:, 0] = last_word
#         output_block[:, 0] = last_word

#         kv_len = context.size(-1) - 1
         
#         Npad = input_block.size(-1) - 1
#         max_Ndraft = input_block.size(0)

#         Ndraft = self.pareto.get_optimal_bs(kv_len=kv_len, Npad=Npad)
#         Ndraft = min(Ndraft, max_Ndraft)

#         step_block = torch.narrow(input_block, 0, 0, Ndraft)

        
#         # obtain the matching grams
#         matches = context_ngram_matcher(context=context, query=context[-self.query_size:], Ndraft=Ndraft, Npad=Npad)
#         if matches is None:
#             ng, rs = (0, Ndraft)
#             self.num_context_drafts = -1
#         else:
#             ng = matches.size(0)
#             self.num_context_drafts = ng
#             rs = Ndraft - ng
#             # residual space left
#             step_block[:ng, 1:] = matches[:, self.query_size:]
#         if rs > 0:
#             bi_ext = self.ext_bigram[last_word, :rs, :Npad].to(input_block.device)
#             step_block[ng:, 1:]  = bi_ext


#         return step_block



#     def update_(self, input_block, output_block, last_word, context, **kwargs):

#         input_block[:, 0] = last_word
#         output_block[:, 0] = last_word

#         kv_len = context.size(-1) - 1
         
#         Npad = input_block.size(-1) - 1
#         max_Ndraft = input_block.size(0)

#         Ndraft = self.pareto.get_optimal_bs(kv_len=kv_len, Npad=Npad)
#         Ndraft = min(Ndraft, max_Ndraft)

#         step_block = torch.narrow(input_block, 0, 0, Ndraft)

        
#         # obtain the matching grams
#         matches = context_ngram_matcher(context=context, query=context[-self.query_size:], Ndraft=Ndraft, Npad=Npad)
#         if matches is None:
#             ng, rs = (0, Ndraft)
#             self.num_context_drafts = -1
#         else:
#             ng = matches.size(0)
#             self.num_context_drafts = ng
#             rs = Ndraft - ng
#             # residual space left
#             step_block[:ng, 1:] = matches[:, self.query_size:]
#             draft_ids += [f'Context_q{self.query_size}' for _ in range(ng)]
#         if rs > 0:
#             bi_ext = self.ext_bigram[last_word, :rs, :Npad].to(input_block.device)
#             step_block[ng:, 1:]  = bi_ext
#             draft_ids += ['ModelBigramModelExt' for _ in range(rs)]

#         return step_block


#     def get_strat_keys_(self):
#         return [f'Context_q{self.query_size}' , 'ModelBigramModelExt']






# # ------------------------------------------------------------ LEGACY BELOW -------------------------------------------------------------------------------







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



    
# # PNG = prompt N gram
# class PNG_ModelBigramModelExt:
#     '''
#     Prompt lookup and pad everything else with unigram model
#     '''
#     def __init__(self, model=None, device=None, **kwargs):
#             model_name = model.config._name_or_path.split("/")[-1]
#             dtype_str = str(model.config.torch_dtype).split('.')[-1]
#             self.ext_bigram = torch.load(
#                 Path('src/strategies/model2gram').joinpath(model_name, dtype_str, 'model_extended_2gram_rankings.pth')
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
#             draft_ids += ['ModelBigramModelExt' for _ in range(rs)]

#         self.draft_ids = draft_ids
 

#     def get_strat_keys_(self):
#         return ['Prompt', 'ModelBigramModelExt']
