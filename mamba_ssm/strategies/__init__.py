from .model_bigram import ModelBigram, ModelBigramModelExt
from .zerolayerunigram import ZeroLayerUnigram
from .promptgrams import StaticContextBiModelExt
# from .dynamic import MomContextBiModelExt, MomTest
from .dynamic import NpadMom, FullMom
from functools import partial
from .promptgrams import PNG_ModelBigramModelExt, Debugger
from .rampdown import Rampdown

# from .promptgrams import PNG_ZeroLayerUnigram, PNG_ModelBigram_UniExt, PNG_ModelBigramExt, PNG_ModelBigramModelExt
# from .model_nm_gram import Gram_2_5_ModelBigram, TwoLGram_ModelBigramModelExt, OneLGram_ModelBigramModelExt


STRAT_LIST = {
    'ModelBigram' : ModelBigram,
    'ZeroLayerUnigram' : ZeroLayerUnigram,
    'ModelBigramModelExt' : ModelBigramModelExt,
    'PNG_ModelBigramModelExt' : PNG_ModelBigramModelExt,
    'Debugger' : partial(Debugger, query_size=1),
    'FullMom' : FullMom,
    # 'StaticContextBiModelExtQ1' : StaticContextBiModelExtQ1,
    # 'StaticContextBiModelExtQ2' : StaticContextBiModelExtQ2,
    # 'StaticContextBiModelExtQ3' : StaticContextBiModelExtQ3,
    # 'MomContextBiModelExt': MomContextBiModelExt,

}

STRAT_LIST['NpadMomStatic'] = partial(NpadMom, mom_Npad=0, init_Npad=10)
STRAT_LIST['NpadMom_m2_i2'] = partial(NpadMom, mom_Npad=2, init_Npad=2)
STRAT_LIST['NpadMom_m3_i2'] = partial(NpadMom, mom_Npad=3, init_Npad=2)
STRAT_LIST['NpadMom_m4_i2'] = partial(NpadMom, mom_Npad=4, init_Npad=2)
STRAT_LIST['NpadMom_m2_i3'] = partial(NpadMom, mom_Npad=3, init_Npad=2)


pairs = []
# 25
for finish in [1, 5, 10, 15]:
    STRAT_LIST[f'Rampdown_{25}_{finish}'] = partial(Rampdown, start=25, finish=finish)
# 20
for finish in [1, 5, 10, 15]:
    STRAT_LIST[f'Rampdown_{20}_{finish}'] = partial(Rampdown, start=20, finish=finish)
# 15
for finish in [1, 5, 10]:
    STRAT_LIST[f'Rampdown_{15}_{finish}'] = partial(Rampdown, start=15, finish=finish)

# 10
for finish in [1, 5]:
    STRAT_LIST[f'Rampdown_{10}_{finish}'] = partial(Rampdown, start=10, finish=finish)



# for query_size in range(1, 4):
#     STRAT_LIST[f'StaticContextBiModelExtQ{query_size}'] =  partial(StaticContextBiModelExt, query_size=query_size)

# for mom_Npad in [2, 4]:
#     STRAT_LIST[f'MomTestM{mom_Npad}'] = partial(MomTest, mom_Npad=mom_Npad)



# for query_size in range(1, 4):
#     for slowdown in [5, 10, 15]:
#         STRAT_LIST[f'DynContextBiModelExtQ{query_size}T{slowdown}'] =  partial(DynContextBiModelExt, query_size=query_size, slowdown=slowdown)



    # 'PNG_ZeroLayerUnigram' : PNG_ZeroLayerUnigram,
    # 'PNG_ModelBigram_UniExt' : PNG_ModelBigram_UniExt,
    # 'PNG_ModelBigramExt' : PNG_ModelBigramExt,
    # 'PNG_ModelBigramModelExt' : PNG_ModelBigramModelExt,
    # 'Gram_2_5_ModelBigram' : Gram_2_5_ModelBigram,
    # 'TwoLGram_ModelBigramModelExt' : TwoLGram_ModelBigramModelExt,
    # 'OneLGram_ModelBigramModelExt' : OneLGram_ModelBigramModelExt,
