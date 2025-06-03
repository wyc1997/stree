# STree

This repo contains the code for STree: Speculative Tree Decoding for Hybrid State-Space Models.
It is adapted from [MambaInLlama](https://github.com/jxiw/MambaInLlama) repo. 

# Setup

We ran our experiments on a Nvidia RTX3090 machine with CUDA 11.8

Creating

```
pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu118
pip install packaging
pip install evaluate
pip install causal_conv1d==1.4
pip install flash-attn==2.6.3
pip install transformers==4.49.0
pip install triton==3.0.0
```

# Usage 

## Distilling small mamba2 model for drafting

```
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file deepspeed_zero3.yaml train_mamba2/train_distill_mamba2.py llama3.2_3B/distilled_llama2.yaml
```

## Inference 

The inference entry point is located at `benchmarks/benchmark_generation_mamba_simple.py`. It provides a lot of options:


Baseline auto-regressive inference with Mamba2InLlama8B (use JunxiongWang/Llama3.2-Mamba2-3B-distill for 3B models for testing)
```
python benchmarks/benchmark_generation_mamba_simple.py --model-name JunxiongWang/Llama3.1-Mamba2-8B-distill --prompt "Earth is a planet." --cg
```

Speculative tree decoding with static tree as draft
```
python benchmarks/benchmark_generation_mamba_simple.py --model-name JunxiongWang/Llama3.1-Mamba2-8B-distill --prompt "Earth is a planet." --cg --spec_Ngram --use_tree_decoding --activation_replay --jit_state_copy --npad=4 --ndraft=1 --draft_num_beam=3 --strategy MIL-st
```

## Evaluation





