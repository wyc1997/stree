CUDA_VISIBLE_DEVICES=3 python llm_judge/gen_model_answer_mamba.py \
--model-path JunxiongWang/Llama3.1-Mamba2-8B-distill \
--model-id MIL-8B-max1024-2 \
--bench-name human_eval \
--max-new-token 1024 \
--cg 

CUDA_VISIBLE_DEVICES=1 python llm_judge/gen_model_answer_mamba.py \
--model-path JunxiongWang/Llama3.1-Mamba2-8B-distill \
--model-id MIL-8B-Npad4-Ndraft1-StratMIL-max1024-ckpt48000-2 \
--bench-name human_eval \
--max-new-token 1024 \
--cg \
--spec_Ngram \
--use_Nstep_kernel \
--jit_state_copy \
--activation_replay \
--strategy MIL \
--npad 4 \
--ndraft 1 

CUDA_VISIBLE_DEVICES=2 python llm_judge/gen_model_answer_mamba.py \
--model-path JunxiongWang/Llama3.1-Mamba2-8B-distill \
--model-id MIL-8B-td-Npad4-Ndraft1-StratMILst-draft3-max1024-ckpt48000-2 \
--bench-name human_eval \
--max-new-token 1024 \
--cg \
--spec_Ngram \
--use_tree_decoding \
--jit_state_copy \
--activation_replay \
--strategy MIL-st \
--npad 4 \
--ndraft 1 \
--draft_num_beam=3

CUDA_VISIBLE_DEVICES=2 python llm_judge/gen_model_answer_mamba.py \
--model-path JunxiongWang/Llama3.1-Mamba2-8B-distill \
--model-id MIL-8B-max1024-ckpt48000-temp1-2 \
--bench-name human_eval \
--max-new-token 1024 \
--cg \
--temperature=1 \
--top_k=0

CUDA_VISIBLE_DEVICES=1 python llm_judge/gen_model_answer_mamba.py \
--model-path JunxiongWang/Llama3.1-Mamba2-8B-distill \
--model-id MIL-8B-Npad4-Ndraft1-StratMIL-max1024-ckpt48000-temp1-2 \
--bench-name human_eval \
--max-new-token 1024 \
--cg \
--spec_Ngram \
--use_Nstep_kernel \
--jit_state_copy \
--activation_replay \
--strategy MIL \
--npad 4 \
--ndraft 1 \
--temperature=1 \
--top_k=0


CUDA_VISIBLE_DEVICES=2 python llm_judge/gen_model_answer_mamba.py \
--model-path JunxiongWang/Llama3.1-Mamba2-8B-distill \
--model-id MIL-8B-td-Npad4-Ndraft1-StratMILst-draft3-max1024-ckpt48000-temp1-2 \
--bench-name human_eval \
--max-new-token 1024 \
--cg \
--spec_Ngram \
--use_tree_decoding \
--jit_state_copy \
--activation_replay \
--strategy MIL-st \
--npad 4 \
--ndraft 1 \
--draft_num_beam=3 \
--temperature=1 \
--top_k=0