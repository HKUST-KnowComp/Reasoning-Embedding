#!/usr/bin/env bash

export TORCH_CUDA_ARCH_LIST="8.9"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export CUDA_VISIBLE_DEVICES=0
export HF_HOME="~/.cache/huggingface/"
# export MTEB_CACHE="/data/wychanbu/mteb"

model_path=lucaswychan/Qwen-2.5-1.5B-SimpleRL-Zoo-Reasoning-Embedding
model_name=fine_tuned_qwen2p5_1p5b_simplerlzoo_seed_1016

# benchmark="RTEB(beta)"
benchmark="MTEB(Multilingual, v2)"
# benchmark="MTEB(Code, v1)"

device=cuda
seed=1016

clear
python3 evaluation/evaluate_mteb.py \
  --model $model_path \
  --config_kwargs "{\"max_length\": 8192}" \
  --model_kwargs "{\"attn_implementation\": \"flash_attention_2\", \"dtype\": \"bfloat16\"}" \
  --encode_kwargs "{\"normalize_embeddings\": true, \"convert_to_numpy\": true}" \
  --output_dir $model_name \
  --batch_size 4 \
  --benchmark "$benchmark" \
  --device "$device" \
  --seed $seed