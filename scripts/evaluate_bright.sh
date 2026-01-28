#!/usr/bin/env bash

export TORCH_CUDA_ARCH_LIST="8.9"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export CUDA_VISIBLE_DEVICES=1
export HF_HOME="~/.cache/huggingface/"

# model_path=lucaswychan/Qwen3-0.6B-Base-Reasoning-Embedding
# model_name=code_fine_tuned_qwen3_0p6b_base_hard_neg_no_lora

model_path=lucaswychan/Qwen3-0.6B-Reasoning-Embedding
model_name=code_fine_tuned_qwen3_0p6b_hard_neg_no_lora

device=cuda

clear
python3 evaluation/evaluate_bright.py \
  --model $model_path \
  --config_kwargs "{\"max_length\": 8192}" \
  --model_kwargs "{\"attn_implementation\": \"flash_attention_2\", \"dtype\": \"bfloat16\"}" \
  --encode_kwargs "{\"normalize_embeddings\": true, \"convert_to_numpy\": true}" \
  --output_dir $model_name \
  --batch_size 4 \
  --device "$device"