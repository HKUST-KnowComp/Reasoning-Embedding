#!/bin/bash
# IMPORTANT: Run this script with bash, not sh
# Usage: bash scripts/training/train_embonly.sh
#        NOT: sh scripts/training/train_embonly.sh
#
#SBATCH --job-name=gritlm
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --partition=a3
#SBATCH --gres=gpu:8                 # number of gpus
#SBATCH --time 999:00:00             # maximum execution time (HH:MM:SS)
#SBATCH --output=/data/niklas/jobs/%x-%j.out           # output file name
#SBATCH --exclusive

######################
### Set enviroment ###
######################
cd /home/wychanbu/gritlm-own/gritlm
source /home/wychanbu/gritlm-own/.gritvenv_old/bin/activate
# export WANDB_PROJECT="gritlm"

export CUDA_VISIBLE_DEVICES=0,5,6,7
export GPUS_PER_NODE=4

export TORCH_CUDA_ARCH_LIST="8.9"
export HF_HOME=/data/wychanbu/huggingface

# Optimized NCCL settings for multi-GPU training
# export NCCL_P2P_DISABLE=1 
# export NCCL_IB_DISABLE=1   # Disable InfiniBand (not available on most setups)
# export NCCL_DEBUG=INFO     # Enable debug output to see what's happening
# export NCCL_TIMEOUT=3600   # Increase timeout to 1 hour
# # export NCCL_BLOCKING_WAIT=1  # Use blocking wait for better error messages
# # Avoid problematic network interfaces
# export NCCL_SOCKET_IFNAME=^lo,docker0,virbr0
export NCCL_P2P_DISABLE=1

# Memory and performance optimizations
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_CUDNN_V8_API_ENABLED=1  # Use cuDNN v8 API

# Disable torch.compile/dynamo (incompatible with FSDP + GradCache)
export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1

# Enable TF32 for faster matmul on Ampere+ GPUs
export NVIDIA_TF32_OVERRIDE=1

# Note: Use --report_to none flag instead of WANDB_DISABLED env var (deprecated in wandb v5)

LAUNCHER="accelerate launch \
    --config_file /home/wychanbu/gritlm-own/scripts/configs/config_8gpusfsdp_qwen.yml \
    --num_machines 1 \
    --num_processes $GPUS_PER_NODE \
    --main_process_port 8001 \
    --machine_rank 0 \
    --role localhost: \
    --tee 1 \
    "

TRAIN_DATA=/data/wychanbu/re_data/hard-neg-with-stem # replace with the directory of your training data

export CMD=" \
    -m training.run \
    --output_dir /data/wychanbu/re_models/Qwen3-0.6B-hard_neg_with_stem_2 \
    --model_name_or_path Qwen/Qwen3-0.6B \
    --train_data $TRAIN_DATA \
    --learning_rate 2e-5 \
    --weight_decay 0.05 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 512 \
    --gradient_accumulation_steps 2 \
    --dataloader_drop_last \
    --normalized \
    --temperature 0.02 \
    --train_group_size 4 \
    --negatives_cross_device \
    --query_max_len 512 \
    --passage_max_len 512 \
    --mode embedding \
    --logging_steps 1 \
    --bf16 \
    --pooling_method mean \
    --use_unique_indices \
    --attn bbcc \
    --attn_implementation flash_attention_2 \
    --save_steps 200 \
    --dataloader_num_workers 4 \
    --dataloader_pin_memory \
    --report_to none \
    --gradient_checkpointing \
    "
# Note: torch.compile is automatically disabled with FSDP due to compatibility issues
# For single-GPU training, you can add: --torch_compile --torch_compile_mode reduce-overhead
# Optional: Add --use_fused_adamw for 5-10% optimizer speedup (don't use with --use_muon)

clear; $LAUNCHER $CMD