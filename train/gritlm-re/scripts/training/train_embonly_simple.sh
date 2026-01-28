#!/bin/bash
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
cd /home/wychanbu/gritlm/gritlm
source /home/wychanbu/gritlm/.gritvenv/bin/activate
# export WANDB_PROJECT="gritlm"
export CUDA_VISIBLE_DEVICES=4,5,6,7
export HF_HOME=/data/wychanbu/huggingface
export NCCL_P2P_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Training setup
GPUS_PER_NODE=4

LAUNCHER="accelerate launch \
    --config_file /home/wychanbu/gritlm/scripts/configs/config_4gpus_ddp.yml \
    --num_machines 1 \
    --num_processes $GPUS_PER_NODE \
    --main_process_port 6000 \
    --machine_rank 0 \
    --role localhost: \
    --tee 1 \
    "

TRAIN_DATA=/data/wychanbu/test_data/ # replace with the directory of your training data

export CMD=" \
    -m training.run \
    --output_dir /data/wychanbu/re_models/Qwen2.5_7B_gritlm/ \
    --model_name_or_path Qwen/Qwen2.5-7B \
    --train_data $TRAIN_DATA \
    --learning_rate 2e-5 \
    --lr_scheduler_type constant_with_warmup \
    --warmup_ratio 0.06 \
    --max_steps 10000 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --dataloader_drop_last \
    --normalized \
    --temperature 0.02 \
    --train_group_size 2 \
    --query_max_len 1024 \
    --passage_max_len 1024 \
    --mode embedding \
    --logging_steps 1 \
    --bf16 \
    --pooling_method mean \
    --attn bbcc \
    --gradient_checkpointing \
    --attn_implementation sdpa \
    --save_steps 500 \
    --lora
    "

clear; $LAUNCHER $CMD 