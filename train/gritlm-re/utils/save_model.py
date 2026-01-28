import argparse
import json
import logging
import os
import sys

import torch
from dotenv import load_dotenv
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoConfig, AutoModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)

load_dotenv()


def save_model(checkpoint_dir, output_dir=None, base_model_name=None, is_peft=False) -> None:
    if output_dir is None:
        output_dir = checkpoint_dir

    config_path = os.path.join(checkpoint_dir, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)

    if base_model_name is None:
        base_model_name = config["_name_or_path"]
        if base_model_name is None:
            raise ValueError("base_model_name is None while config._name_or_path is also None")

    target_path = os.path.join(checkpoint_dir, "pytorch_model.bin")

    logger.info(f"Loading base model from {base_model_name}")
    base_model = AutoModel.from_pretrained(base_model_name, cache_dir=os.getenv("model_cache_dir"), dtype=torch.bfloat16)

    if is_peft:
        logger.info(f"Loading peft config from {checkpoint_dir}")
        peft_config = LoraConfig(inference_mode=False, r=16, lora_alpha=32, lora_dropout=0.1, target_modules="all-linear", task_type=TaskType.FEATURE_EXTRACTION)

        logger.info(f"Loading peft model from {checkpoint_dir}")
        peft_model = get_peft_model(base_model, peft_config)

        logger.info(f"Loading state dict from {target_path}")
        loaded_res = peft_model.load_state_dict(torch.load(target_path))
        logger.info(f"Loaded result: {loaded_res}")

        logger.info(f"Merging and unloading peft model")
        base_model = peft_model.merge_and_unload(progressbar=True)
    else:
        logger.info(f"Loading state dict from {target_path}")
        base_model.load_state_dict(torch.load(target_path))

    logger.info(f"Saving merged model to {output_dir}")
    base_model.save_pretrained(output_dir, max_shard_size="5GB", safe_serialization=True)


if __name__ == "__main__":
    """
    example usage:

    python3 models/save_model.py /data/wychanbu/re_models/Qwen2.5-7B_toy_data/checkpoint-5 --base_model_name Qwen/Qwen2.5-7B-Instruct

    """
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_dir", type=str)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--base_model_name", type=str, default=None)
    parser.add_argument("--is_peft", action="store_true", default=False)
    args = parser.parse_args()

    save_model(args.checkpoint_dir, output_dir=args.output_dir, base_model_name=args.base_model_name, is_peft=args.is_peft)
