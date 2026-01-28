#!/usr/bin/env python
"""
Command-line wrapper for running different datasets.
Usage:
    python run_dataset.py math500 --limit 500
    python run_dataset.py numina_math --limit 200
    python run_dataset.py gsm8k --limit 500
"""

import argparse

from dataset_config import DEFAULT_DATASET_CONFIG, GSM8K_CONFIG, MATH500_DATASET_CONFIG, NUMINAMATH_MATH_CONFIG, DatasetConfig
from run_qwen3_competition_math import main

DATASET_PRESETS = {
    "math500": MATH500_DATASET_CONFIG,
    "numina_math": NUMINAMATH_MATH_CONFIG,
    "gsm8k": GSM8K_CONFIG,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Qwen3-32B on math datasets")
    parser.add_argument("dataset", choices=list(DATASET_PRESETS.keys()), help="Dataset preset to use")
    parser.add_argument("--limit", type=int, default=None, help="Number of problems to solve (default: all)")
    parser.add_argument("--offset", type=int, default=0, help="Starting index (ignored if dataset uses random sampling)")
    parser.add_argument("--model", type=str, default="qwen3-32b", help="Model name (default: qwen3-32b)")
    parser.add_argument("--api-key-suffix", type=str, default="", help="API key suffix (e.g., '_2' for DASHSCOPE_API_KEY_2)")
    parser.add_argument("--no-evaluation", action="store_true", help="Disable DeepSeek-V3.2 answer evaluation")

    args = parser.parse_args()

    dataset_cfg = DATASET_PRESETS[args.dataset]

    main(
        dataset_cfg=dataset_cfg,
        limit=args.limit,
        offset=args.offset,
        model=args.model,
        api_key_suffix=args.api_key_suffix,
        enable_evaluation=not args.no_evaluation,
    )
