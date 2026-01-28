#!/usr/bin/env python3
"""
Batch evaluation script for existing unevaluated outputs using DeepSeek-V3.2.

This script:
1. Scans output directories for JSONL files
2. Loads the original datasets to get official answers
3. For records without correctness_label, compares model answer with official answer using DeepSeek-V3.2
4. Saves evaluated versions with _evaluated.jsonl suffix
"""

import json
import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI

from datasets import load_dataset


def make_deepseek_client() -> OpenAI:
    """Create OpenAI client for DeepSeek-V3.2 evaluation."""
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))

    deepseek_key = os.getenv("DASHSCOPE_API_KEY")
    if not deepseek_key:
        raise RuntimeError("Missing DASHSCOPE_API_KEY in environment/.env")

    return OpenAI(api_key=deepseek_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")


def load_dataset_answers(dataset_name: str, source_filter: Optional[str] = None) -> Dict[str, str]:
    """Load official answers from the dataset.

    Args:
        dataset_name: Name of the HuggingFace dataset
        source_filter: Optional source filter for NuminaMath-CoT

    Returns:
        Dictionary mapping question_id to official answer
    """
    print(f"Loading official answers from {dataset_name}...")
    answers = {}

    if dataset_name == "openai/gsm8k":
        ds = load_dataset(dataset_name, "main")
        for idx, row in enumerate(ds["test"]):
            question_id = f"test_{idx}"
            # GSM8K has 'answer' field
            answers[question_id] = str(row.get("answer", ""))

    elif dataset_name == "HuggingFaceH4/MATH-500":
        ds = load_dataset(dataset_name)
        for idx, row in enumerate(ds["test"]):
            question_id = f"test_{idx}"
            # MATH-500 has 'answer' field
            answers[question_id] = str(row.get("answer", ""))

    elif dataset_name == "opencompass/LiveMathBench":
        ds = load_dataset(dataset_name, "v202505_hard_en")
        for idx, row in enumerate(ds["test"]):
            question_id = f"test_{idx}"
            # LiveMathBench has 'answer' field
            answers[question_id] = str(row.get("answer", ""))

    elif dataset_name == "AI-MO/NuminaMath-CoT":
        ds = load_dataset(dataset_name)
        data = ds["train"]

        # Filter by source if specified - MUST match the order used during generation
        if source_filter:
            print(f"Filtering by source='{source_filter}'...")
            data = data.filter(lambda x: x.get("source") == source_filter)
            print(f"Filtered to {len(data)} problems.")

        # Iterate through the FILTERED dataset (indices are relative to filtered data)
        for idx in range(len(data)):
            row = data[idx]
            question_id = f"train_{idx}"
            # NuminaMath-CoT has 'solution' field at the top level
            answers[question_id] = str(row.get("solution", ""))

    print(f"Loaded {len(answers)} official answers.")
    return answers


def classify_answer_correctness(evaluator_client: OpenAI, problem: str, model_answer: str, official_answer: str) -> int:
    """Use DeepSeek-V3.2 to compare model answer with official answer.

    Args:
        evaluator_client: OpenAI client configured for DeepSeek-V3.2
        problem: The math problem statement
        model_answer: The model's answer to evaluate
        official_answer: The official/ground truth answer from the dataset

    Returns:
        int: 1 if answers match, 0 if they don't match, -1 if cannot parse response
    """
    evaluation_prompt = f"""You are an expert math answer comparator. Your task is to determine if the model's answer matches the official answer.

Problem:
{problem}

Official Answer:
{official_answer}

Model's Answer:
{model_answer}

Compare these two answers. They should be considered matching if they represent the same mathematical result, even if formatting differs slightly. Respond with ONLY "1" if the answers match, or "0" if they don't match. Do not provide any explanation."""

    try:
        response = evaluator_client.chat.completions.create(
            model="deepseek-v3.2-exp",
            messages=[{"role": "system", "content": "You are a precise math answer comparator. Respond only with 0 or 1."}, {"role": "user", "content": evaluation_prompt}],
            temperature=0,
            max_tokens=10,
            extra_body={"enable_thinking": False},
        )

        result = response.choices[0].message.content.strip()
        if "1" in result:
            return 1
        else:
            return 0
    except Exception as e:
        print(f"Warning: DeepSeek evaluation failed: {e}")
        return -1


def evaluate_file(input_path: str, output_path: str, evaluator_client: OpenAI, official_answers: Dict[str, str]) -> None:
    """Evaluate records in a file and save results.

    Args:
        input_path: Path to input JSONL file
        output_path: Path to output JSONL file
        evaluator_client: DeepSeek client for evaluation
        official_answers: Dictionary mapping question_id to official answer
    """
    records = []
    evaluated_count = 0
    skipped_count = 0

    # Load existing records
    if os.path.exists(input_path):
        with open(input_path, "r") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))

    print(f"Processing {len(records)} records from {input_path}...")

    # Evaluate unevaluated records
    for record in records:
        # Skip if already evaluated
        if "correctness_label" in record and record["correctness_label"] != -1:
            continue

        # Skip if marked as skipped/incomplete
        if record.get("skipped", False):
            # Set correctness_label to -1 for skipped records
            record["correctness_label"] = -1
            record["evaluator_model"] = "N/A (skipped)"
            skipped_count += 1
            continue

        # Get official answer
        question_id = record.get("question_id", "")
        official_answer = official_answers.get(question_id, "")

        if not official_answer:
            print(f"Warning: No official answer found for {question_id}")
            record["correctness_label"] = -1
            record["evaluator_model"] = "N/A (no official answer)"
            continue

        # Evaluate
        problem = record.get("problem", "")
        model_answer = record.get("answer", "")

        correctness = classify_answer_correctness(evaluator_client, problem, model_answer, official_answer)

        record["correctness_label"] = correctness
        record["evaluator_model"] = "deepseek-v3.2-exp"
        evaluated_count += 1

    # Write evaluated records
    with open(output_path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    print(f"Evaluated {evaluated_count} records, skipped {skipped_count}, saved to {output_path}")


def main():
    """Main evaluation loop."""
    # Initialize DeepSeek client
    evaluator_client = make_deepseek_client()
    print("DeepSeek-V3.2 evaluator initialized.")

    # Define directories to scan
    base_dir = os.path.dirname(__file__)
    output_dirs = [
        # os.path.join(base_dir, "outputs"),
        # os.path.join(base_dir, "outputs", "hard-level"),
        os.path.join(base_dir, "outputs", "medium-level"),
    ]

    # Process each directory
    for output_dir in output_dirs:
        if not os.path.exists(output_dir):
            continue

        print(f"\n{'='*60}")
        print(f"Scanning {output_dir}...")
        print(f"{'='*60}")

        # Find all JSONL files (handle both _outputs.jsonl and .jsonl patterns)
        jsonl_files = [f for f in os.listdir(output_dir) if f.endswith(".jsonl") and not f.endswith("_evaluated.jsonl")]

        for filename in jsonl_files:
            input_path = os.path.join(output_dir, filename)
            # Handle both naming patterns
            if filename.endswith("_outputs.jsonl"):
                output_filename = filename.replace("_outputs.jsonl", "_outputs_evaluated.jsonl")
            else:
                output_filename = filename.replace(".jsonl", "_evaluated.jsonl")
            output_path = os.path.join(output_dir, output_filename)

            # Skip if already evaluated
            if os.path.exists(output_path):
                print(f"\n--- Skipping {filename} (already evaluated) ---")
                continue

            print(f"\n--- Processing {filename} ---")

            # Determine dataset name and source filter from filename
            if "gsm8k" in filename.lower():
                dataset_name = "openai/gsm8k"
                source_filter = None
            elif "math-500" in filename.lower() or "math_500" in filename.lower():
                dataset_name = "HuggingFaceH4/MATH-500"
                source_filter = None
            elif "livemathbench" in filename.lower():
                dataset_name = "opencompass/LiveMathBench"
                source_filter = None
            elif "numinamath" in filename.lower() or "AI-MO" in filename:
                dataset_name = "AI-MO/NuminaMath-CoT"
                # Extract source filter from filename
                if "cn_k12" in filename:
                    source_filter = "cn_k12"
                elif "olympiads" in filename:
                    source_filter = "olympiads"
                elif "math" in filename.lower():
                    source_filter = "math"
                else:
                    source_filter = None
            else:
                print(f"Warning: Unknown dataset in filename {filename}, skipping...")
                continue

            # Load official answers
            try:
                official_answers = load_dataset_answers(dataset_name, source_filter)
            except Exception as e:
                print(f"Error loading dataset {dataset_name}: {e}")
                print(f"Skipping {filename}...")
                continue

            # Evaluate
            evaluate_file(input_path, output_path, evaluator_client, official_answers)

    print("\n" + "=" * 60)
    print("Batch evaluation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
