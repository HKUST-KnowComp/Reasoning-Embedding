#!/usr/bin/env python3
"""
Filter CodeSearchNet dataset by margin scores (positive score - negative score).
Uses batch processing for efficient computation.
"""

import argparse
import gc
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from datasets import Dataset

# Set HuggingFace cache directory
os.environ["HF_HOME"] = "~/.cache/huggingface/"


def load_data_as_dataset(input_file):
    """Load data from JSONL file and convert to HuggingFace Dataset."""
    print(f"Loading data from {input_file}...")
    all_data = []
    with open(input_file, "r") as f:
        for line in f:
            all_data.append(json.loads(line))

    print(f"Loaded {len(all_data)} samples")

    # Filter data that has exactly 3 hard negatives
    data_with_3_hard_neg = [data for data in all_data if len(data["neg"]) == 3]
    print(f"Filtered to {len(data_with_3_hard_neg)} samples with 3 hard negatives")

    # Convert to HuggingFace Dataset
    dataset = Dataset.from_list(data_with_3_hard_neg)
    print(f"Created HuggingFace Dataset with {len(dataset)} samples")

    return dataset


def process_single_batch(dataset, model, batch_start, batch_end, internal_batch_size):
    """Process a single batch with specified internal batch size."""
    batch = dataset[batch_start:batch_end]
    margin_scores = []

    # Extract queries (handle both string and list formats)
    queries = [query[1] if isinstance(query, list) else query for query in batch["query"]]

    # Encode queries in batch
    query_embeddings = model.encode(queries, convert_to_numpy=True, normalize_embeddings=True, batch_size=internal_batch_size, show_progress_bar=False)

    # Encode positives in batch
    positives = [pos[0] for pos in batch["pos"]]
    pos_embeddings = model.encode(positives, convert_to_numpy=True, normalize_embeddings=True, batch_size=internal_batch_size, show_progress_bar=False)

    # Process each sample in the batch for negatives (since they're lists)
    for i in range(len(queries)):
        # Encode negatives for this sample
        neg_embeddings = model.encode(batch["neg"][i], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)

        # Calculate margin: positive score - max negative score
        query_neg_scores = query_embeddings[i] @ neg_embeddings.T
        max_neg_score = np.max(query_neg_scores)
        pos_score = query_embeddings[i] @ pos_embeddings[i]
        pos_margin = pos_score - max_neg_score

        margin_scores.append((float(pos_margin), batch_start + i))

    return margin_scores


def process_dataset_in_batches(dataset, model, batch_size=32):
    """Process dataset in batches to compute margin scores."""
    print(f"Processing {len(dataset)} samples in batches of {batch_size}...")

    margin_scores = []
    current_pos = 0
    pbar = tqdm(total=len(dataset), desc="Processing samples")

    # Process in batches with adaptive batch size
    while current_pos < len(dataset):
        batch_end = min(current_pos + batch_size, len(dataset))
        current_batch_size = batch_end - current_pos

        # Defense against OOM errors - reduce batch size if needed
        temp_batch_size = current_batch_size
        success = False

        while not success and temp_batch_size > 0:
            temp_batch_end = current_pos + temp_batch_size
            try:
                # Process this batch
                batch_results = process_single_batch(dataset, model, current_pos, temp_batch_end, internal_batch_size=min(temp_batch_size, 32))
                margin_scores.extend(batch_results)
                success = True

                # Update progress bar
                pbar.update(temp_batch_size)
                current_pos = temp_batch_end

            except Exception as e:
                # Reduce batch size and retry
                if temp_batch_size == 1:
                    print(f"Error processing batch starting at {current_pos}: {e}")
                    margin_scores.extend([(0, current_pos + i) for i in range(batch_size)])
                    pbar.update(batch_size)
                    current_pos = current_pos + batch_size
                    continue
                temp_batch_size = max(1, temp_batch_size // 2)
                print(f"Error processing batch {current_pos} to {temp_batch_end}: {e}")
                print(f"Retrying with reduced batch size: {temp_batch_size}")
                time.sleep(3)
                gc.collect()
                torch.cuda.empty_cache()

        # Clear memory periodically
        if current_pos % (batch_size * 3) == 0:
            gc.collect()
            torch.cuda.empty_cache()

    pbar.close()
    print(f"Processed {len(margin_scores)} samples in total")
    return margin_scores


def save_results(margin_scores, output_file, dataset):
    """Save filtered results to output file in original format."""
    print(f"Saving results to {output_file}...")

    with open(output_file, "w") as f:
        for margin_score, original_idx in margin_scores:
            # Get original data from dataset
            original_data = dataset[original_idx]
            f.write(json.dumps(original_data) + "\n")

    print(f"Saved {len(margin_scores)} samples to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Filter CodeSearchNet dataset by margin scores")
    parser.add_argument("--input", type=str, default="/data/wychanbu/re_data/codesearchnet-instruction-hard-neg.jsonl", help="Input JSONL file path")
    parser.add_argument("--output", type=str, default="/data/wychanbu/re_data/codesearchnet-filtered-by-margin-hard-neg.jsonl", help="Output JSONL file path")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-Embedding-8B", help="SentenceTransformer model name")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID to use (default: 0)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for encoding (default: 32)")
    parser.add_argument("--sort-order", type=str, choices=["ascending", "descending"], default="descending", help="Sort order for margin scores (default: descending = largest first)")
    parser.add_argument("--max-samples", type=int, default=350000, help="Maximum number of samples to retain after threshold filtering")
    parser.add_argument("--margin-threshold", type=float, default=0.7, help="Margin threshold - only keep samples with margin > threshold")

    args = parser.parse_args()

    # Load data as HuggingFace Dataset
    dataset = load_data_as_dataset(args.input)

    # Check that we have enough data
    if len(dataset) == 0:
        print("No data with 3 hard negatives found!")
        return

    # Load model on single GPU
    print(f"Loading model on GPU {args.gpu}...")
    model = SentenceTransformer(
        args.model,
        device=f"cuda:{args.gpu}",
        trust_remote_code=True,
        model_kwargs={"dtype": torch.bfloat16, "attn_implementation": "flash_attention_2"},
    )
    print(f"Model loaded successfully")

    margin_scores_parallel = process_dataset_in_batches(dataset, model, batch_size=args.batch_size)

    # Sort by margin score (descending to see distribution)
    margin_scores_parallel.sort(key=lambda x: x[0], reverse=True)

    print(f"\nMargin score statistics (before filtering):")
    print(f"  Largest margin score: {margin_scores_parallel[0]}")
    print(f"  Smallest margin score: {margin_scores_parallel[-1]}")
    print(f"  Mean margin score: {np.mean([x[0] for x in margin_scores_parallel]):.4f}")
    print(f"  Median margin score: {np.median([x[0] for x in margin_scores_parallel]):.4f}")
    print(f"  Total samples: {len(margin_scores_parallel)}")

    # Apply margin threshold filter first
    if args.margin_threshold is not None and args.margin_threshold > 0:
        print(f"\nApplying margin threshold <= {args.margin_threshold}...")
        samples_before = len(margin_scores_parallel)
        break_idx = -1
        for i, (score, _) in enumerate(margin_scores_parallel):
            if score <= args.margin_threshold:
                break_idx = i
                break
        margin_scores_parallel = margin_scores_parallel[break_idx + 1 :]
        print(f"  Samples after threshold filtering: {len(margin_scores_parallel)} (removed {samples_before - len(margin_scores_parallel)})")
        print(f"  Filtered margin range: [{margin_scores_parallel[0][0]:.4f}, {margin_scores_parallel[-1][0]:.4f}]")

    # Then apply max_samples limit (take first N samples after threshold)
    if args.max_samples is not None and args.max_samples < len(margin_scores_parallel):
        print(f"\nLimiting to {args.max_samples} samples after threshold...")
        margin_scores_parallel = margin_scores_parallel[: args.max_samples]
        print(f"  Final samples margin range: [{margin_scores_parallel[0][0]:.4f}, {margin_scores_parallel[-1][0]:.4f}]")

    # Apply final sort order
    reverse_sort = args.sort_order == "descending"
    margin_scores_parallel.sort(key=lambda x: x[0], reverse=reverse_sort)

    print(f"\nFinal statistics (after filtering, sorted {args.sort_order}):")
    print(f"  First sample margin score: {margin_scores_parallel[0]}")
    print(f"  Last sample margin score: {margin_scores_parallel[-1]}")
    print(f"  Total samples to save: {len(margin_scores_parallel)}")

    # Save results
    save_results(margin_scores_parallel, args.output, dataset)

    print("\nDone!")


if __name__ == "__main__":
    main()
