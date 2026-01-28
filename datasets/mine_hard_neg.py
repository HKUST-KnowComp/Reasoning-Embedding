import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import logging
import multiprocessing
import time

import torch
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import mine_hard_negatives

from datasets import Dataset, load_dataset

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()


def flatten_data(example):
    """Optimized flatten function with error handling"""
    try:
        # remove the instruction from the query
        example["query"] = example["query"][1]
        # extract the positive from the pos
        example["pos"] = example["pos"][0]
        return example
    except (IndexError, KeyError) as e:
        logger.warning(f"Error flattening data: {e}, skipping example")
        return None


def save_dataset(dataset: list[dict], output_path: str) -> None:
    dataset = Dataset.from_list(dataset)
    dataset.to_json(output_path)


def mine_hard_neg(dataset, model: SentenceTransformer, num_negatives=3, batch_size=32):
    """
    Optimized hard negative mining function.

    The dataset are in the format of:
    {
        "query": [instruction, query],
        "pos": [pos],
        "neg": [neg] / [] (if there is no negative) / [neg1, neg2, ...] (if there are already annotated negatives)
    }
    """
    torch.cuda.empty_cache()

    start_time = time.time()
    logger.info(f"Starting hard negative mining for {len(dataset)} examples")

    # Optimized flattening with parallel processing and filtering
    flattened_dataset = dataset.map(flatten_data, num_proc=min(4, multiprocessing.cpu_count()))
    # Filter out None values from failed flattening
    flattened_dataset = flattened_dataset.filter(lambda x: x is not None)

    logger.info(f"Flattened {len(flattened_dataset)} examples in {time.time() - start_time:.2f}s")

    # Optimized query index mapping - removed unnecessary ThreadPoolExecutor
    # we cannot use only query because there are multiple positives for the same query
    query_index_map = {example["query"] + example["pos"]: idx for idx, example in enumerate(flattened_dataset)}

    logger.info(f"Created query index map with {len(query_index_map)} examples in {time.time() - start_time:.2f}s")

    # filter duplicate sample using query_index_map
    if len(flattened_dataset) != len(query_index_map):
        flattened_dataset = flattened_dataset.filter(lambda x, idx: query_index_map[x["query"] + x["pos"]] == idx, with_indices=True)

    logger.info(f"Remained {len(flattened_dataset)} examples in {time.time() - start_time:.2f}s")

    # Convert original dataset to list more efficiently
    dataset_list = list(dataset)

    # With Faiss, we can process the whole dataset at once without running out of memory.
    # It's much more efficient than computing the full similarity matrix.
    # Note: You may need to install faiss: `pip install faiss-cpu` or `pip install faiss-gpu`
    dataset_after_mining = mine_hard_negatives(
        dataset=flattened_dataset,
        model=model,
        anchor_column_name="query",
        positive_column_name="pos",
        relative_margin=0.05,  # 0.05 means that the negative is at most 95% as similar to the anchor as the positive
        num_negatives=num_negatives,  # 10 or less is recommended
        sampling_strategy="top",  # "top" means that we sample the top candidates as negatives
        batch_size=batch_size,  # Adjust as needed
        use_faiss=True,  # Use Faiss for memory-efficient and fast search
        output_format="n-tuple",
        range_min=0,
        range_max=min(100, int(len(flattened_dataset) * 0.1)),
    )

    logger.info(f"Mined negatives for {len(dataset_after_mining)} examples")

    # Reformat and update the main dataset_list
    reformat_start = time.time()
    for example in dataset_after_mining:
        map_key = example["query"] + example["pos"]
        if map_key in query_index_map:
            original_index = query_index_map[map_key]
            new_negatives = [example[f"negative_{j+1}"] for j in range(num_negatives)]
            dataset_list[original_index]["neg"].extend(new_negatives)

    logger.info(f"Dataset reformatting completed in {time.time() - reformat_start:.2f}s")

    logger.info(f"Total processing time: {time.time() - start_time:.2f}s")

    torch.cuda.empty_cache()

    return dataset_list


def process_single_file(input_file, data_dir, model, num_negatives, batch_size, output_dir=None):
    """Process a single file - designed for parallel execution"""

    try:
        start_time = time.time()
        logger.info(f"Processing file: {input_file}")

        dataset_path = os.path.join(data_dir, input_file)
        dataset = load_dataset("json", data_files=dataset_path, split="train", cache_dir=os.getenv("data_cache_dir"))

        # Check if dataset has existing negatives
        if len(dataset) > 0 and 0 < len(dataset[0]["neg"]) < num_negatives:
            target_num_negatives = num_negatives - len(dataset[0]["neg"])
        else:
            target_num_negatives = num_negatives

        # Mine hard negatives
        dataset = mine_hard_neg(dataset, model, num_negatives=target_num_negatives, batch_size=batch_size)

        # Clean filename
        output_file = input_file
        if "no-neg" in output_file:
            output_file = output_file.replace("-no-neg", "")

        if output_dir is None:
            output_dir = data_dir

        output_path = os.path.join(output_dir, f"{output_file.replace('.jsonl', '-hard-neg.jsonl')}")
        save_dataset(dataset, output_path)

        processing_time = time.time() - start_time
        logger.info(f"Completed processing {input_file} in {processing_time:.2f}s")

        return f"Successfully processed {input_file} in {processing_time:.2f}s"

    except Exception as e:
        error_msg = f"Error processing {input_file}: {str(e)}"
        logger.error(error_msg)
        return error_msg


if __name__ == "__main__":
    overall_start = time.time()

    # Load model once and reuse
    logger.info("Loading SentenceTransformer model...")
    model_start = time.time()
    model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", cache_folder=os.getenv("model_cache_dir"), device="cuda:6")
    logger.info(f"Model loaded in {time.time() - model_start:.2f}s on device: {model.device}")

    # Optimized parameters
    num_negatives = 3
    batch_size = 2  # Increased batch size for better GPU utilization

    data_dir = "/data/wychanbu/re_data"
    output_dir = None
    # output_dir = "/data/wychanbu/re_data/hard-neg"

    # Get all JSONL files that have "no-neg" in the filename
    jsonl_files = [file for file in os.listdir(data_dir) if file.endswith(".jsonl") and "-no-neg" in file]
    logger.info(f"Found {len(jsonl_files)} JSONL files to process")

    # Option 1: Sequential processing (safer for GPU memory)
    # Comment this out if you want to use parallel processing
    for file in jsonl_files:
        result = process_single_file(file, data_dir, model, num_negatives, batch_size, output_dir=output_dir)
        logger.info(result)

    # result = process_single_file(jsonl_files[0], data_dir, model, num_negatives, batch_size, output_dir=output_dir)

    # Option 2: Parallel processing (uncomment if you have enough GPU memory)
    # WARNING: This may cause GPU memory issues if processing large files simultaneously
    # max_workers = min(2, len(jsonl_files))  # Limit to 2 to avoid GPU memory issues
    # with ThreadPoolExecutor(max_workers=max_workers) as executor:
    #     args_list = [(file, data_dir, model, num_negatives, batch_size) for file in jsonl_files]
    #     results = list(executor.map(process_single_file, args_list))
    #     for result in results:
    #         logger.info(result)

    total_time = time.time() - overall_start
    logger.info(f"All files processed in {total_time:.2f}s")
