#!/usr/bin/env python3
"""
Shared helpers for dataset loading.
"""
from __future__ import annotations

import gc
import torch
from colorama import Fore, Style
from typing import List, Optional, Tuple
import os
import re
from typing import Union

def clear_memory():
    """
    Clear memory by collecting garbage, emptying the cache, and synchronizing the GPU.
    """
    gc.collect()
    if torch is not None:
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
    # torch.cuda.synchronize()


PathLike = Union[str, os.PathLike]

def sanitize_model_name(model_path: Union[str, os.PathLike], fallback: str = "unknown_model") -> str:
    """
    Produce a sanitized version of a model identifier that is safe to use in file paths.

    This helper keeps the final component of the provided path, while preserving a
    trailing checkpoint directory (e.g., ``.../model/checkpoint-200`` becomes
    ``model_checkpoint-200``). It also normalizes separators and removes characters
    that are not alphanumeric, ``-``, ``_``, or ``.``.

    Args:
        model_path: Original model identifier or path-like object.
        fallback: Value returned if the input cannot be sanitized (defaults to ``unknown_model``).

    Returns:
        Sanitized model name suitable for directory and file names.
    """
    if model_path is None:
        return fallback

    try:
        normalized_path = os.fspath(model_path)
    except TypeError:
        return fallback

    parts = [part for part in re.split(r"[\\/]+", normalized_path) if part]
    if not parts:
        return fallback

    last_part = parts[-1]
    if "checkpoint-" in last_part and len(parts) >= 2:
        candidate = f"{parts[-2]}_{last_part}"
    else:
        candidate = last_part

    candidate = candidate.strip()
    sanitized = re.sub(r"[^\w\-\.]+", "_", candidate)
    sanitized = sanitized.strip("_")

    return sanitized or fallback

# ---------------------------------------------------------------------------
# Dtype configuration
# ---------------------------------------------------------------------------

DTYPE_ALIASES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
}


def parse_compute_dtype(dtype_name: str) -> torch.dtype:
    """
    Parse a user-provided dtype string to a torch dtype.

    Args:
        dtype_name: One of 'bfloat16', 'float16', 'float32', 'float64'.

    Returns:
        Corresponding torch.dtype.

    Raises:
        ValueError: If dtype_name is not recognized.
    """
    key = dtype_name.lower()
    if key not in DTYPE_ALIASES:
        available = ", ".join(sorted(DTYPE_ALIASES))
        raise ValueError(f"Unknown compute dtype '{dtype_name}'. Available: {available}")
    return DTYPE_ALIASES[key]


def dtype_to_str(dtype: torch.dtype) -> str:
    """Convert a torch dtype to its string name for caching and logging."""
    for name, dt in DTYPE_ALIASES.items():
        if dt == dtype:
            return name
    return str(dtype).replace("torch.", "")

# ---------------------------------------------------------------------------
# Dataset configuration
# ---------------------------------------------------------------------------

def canonical_dataset_slug(dataset_path: PathLike, fallback: str = "dataset") -> str:
    """
    Produce a canonical slug for dataset identifiers or file paths.

    Args:
        dataset_path: Dataset identifier or path-like object.
        fallback: Value returned when the dataset path cannot be resolved.

    Returns:
        Slug with extensions removed and unsafe characters normalized.
    """
    if dataset_path is None:
        return fallback

    try:
        normalized_path = os.fspath(dataset_path)
    except TypeError:
        return fallback

    normalized_path = normalized_path.strip()
    if not normalized_path:
        return fallback

    normalized_path = normalized_path.rstrip("/\\")
    if not normalized_path:
        return fallback

    parts = [part for part in re.split(r"[\\/]+", normalized_path) if part]
    candidate = parts[-1] if parts else normalized_path
    candidate = candidate.strip()

    candidate = re.sub(r"\.(jsonl?|json|txt|csv)$", "", candidate, flags=re.IGNORECASE)
    slug = re.sub(r"[^\w\-\.]+", "_", candidate).strip("_")

    return slug or fallback


def get_model(
    model_name_or_path: str,
    device: str,
    is_causal_attn: bool = False,
    dtype: Optional[torch.dtype] = torch.bfloat16,
):
    """Load a pretrained model and tokenizer for hidden-state extraction."""
    print(f"{Fore.YELLOW}Loading model: {model_name_or_path} (dtype={dtype}){Style.RESET_ALL}")

    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ImportError("Please install transformers: pip install transformers") from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModel.from_pretrained(
        model_name_or_path,
        attn_implementation="flash_attention_2",
        dtype=dtype,
        device_map=device,
    )
    model.eval()

    def f_model_fn(text):
        """Convert text (str or list[str]) to hidden state features."""
        if isinstance(text, str):
            text = [text]

        inputs = tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=False,
            padding=True,
            truncation=True,
            max_length=131072,
        ).to(device)
        with torch.no_grad():
            outputs = model(
                **inputs,
                output_hidden_states=True,
                is_causal=is_causal_attn,
            )
            hidden_states = outputs.hidden_states
            attention_mask = inputs["attention_mask"]

        del inputs, outputs
        clear_memory()
        return hidden_states, attention_mask

    return f_model_fn, model, tokenizer

def load_sentences_and_labels(
    dataset_name_or_path: str,
    text_column: str,
    label_column: Optional[str],
    subset: str,
    split: str,
    num_sentences: int,
) -> Tuple[List[str], Optional[List[str]]]:
    """Load sentences (and optional labels) for downstream analyses."""
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ImportError("Please install datasets: pip install datasets") from exc

    print(f"{Fore.YELLOW}Loading data from: {dataset_name_or_path}{Style.RESET_ALL}")

    if dataset_name_or_path.endswith(".jsonl") or dataset_name_or_path.endswith(".json"):
        dataset = load_dataset("json", data_files=dataset_name_or_path, split=split)
    else:
        dataset = load_dataset(dataset_name_or_path, subset, split=split)

    if text_column not in dataset.column_names:
        raise ValueError(f"Column '{text_column}' not found. Available: {dataset.column_names}")

    labels: Optional[List[str]] = None
    if label_column is not None:
        if label_column not in dataset.column_names:
            raise ValueError(
                f"Label column '{label_column}' not found. Available: {dataset.column_names}"
            )
        labels = [str(label) for label in dataset[label_column][:num_sentences]]

    limit = min(num_sentences, len(dataset))
    if limit < len(dataset):
        dataset = dataset.select(range(limit))

    sentences = dataset[text_column]
    print(f"{Fore.GREEN}Loaded {len(sentences)} sentences{Style.RESET_ALL}")
    return sentences, labels


def get_data(
    dataset_name_or_path: str,
    num_sentences: int = 2000,
    text_column: str = "text",
    subset: str = "main",
    split: str = "train",
) -> List[str]:
    """Load dataset text from HuggingFace or local JSON/JSONL."""
    print(f"{Fore.YELLOW}Loading data from: {dataset_name_or_path}{Style.RESET_ALL}")

    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ImportError("Please install datasets: pip install datasets") from exc

    if dataset_name_or_path.endswith(".jsonl") or dataset_name_or_path.endswith(".json"):
        print(f"{Fore.MAGENTA}Loading local JSON/JSONL file...{Style.RESET_ALL}")
        dataset = load_dataset("json", data_files=dataset_name_or_path, split=split)
    else:
        print(f"{Fore.MAGENTA}Loading HuggingFace dataset...{Style.RESET_ALL}")
        dataset = load_dataset(dataset_name_or_path, subset, split=split)

    if text_column not in dataset.column_names:
        available_cols = dataset.column_names
        raise ValueError(f"Column '{text_column}' not found. Available columns: {available_cols}")

    data = [item[text_column] for item in dataset][:num_sentences]
    print(f"{Fore.GREEN}Loaded {len(data)} sentences{Style.RESET_ALL}")
    return data