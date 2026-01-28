"""
LLM HuggingFace SLERP Merge

Spherical Linear Interpolation (SLERP) for merging Large Language Models.
Retrofitted from dvschultz's script to work with HuggingFace Pretrained Language Models.

Original Credits:
- Script base: dvschultz (https://gist.github.com/dvschultz/3af50c40df002da3b751efab1daddf2c)
- Adaptation: Chasm (AKA Digitous) and CalderaAI
- Linear interpolation methods: Concedo AKA LostRuins

Usage:
    python slerp_merge.py <primary_model_path> <secondary_model_path> <output_path> [--alpha 0.5]
"""

import argparse
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from colorama import Fore, Style, init
from transformers import AutoConfig, AutoModel

# Initialize colorama for cross-platform colored output
init()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def clear_console() -> None:
    """Clear the console screen based on the operating system."""
    if os.name == "nt":  # Windows
        subprocess.call("cls", shell=True)
    else:  # Linux and macOS
        subprocess.call("clear", shell=True)


def lerp(t: float, v0: Union[torch.Tensor, np.ndarray], v1: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """
    Linear interpolation between two vectors.

    Args:
        t: Interpolation factor (0.0 = v0, 1.0 = v1)
        v0: First vector
        v1: Second vector

    Returns:
        Linearly interpolated vector
    """
    return (1 - t) * v0 + t * v1


def slerp(t: float, v0: torch.Tensor, v1: torch.Tensor, dot_threshold: float = 0.9995, epsilon: float = 1e-10) -> torch.Tensor:
    """
    Spherical Linear Interpolation (SLERP) between two tensors.

    SLERP provides smooth interpolation between vectors by following the shortest
    path on the surface of a hypersphere. Falls back to linear interpolation
    when vectors are nearly collinear.

    Args:
        t: Interpolation factor (0.0 = v0, 1.0 = v1)
        v0: First tensor
        v1: Second tensor
        dot_threshold: Threshold for using linear interpolation instead of SLERP
        epsilon: Small value to avoid division by zero

    Returns:
        Interpolated tensor
    """
    # Convert tensors to float32 for consistency
    v0 = v0.to(dtype=torch.float32)
    v1 = v1.to(dtype=torch.float32)

    # Convert to numpy for mathematical operations
    was_tensor = True
    if not isinstance(v0, np.ndarray):
        v0_np = v0.detach().cpu().numpy()
    else:
        v0_np = v0
        was_tensor = False

    if not isinstance(v1, np.ndarray):
        v1_np = v1.detach().cpu().numpy()
    else:
        v1_np = v1

    # Store original vectors for interpolation
    v0_original = np.copy(v0_np)
    v1_original = np.copy(v1_np)

    # Normalize vectors for direction calculation
    norm_v0 = np.linalg.norm(v0_np)
    norm_v1 = np.linalg.norm(v1_np)

    if norm_v0 > epsilon:
        v0_normalized = v0_np / norm_v0
    else:
        logger.warning(f"Vector v0 has very small norm ({norm_v0}). Skipping normalization.")
        v0_normalized = v0_np

    if norm_v1 > epsilon:
        v1_normalized = v1_np / norm_v1
    else:
        logger.warning(f"Vector v1 has very small norm ({norm_v1}). Skipping normalization.")
        v1_normalized = v1_np

    # Calculate dot product of normalized vectors
    dot_product = np.sum(v0_normalized * v1_normalized)

    # Use linear interpolation if vectors are nearly collinear
    if np.abs(dot_product) > dot_threshold:
        logger.info(f"Vectors are nearly collinear, using linear interpolation")
        result = lerp(t, v0_original, v1_original)
    else:
        # Calculate SLERP
        theta_0 = np.arccos(np.clip(dot_product, -1.0, 1.0))  # Clip to avoid numerical errors
        sin_theta_0 = np.sin(theta_0)

        if abs(sin_theta_0) < epsilon:
            # Vectors are parallel, use linear interpolation
            logger.info(f"Vectors are parallel, using linear interpolation")
            result = lerp(t, v0_original, v1_original)
        else:
            logger.info(f"Vectors are not parallel, using SLERP")
            theta_t = theta_0 * t
            sin_theta_t = np.sin(theta_t)

            s0 = np.sin(theta_0 - theta_t) / sin_theta_0
            s1 = sin_theta_t / sin_theta_0
            result = s0 * v0_original + s1 * v1_original

    # Convert back to tensor if input was tensor
    if was_tensor:
        return torch.from_numpy(result)
    else:
        return result


def load_sharded_model(model_path: Union[str, Path]) -> Dict[str, torch.Tensor]:
    """
    Load a sharded PyTorch model from multiple .bin files.

    Args:
        model_path: Path to directory containing model shards

    Returns:
        Dictionary containing the combined state dict

    Raises:
        FileNotFoundError: If no model shards are found
        ValueError: If shard files cannot be parsed properly
    """
    model_path = Path(model_path)
    state_dict = {}

    # Find all shard files
    shard_files = [f for f in os.listdir(model_path) if f.startswith("pytorch_model") and f.endswith(".bin")]

    if not shard_files:
        raise FileNotFoundError(f"No model shards found in {model_path}")

    # Sort shards by number for consistent loading order
    try:
        sorted_shards = sorted(shard_files, key=lambda x: int(x.split("-")[1]))
    except (IndexError, ValueError) as e:
        logger.warning(f"Could not parse shard numbers, using alphabetical order: {e}")
        sorted_shards = sorted(shard_files)

    logger.info(f"Loading {len(sorted_shards)} model shards from {model_path}")

    for shard_file in sorted_shards:
        shard_path = model_path / shard_file
        try:
            shard = torch.load(shard_path, map_location="cpu")
            state_dict.update(shard)
            logger.debug(f"Loaded shard: {shard_file}")
        except Exception as e:
            logger.error(f"Failed to load shard {shard_file}: {e}")
            raise

    return state_dict


def load_model(model_path: Union[str, Path]) -> Dict[str, torch.Tensor]:
    """
    Load a PyTorch model, handling both single-file and sharded models.

    Args:
        model_path: Path to model directory

    Returns:
        Dictionary containing the state dict

    Raises:
        FileNotFoundError: If no model files are found
    """
    model_path = Path(model_path)
    single_file_path = model_path / "pytorch_model.bin"

    if single_file_path.exists():
        logger.info(f"Loading single model file from {single_file_path}")
        try:
            state_dict = torch.load(single_file_path, map_location="cpu")
            return state_dict
        except Exception as e:
            logger.error(f"Failed to load model from {single_file_path}: {e}")
            raise
    else:
        logger.info(f"Single model file not found, attempting to load sharded model")
        return load_sharded_model(model_path)


def save_model(model: torch.Tensor, save_path: Union[str, Path]) -> None:
    """
    Save a PyTorch model to disk.

    Args:
        model: Model tensor to save
        save_path: Path where to save the model

    Raises:
        OSError: If the model cannot be saved
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        torch.save(model, save_path)
        logger.info(f"Model saved to {save_path}")
    except Exception as e:
        logger.error(f"Failed to save model to {save_path}: {e}")
        raise


def pad_state_dicts_for_compatibility(primary_state_dict: Dict[str, torch.Tensor], secondary_state_dict: Dict[str, torch.Tensor]) -> None:
    """
    Pad state dictionaries to ensure compatibility for model merging.

    This function handles size mismatches between corresponding tensors in two
    state dictionaries by padding the smaller tensor with zeros. It also adds
    missing keys from one model to the other with zero-initialized tensors.

    Args:
        primary_state_dict: State dictionary of the primary model (modified in-place)
        secondary_state_dict: State dictionary of the secondary model (modified in-place)

    Raises:
        AssertionError: If vocabulary sizes don't match after padding
        KeyError: If expected embedding keys are missing
    """
    logger.info("Padding state dictionaries for compatibility...")

    with torch.no_grad():
        # Get keys present in both models
        common_keys = set(primary_state_dict.keys()).intersection(set(secondary_state_dict.keys()))
        logger.info(f"There are {len(primary_state_dict.keys())} keys in primary model and {len(secondary_state_dict.keys())} keys in secondary model")
        logger.info(f"Found {len(common_keys)} common parameters between models")

        # Handle size mismatches for common keys
        mismatched_keys = []
        for key in common_keys:
            tensor1 = primary_state_dict[key]
            tensor2 = secondary_state_dict[key]

            if tensor1.size() != tensor2.size():
                mismatched_keys.append(key)
                logger.debug(f"Size mismatch for {key}: {tensor1.size()} vs {tensor2.size()}")

                # Only pad along the first dimension (typically vocabulary/feature dimension)
                if tensor1.size(0) != tensor2.size(0):
                    if tensor1.size(0) < tensor2.size(0):
                        # Pad primary tensor
                        padding_size = tensor2.size(0) - tensor1.size(0)
                        padding_shape = (padding_size,) + tensor1.size()[1:]
                        padding = torch.zeros(padding_shape, device=tensor1.device, dtype=tensor1.dtype)
                        primary_state_dict[key] = torch.cat([tensor1, padding], dim=0)
                        logger.debug(f"Padded primary model {key} with {padding_size} zeros")
                    else:
                        # Pad secondary tensor
                        padding_size = tensor1.size(0) - tensor2.size(0)
                        padding_shape = (padding_size,) + tensor2.size()[1:]
                        padding = torch.zeros(padding_shape, device=tensor2.device, dtype=tensor2.dtype)
                        secondary_state_dict[key] = torch.cat([tensor2, padding], dim=0)
                        logger.debug(f"Padded secondary model {key} with {padding_size} zeros")
                else:
                    logger.warning(f"Size mismatch in non-first dimension for {key}, skipping padding")

        if mismatched_keys:
            logger.info(f"Resolved size mismatches for {len(mismatched_keys)} parameters")

        # Add missing keys from primary to secondary model
        primary_only_keys = set(primary_state_dict.keys()) - set(secondary_state_dict.keys())
        for key in primary_only_keys:
            tensor = primary_state_dict[key]
            secondary_state_dict[key] = torch.zeros_like(tensor)
            logger.debug(f"Added missing parameter {key} to secondary model")

        # Add missing keys from secondary to primary model
        secondary_only_keys = set(secondary_state_dict.keys()) - set(primary_state_dict.keys())
        for key in secondary_only_keys:
            tensor = secondary_state_dict[key]
            primary_state_dict[key] = torch.zeros_like(tensor)
            logger.debug(f"Added missing parameter {key} to primary model")

        if primary_only_keys or secondary_only_keys:
            logger.info(f"Added {len(primary_only_keys) + len(secondary_only_keys)} missing parameters")

        # Verify vocabulary sizes match (if embedding layers exist)
        embedding_key = "embed_tokens.weight"
        if embedding_key in primary_state_dict and embedding_key in secondary_state_dict:
            primary_vocab_size = primary_state_dict[embedding_key].size(0)
            secondary_vocab_size = secondary_state_dict[embedding_key].size(0)

            if primary_vocab_size != secondary_vocab_size:
                raise AssertionError(f"Vocabulary sizes do not match after padding: " f"{primary_vocab_size} vs {secondary_vocab_size}")

            logger.info(f"Vocabulary size verification passed: {primary_vocab_size}")
        else:
            logger.warning("No embedding layer found for vocabulary size verification")


def interpolate_state_dicts(primary_state_dict: Dict[str, torch.Tensor], secondary_state_dict: Dict[str, torch.Tensor], alpha: float = 0.5) -> Dict[str, torch.Tensor]:
    """
    Interpolate between two state dictionaries using SLERP.

    Args:
        primary_state_dict: Primary model state dictionary
        secondary_state_dict: Secondary model state dictionary
        alpha: Interpolation factor (0.0 = primary only, 1.0 = secondary only)

    Returns:
        Interpolated state dictionary
    """
    logger.info(f"Interpolating state dictionaries with alpha={alpha}")
    interpolated_dict = {}

    # Get all unique keys from both dictionaries
    all_keys = set(primary_state_dict.keys()).union(set(secondary_state_dict.keys()))
    logger.info(f"Interpolating {len(all_keys)} parameters")

    skipped_keys = []
    interpolated_keys = []

    for key in all_keys:
        if key in primary_state_dict and key in secondary_state_dict:
            tensor1 = primary_state_dict[key]
            tensor2 = secondary_state_dict[key]

            # Check if both values are tensors
            if isinstance(tensor1, torch.Tensor) and isinstance(tensor2, torch.Tensor):
                # Use SLERP interpolation
                interpolated_dict[key] = slerp(alpha, tensor1, tensor2)
                interpolated_keys.append(key)
            else:
                logger.warning(f"Skipping {key}: not both tensors")
                interpolated_dict[key] = tensor1  # Use primary as default
                skipped_keys.append(key)
        elif key in secondary_state_dict:
            # Key only in secondary model
            interpolated_dict[key] = secondary_state_dict[key]
        else:
            # Key only in primary model
            interpolated_dict[key] = primary_state_dict[key]

    logger.info(f"Successfully interpolated {len(interpolated_keys)} parameters")
    if skipped_keys:
        logger.warning(f"Skipped {len(skipped_keys)} non-tensor parameters")

    return interpolated_dict


def copy_model_files(primary_model_path: Union[str, Path], secondary_model_path: Union[str, Path], output_path: Union[str, Path]) -> None:
    """
    Copy necessary tokenizer and configuration files to the output directory.

    Args:
        primary_model_path: Path to primary model directory
        secondary_model_path: Path to secondary model directory
        output_path: Path to output directory
    """
    primary_path = Path(primary_model_path)
    secondary_path = Path(secondary_model_path)
    output_path = Path(output_path)

    # Files that should be copied from the source model
    files_to_copy = ["special_tokens_map.json", "tokenizer_config.json", "vocab.json", "tokenizer.model", "generation_config.json", "added_tokens.json", "merges.txt"]

    # Determine which model to copy files from based on tokenizer completeness
    special_tokens_primary = (primary_path / "special_tokens_map.json").exists()
    special_tokens_secondary = (secondary_path / "special_tokens_map.json").exists()

    if special_tokens_primary and not special_tokens_secondary:
        source_path = primary_path
        logger.info("Using primary model as source for auxiliary files")
    else:
        source_path = secondary_path
        logger.info("Using secondary model as source for auxiliary files")

    # Copy each file
    copied_files = []
    skipped_files = []

    for filename in files_to_copy:
        src_file = source_path / filename
        dst_file = output_path / filename

        if src_file.exists():
            try:
                shutil.copy2(src_file, dst_file)
                copied_files.append(filename)
                logger.debug(f"Copied {filename}")
            except Exception as e:
                logger.error(f"Failed to copy {filename}: {e}")
                skipped_files.append(filename)
        else:
            skipped_files.append(filename)
            logger.debug(f"File {filename} not found in source model")

    logger.info(f"Copied {len(copied_files)} auxiliary files")
    if skipped_files:
        logger.info(f"Skipped {len(skipped_files)} missing files: {skipped_files}")


def create_merged_model(interpolated_state_dict: Dict[str, torch.Tensor], primary_model_path: Union[str, Path], output_path: Union[str, Path]) -> None:
    """
    Create and save the merged model with updated configuration.

    Args:
        interpolated_state_dict: The interpolated state dictionary
        primary_model_path: Path to primary model (for config)
        output_path: Path to save the merged model
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load configuration from primary model
    try:
        config = AutoConfig.from_pretrained(primary_model_path)
        logger.info(f"Loaded configuration from {primary_model_path}")
    except Exception as e:
        logger.error(f"Failed to load config from {primary_model_path}: {e}")
        raise

    # Convert any numpy arrays back to tensors
    for key, value in interpolated_state_dict.items():
        if isinstance(value, np.ndarray):
            interpolated_state_dict[key] = torch.tensor(value)

    # Update vocabulary size if needed
    embedding_key = "embed_tokens.weight"
    if embedding_key in interpolated_state_dict:
        resulting_vocab_size = interpolated_state_dict[embedding_key].size(0)
        if config.vocab_size != resulting_vocab_size:
            logger.info(f"Updating config vocab size from {config.vocab_size} to {resulting_vocab_size}")
            config.vocab_size = resulting_vocab_size

    # Create model from config and load the interpolated state dict
    try:
        model = AutoModel.from_config(config)
        model.load_state_dict(interpolated_state_dict)
        model.save_pretrained(output_path, safe_serialization=False)
        logger.info(f"Merged model saved to {output_path} as pytorch_model.bin")
    except Exception as e:
        logger.error(f"Failed to create and save merged model: {e}")
        raise


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Merge two HuggingFace language models using Spherical Linear Interpolation (SLERP)", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("primary_model", type=str, help="Path to the primary model directory")

    parser.add_argument("secondary_model", type=str, help="Path to the secondary model directory")

    parser.add_argument("output_path", type=str, help="Path to save the merged model")

    parser.add_argument("--alpha", type=float, default=0.5, help="Interpolation factor (0.0 = primary only, 1.0 = secondary only)")

    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    return parser.parse_args()


def main() -> None:
    """Main function to orchestrate the model merging process."""
    clear_console()
    print(f"{Fore.YELLOW}Starting {Fore.GREEN}spherical linear interpolation{Fore.YELLOW} script...{Style.RESET_ALL}")

    # Parse arguments
    args = parse_arguments()

    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("=" * 60)
    logger.info("SLERP Model Merging")
    logger.info("=" * 60)
    logger.info(f"Primary model: {args.primary_model}")
    logger.info(f"Secondary model: {args.secondary_model}")
    logger.info(f"Output path: {args.output_path}")
    logger.info(f"Alpha (interpolation factor): {args.alpha}")

    try:
        # Load models
        logger.info("\n--- Loading Models ---")
        primary_state_dict = load_model(args.primary_model)
        secondary_state_dict = load_model(args.secondary_model)

        # Pad state dictionaries for compatibility
        logger.info("\n--- Preparing Models ---")
        pad_state_dicts_for_compatibility(primary_state_dict, secondary_state_dict)

        # Interpolate state dictionaries
        logger.info("\n--- Interpolating Models ---")
        merged_state_dict = interpolate_state_dicts(primary_state_dict, secondary_state_dict, args.alpha)

        # Create and save merged model
        logger.info("\n--- Creating Merged Model ---")
        create_merged_model(merged_state_dict, args.primary_model, args.output_path)

        # Copy auxiliary files
        logger.info("\n--- Copying Auxiliary Files ---")
        copy_model_files(args.primary_model, args.secondary_model, args.output_path)

        # Success message
        print(f"\n{Fore.GREEN}✓ Model merging completed successfully!{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Merged model saved to: {args.output_path}{Style.RESET_ALL}")

    except Exception as e:
        logger.error(f"Model merging failed: {e}")
        print(f"\n{Fore.RED}✗ Model merging failed: {e}{Style.RESET_ALL}")
        sys.exit(1)


if __name__ == "__main__":
    main()
