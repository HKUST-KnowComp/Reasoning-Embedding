import json
import os
from datetime import datetime
from typing import Optional

from dataset_config import DEFAULT_DATASET_CONFIG, MATH500_DATASET_CONFIG, DatasetConfig
from openai import OpenAI

from datasets import load_dataset


def _get_output_path(dataset_name: str, source_filter: Optional[str] = None) -> str:
    """Generate dataset-specific output filename."""
    # Sanitize dataset name for filename: replace / with _ and remove special chars
    safe_name = dataset_name.replace("/", "_").replace(":", "_")
    if source_filter:
        filename = f"qwen3_32b_{safe_name}_{source_filter}_outputs.jsonl"
    else:
        filename = f"qwen3_32b_{safe_name}_outputs.jsonl"

    # Store outputs in outputs/ directory
    output_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, filename)


def _get_completed_question_ids(output_path: str) -> set:
    """Load already-processed question IDs from output file for resume support."""
    completed = set()
    if not os.path.exists(output_path):
        return completed

    try:
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    completed.add(record.get("question_id"))
    except (OSError, json.JSONDecodeError):
        # If file is corrupted or unreadable, start fresh
        pass

    return completed


def _load_dotenv(env_path: str = ".env") -> None:
    """Minimal .env loader (KEY=VALUE lines) without extra deps.

    Only sets variables that are not already present in os.environ.
    This lets the script pick up DASHSCOPE_API_KEY / DASHSCOPE_BASE_URL
    from the project-level .env automatically.
    """

    if not os.path.exists(env_path):
        return

    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
    except OSError:
        # Fail silently; user can still rely on real environment vars
        pass


def append_jsonl(path: str, record: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def make_client(api_key_suffix: str = "") -> OpenAI:
    """Create OpenAI client with optional API key suffix for multiple keys.

    Args:
        api_key_suffix: Optional suffix to select different API key.
                       E.g., "_2" looks for DASHSCOPE_API_KEY_2
    """
    # Load env vars from repo-level .env if present
    _load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))

    # Try with suffix first, then fallback to default
    if api_key_suffix:
        api_key = os.getenv(f"DASHSCOPE_API_KEY{api_key_suffix}") or os.getenv(f"BAILIAN_API_KEY{api_key_suffix}")
    else:
        api_key = None

    if not api_key:
        api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("BAILIAN_API_KEY")

    base_url = os.getenv("DASHSCOPE_BASE_URL") or os.getenv("BAILIAN_API_ENDPOINT") or "https://dashscope.aliyuncs.com/compatible-mode/v1"

    if not api_key:
        raise RuntimeError("Missing DASHSCOPE_API_KEY or BAILIAN_API_KEY in environment/.env")

    return OpenAI(api_key=api_key, base_url=base_url)


def make_deepseek_client() -> OpenAI:
    """Create OpenAI client for DeepSeek-V3.2 evaluation via Aliyun Bailian."""
    _load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))

    # DeepSeek-V3.2 is available through Aliyun Bailian with the same API key
    deepseek_key = os.getenv("DASHSCOPE_API_KEY")
    if not deepseek_key:
        raise RuntimeError("Missing DASHSCOPE_API_KEY in environment/.env for DeepSeek evaluation")

    return OpenAI(api_key=deepseek_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")


def classify_answer_correctness(evaluator_client: OpenAI, problem: str, answer: str) -> int:
    """Use DeepSeek-V3.2 to classify if the answer is correct (binary: 0 or 1).

    Args:
        evaluator_client: OpenAI client configured for DeepSeek
        problem: The math problem statement
        answer: The model's answer to evaluate

    Returns:
        1 if correct, 0 if incorrect
    """
    evaluation_prompt = f"""You are an expert math problem evaluator. Your task is to determine if the provided answer correctly solves the given problem.

Problem:
{problem}

Answer:
{answer}

Evaluate whether the answer is correct. Respond with ONLY "1" if the answer is correct, or "0" if it is incorrect. Do not provide any explanation."""

    try:
        response = evaluator_client.chat.completions.create(
            model="deepseek-v3.2-exp",
            messages=[{"role": "system", "content": "You are a precise math answer evaluator. Respond only with 0 or 1."}, {"role": "user", "content": evaluation_prompt}],
            temperature=0,
            max_tokens=10,
            # Disable thinking mode for DeepSeek-V3.2 to get direct answer only
            extra_body={"enable_thinking": False},
        )

        result = response.choices[0].message.content.strip()
        # Parse the result, default to 0 if unclear
        if "1" in result:
            return 1
        else:
            return 0
    except Exception as e:
        # If evaluation fails, mark as 0 and log error
        print(f"Warning: DeepSeek evaluation failed: {e}")
        return 0


def solve_problem(
    client: OpenAI,
    problem: str,
    question_id: str,
    split: str,
    model: str = "qwen3-32b",
    dataset_name: str = None,
    source_filter: Optional[str] = None,
    max_cot_tokens: int = 8000,
    evaluator_client: Optional[OpenAI] = None,
) -> bool:
    """Send one competition math problem to Qwen3-32B with streaming CoT and log the result.

    All logging is written to JSONL; no stdout printing to keep terminal clean.

    Args:
        max_cot_tokens: Maximum tokens for CoT reasoning (default: 8000)
        evaluator_client: Optional DeepSeek client for answer evaluation

    Returns:
        True if problem was processed successfully, False if skipped due to length limit.
    """

    user_prompt = "You are an expert competition math solver. Read the problem carefully and solve it step by step.\n\n " "Problem:\n" + problem

    messages = [
        {"role": "system", "content": "You are a helpful and rigorous math reasoning assistant."},
        {"role": "user", "content": user_prompt},
    ]

    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=11000,
        temperature=0.6,
        top_p=0.95,
        extra_body={"enable_thinking": True, "top_k": 20},
        stream=True,
    )

    reasoning_text = ""
    answer_text = ""
    is_answering = False
    exceeded_limit = False
    reasoning_tokens = 0

    for chunk in completion:
        delta = chunk.choices[0].delta

        # Model's internal reasoning
        if hasattr(delta, "reasoning_content") and delta.reasoning_content is not None:
            if not is_answering and not exceeded_limit:
                chunk_text = delta.reasoning_content
                reasoning_text += chunk_text
                # Rough token count: ~1.3 chars per token for Chinese/English mixed text
                reasoning_tokens = len(reasoning_text) // 1.3
                # Check token count during streaming
                if reasoning_tokens > max_cot_tokens:
                    exceeded_limit = True
                    # Stop consuming the stream
                    break

        # Final answer content
        if hasattr(delta, "content") and delta.content:
            if not is_answering:
                is_answering = True
            answer_text += delta.content

    # Log even if exceeded token limit (to prevent re-processing)
    if exceeded_limit:
        # Log a record with truncated reasoning but still evaluate the answer if available
        record = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "dataset": dataset_name or "unknown",
            "split": split,
            "question_id": question_id,
            "model": model,
            "cot_length_words": len(reasoning_text.split()),
            "cot_length_tokens": int(reasoning_tokens),  # Estimated token count
            "problem": problem,
            "messages": messages,
            "reasoning": reasoning_text,  # Store the truncated CoT that was generated
            "answer": answer_text if answer_text else "[INCOMPLETE: CoT truncated before answer]",
            "skipped": True,
            "skip_reason": "cot_exceeded_max_tokens",
        }

        # Evaluate answer if we got one before truncation
        if answer_text and evaluator_client:
            correctness_label = classify_answer_correctness(evaluator_client, problem, answer_text)
            record["correctness_label"] = correctness_label
            record["evaluator_model"] = "deepseek-v3.2-exp"
        elif not answer_text:
            # No answer available due to early truncation
            record["correctness_label"] = -1  # Marker for no answer to evaluate

        output_path = _get_output_path(dataset_name or "unknown", source_filter)
        append_jsonl(output_path, record)
        return False

    record = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "dataset": dataset_name or "unknown",
        "split": split,
        "question_id": question_id,
        "model": model,
        "cot_length_words": len(reasoning_text.split()),
        "cot_length_tokens": int(len(reasoning_text) // 1.3),  # Estimated token count
        "problem": problem,
        "messages": messages,
        "reasoning": reasoning_text,
        "answer": answer_text,
        "skipped": False,
    }

    # Add DeepSeek-V3.2 correctness label if evaluator is provided
    if evaluator_client:
        correctness_label = classify_answer_correctness(evaluator_client, problem, answer_text)
        record["correctness_label"] = correctness_label
        record["evaluator_model"] = "deepseek-v3.2-exp"

    output_path = _get_output_path(dataset_name or "unknown", source_filter)
    append_jsonl(output_path, record)
    return True


def main(
    dataset_cfg: DatasetConfig = DEFAULT_DATASET_CONFIG,
    limit: Optional[int] = 3,
    offset: int = 0,
    model: str = "qwen3-32b",
    api_key_suffix: str = "",
    enable_evaluation: bool = True,
) -> None:
    """Run Qwen3-32B on a slice of a hard math dataset.

    Args:
        dataset_cfg: DatasetConfig specifying HF name/subset/split (includes random_sample and random_seed).
        limit: Number of problems to run (None for all).
        offset: Start index within the split (ignored if dataset_cfg.random_sample=True).
        model: Qwen model ID in 百炼 (e.g., 'qwen3-32b').
        api_key_suffix: Optional suffix for API key (e.g., "_2" for DASHSCOPE_API_KEY_2).
        enable_evaluation: Whether to use DeepSeek-V3.2 for answer evaluation (default: True).
    """

    client = make_client(api_key_suffix)

    # Initialize DeepSeek evaluator if enabled
    evaluator_client = None
    if enable_evaluation:
        try:
            evaluator_client = make_deepseek_client()
            print("DeepSeek-V3.2 evaluator enabled for answer classification (without thinking mode).")
        except RuntimeError as e:
            print(f"Warning: {e}. Proceeding without evaluation.")
            evaluator_client = None

    name = dataset_cfg.name
    subset = dataset_cfg.subset
    split = dataset_cfg.split

    if subset:
        print(f"Loading dataset {name} subset='{subset}' split='{split}' ...")
        ds = load_dataset(name, subset)
    else:
        print(f"Loading dataset {name} split='{split}' ...")
        ds = load_dataset(name)

    data = ds[split]

    # Filter by source if specified (for NuminaMath-CoT)
    if dataset_cfg.source_filter:
        print(f"Filtering by source='{dataset_cfg.source_filter}' ...")
        data = data.filter(lambda x: x.get("source") == dataset_cfg.source_filter)
        print(f"Filtered to {len(data)} problems with source='{dataset_cfg.source_filter}'.")

    n_total = len(data)

    # Determine indices to process
    if dataset_cfg.random_sample:
        import random

        if dataset_cfg.random_seed is not None:
            random.seed(dataset_cfg.random_seed)
        sample_size = limit if limit is not None else n_total
        sample_size = min(sample_size, n_total)
        indices = random.sample(range(n_total), sample_size)
        print(f"Total problems in split: {n_total}. Randomly sampling {sample_size} problems (seed={dataset_cfg.random_seed}).")
    else:
        if limit is None:
            end = n_total
        else:
            end = min(offset + limit, n_total)
        indices = list(range(offset, end))
        print(f"Total problems in split: {n_total}. Evaluating indices [{offset}, {end}).")

    is_math500 = name == MATH500_DATASET_CONFIG.name

    # Load already-completed question IDs for resume support
    output_path = _get_output_path(name, dataset_cfg.source_filter)
    completed_ids = _get_completed_question_ids(output_path)
    if completed_ids:
        print(f"Resume mode: Found {len(completed_ids)} already-processed questions. Skipping them.")

    for i in indices:
        row = data[i]

        if is_math500:
            # MATH-500 uses 'problem' field
            problem = str(row.get("problem", ""))
        else:
            item = row.get("problem") or row.get("question") or row
            if isinstance(item, dict):
                problem = item.get("query", str(item))
            else:
                problem = str(item)

        question_id = f"{split}_{i}"

        # Skip if already processed
        if question_id in completed_ids:
            continue

        success = solve_problem(
            client,
            problem,
            question_id,
            split=split,
            model=model,
            dataset_name=name,
            source_filter=dataset_cfg.source_filter,
            max_cot_tokens=8000,
            evaluator_client=evaluator_client,
        )
        # Note: problems exceeding 8000 tokens are skipped but still logged


if __name__ == "__main__":
    main(dataset_cfg=MATH500_DATASET_CONFIG, limit=500, model="qwen3-32b")
