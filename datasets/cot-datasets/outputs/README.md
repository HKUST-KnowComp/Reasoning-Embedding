# Qwen3-32B Math Reasoning Outputs

This directory contains the output results from running Qwen3-32B on various math reasoning datasets with Chain-of-Thought (CoT) reasoning and answer evaluation.

## Directory Structure

```
outputs/
├── easy/                   # Elementary-level math problems
│   └── qwen3_32b_openai_gsm8k_outputs_evaluated.jsonl
├── moderate/               # Intermediate-level math problems
│   ├── qwen3_32b_HuggingFaceH4_MATH-500_outputs_evaluated.jsonl
│   └── qwen3_32b_AI-MO_NuminaMath-CoT_cn_k12_outputs_evaluated.jsonl
└── hard/                   # Advanced competition-level problems
    └── qwen3_32b_LiveMathbench_evaluated.jsonl
```

## Datasets Overview

| Dataset | Level | Total Records | Evaluated | Accuracy (excl. skip) | Accuracy (w/ skip) | Description |
|---------|-------|---------------|-----------|----------------------|-------------------|-------------|
| **GSM8K** | Easy | 500 | 471 (94.2%) | 98.51% | 92.80% | Grade school math word problems |
| **MATH-500** | Moderate | 479 | 364 (76.0%) | 99.73% | 75.78% | Competition math problems (AMC, AIME) |
| **NuminaMath cn_k12** | Moderate | 161 | 154 (95.7%) | 95.45% | 91.30% | Chinese K-12 curriculum math problems |
| **LiveMathBench** | Hard | 57* | 57 (100%) | 56.14% | 56.14% | Recent hard competition problems |
| **OVERALL** | - | **1197** | **1046 (87.4%)** | **96.18%** | **84.04%** | Combined statistics |

\* LiveMathBench: Only 57 out of 100 problems completed (in progress). **Note:** LiveMathBench generation runs without CoT token limits to allow extremely long reasoning chains for hard problems.

## Data Format

Each JSONL file contains one JSON object per line with the following structure:

### Core Fields

```json
{
  "timestamp": "2025-11-20T16:32:23.195088Z",
  "dataset": "openai/gsm8k",
  "split": "test",
  "question_id": "test_1309",
  "model": "qwen3-32b",
  "cot_length_words": 769,
  "cot_length_tokens": 3142,
  "problem": "The girls are trying to raise money...",
  "messages": [...],
  "reasoning": "Okay, let me try to figure out...",
  "answer": "To determine the total amount...",
  "skipped": false,
  "correctness_label": 1,
  "evaluator_model": "deepseek-v3.2-exp"
}
```

### Field Descriptions

| Field | Type | Description |
|-------|------|-------------|
| `timestamp` | string | ISO 8601 timestamp of generation |
| `dataset` | string | HuggingFace dataset identifier |
| `split` | string | Dataset split (test/train) |
| `question_id` | string | Unique identifier (e.g., "test_123") |
| `model` | string | Model used for generation ("qwen3-32b") |
| `cot_length_words` | integer | Word count of reasoning trace |
| `cot_length_tokens` | integer | Estimated token count of reasoning (~1.3 chars/token) |
| `problem` | string | Original problem statement |
| `messages` | array | Full conversation history (system + user prompt) |
| `reasoning` | string | Internal Chain-of-Thought reasoning from model |
| `answer` | string | Final formatted answer |
| `skipped` | boolean | Whether problem was skipped (true if error occurred) |
| `skip_reason` | string | (Optional) Reason for skipping (e.g., "cot_exceeded_max_tokens") |
| `correctness_label` | integer | Evaluation result: 1=correct, 0=incorrect, -1=error/skipped |
| `evaluator_model` | string | Model used for evaluation ("deepseek-v3.2-exp") |

### Optional Fields (for skipped records)

- `skip_reason`: Reason for skipping (e.g., "cot_exceeded_max_tokens", "api_internal_error")
- `error`: Error message if API failure occurred

## Generation Parameters

**Model**: Qwen3-32B via Aliyun DashScope API
- `temperature`: 0.6
- `top_p`: 0.95
- `top_k`: 20
- `max_tokens`: 11000
- `enable_thinking`: true (exposes internal reasoning)
- `max_cot_tokens`: 8000 (problems exceeding this are marked as skipped)
  - **Exception**: LiveMathBench has no CoT token limit to allow extremely long reasoning for hard problems

**Evaluation**: DeepSeek-V3.2-exp via Aliyun DashScope API
- `temperature`: 0
- `enable_thinking`: false
- Method: Compare model answer with official dataset answer

## Accuracy Metrics

Two accuracy metrics are reported:

1. **Accuracy (excluding skipped)**: Correctness rate among successfully evaluated problems
   - Formula: `correct / (correct + incorrect)`
   - Represents model's performance on problems it could process

2. **Accuracy (with skipped as wrong)**: Correctness rate treating all skipped problems as failures
   - Formula: `correct / total`
   - Represents overall robustness including API/generation failures

## Skipped Records

Records may be skipped for:

1. **CoT Token Limit Exceeded** (`cot_exceeded_max_tokens`): Reasoning chain exceeded 8000 tokens during generation

Skipped records are still logged with partial data and `correctness_label = -1`.

## Dataset Sources

- **GSM8K**: [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) - Grade school math word problems
- **MATH-500**: [HuggingFaceH4/MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) - Curated subset of MATH benchmark
- **NuminaMath-CoT**: [AI-MO/NuminaMath-CoT](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT) - Math problems with chain-of-thought annotations (filtered by source="cn_k12")
- **LiveMathBench**: [opencompass/LiveMathBench](https://huggingface.co/datasets/opencompass/LiveMathBench) - Recent competition problems (v202505_hard_en)

## Usage Examples

### Reading a file

```python
import json

with open('outputs/easy/qwen3_32b_openai_gsm8k_outputs_evaluated.jsonl', 'r') as f:
    for line in f:
        record = json.loads(line)
        print(f"Problem: {record['problem']}")
        print(f"Correct: {record['correctness_label'] == 1}")
```

### Calculate accuracy

```bash
python check_accuracy.py outputs/easy/qwen3_32b_openai_gsm8k_outputs_evaluated.jsonl
```

### Check all datasets

```bash
python check_accuracy.py  # Scans all evaluated files
```

## Notes

- **Random Sampling**: GSM8K and NuminaMath datasets use random sampling with seed=42 for reproducibility
- **Resume Support**: Generation scripts automatically skip already-processed questions
- **Evaluation Independence**: Evaluation is performed separately after generation to allow batch re-evaluation
- **Token Estimates**: `cot_length_tokens` uses approximation of 1.3 characters per token for mixed Chinese/English text

## Generation Date

- GSM8K: November 20-21, 2025
- MATH-500: November 20-21, 2025
- NuminaMath cn_k12: November 20-21, 2025
- LiveMathBench: November 20-21, 2025 (partial, ongoing)

---

For questions or issues, refer to the parent directory's README or generation scripts.
