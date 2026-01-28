import json
import logging
import random
from dataclasses import dataclass, field
from typing import Optional

import mteb
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.model_card import SentenceTransformerModelCardData
from sentence_transformers.models import Normalize, Pooling, Transformer


@dataclass
class EvalArguments:
    """
    Arguments.
    """

    model: Optional[str] = field(default=None, metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"})
    model_name: Optional[str] = field(default=None, metadata={"help": "Model name for the save path"})
    model_kwargs: Optional[str] = field(
        default=None,
        metadata={"help": "The specific model kwargs, json string."},
    )
    config_kwargs: Optional[str] = field(
        default=None,
        metadata={"help": "The specific config kwargs, json string."},
    )
    encode_kwargs: Optional[str] = field(
        default=None,
        metadata={"help": "The specific encode kwargs, json string."},
    )
    run_kwargs: Optional[str] = field(
        default=None,
        metadata={"help": "The specific kwargs for `MTEB.run()`, json string."},
    )

    output_dir: Optional[str] = field(default=None, metadata={"help": "output dir of results (which will be concatenated with OUTPUT_DIR)"})
    benchmark: Optional[str] = field(default=None, metadata={"help": "Benchmark name"})
    tasks: Optional[str] = field(default=None, metadata={"help": "',' separated"})
    langs: Optional[str] = field(default=None, metadata={"help": "',' separated"})

    batch_size: int = field(default=8, metadata={"help": "Will be set to `encode_kwargs`"})
    device: str = field(default="cuda", metadata={"help": "cuda, cpu"})
    seed: Optional[int] = field(default=None, metadata={"help": "Random seed for reproducibility"})
    prompts_path: Optional[str] = field(default="evaluation/task_prompts.json", metadata={"help": "Path to prompts"})
    instruction_template: Optional[str] = field(default="Instruct: {instruction}\nQuery: ", metadata={"help": "Instruction template with {instruction}"})

    def __post_init__(self):
        if isinstance(self.tasks, str):
            self.tasks = self.tasks.split(",")
        if isinstance(self.langs, str):
            self.langs = self.langs.split(",")
        for name in ("model", "config", "encode", "run"):
            name = name + "_kwargs"
            attr = getattr(self, name)
            if attr is None:
                setattr(self, name, dict())
            elif isinstance(attr, str):
                setattr(self, name, json.loads(attr))
            print(f"self.{name}: {getattr(self, name)}")
            print(self.benchmark)


def get_model(model_path: str, device: str = "cuda", model_kwargs: dict = None, config_kwargs: dict = None, prompts=None, **kwargs):
    # model = SentenceTransformer(model_path, device=device, model_kwargs=model_kwargs, config_kwargs=config_kwargs, **kwargs)
    # Fix device_map: use 'auto' for cuda, or 'cpu' for cpu
    device_map = "auto" if device == "cuda" else device
    model_kwargs = dict(**model_kwargs, device_map=device_map)
    model = Transformer(model_name_or_path=model_path, model_args=model_kwargs, config_args=config_kwargs)
    pool = Pooling(word_embedding_dimension=model.get_word_embedding_dimension(), pooling_mode_mean_tokens=True, include_prompt=False)
    normalize = Normalize()

    model_card_data = SentenceTransformerModelCardData(
        model_name=model_path,
    )
    model = SentenceTransformer(modules=[model, pool, normalize], prompts=prompts, model_card_data=model_card_data, **kwargs)

    return model


def get_tasks(names: list[str] | None, languages: list[str] | None = None, benchmark: str | None = None):
    if benchmark:
        tasks = mteb.get_benchmark(benchmark).tasks
    else:
        tasks = mteb.get_tasks(languages=languages, tasks=names)
    return tasks


def set_random_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
