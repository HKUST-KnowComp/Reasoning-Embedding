import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gc
import json
import logging

import mteb
import torch
from dotenv import load_dotenv
from mteb_utils import EvalArguments, get_model, get_tasks, set_random_seed
from transformers import HfArgumentParser

from hrsa.config import METRIC_RESULTS_FOLDER

load_dotenv()

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%Y/%m/%d %H:%M:%S", level=logging.INFO)
logger = logging.getLogger("evaluate_mteb.py")

OUTPUT_DIR = os.path.join(METRIC_RESULTS_FOLDER, "mteb_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def clean_and_add_instruction_template(prompts, tasks, instruction_template):
    """
    First we clean the prompts by extracting the tasks we want to run
    And then we add the instruction template to the prompts for the task

    The reason of add the suffix `-query` and `-document` is because some tasks have different prompts for query and document.
    Also, the priority of `task_name-query` / `task_name-document` is higher than `task_name`,
    which means the `task_name-query` / `task_name-document` will be checked first.

    E.g. for QuoraRetrieval, the order of checking is
    QuoraRetrieval-query --> QuoraRetrieval --> ...
    So if we don't want to add instructions to the document, we need to add `-document` with empty string,
    otherwise the normal prompt will be used.

    For details, see mteb/models/sentence_transformer_wrapper.py:get_prompt_name()

    Args:
        prompts: Dict[str, str]
        tasks: List[mteb.task.Task]
        instruction_template: str

    Returns:
        cleaned prompts: Dict[str, str]
    """
    cleaned_prompts = {}

    for task in tasks:
        task_name = task.metadata.name
        # if task name is not in the prompts, use the default prompt
        if task_name not in prompts:
            prompts[task_name] = prompts["default"]

        cleaned_prompts[task_name] = instruction_template.format(instruction=prompts[task_name])
        cleaned_prompts[f"{task_name}-query"] = cleaned_prompts[task_name]
        cleaned_prompts[f"{task_name}-document"] = ""  # since we don't want to add document prompt for the task

    return cleaned_prompts


def evaluate_mteb(model, tasks, prompts, args, **kwargs):
    if not tasks:
        raise RuntimeError("No task selected")

    encode_kwargs = args.encode_kwargs or dict()

    all_results = []
    cache = mteb.cache.ResultCache(args.output_dir)

    for t in tasks:
        print("=" * 100, end="\n\n")
        logger.info(f"Running task: {t.metadata.name} with prompt: {prompts[t.metadata.name]}")

        try:
            os.environ["HF_DATASETS_OFFLINE"] = "1"
            # results = evaluation.run(
            #     model,
            #     output_folder=args.output_dir,
            #     encode_kwargs=encode_kwargs,
            #     save_predictions=False,
            #     **kwargs
            # )
            print(f"Output directory: {args.output_dir}")
            results = mteb.evaluate(model, tasks=[t], cache=cache, encode_kwargs=encode_kwargs, **kwargs)
            all_results.append(results)
        except Exception as e:
            try:
                os.environ["HF_DATASETS_OFFLINE"] = "0"
                # results = evaluation.run(
                #     model,
                #     output_folder=args.output_dir,
                #     encode_kwargs=encode_kwargs,
                #     **kwargs
                # )
                results = mteb.evaluate(model, tasks=[t], cache=cache, encode_kwargs=encode_kwargs, **kwargs)
                all_results.append(results)
            except Exception as e:
                print(f"meet error when running task: {t.metadata.name}. {str(e)}")
                continue

        print("=" * 100, end="\n\n")

        gc.collect()
        torch.cuda.empty_cache()

    return all_results


def main():
    parser = HfArgumentParser(EvalArguments)
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        with open(os.path.abspath(sys.argv[1])) as f:
            config = json.load(f)
        logger.info(f"Json config {f.name} : \n{json.dumps(config, indent=2)}")
        args, *_ = parser.parse_dict(config)
        del config, f
    else:
        args, *_ = parser.parse_args_into_dataclasses()
        logger.info(f"Args {args}")
    del parser

    # Set random seed for reproducibility
    set_random_seed(args.seed)

    # get the tasks
    tasks = get_tasks(args.tasks, args.langs, args.benchmark)

    # load the prompts with different instruction templates for different tasks
    with open(args.prompts_path, "r") as f:
        prompts = json.load(f)

    # remove the prompts not in the tasks
    prompts = clean_and_add_instruction_template(prompts, tasks, args.instruction_template)

    if args.output_dir is None:
        args.output_dir = OUTPUT_DIR
    else:
        args.output_dir = os.path.join(OUTPUT_DIR, args.output_dir)

    logger.info(f"Selected {len(tasks)} tasks:\n" + "\n".join(str(t) for t in tasks))

    # load the model
    model = get_model(args.model, args.device, model_kwargs=args.model_kwargs, config_kwargs=args.config_kwargs, prompts=prompts)

    args.encode_kwargs.update(batch_size=args.batch_size)

    # benchmark = mteb.get_benchmark(args.benchmark)
    # results = mteb.evaluate(model, tasks=benchmark, cache=mteb.cache.ResultCache(args.output_dir), encode_kwargs=args.encode_kwargs, **args.run_kwargs)

    # evaluate the model
    results = evaluate_mteb(model, tasks, prompts, args, **args.run_kwargs)

    logger.info(f"Results: {results}")
    logger.info(f"Done {len(tasks)} tasks.")


if __name__ == "__main__":
    main()
