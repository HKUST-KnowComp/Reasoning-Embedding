import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import logging

import mteb
from dotenv import load_dotenv
from mteb_utils import EvalArguments, get_model
from transformers import HfArgumentParser

from hrsa.config import METRIC_RESULTS_FOLDER

load_dotenv()

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%Y/%m/%d %H:%M:%S", level=logging.INFO)
logger = logging.getLogger("evaluate_bright.py")

OUTPUT_DIR = os.path.join(METRIC_RESULTS_FOLDER, "bright_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def evaluate_bright(t, model, args, **kwargs):
    # task_instructions = {}
    Instructions = {
        "aops": "Given a Math problem, retrieve relevant examples that help answer the problem.",
        "biology": "Given a post, retrieve relevant passages that help answer the post.",
        "earth_science": "Given a post, retrieve relevant passages that help answer the post.",
        "economics": "Given a economics post, retrieve relevant passages that help answer the post.",
        "leetcode": "Given a coding problem, retrieve relevant examples that help answer the problem.",
        "pony": "Given a question about pony program language, retrieve relevant passages that help answer the question.",
        "psychology": "Given a psychology post, retrieve relevant passages that help answer the post.",
        "theoremqa_questions": "Given a Math problem, retrieve relevant examples that help answer the problem.",
        "theoremqa_theorems": "Given a Math problem, retrieve relevant theorems that help answer the problem.",
        "robotics": "Given a robotics post, retrieve relevant passages that help answer the post.",
        "stackoverflow": "Given a stackoverflow post, retrieve relevant passages that help answer the post.",
        "sustainable_living": "Given a sustainable_living post, retrieve relevant passages that help answer the post.",
    }
    encode_kwargs = args.encode_kwargs or dict()

    all_results = []
    cache = mteb.cache.ResultCache(args.output_dir)

    for task in Instructions.keys():
        logger.info(f"Running task: {t.metadata.name} with prompt: {Instructions[task]}")

        instruct = args.instruction_template.format(instruction=Instructions[task])
        model.prompts = {"BrightRetrieval-query": instruct, "BrightRetrieval-document": ""}
        results = mteb.evaluate(model, tasks=[t], cache=cache, encode_kwargs=encode_kwargs, **kwargs)
        all_results.append(results)

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

    # get the tasks
    tasks = mteb.get_tasks(tasks=["BrightRetrieval"])[0]

    if args.output_dir is None:
        args.output_dir = OUTPUT_DIR
    else:
        args.output_dir = os.path.join(OUTPUT_DIR, args.output_dir)

    # load the model
    model = get_model(args.model, args.device, model_kwargs=args.model_kwargs, config_kwargs=args.config_kwargs)
    # model.set_pooling_include_prompt(False)

    args.encode_kwargs.update(batch_size=args.batch_size)

    # # evaluate the model
    results = evaluate_bright(tasks, model, args, **args.run_kwargs)

    logger.info(f"Results: {results}")
    logger.info("Finished evaluating BrightRetrieval tasks.")


if __name__ == "__main__":
    main()
