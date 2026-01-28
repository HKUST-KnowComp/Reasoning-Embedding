import argparse
import json
import os
from typing import Dict, List, Optional

import mteb
from colorama import Fore, Style, init

# Initialize colorama for cross-platform colored output
init()


def load_results(path: str, task_names: List[str]) -> Dict[str, float]:
    """Load and process results from json files"""
    results = {}
    for task_file in os.listdir(path):
        name = task_file.split(".json")[0]
        if name not in task_names:
            continue

        with open(os.path.join(path, task_file)) as f:
            result = json.load(f)

        eval_split = list(result["scores"].keys())[0]
        score = sum(ele["main_score"] for ele in result["scores"][eval_split]) / len(result["scores"][eval_split])
        results[name] = round(score * 100, 2)

    return results


def get_task_info(benchmark: Optional[str] = None, task_names: Optional[List[str]] = None, languages: Optional[List[str]] = None):
    """Get task information from MTEB"""
    if benchmark:
        tasks = mteb.get_benchmark(benchmark).tasks
    else:
        tasks = mteb.get_tasks(languages=languages, tasks=task_names)

    names = [t.metadata.name for t in tasks]
    return {name: task for name, task in zip(names, tasks)}


def analyze_results(results: Dict[str, float], tasks) -> Dict[str, List[float]]:
    """Group and analyze results by task type"""
    split_tasks = {}
    for name, score in results.items():
        task_type = tasks[name].metadata.type
        if task_type not in split_tasks:
            split_tasks[task_type] = []
        split_tasks[task_type].append(score / 100)  # Convert back to 0-1 scale
    return split_tasks


def main(path, benchmark, tasks):
    # Configuration
    benchmark = benchmark or None
    tasks = tasks or None

    if benchmark and tasks:
        raise ValueError("Benchmark and tasks cannot be set at the same time")

    output_file_name = os.path.join(path, "summary.txt")
    output_file = open(output_file_name, "w")

    # Model name
    stripped_path = path.strip("/")
    splitted_path = stripped_path.split("/")
    if len(splitted_path) > 3:
        model_path = splitted_path[-3]
        model_name = splitted_path[-2]
        revision_name = splitted_path[-1]
    else:
        model_name = splitted_path[-1]
        model_path = None
        revision_name = None

    print("Model path: ", model_path)
    print(" --- Model name: ", model_name)
    print(" --- Revision name: ", revision_name, "\n")
    output_file.write("Model path: " + model_path + "\n")
    output_file.write(" --- Model name: " + model_name + "\n")
    output_file.write(" --- Revision name: " + revision_name + "\n\n")

    # Get task information
    tasks = get_task_info(benchmark=benchmark, task_names=tasks)
    task_names = list(tasks.keys())

    # Load and process results
    results = load_results(path, task_names)

    # Analyze results
    split_tasks = analyze_results(results, tasks)

    # Print summary statistics
    missed_tasks = [name for name in task_names if name not in results]
    print("Missed tasks:", missed_tasks, "\n")
    output_file.write("Missed tasks:" + str(missed_tasks) + "\n\n")
    print(f"{Fore.BLUE}Final score: {len(results)} {sum(results.values()) / len(results)}{Style.RESET_ALL}", "\n")
    output_file.write(f"Final score: {len(results)} {sum(results.values()) / len(results)}\n\n")

    type_scores = []
    for task_type, scores in split_tasks.items():
        avg_score = sum(scores) / len(scores) * 100
        print(f" * {task_type}: {len(scores)} tasks, average score {avg_score:.3f}")
        output_file.write(f" * {task_type}: {len(scores)} tasks, average score {avg_score:.3f}" + "\n")
        type_scores.append(avg_score)

    print(f"{Fore.BLUE}\nMean(Type): {sum(type_scores) / len(type_scores)}{Style.RESET_ALL}\n")
    output_file.write(f"\nMean(Type): {sum(type_scores) / len(type_scores)}\n\n")
    print("=" * 100, "\n")
    output_file.write("=" * 100 + "\n\n")
    # sort result by score
    results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    for i, (name, score) in enumerate(results, 1):
        print(f"{i}. {name}: {score}")
        output_file.write(f"{i}. {name}: {score}" + "\n")

    output_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str)
    parser.add_argument("--benchmark", type=str, default=None)
    parser.add_argument("--tasks", type=str, default=None, help="Comma-separated string of tasks, or None")
    args = parser.parse_args()

    # If tasks is not None, split it by comma
    if args.tasks is not None:
        args.tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]

    main(args.path, args.benchmark, args.tasks)
