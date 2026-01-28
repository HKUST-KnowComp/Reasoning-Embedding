#!/usr/bin/env python3
"""
Simple script to calculate accuracy from evaluated output files.
"""

import json
import sys
from collections import defaultdict


def calculate_accuracy(filepath):
    """Calculate accuracy metrics from a JSONL file."""
    total = 0
    correct = 0
    incorrect = 0
    skipped = 0
    error = 0

    with open(filepath, "r") as f:
        for line in f:
            if not line.strip():
                continue

            record = json.loads(line)
            total += 1

            label = record.get("correctness_label", -1)

            if label == 1:
                correct += 1
            elif label == 0:
                incorrect += 1
            elif label == -1:
                if record.get("skipped", False):
                    skipped += 1
                else:
                    error += 1

    evaluated = correct + incorrect
    accuracy = (correct / evaluated * 100) if evaluated > 0 else 0

    # Accuracy treating skipped as wrong
    evaluated_with_skipped = correct + incorrect + skipped
    accuracy_with_skipped = (correct / evaluated_with_skipped * 100) if evaluated_with_skipped > 0 else 0

    return {
        "total": total,
        "correct": correct,
        "incorrect": incorrect,
        "skipped": skipped,
        "error": error,
        "evaluated": evaluated,
        "accuracy": accuracy,
        "evaluated_with_skipped": evaluated_with_skipped,
        "accuracy_with_skipped": accuracy_with_skipped,
    }


def main():
    import glob
    import os
    from datetime import datetime

    if len(sys.argv) > 1:
        # Process specific files
        files = sys.argv[1:]
    else:
        # Default: scan for all evaluated files
        base_dir = os.path.dirname(__file__)
        patterns = [
            os.path.join(base_dir, "outputs", "*_evaluated.jsonl"),
            os.path.join(base_dir, "outputs", "hard-level", "*_evaluated.jsonl"),
            os.path.join(base_dir, "outputs", "medium-level", "*_evaluated.jsonl"),
        ]

        files = []
        for pattern in patterns:
            files.extend(glob.glob(pattern))

    if not files:
        print("No evaluated files found!")
        print("Usage: python check_accuracy.py [file1.jsonl file2.jsonl ...]")
        return

    # Prepare output file
    base_dir = os.path.dirname(__file__)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(base_dir, f"accuracy_report_{timestamp}.txt")

    def print_and_write(text, f=None):
        """Print to console and write to file."""
        print(text)
        if f:
            f.write(text + "\n")

    with open(output_file, "w") as out_f:
        print_and_write("=" * 80, out_f)
        print_and_write("Accuracy Report", out_f)
        print_and_write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", out_f)
        print_and_write("=" * 80, out_f)

        all_stats = defaultdict(lambda: {"total": 0, "correct": 0, "incorrect": 0, "skipped": 0, "error": 0, "evaluated": 0})

        for filepath in sorted(files):
            filename = filepath.split("/")[-1]
            stats = calculate_accuracy(filepath)

            print_and_write(f"\n{filename}", out_f)
            print_and_write("-" * 80, out_f)
            print_and_write(f"  Total records:     {stats['total']}", out_f)
            print_and_write(f"  Correct:           {stats['correct']}", out_f)
            print_and_write(f"  Incorrect:         {stats['incorrect']}", out_f)
            print_and_write(f"  Skipped:           {stats['skipped']}", out_f)
            print_and_write(f"  Error/Not eval:    {stats['error']}", out_f)
            print_and_write(f"  Evaluated:         {stats['evaluated']} ({stats['evaluated']/stats['total']*100:.1f}%)", out_f)
            print_and_write(f"  Accuracy:          {stats['accuracy']:.2f}% (excluding skipped)", out_f)
            print_and_write(f"  Accuracy (w/ skip):{stats['accuracy_with_skipped']:.2f}% (treating skipped as wrong)", out_f)

            # Aggregate stats
            all_stats["all"]["total"] += stats["total"]
            all_stats["all"]["correct"] += stats["correct"]
            all_stats["all"]["incorrect"] += stats["incorrect"]
            all_stats["all"]["skipped"] += stats["skipped"]
            all_stats["all"]["error"] += stats["error"]
            all_stats["all"]["evaluated"] += stats["evaluated"]

        # Print overall stats
        if len(files) > 1:
            print_and_write("\n" + "=" * 80, out_f)
            print_and_write("OVERALL", out_f)
            print_and_write("=" * 80, out_f)
            total = all_stats["all"]["total"]
            correct = all_stats["all"]["correct"]
            evaluated = all_stats["all"]["evaluated"]
            accuracy = (correct / evaluated * 100) if evaluated > 0 else 0

            skipped = all_stats["all"]["skipped"]
            evaluated_with_skipped = correct + all_stats["all"]["incorrect"] + skipped
            accuracy_with_skipped = (correct / evaluated_with_skipped * 100) if evaluated_with_skipped > 0 else 0

            print_and_write(f"  Total records:     {total}", out_f)
            print_and_write(f"  Correct:           {correct}", out_f)
            print_and_write(f"  Incorrect:         {all_stats['all']['incorrect']}", out_f)
            print_and_write(f"  Skipped:           {skipped}", out_f)
            print_and_write(f"  Error/Not eval:    {all_stats['all']['error']}", out_f)
            print_and_write(f"  Evaluated:         {evaluated} ({evaluated/total*100:.1f}%)", out_f)
            print_and_write(f"  Accuracy:          {accuracy:.2f}% (excluding skipped)", out_f)
            print_and_write(f"  Accuracy (w/ skip):{accuracy_with_skipped:.2f}% (treating skipped as wrong)", out_f)

        print_and_write("\n" + "=" * 80, out_f)
        print_and_write(f"\nReport saved to: {output_file}", out_f)


if __name__ == "__main__":
    main()
