"""Homework 1, Q13: Find the median Euclidean norm of the final PLA weight vectors."""

import numpy as np

from pla import (load_data, parse_args, report, run_experiments,
                 save_answers, save_experiments)


def solve(runs=1000, start_seed=0, max_checks=100_000_000, output_dir=None):
    """Run Q13's experiment and return its measured multiple-choice answer.

    Args:
        runs: Number of independent experiments (1000 for the assignment).
        start_seed: First of the consecutive random seeds.
        max_checks: Maximum sampled checks per experiment.
        output_dir: Result directory; defaults to results beside this script.
    """
    features, labels = load_data()
    # Use all 13 coordinates, including the learned bias weight.
    x = np.column_stack((np.full(len(labels), 1.0), features))
    results = run_experiments(x, labels, runs, start_seed, max_checks)
    value = np.median(np.linalg.norm(results["weights"], axis=1))
    answer = report(13, "median_norm", value,
                    [65, 75, 85, 95, 105])
    save_experiments("baseline", results, labels, start_seed,
                     bias=1.0, scale=1.0, output_dir=output_dir)
    save_answers("q13_answer", [answer], runs, start_seed, output_dir)
    return answer


def main():
    """Run this question with command-line experiment options."""
    args = parse_args(__doc__)
    solve(args.runs, args.start_seed, args.max_checks, args.output_dir)


if __name__ == "__main__":
    main()
