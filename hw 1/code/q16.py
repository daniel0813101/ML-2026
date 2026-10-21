"""Homework 1, Q16: Find the median PLA updates with zero bias and unscaled raw features."""

import numpy as np

from pla import (load_data, parse_args, report, run_experiments,
                 save_answers, save_experiments)


def solve(runs=1000, start_seed=0, max_checks=100_000_000, output_dir=None):
    """Run Q16's experiment and return its measured multiple-choice answer.

    Args:
        runs: Number of independent experiments (1000 for the assignment).
        start_seed: First of the consecutive random seeds.
        max_checks: Maximum sampled checks per experiment.
        output_dir: Result directory; defaults to results beside this script.
    """
    features, labels = load_data()
    # A zero bias column forces the separating hyperplane through the origin.
    x = np.column_stack((np.full(len(labels), 0.0), features))
    results = run_experiments(x, labels, runs, start_seed, max_checks)
    value = np.median(results["updates"])
    answer = report(16, "median_updates", value,
                    [7500, 10000, 12500, 15000, 17500])
    save_experiments("zero_bias", results, labels, start_seed,
                     bias=0.0, scale=1.0, output_dir=output_dir)
    save_answers("q16_answer", [answer], runs, start_seed, output_dir)
    return answer


def main():
    """Run this question with command-line experiment options."""
    args = parse_args(__doc__)
    solve(args.runs, args.start_seed, args.max_checks, args.output_dir)


if __name__ == "__main__":
    main()
