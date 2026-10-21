"""Homework 2, Q14: median generalization gap for N=3, tau=0.1."""

import numpy as np

from decision_stump import (fit_stump, generate_data, out_error,
                            parse_args, write_report)


def solve(trials: int = 10000, seed: int = 2026) -> dict:
    """Generate Q14's datasets, train stumps, and select the closest answer.

    Args:
        trials: Number of independent datasets (10000 for the assignment).
        seed: Base seed; add 14 to obtain this question's PCG64 seed.

    Returns:
        Experiment settings, median E_out - E_in, and selected answer.

    Raises:
        ValueError: If trials is not positive or seed is negative.
    """
    if trials < 1 or seed < 0:
        raise ValueError("Require positive trials and a nonnegative base seed.")
    n, tau = 3, 0.1
    choices = (0.10, 0.15, 0.20, 0.25, 0.30)
    actual_seed = seed + 14
    rng = np.random.Generator(np.random.PCG64(actual_seed))
    gaps = np.empty(trials)
    # Draw a fresh dataset for each trial; keep one random stream throughout.
    for index in range(trials):
        x, y = generate_data(n, tau, rng)
        stump = fit_stump(x, y)
        gaps[index] = out_error(stump, tau) - stump.ein

    # The question asks for the median of the per-dataset gaps.
    median = float(np.median(gaps))
    choice_index = int(np.argmin(np.abs(np.asarray(choices) - median)))
    return {
        "number": 14,
        "n": n,
        "tau": tau,
        "choices": choices,
        "trials": trials,
        "seed": actual_seed,
        "median_gap": median,
        "choice_number": choice_index + 1,
        "closest_choice": choices[choice_index],
    }


def main() -> None:
    """Run this question and print or save its result."""
    args = parse_args(__doc__)
    write_report([solve(args.trials, args.seed)], args.output)


if __name__ == "__main__":
    main()
