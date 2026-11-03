"""Homework 3, Q14: logistic SGD initialized with linear regression weights."""

import numpy as np

from linear_models import (closest_answer, cross_entropy, fit_linear, load_data,
                           parse_args, train_logistic_sgd, validate_experiment,
                           write_report)


def solve(trials: int = 1000, seed: int = 2026, iterations: int = 800,
          learning_rate: float = 0.001, data_dir=None) -> dict:
    """Average Q14's logistic losses using trial seeds seed + 14 + i.

    Args:
        trials: Number of independently seeded experiments.
        seed: Nonnegative base seed; also used by Q15 to reproduce these models.
        iterations: Number of SGD updates per experiment.
        learning_rate: Fixed SGD step size.
        data_dir: Optional directory containing the assignment data.

    Returns:
        Settings, per-trial cross-entropies, their average, and closest option.
    """
    validate_experiment(trials, seed, iterations, learning_rate)
    x, y = load_data("train", data_dir)
    initial = fit_linear(x, y)
    errors = np.empty(trials)
    for trial in range(trials):
        rng = np.random.Generator(np.random.PCG64(seed + 14 + trial))
        w = train_logistic_sgd(x, y, rng, initial, iterations, learning_rate)
        errors[trial] = cross_entropy(x, y, w)
    value = float(np.mean(errors))
    choices = (0.60, 0.63, 0.66, 0.69, 0.72)
    return {"number": 14, "metric": "mean_in_cross_entropy", "value": value,
            "trials": trials, "first_seed": seed + 14,
            "iterations": iterations, "learning_rate": learning_rate,
            "initialization": "linear_regression", "trial_errors": errors.tolist(),
            **closest_answer(value, choices)}


def main() -> None:
    """Run this question and print or save its result."""
    args = parse_args(__doc__)
    write_report([solve(args.trials, args.seed, args.iterations,
                        args.learning_rate, args.data_dir)], args.output)


if __name__ == "__main__":
    main()
