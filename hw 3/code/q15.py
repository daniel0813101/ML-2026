"""Homework 3, Q15: average absolute classification gap after Q14 training."""

import numpy as np

from linear_models import (classification_error, closest_answer, fit_linear,
                           load_data, parse_args, train_logistic_sgd,
                           validate_experiment, write_report)


def solve(trials: int = 1000, seed: int = 2026, iterations: int = 800,
          learning_rate: float = 0.001, data_dir=None) -> dict:
    """Reproduce Q14's models and average their absolute train/test error gaps.

    Args:
        trials: Number of independently seeded experiments.
        seed: Base seed; use seed + 14 + i to match Q14 exactly.
        iterations: Number of SGD updates per experiment.
        learning_rate: Fixed SGD step size.
        data_dir: Optional directory containing the assignment data.

    Returns:
        Per-trial classification errors/gaps, their mean gap, and closest option.
    """
    validate_experiment(trials, seed, iterations, learning_rate)
    x, y = load_data("train", data_dir)
    test_x, test_y = load_data("test", data_dir)
    initial = fit_linear(x, y)
    in_errors, out_errors = np.empty(trials), np.empty(trials)
    for trial in range(trials):
        # The test set is used only for evaluation after all training updates.
        rng = np.random.Generator(np.random.PCG64(seed + 14 + trial))
        w = train_logistic_sgd(x, y, rng, initial, iterations, learning_rate)
        in_errors[trial] = classification_error(x, y, w)
        out_errors[trial] = classification_error(test_x, test_y, w)
    # Average absolute gaps, rather than taking the absolute difference of means.
    gaps = np.abs(in_errors - out_errors)
    value = float(np.mean(gaps))
    choices = (0.03, 0.04, 0.05, 0.06, 0.07)
    return {"number": 15, "metric": "mean_absolute_classification_gap",
            "value": value, "trials": trials, "first_seed": seed + 14,
            "iterations": iterations, "learning_rate": learning_rate,
            "initialization": "linear_regression",
            "trial_in_errors": in_errors.tolist(),
            "trial_out_errors": out_errors.tolist(), "trial_gaps": gaps.tolist(),
            **closest_answer(value, choices)}


def main() -> None:
    """Run this question and print or save its result."""
    args = parse_args(__doc__)
    write_report([solve(args.trials, args.seed, args.iterations,
                        args.learning_rate, args.data_dir)], args.output)


if __name__ == "__main__":
    main()
