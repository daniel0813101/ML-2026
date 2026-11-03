"""Homework 3, Q12: SGD for linear regression, initialized at zero."""

import numpy as np

from linear_models import (closest_answer, load_data, parse_args, squared_error,
                           validate_experiment, write_report)


def train_linear_sgd(x: np.ndarray, y: np.ndarray, rng: np.random.Generator,
                     iterations: int, learning_rate: float) -> np.ndarray:
    """Train from zero with uniformly sampled, with-replacement SGD updates.

    Args:
        x: Design matrix, already including its bias column.
        y: Training labels.
        rng: Independent random generator for this experiment.
        iterations: Number of single-example updates.
        learning_rate: Step size for the gradient of squared error.

    Returns:
        The weight vector after exactly iterations updates.
    """
    w = np.zeros(x.shape[1])
    for index in rng.integers(len(y), size=iterations):
        row = x[index]
        # Gradient of (w^T x - y)^2; there is no 1/2 in the assigned error.
        w -= 2 * learning_rate * (w @ row - y[index]) * row
    return w


def solve(trials: int = 1000, seed: int = 2026, iterations: int = 800,
          learning_rate: float = 0.001, data_dir=None) -> dict:
    """Run Q12's trials and average the final in-sample squared errors.

    Args:
        trials: Number of independently seeded experiments.
        seed: Base seed; trial i uses seed + 12 + i (zero-based).
        iterations: Number of SGD updates per experiment.
        learning_rate: Fixed SGD step size.
        data_dir: Optional directory containing the assignment data.

    Returns:
        Settings, per-trial errors, their average, and the closest option.
    """
    validate_experiment(trials, seed, iterations, learning_rate)
    x, y = load_data("train", data_dir)
    errors = np.empty(trials)
    for trial in range(trials):
        rng = np.random.Generator(np.random.PCG64(seed + 12 + trial))
        w = train_linear_sgd(x, y, rng, iterations, learning_rate)
        errors[trial] = squared_error(x, y, w)
    value = float(np.mean(errors))
    choices = (0.76, 0.78, 0.80, 0.82, 0.84)
    return {"number": 12, "metric": "mean_in_squared_error", "value": value,
            "trials": trials, "first_seed": seed + 12,
            "iterations": iterations, "learning_rate": learning_rate,
            "initialization": "zero", "trial_errors": errors.tolist(),
            **closest_answer(value, choices)}


def main() -> None:
    """Run this question and print or save its result."""
    args = parse_args(__doc__)
    write_report([solve(args.trials, args.seed, args.iterations,
                        args.learning_rate, args.data_dir)], args.output)


if __name__ == "__main__":
    main()
