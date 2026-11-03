"""Reusable data, linear-model, error, and command-line utilities for HW3."""

import argparse
import json
from pathlib import Path
import platform

import numpy as np


DATA_DIR = Path(__file__).resolve().parent / "data"


def load_data(split: str, data_dir=None) -> tuple[np.ndarray, np.ndarray]:
    """Load a train/test file and prepend the required constant bias feature.

    Args:
        split: Either "train" or "test".
        data_dir: Optional directory containing the assignment's .dat files.

    Returns:
        An N-by-11 design matrix and its N labels.

    Raises:
        ValueError: If the split or dataset format is invalid.
    """
    if split not in ("train", "test"):
        raise ValueError("split must be 'train' or 'test'.")
    directory = DATA_DIR if data_dir is None else Path(data_dir)
    data = np.loadtxt(directory / f"hw3_{split}.dat", ndmin=2)
    if (data.shape[1] != 11 or not len(data)
            or not np.isfinite(data).all()
            or not np.isin(data[:, -1], (-1, 1)).all()):
        raise ValueError("Require 10 finite features and a +/-1 label per row.")
    return np.column_stack((np.ones(len(data)), data[:, :-1])), data[:, -1]


def fit_linear(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Minimize averaged squared error using the least-squares pseudoinverse."""
    # lstsq uses a stable factorization instead of explicitly inverting X^T X.
    return np.linalg.lstsq(x, y, rcond=None)[0]


def validate_experiment(trials: int, seed: int, iterations: int,
                        learning_rate: float) -> None:
    """Reject invalid experiment settings before starting any trials."""
    if trials < 1 or seed < 0 or iterations < 1:
        raise ValueError("Require positive trials/iterations and a nonnegative seed.")
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive.")


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


def train_logistic_sgd(x: np.ndarray, y: np.ndarray, rng: np.random.Generator,
                       initial: np.ndarray, iterations: int,
                       learning_rate: float) -> np.ndarray:
    """Train logistic regression using independent uniform example selections.

    Args:
        x: Design matrix, already including its bias column.
        y: Labels in {-1, +1}.
        rng: Independent random generator for this experiment.
        initial: Starting weights; copied so trials cannot modify one another.
        iterations: Number of single-example updates.
        learning_rate: Step size for the cross-entropy gradient.

    Returns:
        The weight vector after exactly iterations updates.
    """
    w = initial.copy()
    for index in rng.integers(len(y), size=iterations):
        row, label = x[index], y[index]
        margin = label * (w @ row)
        # Stable evaluation of 1 / (1 + exp(margin)).
        probability = np.exp(-np.logaddexp(0.0, margin))
        w += learning_rate * label * row * probability
    return w


def squared_error(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    """Return averaged squared error, including the factor 1/N."""
    return float(np.mean((x @ w - y) ** 2))


def cross_entropy(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    """Return averaged logistic cross-entropy without exponential overflow."""
    return float(np.mean(np.logaddexp(0.0, -y * (x @ w))))


def classification_error(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    """Return averaged 0/1 error, taking sign(0) as +1."""
    return float(np.mean(np.where(x @ w >= 0, 1, -1) != y))


def closest_answer(value: float, choices: tuple) -> dict:
    """Select a 1-based option, taking the first option in an exact tie."""
    index = int(np.argmin(np.abs(np.asarray(choices) - value)))
    return {"choices": choices, "choice_number": index + 1,
            "closest_choice": choices[index]}


def parse_args(description: str) -> argparse.Namespace:
    """Parse common experiment and output options for the question scripts."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--trials", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026,
                        help="Base seed; each question documents its offset.")
    parser.add_argument("--iterations", type=int, default=800)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def write_report(results: list[dict], output=None) -> None:
    """Print a JSON report and optionally save it relative to the working directory."""
    report = {"python_version": platform.python_version(),
              "numpy_version": np.__version__, "experiments": results}
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    print(text, end="")
    if output is not None:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
