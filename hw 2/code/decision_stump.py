"""Shared decision stump implementation and reproducible homework experiments."""

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Stump:
    """A fitted hypothesis h(x) = s * sign(x - theta).

    Attributes:
        s: Ray direction, either -1 or +1.
        theta: Threshold in [-1, 1].
        ein: Training classification error.
    """

    s: int
    theta: float
    ein: float

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict labels, defining sign(0) as -1.

        Args:
            x: Input coordinates.

        Returns:
            An array of labels in {-1, +1}.
        """
        return self.s * np.where(np.asarray(x) > self.theta, 1, -1)


def generate_data(
    n: int, tau: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Draw uniform inputs and independently flip target labels.

    Args:
        n: Positive number of examples.
        tau: Label flip probability in [0, 1].
        rng: Random generator owned by the caller.

    Returns:
        Input and label arrays, each with shape (n,).

    Raises:
        ValueError: If n or tau is outside its valid range.
    """
    if n < 1 or not 0 <= tau <= 1:
        raise ValueError("Require n >= 1 and 0 <= tau <= 1.")
    x = rng.uniform(-1.0, 1.0, n)
    y = np.where(x > 0.0, 1, -1)
    y = np.where(rng.random(n) < tau, -y, y)
    return x, y


def fit_stump(x: np.ndarray, y: np.ndarray) -> Stump:
    """Minimize training errors over the exact candidate set in Q11.

    Sorting costs O(N log N); candidate evaluation costs O(N) using
    prefix counts. Ties are resolved by the smallest s + theta.

    Args:
        x: Nonempty one-dimensional input array in [-1, 1].
        y: Matching labels in {-1, +1}.

    Returns:
        The fitted stump and its training error.

    Raises:
        ValueError: If array shapes, inputs, or labels are invalid.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y)
    if (
        x.ndim != 1 or x.size == 0 or y.shape != x.shape
        or not np.all(np.isfinite(x)) or np.any(np.abs(x) > 1)
        or not np.all(np.isin(y, [-1, 1]))
    ):
        raise ValueError("Require matching 1-D inputs in [-1, 1] and +/-1 labels.")
    order = np.argsort(x, kind="stable")
    xs, ys = x[order], y[order]
    splits = np.flatnonzero(xs[:-1] < xs[1:]) + 1
    theta = np.r_[-1.0, (xs[splits - 1] + xs[splits]) / 2.0]
    # Inputs equal to -1 belong on the left because sign(0) = -1.
    left_count = np.r_[np.searchsorted(xs, -1.0, side="right"), splits]
    positive_prefix = np.r_[0, np.cumsum(ys == 1)]
    positive_left = positive_prefix[left_count]
    negative_right = (x.size - positive_prefix[-1]) - (left_count - positive_left)
    errors_plus = positive_left + negative_right
    errors = np.r_[x.size - errors_plus, errors_plus]
    directions = np.repeat([-1, 1], theta.size)
    thresholds = np.tile(theta, 2)
    best_errors = int(errors.min())
    tied = np.flatnonzero(errors == best_errors)
    winner = tied[np.argmin((directions + thresholds)[tied])]
    return Stump(int(directions[winner]), float(thresholds[winner]), best_errors / x.size)


def out_error(stump: Stump, tau: float) -> float:
    """Compute population error analytically for the homework distribution.

    Args:
        stump: Hypothesis with direction +/-1 and threshold in [-1, 1].
        tau: Independent label flip probability in [0, 1].

    Returns:
        Exact E_out against noisy labels, without sampling test data.

    Raises:
        ValueError: If the hypothesis or probability is invalid.
    """
    if stump.s not in (-1, 1) or not -1 <= stump.theta <= 1 or not 0 <= tau <= 1:
        raise ValueError("Invalid hypothesis or noise probability.")
    clean_error = abs(stump.theta) / 2.0
    if stump.s == -1:
        clean_error = 1.0 - clean_error
    return float(tau + (1.0 - 2.0 * tau) * clean_error)


@dataclass(frozen=True)
class Question:
    """Configuration for one homework experiment.

    Attributes:
        number: Homework question number.
        n: Training data size.
        tau: Independent label flip probability.
        choices: Multiple-choice values in their original order.
    """

    number: int
    n: int
    tau: float
    choices: tuple[float, ...]


QUESTIONS = {
    12: Question(12, 3, 0.0, (0.10, 0.15, 0.20, 0.25, 0.30)),
    13: Question(13, 30, 0.0, (0.00, 0.01, 0.02, 0.03, 0.04)),
    14: Question(14, 3, 0.1, (0.10, 0.15, 0.20, 0.25, 0.30)),
    15: Question(15, 30, 0.1, (0.00, 0.01, 0.02, 0.03, 0.04)),
    16: Question(16, 300, 0.1, (0.000, 0.002, 0.004, 0.006, 0.008)),
}


def run_experiment(question: Question, trials: int = 10000, seed: int = 2026) -> dict:
    """Repeat training and summarize the median generalization gap.

    Args:
        question: Dataset configuration and answer choices.
        trials: Positive number of independently generated datasets.
        seed: Seed for NumPy's PCG64 random generator.

    Returns:
        JSON-compatible settings, median gap, and closest answer.

    Raises:
        ValueError: If trials is not positive.
    """
    if trials < 1:
        raise ValueError("trials must be positive.")
    rng = np.random.Generator(np.random.PCG64(seed))
    gaps = np.empty(trials)
    for index in range(trials):
        x, y = generate_data(question.n, question.tau, rng)
        stump = fit_stump(x, y)
        gaps[index] = out_error(stump, question.tau) - stump.ein
    median = float(np.median(gaps))
    choice_index = int(np.argmin(np.abs(np.asarray(question.choices) - median)))
    return {
        **asdict(question),
        "trials": trials,
        "seed": seed,
        "median_gap": median,
        "choice_number": choice_index + 1,
        "closest_choice": question.choices[choice_index],
    }


def main(question_number: int | None = None) -> None:
    """Run one question or all questions through a command-line interface.

    Args:
        question_number: Fixed question number, or None to run Q12-Q16.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026, help="Base seed; add question number.")
    parser.add_argument("--output", type=Path, help="Optional JSON output path.")
    args = parser.parse_args()
    numbers = list(QUESTIONS) if question_number is None else [question_number]
    results = [
        run_experiment(QUESTIONS[number], args.trials, args.seed + number)
        for number in numbers
    ]
    report = {
        "numpy_version": np.__version__,
        "bit_generator": "PCG64",
        "results": results,
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")

