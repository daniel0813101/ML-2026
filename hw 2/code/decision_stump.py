"""Reusable decision stump learning, data generation, and output utilities."""

import argparse
from dataclasses import dataclass
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


def parse_args(description: str) -> argparse.Namespace:
    """Parse the shared experiment options without selecting a question.

    Args:
        description: Help text supplied by the calling script.

    Returns:
        Validated trial count, base seed, and optional JSON output path.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--trials", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026,
                        help="Base seed; each question adds its number.")
    parser.add_argument("--output", type=Path, help="Optional JSON output path.")
    args = parser.parse_args()
    if args.trials < 1:
        parser.error("trials must be positive")
    if args.seed < 0:
        parser.error("seed must be nonnegative")
    return args


def write_report(results: list[dict], output: Path | None = None) -> None:
    """Print and optionally save experiment records with environment metadata.

    Args:
        results: Records computed by the calling question implementations.
        output: Optional JSON path, relative to the current working directory.
    """
    report = {
        "numpy_version": np.__version__,
        "bit_generator": "PCG64",
        "results": results,
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
