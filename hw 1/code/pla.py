"""Shared reproducible PLA experiments for Homework 1, Q11 through Q16."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

import numpy as np
from numba import njit

ROOT = Path(__file__).resolve().parent
DATA_URL = "https://www.csie.ntu.edu.tw/~htlin/course/ml26fall/hw1/hw1_train.dat"
CONFIGS = {"baseline": (1.0, 1.0), "half_scale": (1.0, 0.5),
           "half_bias": (0.5, 1.0), "zero_bias": (0.0, 1.0)}
QUESTIONS = {
    11: ("baseline", "zero_error_percent", [0, 2, 4, 6, 8]),
    12: ("baseline", "median_updates", [600, 850, 1100, 1350, 1600]),
    13: ("baseline", "median_norm", [65, 75, 85, 95, 105]),
    14: ("half_scale", "median_updates", [600, 850, 1100, 1350, 1600]),
    15: ("half_bias", "median_updates", [2200, 2400, 2600, 2800, 3000]),
    16: ("zero_bias", "median_updates", [7500, 10000, 12500, 15000, 17500]),
}


def load_data():
    """Download once and load the original training examples.

    Returns:
        A tuple containing the feature matrix and integer labels.

    Raises:
        ValueError: If the downloaded data has an unexpected shape or labels.
    """
    path = ROOT / "data" / "hw1_train.dat"
    path.parent.mkdir(exist_ok=True)
    if not path.exists():
        with urlopen(DATA_URL, timeout=60) as response:
            payload = response.read()
        values = np.loadtxt(payload.splitlines())
        if values.shape != (256, 13):
            raise ValueError("Expected 256 rows and 13 columns.")
        path.write_bytes(payload)
    data = np.loadtxt(path)
    if data.shape != (256, 13) or not np.isfinite(data).all():
        raise ValueError("Expected a finite 256-by-13 dataset.")
    if not np.isin(data[:, -1], [-1, 1]).all():
        raise ValueError("Labels must be -1 or +1.")
    return data[:, :-1], data[:, -1].astype(np.int64)


@njit(cache=True)
def train_pla(x, y, seed, max_checks):
    """Train PLA with replacement sampling and the consecutive-correct stop.

    Args:
        x: Augmented, scaled feature matrix.
        y: Labels in {-1, +1}.
        seed: Seed for Numba's NumPy random generator.
        max_checks: Safety limit on sampled examples; never a successful stop.

    Returns:
        Final weights, number of updates, and number of checked examples.

    Raises:
        RuntimeError: If the safety limit is reached before the required stop.
    """
    np.random.seed(seed)
    n, d = x.shape
    w = np.zeros(d, dtype=np.float64)
    consecutive = 0
    updates = 0
    for checks in range(1, max_checks + 1):
        index = np.random.randint(n)
        score = 0.0
        for j in range(d):
            score += w[j] * x[index, j]
        prediction = 1 if score >= 0.0 else -1
        if prediction != y[index]:
            for j in range(d):
                w[j] += y[index] * x[index, j]
            updates += 1
            consecutive = 0
        else:
            consecutive += 1
        if consecutive == 2 * n:
            return w, updates, checks
    raise RuntimeError("PLA exceeded max_checks before 2N consecutive correct checks.")


def run_experiments(config, runs=1000, start_seed=0, max_checks=100_000_000):
    """Run independent seeded experiments and persist the complete results.

    Args:
        config: Key identifying the bias and global scaling configuration.
        runs: Number of experiments.
        start_seed: First seed in the consecutive seed sequence.
        max_checks: Maximum sampled examples allowed per experiment.

    Returns:
        A dictionary with aggregate statistics for this configuration.
    """
    features, labels = load_data()
    bias, scale = CONFIGS[config]
    x = np.ascontiguousarray(
        np.column_stack((np.full(len(labels), bias), features)) * scale)
    weights = np.empty((runs, x.shape[1]))
    updates = np.empty(runs, dtype=np.int64)
    checks = np.empty(runs, dtype=np.int64)
    errors = np.empty(runs, dtype=np.int64)
    for i in range(runs):
        weights[i], updates[i], checks[i] = train_pla(
            x, labels, start_seed + i, max_checks)
        # Recompute predictions over all N examples after the randomized stop.
        scores = np.sum(x * weights[i], axis=1)
        errors[i] = np.count_nonzero(np.where(scores >= 0, 1, -1) != labels)
    norms = np.linalg.norm(weights, axis=1)
    summary = {
        "configuration": config, "runs": runs, "start_seed": start_seed,
        "bias_before_scaling": bias, "global_scale": scale,
        "dataset_sha256": hashlib.sha256(
            (ROOT / "data" / "hw1_train.dat").read_bytes()).hexdigest(),
        "zero_error_percent": float(100 * np.mean(errors == 0)),
        "median_updates": float(np.median(updates)),
        "median_norm": float(np.median(norms)),
    }
    output = ROOT / "results" / f"{config}_{runs}_{start_seed}"
    output.parent.mkdir(exist_ok=True)
    np.savez(output.with_suffix(".npz"), weights=weights, updates=updates,
             checks=checks, errors=errors, norms=norms)
    with output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["seed", "updates", "checks", "weight_norm", "error_count", "ein"])
        for i in range(runs):
            writer.writerow([start_seed + i, updates[i], checks[i], norms[i],
                             errors[i], errors[i] / len(labels)])
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def report(question, summary):
    """Print the measured statistic and closest multiple-choice answer.

    Args:
        question: Question number from 11 through 16.
        summary: Aggregate statistics returned by run_experiments.

    Returns:
        A dictionary describing the numerical result and selected choice.
    """
    _, metric, choices = QUESTIONS[question]
    value = summary[metric]
    index = min(range(len(choices)), key=lambda i: abs(choices[i] - value))
    answer = {"question": question, "metric": metric, "value": value,
              "choice_number": index + 1, "choice_value": choices[index]}
    print(json.dumps(answer, indent=2))
    return answer


def main(question=None):
    """Run one question or all questions from command-line arguments.

    Args:
        question: Fixed question number for a per-question entry point, or None.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=1000)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--max-checks", type=int, default=100_000_000)
    args = parser.parse_args()
    if args.runs < 1 or args.max_checks < 1:
        parser.error("runs and max-checks must be positive")
    if args.start_seed < 0 or args.start_seed + args.runs - 1 > 2**32 - 1:
        parser.error("seeds must be within the unsigned 32-bit range")
    questions = [question] if question is not None else list(QUESTIONS)
    summaries = {}
    answers = []
    for number in questions:
        config = QUESTIONS[number][0]
        if config not in summaries:
            print(f"Running {config}: {args.runs} experiments...", flush=True)
            summaries[config] = run_experiments(
                config, args.runs, args.start_seed, args.max_checks)
        answers.append(report(number, summaries[config]))
    name = "answers" if question is None else f"q{question}_answer"
    destination = ROOT / "results" / f"{name}_{args.runs}_{args.start_seed}.json"
    destination.write_text(json.dumps(answers, indent=2) + "\n")
