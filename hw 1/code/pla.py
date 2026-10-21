"""Reusable PLA training, experiment collection, and output utilities."""

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


def run_experiments(x, labels, runs=1000, start_seed=0,
                    max_checks=100_000_000):
    """Collect seeded PLA outcomes for an already prepared feature matrix.

    Args:
        x: Augmented feature matrix, prepared by the calling question.
        labels: Training labels in {-1, +1}.
        runs: Number of independent experiments.
        start_seed: First seed in a consecutive sequence.
        max_checks: Safety limit on sampled examples per experiment.

    Returns:
        Arrays of weights, update/check counts, full-data errors, and norms.
    """
    if runs < 1 or max_checks < 1:
        raise ValueError("runs and max_checks must be positive")
    if start_seed < 0 or start_seed + runs - 1 > 2**32 - 1:
        raise ValueError("seeds must be within the unsigned 32-bit range")
    x = np.ascontiguousarray(x, dtype=np.float64)
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
    return {"weights": weights, "updates": updates, "checks": checks,
            "errors": errors, "norms": norms}


def save_experiments(config, results, labels, start_seed, bias, scale,
                     output_dir=None):
    """Save raw outcomes and descriptive statistics for a named experiment."""
    runs = len(results["updates"])
    summary = {
        "configuration": config, "runs": runs, "start_seed": start_seed,
        "bias_before_scaling": bias, "global_scale": scale,
        "dataset_sha256": hashlib.sha256(
            (ROOT / "data" / "hw1_train.dat").read_bytes()).hexdigest(),
        "zero_error_percent": float(100 * np.mean(results["errors"] == 0)),
        "median_updates": float(np.median(results["updates"])),
        "median_norm": float(np.median(results["norms"])),
    }
    directory = Path(output_dir) if output_dir is not None else ROOT / "results"
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / f"{config}_{runs}_{start_seed}"
    np.savez(output.with_suffix(".npz"), **results)
    with output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["seed", "updates", "checks", "weight_norm", "error_count", "ein"])
        for i in range(runs):
            writer.writerow([start_seed + i, results["updates"][i],
                             results["checks"][i], results["norms"][i],
                             results["errors"][i], results["errors"][i] / len(labels)])
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def report(question, metric, value, choices):
    """Print and return the nearest choice for a caller-supplied statistic."""
    index = min(range(len(choices)), key=lambda i: abs(choices[i] - value))
    answer = {"question": question, "metric": metric, "value": float(value),
              "choice_number": index + 1, "choice_value": choices[index]}
    print(json.dumps(answer, indent=2))
    return answer


def save_answers(name, answers, runs, start_seed, output_dir=None):
    """Persist question answers using the existing JSON list format."""
    directory = Path(output_dir) if output_dir is not None else ROOT / "results"
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{name}_{runs}_{start_seed}.json"
    destination.write_text(json.dumps(answers, indent=2) + "\n")


def parse_args(description):
    """Parse and validate the experiment options shared by the scripts."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--runs", type=int, default=1000)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--max-checks", type=int, default=100_000_000)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    if args.runs < 1 or args.max_checks < 1:
        parser.error("runs and max-checks must be positive")
    if args.start_seed < 0 or args.start_seed + args.runs - 1 > 2**32 - 1:
        parser.error("seeds must be within the unsigned 32-bit range")
    return args
