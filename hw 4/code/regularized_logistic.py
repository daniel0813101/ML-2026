"""Shared data, LIBLINEAR, model selection, and reporting utilities for HW4."""

import argparse
import json
from decimal import Decimal
from pathlib import Path

import numpy as np
from liblinear.liblinearutil import predict, train

DATA_DIR = Path(__file__).resolve().parent / "data"
LOG_LAMBDAS = (-4, -2, 0, 2, 4)
ERROR_CHOICES = (0.10, 0.11, 0.12, 0.13, 0.14)


def polynomial_features(x):
    """Map each input row to its constant, linear, and quadratic features.

    Args:
        x: Two-dimensional array of original feature rows.

    Returns:
        Transformed rows, with x_i*x_j ordered by i and then j for i <= j.
    """
    i, j = np.triu_indices(x.shape[1])
    return np.column_stack((np.ones(len(x)), x, x[:, i] * x[:, j]))


def load_data(split, data_dir=None):
    """Load an assignment dataset and apply the polynomial transformation.

    Args:
        split: Dataset name, either "train" or "test".
        data_dir: Directory containing the assignment files, or None for data/.

    Returns:
        A pair of arrays containing the 28-feature design matrix and the labels.

    Raises:
        ValueError: If the split or file contents violate the assignment format.
    """
    if split not in ("train", "test"):
        raise ValueError("split must be 'train' or 'test'.")
    directory = DATA_DIR if data_dir is None else Path(data_dir)
    path = directory / f"hw4_{split}.dat"
    data = np.loadtxt(path, ndmin=2)
    if (data.shape[1] != 7 or not len(data) or not np.isfinite(data).all()
            or not np.isin(data[:, -1], (-1, 1)).all()):
        raise ValueError(f"{path} must contain six finite features and a +/-1 label per row.")
    if split == "train" and len(data) != 200:
        raise ValueError(f"{path} must contain exactly 200 training rows.")
    return polynomial_features(data[:, :-1]), data[:, -1]


def fit_logistic(x, y, log_lambda):
    """Fit L2-regularized logistic regression using LIBLINEAR solver 0.

    Args:
        x: Transformed training features, including the regularized constant.
        y: Training labels in {-1, +1}.
        log_lambda: Base-10 exponent of the regularization strength.

    Returns:
        The fitted LIBLINEAR model.
    """
    # Multiplying the homework objective by N/(2*lambda) gives LIBLINEAR's.
    cost = 1 / (2 * 10.0 ** log_lambda)
    return train(y, x, f"-s 0 -c {cost} -e 0.000001 -q")


def count_errors(model, x, y):
    """Count 0/1 mistakes on an evaluation set.

    Args:
        model: Fitted LIBLINEAR model.
        x: Transformed evaluation features.
        y: Evaluation labels in {-1, +1}.

    Returns:
        Integer number of misclassified rows.
    """
    labels, _, _ = predict(y, x, model, "-q")
    return int(np.count_nonzero(np.asarray(labels) != y))


def select_log_lambda(errors):
    """Choose the minimum-error exponent, preferring larger lambda on ties.

    Args:
        errors: Mapping from log10(lambda) to integer misclassification counts
            measured on the same number of evaluation examples per candidate.

    Returns:
        The selected log10(lambda).
    """
    return min(errors, key=lambda exponent: (errors[exponent], -exponent))


def fit_candidates(x_train, y_train, x_eval, y_eval):
    """Fit the candidate grid and score each model on a fixed evaluation set.

    Args:
        x_train: Transformed fitting features.
        y_train: Fitting labels.
        x_eval: Transformed evaluation features.
        y_eval: Evaluation labels.

    Returns:
        The selected exponent, its fitted model, and all candidate error counts.
    """
    models = {
        exponent: fit_logistic(x_train, y_train, exponent)
        for exponent in LOG_LAMBDAS
    }
    errors = {
        exponent: count_errors(model, x_eval, y_eval)
        for exponent, model in models.items()
    }
    selected = select_log_lambda(errors)
    return selected, models[selected], errors


def fit_validation_candidates(x, y):
    """Fit and select candidates using the assignment's fixed 120/80 split.

    Args:
        x: Transformed features for the 200 training examples in file order.
        y: Labels for the 200 training examples in file order.

    Returns:
        The selected exponent, its 120-row model, and validation error counts.
    """
    return fit_candidates(x[:120], y[:120], x[120:], y[120:])


def answer_fields(value, choices):
    """Find the closest available answer, preferring the first exact tie.

    Args:
        value: Numeric answer before rounding to a multiple-choice option.
        choices: Available numeric options in the order listed by the homework.

    Returns:
        A dictionary with the options, the closest choice, and its 1-based number.
    """
    target = Decimal(str(value))
    index = min(
        range(len(choices)),
        key=lambda i: abs(Decimal(str(choices[i])) - target),
    )
    return {"choices": choices, "closest_choice": choices[index], "choice_number": index + 1}


def parse_args(description):
    """Parse the data directory and JSON output options for a CLI.

    Args:
        description: Question or runner description shown in --help.

    Returns:
        The parsed argument namespace.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output", type=Path, help="Also save JSON to this path.")
    return parser.parse_args()


def write_report(results, output=None):
    """Print experiment results as JSON and optionally write them to a file.

    Args:
        results: List of question result dictionaries in display order.
        output: Optional destination path for the JSON report.
    """
    report = json.dumps({"experiments": results}, indent=2, allow_nan=False) + "\n"
    print(report, end="")
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report, encoding="utf-8")
