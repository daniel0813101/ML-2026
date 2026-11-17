"""Homework 4, Q16: select lambda by fixed five-fold cross-validation."""

import numpy as np

from regularized_logistic import (
    ERROR_CHOICES,
    LOG_LAMBDAS,
    answer_fields,
    count_errors,
    fit_logistic,
    load_data,
    parse_args,
    select_log_lambda,
    write_report,
)


def solve(data_dir=None):
    """Select lambda by fixed five-fold cross-validation.

    Args:
        data_dir: Optional directory containing the assignment data files.

    Returns:
        The measured quantities and the closest multiple-choice answer.
    """
    x, y = load_data("train", data_dir)
    rows = np.arange(len(y))
    folds = np.split(rows, 5)
    fold_errors = {}
    for exponent in LOG_LAMBDAS:
        errors = []
        for fold in folds:
            train_rows = (rows < fold[0]) | (rows > fold[-1])
            model = fit_logistic(x[train_rows], y[train_rows], exponent)
            errors.append(count_errors(model, x[fold], y[fold]))
        fold_errors[exponent] = errors

    # The folds have equal size, so minimizing total mistakes minimizes E_cv.
    totals = {exponent: sum(errors) for exponent, errors in fold_errors.items()}
    selected = select_log_lambda(totals)
    value = totals[selected] / len(y)
    return {
        "number": 16,
        "selected_log_lambda": selected,
        "fold_error_counts": fold_errors,
        "fold_size": len(folds[0]),
        "cv_error_count": totals[selected],
        "cv_error": value,
        **answer_fields(value, ERROR_CHOICES),
    }


def main():
    """Run this question and print or save its JSON result."""
    args = parse_args(__doc__)
    write_report([solve(args.data_dir)], args.output)


if __name__ == "__main__":
    main()
