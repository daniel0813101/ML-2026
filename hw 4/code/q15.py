"""Homework 4, Q15: refit on all training rows at the validation-selected lambda."""

from regularized_logistic import (
    ERROR_CHOICES,
    answer_fields,
    count_errors,
    fit_logistic,
    fit_validation_candidates,
    load_data,
    parse_args,
    write_report,
)


def solve(data_dir=None):
    """Refit on all training rows at the validation-selected lambda.

    Args:
        data_dir: Optional directory containing the assignment data files.

    Returns:
        The measured quantities and the closest multiple-choice answer.
    """
    x, y = load_data("train", data_dir)
    x_test, y_test = load_data("test", data_dir)
    selected, _, errors = fit_validation_candidates(x, y)
    model = fit_logistic(x, y, selected)
    test_errors = count_errors(model, x_test, y_test)
    value = test_errors / len(y_test)
    return {
        "number": 15,
        "selected_log_lambda": selected,
        "validation_error_counts": errors,
        "validation_size": len(y) - 120,
        "train_size": len(y),
        "test_size": len(y_test),
        "test_error_count": test_errors,
        "test_error": value,
        **answer_fields(value, ERROR_CHOICES),
    }


def main():
    """Run this question and print or save its JSON result."""
    args = parse_args(__doc__)
    write_report([solve(args.data_dir)], args.output)


if __name__ == "__main__":
    main()
