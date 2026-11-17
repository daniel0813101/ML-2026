"""Homework 4, Q12: select lambda by minimizing test-set error."""

from regularized_logistic import (
    LOG_LAMBDAS,
    answer_fields,
    fit_candidates,
    load_data,
    parse_args,
    write_report,
)


def solve(data_dir=None):
    """Select lambda by minimizing test-set error.

    Args:
        data_dir: Optional directory containing the assignment data files.

    Returns:
        The measured quantities and the closest multiple-choice answer.
    """
    x, y = load_data("train", data_dir)
    x_test, y_test = load_data("test", data_dir)
    selected, _, errors = fit_candidates(x, y, x_test, y_test)
    return {
        "number": 12,
        "selected_log_lambda": selected,
        "test_error_counts": errors,
        "test_size": len(y_test),
        **answer_fields(selected, LOG_LAMBDAS),
    }


def main():
    """Run this question and print or save its JSON result."""
    args = parse_args(__doc__)
    write_report([solve(args.data_dir)], args.output)


if __name__ == "__main__":
    main()
