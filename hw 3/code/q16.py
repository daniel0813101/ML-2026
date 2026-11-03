"""Homework 3, Q16: absolute classification gap of Q11's linear model."""

from linear_models import (classification_error, closest_answer, fit_linear,
                           load_data, parse_args, write_report)


def solve(data_dir=None) -> dict:
    """Fit Q11's model on training data and evaluate its train/test 0/1 gap."""
    x, y = load_data("train", data_dir)
    test_x, test_y = load_data("test", data_dir)
    w = fit_linear(x, y)
    in_error = classification_error(x, y, w)
    out_error = classification_error(test_x, test_y, w)
    value = abs(in_error - out_error)
    choices = (0.03, 0.04, 0.05, 0.06, 0.07)
    return {"number": 16, "metric": "absolute_classification_gap", "value": value,
            "in_classification_error": in_error,
            "out_classification_error": out_error,
            "train_size": len(y), "test_size": len(test_y),
            **closest_answer(value, choices)}


def main() -> None:
    """Run this question and print or save its result."""
    args = parse_args(__doc__)
    write_report([solve(args.data_dir)], args.output)


if __name__ == "__main__":
    main()
