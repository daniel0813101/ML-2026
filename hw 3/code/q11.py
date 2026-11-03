"""Homework 3, Q11: in-sample squared error of linear regression."""

from linear_models import (closest_answer, fit_linear, load_data, parse_args,
                           squared_error, write_report)


def solve(data_dir=None) -> dict:
    """Fit Q11's least-squares model and select its closest squared-error option."""
    x, y = load_data("train", data_dir)
    w = fit_linear(x, y)
    value = squared_error(x, y, w)
    choices = (0.76, 0.78, 0.80, 0.82, 0.84)
    return {"number": 11, "metric": "in_squared_error", "value": value,
            "weights": w.tolist(), "train_size": len(y),
            **closest_answer(value, choices)}


def main() -> None:
    """Run this question and print or save its result."""
    args = parse_args(__doc__)
    write_report([solve(args.data_dir)], args.output)


if __name__ == "__main__":
    main()
