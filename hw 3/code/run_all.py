"""Run the independent implementations for Homework 3, Q11-Q16."""

from linear_models import parse_args, write_report
from q11 import solve as solve_q11
from q12 import solve as solve_q12
from q13 import solve as solve_q13
from q14 import solve as solve_q14
from q15 import solve as solve_q15
from q16 import solve as solve_q16


def main() -> None:
    """Run all six questions using the same settings as their individual CLIs."""
    args = parse_args(__doc__)
    results = [solve_q11(args.data_dir)]
    for solve in (solve_q12, solve_q13, solve_q14, solve_q15):
        results.append(solve(args.trials, args.seed, args.iterations,
                             args.learning_rate, args.data_dir))
    results.append(solve_q16(args.data_dir))
    write_report(results, args.output)


if __name__ == "__main__":
    main()
