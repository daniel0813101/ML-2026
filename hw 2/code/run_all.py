"""Run the independent implementations for Homework 2, Q12-Q16."""

from decision_stump import parse_args, write_report
from q12 import solve as solve_q12
from q13 import solve as solve_q13
from q14 import solve as solve_q14
from q15 import solve as solve_q15
from q16 import solve as solve_q16


def main() -> None:
    """Call each question implementation and print or save the combined report."""
    args = parse_args(__doc__)
    results = [solve(args.trials, args.seed) for solve in
               (solve_q12, solve_q13, solve_q14, solve_q15, solve_q16)]
    write_report(results, args.output)


if __name__ == "__main__":
    main()
