"""Run the independent implementations for Homework 1, Q11-Q16."""

from pla import parse_args, save_answers
from q11 import solve as solve_q11
from q12 import solve as solve_q12
from q13 import solve as solve_q13
from q14 import solve as solve_q14
from q15 import solve as solve_q15
from q16 import solve as solve_q16


def main():
    """Call each question's implementation and save the combined answers."""
    args = parse_args(__doc__)
    answers = []
    for solve in (solve_q11, solve_q12, solve_q13,
                  solve_q14, solve_q15, solve_q16):
        answers.append(solve(args.runs, args.start_seed,
                             args.max_checks, args.output_dir))
    save_answers("answers", answers, args.runs, args.start_seed, args.output_dir)


if __name__ == "__main__":
    main()
