"""Run the five Homework 4 questions and combine their results."""

import q12
import q13
import q14
import q15
import q16
from regularized_logistic import parse_args, write_report


def main():
    """Run Q12–Q16 with the same data directory and report all answers."""
    args = parse_args(__doc__)
    questions = (q12, q13, q14, q15, q16)
    write_report([question.solve(args.data_dir) for question in questions], args.output)


if __name__ == "__main__":
    main()
