# Homework 4 — Regularized logistic regression (Q12–Q16)

The problem statements and choices are in [question.md](../question.md). Each question has its own Python script and Google-style function docstrings. Shared data loading, feature transformation, model fitting, model selection, and reporting are in `regularized_logistic.py`.

## Reproduce the results

Run these commands from this directory with Python 3.13.5 or an environment with compatible versions of the pinned packages.

```bash
python -m pip install -r requirements.txt
python run_all.py --output results.json
python q14.py
```

The scripts print JSON and optionally save the same report with `--output PATH`. Use `--data-dir PATH` to read separate copies of `hw4_train.dat` and `hw4_test.dat`. The default `data/` directory is resolved relative to the scripts. The bundled course files contain 200 training rows and 300 test rows, each with six features followed by a label in $\{-1,+1\}$. [Training data](https://www.csie.ntu.edu.tw/~htlin/course/ml26fall/hw4/hw4_train.dat) · [Test data](https://www.csie.ntu.edu.tw/~htlin/course/ml26fall/hw4/hw4_test.dat)

## Implementation

The feature map contains the constant, six original coordinates, and the 21 products $x_i x_j$ for $i\le j$. The constant is a regularized feature and LIBLINEAR's additional bias stays disabled. Multiplying the homework objective by $N/(2\lambda)$ gives LIBLINEAR's objective $\frac12\|\mathbf w\|^2+C\sum_n\log(1+\exp(-y_n\mathbf w^T\Phi_2(\mathbf x_n)))$, so $C=1/(2\lambda)$ for every training split. The solver uses `-s 0 -e 0.000001`. [LIBLINEAR README](https://raw.githubusercontent.com/cjlin1/liblinear/master/README)

Candidate selection compares integer mistake counts across equal-sized evaluation sets and chooses the largest $\lambda$ on a tie. Q14 fits on rows 1–120 and validates on rows 121–200. Q15 repeats that selection and fits a fresh model on all 200 rows. Q16 holds out each of five consecutive 40-row folds in turn and divides the total holdout mistakes by 200. The JSON reports include the candidate mistake counts and the selected multiple-choice option.

## Verified results

| Question | Selection criterion | Selected $\log_{10}\lambda$ | Measured quantity | Closest option |
| --- | --- | ---: | --- | ---: |
| Q12 | Test error | −2 | $\log_{10}\lambda^*=-2$ | −2 |
| Q13 | Training error | −4 | $\log_{10}\lambda^*=-4$ | −4 |
| Q14 | Fixed validation split | −2 | Test error $43/300\approx0.14333$ | 0.14 |
| Q15 | Fixed validation split, full refit | −2 | Test error $39/300=0.13$ | 0.13 |
| Q16 | Fixed five-fold cross-validation | −2 | Cross-validation error $24/200=0.12$ | 0.12 |

Verified with Python 3.13.5, NumPy 2.2.6, SciPy 1.16.2, and liblinear-official 2.50.0. All five standalone reports match `run_all.py`.
