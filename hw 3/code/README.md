# Homework 3: Linear models (Q11–Q16)

Each question has an independent `solve()` function and command-line entry point. Question settings, experiment loops, aggregation, and answer choices live in `q11.py` through `q16.py`. `linear_models.py` contains only utilities reused by multiple scripts; the linear SGD routine used only by Q12 stays in `q12.py`.

## Environment and reproduction

Verified using the WSL conda environment `ML2026`, Python 3.15.0, and NumPy 2.5.4. From this directory:

```bash
conda activate ML2026
python -m pip install -r requirements.txt
python -B run_all.py --output results.json
python -B -m unittest -v test_linear_models.py
```

On Windows, create a virtual environment using a compatible Python version:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe run_all.py --output results.json
```

Run any question independently:

```bash
python q11.py
python q12.py --output results/q12.json
python q15.py --trials 1000 --seed 2026 --iterations 800 --learning-rate 0.001
```

The defaults are the assignment settings: 1,000 trials, 800 SGD updates per trial, and learning rate 0.001. Q11 and Q16 are deterministic and use only `--data-dir` and `--output`; the common SGD arguments do not affect them. Each `solve()` returns a dictionary without printing or writing files. The entry points print JSON and optionally save it. Relative output paths use the current working directory; data paths default to `data/` beside the scripts, so scripts can also run from the repository root.

## Files

| File | Purpose |
| --- | --- |
| `q11.py` | Fit linear regression and compute training squared error |
| `q12.py` | Implement linear SGD from zero and average training squared errors |
| `q13.py` | Run logistic SGD from zero and average training cross-entropies |
| `q14.py` | Run logistic SGD from Q11's least-squares weights |
| `q15.py` | Repeat Q14's training and average absolute classification error gaps |
| `q16.py` | Evaluate the classification error gap of Q11's linear model |
| `linear_models.py` | Shared loading, least squares, logistic SGD, losses, validation, and CLI/JSON utilities |
| `run_all.py` | Call the six question implementations and combine their reports |
| `data/` | Original assignment training and test files |
| `results.json` | Full default-run results, including per-trial measurements |
| `test_linear_models.py` | Mathematical and experiment consistency checks |
| `requirements.txt` | Verified NumPy version |

To submit an individual question, include its `qNN.py`, `linear_models.py`, and instructions for obtaining the data. Each question runs without the other question scripts. `run_all.py` requires all six.

## Data

The bundled data were downloaded unchanged from the assignment URLs on November 3, 2026:

- Training: https://www.csie.ntu.edu.tw/~htlin/course/ml26fall/hw3/hw3_train.dat
- Test: https://www.csie.ntu.edu.tw/~htlin/course/ml26fall/hw3/hw3_test.dat

There are 100 training rows and 400 test rows. Each row has 10 features followed by a label in {-1, +1}. The loader prepends a bias feature of 1, giving 11 weights. Pass `--data-dir PATH` to use a different directory containing these two filenames. Training always uses the training file; the test set is used only to evaluate Q15 and Q16.

SHA-256 checksums:

```text
dc043eaceebf13db56375045359d738a3400a6a322469faaed3bedc967999135  hw3_train.dat
ff840d79a2d77773ec7ea655e7105bb898e244f10f3ff9b9a6d373a389ef900b  hw3_test.dat
```

## Algorithms and random seeds

- Q11 minimizes `mean((X @ w - y)**2)` using `numpy.linalg.lstsq`. This computes the least-squares solution without explicitly inverting `X.T @ X`, and also supports rank-deficient matrices.
- Q12 starts every trial at zero and applies `w -= 2 * eta * (w @ x - y) * x` for each sampled example. The factor 2 is required because the squared error has no factor 1/2.
- Q13 starts at zero; Q14 and Q15 start each trial at a fresh copy of Q11's weights. Their logistic update is `w += eta * y * x / (1 + exp(y * (w @ x)))`. The implementation evaluates the denominator through `logaddexp` to avoid overflow.
- Cross-entropy is `mean(log(1 + exp(-y * (X @ w))))`, evaluated using `logaddexp`. Every error function averages over the dataset.
- Classification predicts +1 for nonnegative scores and -1 otherwise. Q15 averages each trial's absolute train/test 0/1 error difference. Q16 computes that absolute difference once using Q11's model.

Each trial samples 800 row indices uniformly with replacement, independently of the labels and current weights. Row indices are generated before the update loop, which is equivalent to drawing one independently each iteration. Each trial gets a fresh NumPy PCG64 generator with seed `base_seed + question_number + trial_index`, where the trial index starts at zero. Q15 deliberately uses question number **14** to reproduce exactly the models in Q14. The default first seeds are 2038 for Q12, 2039 for Q13, and 2040 for Q14/Q15. Trials within each question have different seeds. Fixed data, versions, settings, and seeds reproduce the saved results; changing the seed can slightly change the averages.

## Verified results

All stochastic questions used 1,000 trials of 800 updates with learning rate 0.001 and base seed 2026.

| Question | Quantity | Measured value | Closest option | Option number |
| --- | --- | --- | --- | --- |
| Q11 | Training squared error | 0.792234776111 | 0.80 | 3 |
| Q12 | Mean training squared error | 0.823035044106 | 0.82 | 4 |
| Q13 | Mean training cross-entropy | 0.657161904722 | 0.66 | 3 |
| Q14 | Mean training cross-entropy | 0.605196152071 | 0.60 | 1 |
| Q15 | Mean absolute classification gap | 0.030942500000 | 0.03 | 1 |
| Q16 | Absolute classification gap | 0.040000000000 | 0.04 | 2 |

Q16's training and test classification errors are 0.30 and 0.34. The JSON report includes unrounded measurements, choices, experiment settings, Python/NumPy versions, and all trial errors or gaps.

All seven unit tests pass. They check data/bias handling, least-squares optimality and rank deficiency, both SGD updates against finite-difference gradients, loss averaging and extreme margins, Q14/Q15 model identity and absolute-gap aggregation, reproducibility, and invalid settings. All six standalone CLIs were also run at the full assignment settings and their complete experiment records matched `run_all.py` exactly.
