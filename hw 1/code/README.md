# Homework 1: PLA experiments (Q11-Q16)

## Verified environment

The complete experiments and all six individual question entry points were rerun successfully in WSL Ubuntu using Conda environment `ML2026` on October 20, 2026.

| Component | Version |
| --- | --- |
| Python | 3.15.0 |
| NumPy | 2.5.4 |
| Numba | 0.68.0 |
| llvmlite | 0.50.0 |

`requirements.txt` now pins these verified package versions. The previous run used Python 3.12.7, NumPy 1.26.4, and Numba 0.60.0. All saved numerical arrays from the new run matched the previous run exactly; see `results/ML2026_validation.json`.

## Reproduce in WSL

```bash
conda activate ML2026
cd "/home/daniel/Study/ML-2026/hw 1/code"
python run_all.py
```

To create the same environment on another machine, run this from the code folder:

```bash
conda env create -f environment.yml
conda activate ML2026
python run_all.py
```

For an existing Python 3.15.0 environment, packages can also be installed with `python -m pip install -r requirements.txt`. The requirements file specifies packages, not the Python interpreter version.

The data is downloaded once to `data/hw1_train.dat`, using the HTTPS equivalent of the assignment URL. Later runs use the local file. Paths are relative to the scripts, so changing the shell working directory does not change output locations. Keep the data and its SHA-256 in the result JSON files for reproducibility.

## Files

| File | Purpose |
| --- | --- |
| `pla.py` | Reusable data loader, PLA trainer, experiment collection, and output utilities |
| `q11.py` through `q16.py` | Independent question implementations: input transformation, statistic, and answer choices |
| `run_all.py` | Calls each question’s solve function and saves the combined answers |
| `report.md` | English tutorial: derive, implement, validate, and interpret each question |
| `requirements.txt` | Exact tested Python package versions |
| `environment.yml` | Conda environment including the tested Python version |

To rerun an individual question:

```bash
python q11.py
python q16.py
```

Each question owns a `solve()` function that prepares its inputs, calls the library, computes its requested statistic, and selects its answer. `pla.py` contains no question table or question dispatcher. Every individual script performs fresh experiments. With identical inputs, seeds, and environment, it reproduces the corresponding result from `run_all.py`.

The batch runner also performs fresh experiments for each question, including separate reproducible baseline runs for Q11-Q13. This takes longer than sharing one baseline run, but exercises each question’s implementation. Use `--output-dir /tmp/hw1-practice` to keep practice outputs separate from the saved assignment results.

For a small practice run:

```bash
python run_all.py --runs 10
```

Use the default 1000 experiments, seeds 0-999, for the reported answers:

```bash
python run_all.py --runs 1000 --start-seed 0
```

## Verified answers

| Question | Measured statistic | Closest option | Choice number |
| --- | ---: | ---: | ---: |
| Q11 | 0% | 0% | 1 |
| Q12 | 1386 updates | 1350 | 4 |
| Q13 | 75.34345049771655 | 75 | 2 |
| Q14 | 1386 updates | 1350 | 4 |
| Q15 | 2194 updates | 2200 | 1 |
| Q16 | 12437 updates | 12500 | 3 |

## Outputs and execution rules

Each configuration writes CSV (per-run statistics), NPZ (full weights and arrays), and JSON (aggregate statistics and data hash) files under `results/`. The complete answer list is `answers_1000_0.json`. Individual scripts write files such as `q11_answer_1000_0.json`. Files with identical parameters are overwritten on rerun. Results from before this migration were backed up under `results/previous_20261020_214528/`.

Weights start at zero; sign(0) is +1. Sampling is uniform with replacement. Only mistakes trigger updates. Each update resets the consecutive-correct counter; stopping requires 2N = 512 consecutive correct sampled predictions. Training errors are then recalculated on the entire dataset. Each experiment has a 100,000,000-check safety limit; reaching it raises an error rather than returning an incomplete result.

Q14 scales all augmented features by 0.5. Q15 changes only the bias feature to 0.5. Q16 changes only the bias feature to zero. Q15/Q16's inherited wording "scaled examples" is interpreted according to their explicit instruction not to scale the original features.

Execution logs are saved as `results/*_ML2026.log`. The validation record includes version metadata, comparison against previous arrays, and key correctness checks. Standard-library modules require no additional installation. If submitting a question entry point, include `pla.py` and `requirements.txt` with it.
