# Learning to implement and analyze PLA: Homework 1, Q11-Q16

This report is a guide to solving the assignment yourself: translate the wording into a specification, build the smallest correct algorithm, check its behavior, repeat experiments, and interpret the numbers. Read it alongside `pla.py`; the six `q*.py` files select a question and delegate to that shared implementation.

## 1. Results and experimental context

The full experiment was rerun on October 20, 2026, in WSL Ubuntu, using Conda environment `ML2026`: Python 3.15.0, NumPy 2.5.4, Numba 0.68.0, and llvmlite 0.50.0. Each configuration used 1000 experiments with seeds 0 through 999.

| Question | Statistic | Measured value | Closest listed value | Select |
| --- | --- | ---: | ---: | ---: |
| Q11 | Percentage of final weights with zero training error | 0% | 0% | Choice 1 |
| Q12 | Median update count | 1386 | 1350 | Choice 4 |
| Q13 | Median final weight norm | 75.34345049771655 | 75 | Choice 2 |
| Q14 | Median update count, all augmented features halved | 1386 | 1350 | Choice 4 |
| Q15 | Median update count, bias feature 0.5 | 2194 | 2200 | Choice 1 |
| Q16 | Median update count, bias feature zero | 12437 | 12500 | Choice 3 |

`run_all.py` and each of `q11.py` through `q16.py` completed successfully. Every saved numerical array in the four configurations matched the earlier Windows run exactly, including weights, update counts, check counts, error counts, and norms. This is an observed result for these environments and data, not a promise that arbitrary future environments produce identical random experiments.

## 2. Start by turning the question into a checklist

Before coding, extract the rules that would change your answer if misunderstood:

1. There are N = 256 rows. Each row contains 12 features followed by one label.
2. Labels are -1 or +1. Add a bias feature before training.
3. Initialize every experiment with a fresh zero weight vector.
4. A score of exactly zero predicts +1.
5. Sample one row uniformly **with replacement** at each iteration.
6. Update only when the prediction is wrong.
7. Stop after 2N consecutive correct sampled predictions, resetting the count on every mistake.
8. Repeat with 1000 distinct seeds, and compute the requested statistic.
9. For Q11, evaluate all training rows again after stopping.

Then identify what changes between questions:

| Configuration | Augmented input | Used by |
| --- | --- | --- |
| Baseline | (1, x1, ..., x12) | Q11-Q13 |
| Half scale | (0.5, 0.5*x1, ..., 0.5*x12) | Q14 |
| Half bias | (0.5, x1, ..., x12) | Q15 |
| Zero bias | (0, x1, ..., x12) | Q16 |

This table suggests the program structure: one data loader, one training function, one experiment driver, and different configuration parameters. Writing six independent PLA implementations would make it easier for an accidental difference to influence the comparison.

## 3. Understand the classifier before implementing the loop

For an augmented input x, compute the score and prediction:

$$s = w^T x, \qquad \hat y = \begin{cases}+1 & s\geq 0,\-1 & s<0.\end{cases}$$

With bias feature 1, this score is

$$w_0 + \sum_{j=1}^{12} w_j x_j.$$

The extra coordinate lets a single dot product represent an intercept. The augmented matrix has shape (256, 13), and the weight vector has length 13. Its column count happens to equal the raw file's column count, but the last raw column is a label, while every augmented column is an input.

Load the file with `np.loadtxt`, split `data[:, :-1]` and `data[:, -1]`, check the shape and labels, then add the bias column. In `run_experiments`, the construction is:

```python
x = np.column_stack((np.full(len(labels), bias), features)) * scale
```

The order matters for Q14: augment first, then scale, so the bias is halved too. `np.ascontiguousarray` prepares a consistent memory layout for the compiled loop. `float64` gives a consistent floating-point representation for features and weights.

Avoid `np.sign(score)` here: NumPy returns zero for a zero score, whereas this assignment requires +1. Use `1 if score >= 0 else -1`.

## 4. Derive and trace one PLA update

If a prediction is wrong, update

$$w_{t+1}=w_t+y_n x_n.$$

For a positive label, adding x increases the score on that example by its squared norm. For a negative label, subtracting x decreases that score. More generally, its signed score changes by

$$y_n w_{t+1}^T x_n = y_n w_t^T x_n + \|x_n\|^2.$$

This improves the signed score on the selected example, but can worsen predictions on other examples. Training error therefore need not decrease after every update.

A manual trace helps catch mistakes. Suppose the augmented example is x = (1, 2), its label is -1, and w = (0, 0):

1. The score is 0, so the prediction is +1.
2. This is incorrect, so w becomes (-1, -2).
3. The score on that same example is now -5, which predicts -1 correctly.

If the initial label had been +1, the zero vector would already classify the example correctly; there would be no update.

## 5. Write the training loop in plain language first

A minimal algorithm is:

```text
set this experiment's random seed
initialize w to zero
initialize updates and consecutive_correct to zero
repeat:
    sample one index uniformly from 0 through N-1
    compute its prediction, with zero predicting +1
    if prediction differs from its label:
        w = w + label * example
        updates += 1
        consecutive_correct = 0
    else:
        consecutive_correct += 1
    if consecutive_correct == 2*N:
        return w, updates, checks
```

There are three different quantities: `checks` counts every sampled row, `updates` counts only mistakes, and `consecutive` counts the current uninterrupted streak of correct predictions. Q12 asks for updates, so counting iterations would answer a different question.

`np.random.randint(n)` samples uniformly with replacement. Do not replace it with a shuffled pass through all rows: that would change the sampling procedure. Resetting the seed inside each training call ensures a run does not depend on which other experiments were executed beforehand.

`train_pla` implements this algorithm and returns its final weights and counters. Its 100,000,000-check limit is a guard against an unexpectedly long run. Hitting that limit raises an exception; it is never treated as successful completion.

Numba's `@njit(cache=True)` compiles the loop for speed. The explicit coordinate loops compute the dot product and update without allocating temporary arrays at every check. They also avoid using parallel reductions or fast-math reassociation. First understand the ordinary Python function; compilation should accelerate an understood algorithm. The undecorated behavior is available as `train_pla.py_func`, which was compared with the compiled function during validation.

## 6. Build experiments around the training function

`run_experiments` is responsible for the scientific experiment rather than the individual updates:

1. Load and transform the data for the selected configuration.
2. Allocate arrays for 1000 independent outcomes.
3. For each seed, call `train_pla` with fresh zero weights.
4. Save final weights, updates, and checks.
5. Recompute predictions on all 256 rows and record the number of errors.
6. Compute weight norms and summary statistics.
7. Save raw outcomes and summaries.

Using the same seed set across configurations makes comparisons paired. In particular, configurations with the same update decisions see the same sampled indices until stopping. Separate random seeds need not mean separate machines or separate files: seeds 0-999 give the requested distinct initialization states for the pseudorandom generator.

Q11-Q13 summarize the same baseline experiment, so `run_all.py` runs it once and extracts three statistics. Each individual entry point deliberately reruns the experiment, rather than depending on an existing result file.

## 7. Q11: evaluate the final classifier, not the stopping streak

For each final weight vector, compute

$$E_{in}(w)=\frac{1}{256}\sum_{n=1}^{256}\mathbf{1}[\operatorname{sign}(w^T x_n)\ne y_n].$$

The requested answer is the percentage of experiments with **exactly zero error**:

```python
zero_error_percent = 100 * np.mean(errors == 0)
```

This is different from averaging training accuracy. For example, classifiers with one mistake each are highly accurate but contribute zero to the zero-error percentage.

The measured value was 0%: none of the 1000 baseline runs ended with a perfectly correct classifier. Select **Choice 1: 0%**. The baseline final error counts ranged from 1 to 6, with median 2.

Why can this happen after 512 correct checks? The checks are random and can repeatedly miss the remaining mistakes. If a fixed classifier misclassifies k of N rows, the probability its next 2N independent samples are all correct is

$$(1-k/N)^{2N}.$$

For one wrong row among 256, this is about 13.5%. Thus a long correct streak is evidence, not a complete verification. The formula describes a fixed classifier's next block of checks; the entire PLA stopping process includes many changing classifiers, so it is not directly the probability distribution of the final result.

The observed 0% does not prove the dataset is inseparable. This randomized stopping rule can terminate before all mistakes have been eliminated.

## 8. Q12: summarize how many corrections were made

Store the update count for every baseline run and use

```python
median_updates = np.median(updates)
```

For 1000 outcomes, sort them conceptually and average the 500th and 501st values. NumPy handles this even-sized case. Do not substitute the mean or the number of checked examples.

The measured median was 1386. Its distance from 1350 is 36, smaller than its distance from the other listed values. Select **Choice 4: 1350**.

This is the work performed until the assignment's randomized stopping event. It is not a measurement of the number of updates needed to certify perfect separation.

## 9. Q13: measure the final weights, including the bias

For each run, calculate the Euclidean norm:

$$\|w\|_2 = \sqrt{\sum_{j=0}^{12}w_j^2}.$$

`np.linalg.norm(weights, axis=1)` computes one norm for each row of the saved weight matrix. Then take the median of those 1000 norms. Do not compute a single norm of the entire matrix, omit the bias coordinate, or take the norm of the median weight vector.

The median was 75.34345049771655, so select **Choice 2: 75**.

A large norm alone does not mean a better classifier: multiplying all weights by a positive constant preserves predicted labels. The norm here describes the specific weights produced by the specified update procedure.

## 10. Q14: reason about global scaling before running it

Let the scaled augmented inputs be x' = c*x, where c = 0.5. Starting from zero, suppose the paired runs have w' = c*w. Their scores satisfy

$$w'^T x' = c^2 w^T x.$$

Since c squared is positive, both scores have the same sign, including the zero tie rule. With the same sampled example, both runs therefore make the same update decision. When they update,

$$w' + yx' = c(w+yx),$$

so the relation is preserved. Starting from the same seeds, this induction predicts equal update counts and stopping times in exact arithmetic.

The measured median was again 1386; select **Choice 4: 1350**. We checked all 1000 pairs: updates, checks, and error counts matched exactly, and scaled final weights were exactly half the baseline weights. Scaling by a power of two also avoids introducing a new decimal approximation for these represented inputs under ordinary floating-point conditions.

This reasoning is a useful debugging tool: a disagreement would suggest that the bias was not scaled, the seeds differed, or some other rule changed.

## 11. Q15: changing only the bias changes the update geometry

Here x0 becomes b = 0.5 while the original 12 features remain unchanged. The score is

$$b w_0 + \sum_{j=1}^{12}w_jx_j.$$

On a mistake, w0 changes by y*b. The effective intercept a = b*w0 therefore changes by

$$\Delta a = y b^2.$$

For b = 1, this is y. For b = 0.5, it is y/4. The effective intercept adapts more slowly relative to the unchanged feature weights. There is no single global scaling factor relating all input coordinates, so Q14's invariance proof no longer applies.

The measured update median was 2194; select **Choice 1: 2200**. This dataset needed more updates under the changed representation before the randomized stopping event. The intercept analysis explains a mechanism for the difference; it does not prove that halving the bias always increases update counts on every dataset or seed.

## 12. Q16: removing the bias restricts the classifier

With x0 = 0, w0 starts at zero and never changes. The score reduces to

$$\sum_{j=1}^{12}w_j x_j,$$

and the decision boundary must pass through the origin. Keeping an all-zero first column is mathematically equivalent to removing that coordinate and training a 12-dimensional weight vector. The shared code retains the column so all configurations have a consistent shape.

The measured update median was 12437; select **Choice 3: 12500**. This is about nine times the baseline median. Removing the intercept changes the available separating boundaries and the geometry seen by PLA; the empirical cost here is much larger.

Do not conclude that a perfect origin-passing separator exists merely because the program returned. The randomized stop does not certify separability. The final error counts in this configuration ranged from 3 to 7, with median 3.

## 13. Validate behavior before trusting the answer table

The following checks were performed in `ML2026`:

- Two identical positive examples required zero updates and exactly 2N checks from zero weights. This checks sign(0) and the stopping threshold.
- Two identical negative examples required one initial update followed by 2N correct checks. This checks mistake-only updates and streak resetting.
- A deliberately insufficient safety limit raised an error.
- For seed 0 in each configuration, the compiled trainer matched its ordinary Python version in final weights and both counters.
- All 4000 saved classifiers had their full-dataset error counts independently recomputed using matrix multiplication.
- All 1000 baseline/half-scale pairs satisfied the predicted scaling relations.
- Each individual question result matched the corresponding `run_all.py` answer.
- All saved numerical arrays matched the backed-up results from the previous environment exactly.

These checks target different failure modes. Merely obtaining a plausible multiple-choice answer would not establish correctness.

The details are recorded in `results/ML2026_validation.json`; logs from all seven entry points are in `results/*_ML2026.log`. CSV files expose the per-run observations, and NPZ files preserve the full weight arrays. The dataset SHA-256 is included in every configuration summary JSON.

## 14. A workflow you can follow without AI

1. Write the rule checklist and configuration table yourself before opening an editor.
2. Load the data and print shapes and label values. Confirm which column is the label.
3. Write the prediction rule and manually trace one positive and one negative example from zero weights.
4. Implement one ordinary Python PLA run. Keep checks, updates, and the consecutive-correct streak separate.
5. Verify small synthetic examples with known behavior, including zero scores and mistakes that reset the streak.
6. Run one seed on the real data. Recalculate full-data errors and inspect the output.
7. Add the 1000-seed loop and store observations before summarizing them.
8. Calculate each question's statistic directly from those arrays.
9. Parameterize the input transformations. Predict Q14's behavior algebraically before measuring it.
10. Compile the slow loop only after understanding its correctness, then compare against ordinary Python.
11. Inspect discrepancies, record environment versions and seeds, and explain what the measurements do and do not establish.

To inspect the saved baseline results yourself:

```python
import numpy as np

result = np.load("results/baseline_1000_0.npz")
print(result["updates"][:10])
print(result["errors"][:10])
print(np.median(result["updates"]))
print(100 * np.mean(result["errors"] == 0))
```

For practice, use `python run_all.py --runs 10`, inspect the smaller output, and then return to the required 1000 runs. Changing the seed set is a useful way to explore sampling variability, but do not replace the documented final experiment without recording that change.

## 15. Reproduce this report's numerical results

From a WSL shell:

```bash
conda activate ML2026
cd "/home/daniel/Study/ML 2026/hw 1/code"
python run_all.py --runs 1000 --start-seed 0
```

The package versions in `requirements.txt` were updated only after the complete new-environment execution and validation succeeded. `environment.yml` also records Python 3.15.0 because pip requirements alone cannot select the Python interpreter. See `README.md` for environment creation and the output file layout.
