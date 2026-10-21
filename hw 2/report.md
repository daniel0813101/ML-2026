# Homework 2: Learning to Build and Analyze a Decision Stump Experiment

This report explains how to solve Q11–Q16 from the problem statement: identify the statistical task, derive the necessary formulas, implement a correct learner, verify it, and interpret the experiment. Read the reasoning before looking at the implementation details. The aim is to make the solution reproducible and understandable enough to rebuild independently.

## 1. Verified answers and execution record

All experiments below were rerun in WSL Ubuntu on October 20, 2026. Each question uses 10,000 independently generated training datasets.

| Question | Training size N | Noise tau | Actual seed | Median of E_out(g, tau) - E_in(g) | Closest value | Select |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Q12.1 | 3 | 0 | 2038 | 0.1047779532233962 | 0.10 | Choice 1 |
| Q13.1 | 30 | 0 | 2039 | 0.011443475001547715 | 0.01 | Choice 2 |
| Q14.1 | 3 | 0.1 | 2040 | 0.20914772539144427 | 0.20 | Choice 3 |
| Q15.1 | 30 | 0.1 | 2041 | 0.037459642733907624 | 0.04 | Choice 5 |
| Q16.1 | 300 | 0.1 | 2042 | 0.0038537073288949994 | 0.004 | Choice 3 |

Q11.1 is theoretical: select **Choice 2, |theta| / 2**. Its proof appears in Section 5.

Verified environment:

- Working directory: `/home/daniel/Study/ML-2026/hw 2/code`.
- Python executable: `/home/daniel/miniconda3/bin/python`.
- Python version: 3.13.5.
- NumPy version: 2.2.6.
- Random generator: explicitly constructed NumPy PCG64.
- Base seed: 2026; each question uses base seed + question number.
- Four unit tests passed.
- `run_all.py` and all five individual question entry points were executed. Every individual result exactly matched its entry in the combined output.

The default WSL shell did not expose a command named `python`, so the commands below use the absolute interpreter path. No package installation was needed.

Evidence files, relative to this report:

- [Combined experiment results](code/results.json)
- [Environment and successful command records](code/results/wsl_run.json)
- [Captured execution output, including tests](code/results/execution_log.txt)
- Individual results: [Q12](code/results/q12.json), [Q13](code/results/q13.json), [Q14](code/results/q14.json), [Q15](code/results/q15.json), [Q16](code/results/q16.json).

The earlier Windows run used Python 3.12.7 and NumPy 1.26.4. This WSL run produced the same five numerical results. That is an observed agreement for this implementation and these seeds, not a promise that arbitrary versions or different random sampling methods always agree.

## 2. First turn the problem statement into a specification

Before coding, write down the inputs, output, and rules.

A training dataset contains N pairs (x, y). Draw x uniformly from [-1, 1]. The clean target is sign(x), and each label is independently flipped with probability tau.

The learner may choose only hypotheses of the form:

$$
h_{s,\theta}(x) = s\,\operatorname{sign}(x-\theta),
\qquad s\in\{-1,+1\}.
$$

Use sign(0) = -1. Therefore:

| Direction | x <= theta | x > theta |
| --- | --- | --- |
| s = +1 | Predict -1 | Predict +1 |
| s = -1 | Predict +1 | Predict -1 |

The learner chooses the hypothesis with the smallest training error. If several hypotheses tie, choose the one with the smallest s + theta. The threshold candidates are exactly those specified in Q11: -1 and the midpoints between adjacent distinct sorted inputs.

One repetition must perform these operations:

1. Generate a new training dataset.
2. Train a new stump g on that dataset.
3. Compute E_in(g) on that same training dataset.
4. Compute population error E_out(g, tau).
5. Store their difference.

Repeat this entire process 10,000 times, then take the median of the stored differences.

Do not reuse one training dataset 10,000 times. Do not take the median of E_out alone. Also, in general:

$$
\operatorname{median}(E_{\mathrm{out}}-E_{\mathrm{in}})
\ne
\operatorname{median}(E_{\mathrm{out}})
-
\operatorname{median}(E_{\mathrm{in}}).
$$

This specification reveals that Q12–Q16 use one algorithm with different parameters. It motivates one shared module rather than five copied implementations.

## 3. Implement a transparent learner before optimizing it

### 3.1 Why sorting reduces the search to finitely many thresholds

Imagine sliding theta along the number line. A training prediction changes only when the threshold crosses a training input. Moving theta inside a gap between consecutive inputs leaves every training prediction unchanged.

Consequently, one representative threshold per relevant gap is sufficient. The assignment selects the midpoint of each gap, plus -1.

For sorted inputs [-0.8, -0.2, 0.1, 0.7], the candidate thresholds are:

```text
-1.0, -0.5, -0.05, 0.4
```

Try both directions at each candidate. This gives eight candidates in this example and at most 2N candidates in general.

There is no need to search a dense numerical grid. A grid can miss the exact required candidates and introduce an arbitrary resolution.

There is also no need to add +1: with inputs strictly inside (-1, 1), the two constant prediction patterns are already represented by theta = -1 and the two directions. More fundamentally, the implementation follows the assignment's specified candidate set. Equality at endpoints is handled according to sign(0) = -1.

Although every clean label is generated using threshold zero, the training algorithm must still choose from its prescribed candidates. Hard-coding theta = 0 would solve a different problem.

### 3.2 A direct version you should be able to write first

This reference implementation prioritizes readability. It is a learning example, not a separate experiment entry point.

```python
import numpy as np


def fit_stump_reference(x, y):
    """Find the best candidate by directly checking every prediction.

    Args:
        x: One-dimensional training coordinates.
        y: Matching labels in {-1, +1}.

    Returns:
        A tuple containing direction, threshold, and training error.
    """
    order = np.argsort(x)
    xs = x[order]
    thresholds = [-1.0]

    for i in range(len(xs) - 1):
        if xs[i] != xs[i + 1]:
            thresholds.append((xs[i] + xs[i + 1]) / 2.0)

    best_key = None
    best_model = None
    for s in (-1, 1):
        for theta in thresholds:
            predictions = s * np.where(x > theta, 1, -1)
            mistakes = int(np.count_nonzero(predictions != y))
            key = (mistakes, s + theta)
            if best_key is None or key < best_key:
                best_key = key
                best_model = (s, theta, mistakes / len(x))

    return best_model
```

The tuple comparison expresses the objective precisely: minimize mistakes first, and s + theta only among ties. Do not minimize a weighted sum such as mistakes + s + theta; that could sacrifice training accuracy.

Use integer mistake counts while comparing candidates. Every candidate has the same denominator N, so minimizing counts is equivalent to minimizing E_in and avoids unnecessary floating-point comparisons.

This version costs O(N^2) after sorting because there are O(N) candidates and each candidate examines N examples. It is an excellent correctness reference even if the final implementation is faster.

## 4. Derive the O(N) candidate evaluation

### 4.1 Count errors by side

For s = +1, points on the left receive prediction -1 and points on the right receive prediction +1. Therefore the mistakes are:

```text
positive labels on the left + negative labels on the right
```

Let k be the number of inputs on the left, P[k] the number of positive labels among the first k sorted examples, and P[N] the total positive count.

There are k - P[k] negatives on the left and N - P[N] negatives overall. Thus:

$$
m_+(k)=P[k]+\bigl(N-P[N]-(k-P[k])\bigr).
$$

For the opposite direction every binary prediction is reversed, so:

$$
m_-(k)=N-m_+(k).
$$

A prefix-count array is built once in O(N). Each candidate then needs only constant-time count lookups. The full training cost becomes O(N log N) for sorting plus O(N) for candidate evaluation.

This is the reusable algorithmic idea: when adjacent candidates differ by moving a boundary, maintain information about the two sides instead of recomputing every prediction.

### 4.2 Work through a small dataset by hand

Use these sorted data:

```text
x = [-0.8, -0.2,  0.1, 0.7]
y = [  -1,   +1,   -1,  +1]
```

The positive prefix counts, including the empty prefix, are:

```text
P = [0, 0, 1, 1, 2]
```

| theta | Left count k | Positive left | Negative right | Mistakes for +1 | Mistakes for -1 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| -1.00 | 0 | 0 | 2 | 2 | 2 |
| -0.50 | 1 | 0 | 1 | 1 | 3 |
| -0.05 | 2 | 1 | 1 | 2 | 2 |
| 0.40 | 3 | 1 | 0 | 1 | 3 |

Two candidates have one mistake: (+1, -0.5) and (+1, 0.4). Their s + theta values are 0.5 and 1.4, so the required answer is (+1, -0.5), with E_in = 1/4.

Doing this example yourself makes the optimized array operations much easier to audit.

### 4.3 Map the derivation to the existing code

In `fit_stump`:

- `order = np.argsort(x, kind="stable")` gives the permutation needed to sort both x and y together. Sorting x without applying the same permutation to y destroys the dataset.
- `splits = np.flatnonzero(xs[:-1] < xs[1:]) + 1` locates boundaries between different coordinates. Duplicate inputs must not introduce an impossible split.
- `theta` contains -1 followed by the midpoint thresholds.
- `left_count` stores k for each candidate.
- `positive_prefix = np.r_[0, np.cumsum(ys == 1)]` builds P, with P[0] = 0.
- `positive_left` and `negative_right` implement the count formula.
- `errors` concatenates the mistake counts for direction -1 and direction +1.
- `directions` and `thresholds` use matching layouts, so every position describes one candidate.
- `tied` identifies candidates with minimum mistakes; `argmin` then applies the s + theta rule only within that set.

Here `np.r_` concatenates arrays, `np.repeat` repeats each direction across its candidate thresholds, and `np.tile` repeats the threshold list for the two directions.

The special `searchsorted(..., side="right")` call counts any x exactly equal to -1 on the left of theta = -1. It preserves the sign(0) convention even in manually constructed boundary tests.

## 5. Derive E_out before deciding how to estimate it

### 5.1 Prove Q11.1

First assume no noise and direction +1. The target boundary is zero and the learned boundary is theta.

If theta > 0, the hypothesis and target disagree only between 0 and theta. If theta < 0, they disagree only between theta and 0. In both cases, the disagreement interval has length |theta|.

The input distribution is uniform over an interval of total length 2. Therefore:

$$
E_{\mathrm{out}}(h_{+1,\theta},0)=\frac{|\theta|}{2}.
$$

At theta = 0 the error is zero. At theta = -1 or +1 it is 1/2. Endpoint conventions do not affect a continuous population probability because a single input value has probability zero.

This proves **Q11.1, Choice 2**.

For direction -1, all predictions are reversed. Exactly one direction is correct at any given input, so its clean error is:

$$
E_{\mathrm{out}}(h_{-1,\theta},0)=1-\frac{|\theta|}{2}.
$$

### 5.2 Add independent label noise

Let e be a hypothesis's error against the clean target.

There are two ways to be wrong against the observed noisy label:

| Clean prediction | Label event | Probability of noisy mistake |
| --- | --- | --- |
| Wrong, probability e | Label is not flipped | e(1 - tau) |
| Correct, probability 1 - e | Label is flipped | (1 - e)tau |

Adding them gives:

$$
E_{\mathrm{out}}(h,\tau)
=e(1-\tau)+(1-e)\tau
=\tau+(1-2\tau)e.
$$

The code in `out_error` first computes e using the direction and threshold, then applies this formula.

For tau = 0.1:

$$
E_{\mathrm{out}}=0.1+0.8e.
$$

For the hand-worked model (+1, -0.5), e = 0.25, so E_out = 0.30. Its training error was 0.25, making this illustrative dataset's gap 0.05.

### 5.3 Why use the formula rather than a large test set?

Once g is fixed, the formula gives its exact population error under the stated distribution. A separate finite test set would add sampling variability and substantial computation.

Evaluating 100,000 test examples for each of 10,000 repetitions would process one billion test examples per question. The formula replaces that test evaluation with a constant amount of arithmetic.

The overall experiment is still random: each training dataset changes g, and the median over 10,000 runs is an empirical median. Exact evaluation of each model does not make the distribution of trained models nonrandom.

## 6. Build the data generator carefully

The generator is short because each line corresponds to one sentence in the problem:

```python
x = rng.uniform(-1.0, 1.0, n)
y = np.where(x > 0.0, 1, -1)
y = np.where(rng.random(n) < tau, -y, y)
```

The first line samples coordinates. The second creates the clean target labels with the required sign(0) convention. The third generates an independent flip event for each example.

`rng.random(n) < tau` creates n independent Bernoulli decisions. It does not force exactly tau * N labels to flip. For N = 3 and tau = 0.1, many datasets have no flips, some have one, and a few have more.

Avoid `np.sign` here: NumPy's sign at zero is 0, whereas this assignment requires -1.

Create the random generator once per question and use it throughout its repetitions. Resetting the same seed inside the repetition loop would generate the same dataset repeatedly.

Even when tau = 0, the current generator draws the flip uniforms. Removing those unused draws preserves the distribution but changes the later random sequence and therefore changes exact reproducibility.

## 7. Assemble the experiment and keep its responsibilities separate

Each question implements its own experiment in `solve()`. It sets N, tau, the answer choices, and its PCG64 seed (base seed + question number), then calls the reusable generator, learner, and population-error function. The core logic is:

```python
gaps = np.empty(trials)
for index in range(trials):
    x, y = generate_data(n, tau, rng)
    stump = fit_stump(x, y)
    gaps[index] = out_error(stump, tau) - stump.ein

median_gap = float(np.median(gaps))
```

For 10,000 values, NumPy sorts conceptually and averages the 5,000th and 5,001st values in one-based indexing. After calculating the median, choose the option with the smallest absolute distance.

For example, Q14 gives approximately 0.209148. Its distances to 0.20 and 0.25 are approximately 0.009148 and 0.040852, so 0.20 is the closest choice.

The files reflect different responsibilities:

| File or object | Responsibility | Why keep it separate? |
| --- | --- | --- |
| `generate_data` | Sample the stated distribution | Check sampling without involving training |
| `Stump` | Store s, theta, and training error; predict labels | Keep one learned model explicit |
| `fit_stump` | Select the hypothesis using training data | Keep optimization independent of population evaluation |
| `out_error` | Evaluate a model analytically | Avoid mixing clean and noisy errors |
| `q12.py`–`q16.py`: `solve` | Own N, tau, choices, seed, trial loop, median, and answer selection | Make each question independently readable and executable |
| `parse_args`, `write_report` | Validate shared CLI options and print/save JSON records | Keep reusable I/O separate from question logic |
| Each script’s `main` | Call its own `solve` and output the result | Provide a standalone executable without a question dispatcher |
| `run_all.py` | Call each question’s `solve` and combine the records | Exercise the same implementations used by standalone scripts |

`@dataclass(frozen=True)` provides named fields and prevents reassignment after construction. It makes it harder to accidentally change a fitted model halfway through an experiment. Question settings now belong to the individual scripts; the library contains no `Question` class, `QUESTIONS` table, or question dispatcher.

Google style docstrings describe what a function does, its inputs, outputs, and invalid-input behavior. Comments should explain a non-obvious reason, such as endpoint handling, rather than merely repeat the next line of code.

## 8. Verify the implementation before trusting its answers

A plausible answer is weak evidence of correctness. Check properties with known outcomes and compare independent implementations.

The executed tests cover:

1. **Direct enumeration versus prefix counts.** The test explicitly predicts every training point for every candidate, then compares the selected s, theta, and E_in with the optimized version. It uses 90 random datasets across N = 3, 30, and 300, plus duplicate and endpoint examples.
2. **Population-error formula.** A dense deterministic midpoint grid approximates the input integral for both directions, several thresholds, and noise values 0, 0.1, 0.5, and 1.
3. **Sign and tie behavior.** Check x = theta, equal-error candidates, and single-example data.
4. **Repeatability.** The same settings and seed must return the same summary.

All four tests passed in WSL. The batch program and each individual entry point were also executed and compared: their corresponding result records were identical.

Useful checks when rebuilding:

- Predictions must always be -1 or +1.
- E_in must be between zero and one.
- With tau = 0, continuous random training data are separable by the specified candidate set, so E_in should be zero.
- At tau = 0.5, E_out must be 0.5 for every stump.
- For tau = 0.1, every stump's E_out must lie in [0.1, 0.9].
- A single dataset's gap can be negative. A nonnegative gap is not a universal invariant.

Do not use the multiple-choice options as targets while debugging. Establish correctness from definitions and independent checks first.

## 9. Interpret each result

### Q12: N = 3, tau = 0

The data are clean and a training-perfect stump is available, so E_in = 0. The reported gap is therefore the learned model's population error.

Three points leave large unobserved intervals. A boundary can fit all training labels and still be far from zero. If all sampled labels have the same sign, even a constant predictor fits them.

The empirical median is about 0.104778, closest to **0.10, Choice 1**.

### Q13: N = 30, tau = 0

Training error remains zero. With more points, the nearest observed inputs on either side of zero tend to constrain the candidate boundary more tightly.

The empirical median falls to about 0.0114435, closest to **0.01, Choice 2**. Perfect training accuracy alone could not distinguish Q12 from Q13; population evaluation exposes their different generalization.

This is a tendency across random datasets, not a guarantee that every 30-point sample produces a better model than every 3-point sample.

### Q14: N = 3, tau = 0.1

There is both uncertainty from the small dataset and independent label noise. The probability of at least one flipped label is:

$$
1-(1-0.1)^3=0.271.
$$

A learner minimizing training error can fit a misleading label pattern, and even a clean-target-perfect predictor has noisy-label population error 0.1.

The median gap is about 0.209148, closest to **0.20, Choice 3**.

Adding 0.1 to Q12's median is not a valid derivation: noise changes both the distribution of trained hypotheses and their training errors, as well as their population evaluation.

### Q15: N = 30, tau = 0.1

More observations help identify the overall target boundary, but minimizing training error still favors hypotheses that benefit from fluctuations in the particular sample.

The median gap is about 0.0374596, closest to **0.04, Choice 5**.

This number is not the test error. It is a difference between test and training errors. It can be below the noise probability 0.1 without contradicting the population error floor.

### Q16: N = 300, tau = 0.1

The larger dataset provides a more reliable picture of the label distribution and generally reduces the training-versus-population discrepancy.

The median gap is about 0.00385371, closest to **0.004, Choice 3**.

This does not mean the classifier is 99.6% accurate. With tau = 0.1, the best possible population error against the noisy label is 0.1. A small gap says training error is close to population error; it does not say either error is close to zero.

Across these runs, larger N is associated with smaller median gaps. These five experiments do not prove a universal convergence rate, and one fixed-seed run does not quantify uncertainty in the empirical median.

## 10. Reproduce the WSL run

The commands below use the environment that actually completed this execution:

```bash
cd "/home/daniel/Study/ML-2026/hw 2/code"

/home/daniel/miniconda3/bin/python -B -m unittest -v test_decision_stump.py
/home/daniel/miniconda3/bin/python -B run_all.py --output results.json
```

To execute every individual question as well:

```bash
for q in 12 13 14 15 16; do
    /home/daniel/miniconda3/bin/python -B "q${q}.py" \
        --output "results/q${q}.json"
done
```

The seed convention ensures that individual and batch execution generate the same datasets for a given question. `-B` disables Python bytecode-cache writes and does not affect the experiment.

To try a different base seed without replacing the main recorded output:

```bash
/home/daniel/miniconda3/bin/python -B run_all.py \
    --seed 2027 --output results/alternative_seed.json
```

That last command is an optional follow-up exercise; its results are not part of the executed answer table.

The existing `requirements.txt` pins NumPy 1.26.4 for the earlier Python 3.12 environment. The WSL execution used the already-installed NumPy 2.2.6 with Python 3.13.5. To reproduce the WSL package version in a separate compatible Python environment, install `numpy==2.2.6`; do not treat the older requirements file as a description of this WSL run.

## 11. A practical workflow for solving this independently

Work through these stages in order, and require a concrete check before moving on.

1. **Extract a specification.** Write the data distribution, hypothesis, candidate set, tie rule, repetition count, and requested statistic on paper. Check that every sentence of the assignment has a place in your plan.
2. **Solve a tiny case manually.** Use the four-point example in Section 4. Verify the prediction table and tie rule without NumPy.
3. **Write the direct learner.** Enumerate thresholds, predict labels, count mistakes, compare tuples. Confirm that it matches your hand calculation.
4. **Write and inspect the generator.** Print one small dataset and check how labels and independent flips are produced.
5. **Derive evaluation separately.** Draw the disagreement interval to obtain |theta| / 2, then use the two-case probability calculation to add noise.
6. **Connect one complete repetition.** Print x, y, s, theta, E_in, E_out, and their difference. Check every quantity before creating a large loop.
7. **Optimize only after correctness.** Derive prefix counts and compare the optimized learner against the direct version on many small cases.
8. **Run 10–20 repetitions.** Check valid ranges and reproducibility. This catches experiment-loop mistakes cheaply.
9. **Run the required 10,000 repetitions.** Save parameters, seeds, versions, and numerical results before choosing an answer.
10. **Interpret the statistic in words.** Explain why clean training error can be zero, why small samples generalize imperfectly, and why a small generalization gap is different from a small test error.

For further practice, reconstruct the direct learner from the specification without looking at the code, then explain the prefix-count formula aloud. If you can justify each counted group and each random draw, you understand the implementation rather than only its output.
