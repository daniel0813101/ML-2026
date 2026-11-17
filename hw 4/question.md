# Homework 4 — Regularization and validation

## Shared setup

Consider L2-regularized logistic regression with a second-order polynomial transformation.

$$
\mathbf w_\lambda = \arg\min_{\mathbf w}
\left[
\frac{\lambda}{N}\|\mathbf w\|^2 +
\frac{1}{N}\sum_{n=1}^N \ln\left(1 + \exp(-y_n\mathbf w^T\Phi_2(\mathbf x_n))\right)
\right].
$$

Use the transformation from Lecture 12, page 2, with $Q=2$. The features are the constant $1$, all original coordinates, and every product $x_i x_j$ for $1 \le i \le j \le d$, in order of $i$ and then $j$. For $d=6$, this gives 28 features including the constant.

- Training set $\mathcal D$ — [hw4_train.dat](https://www.csie.ntu.edu.tw/~htlin/course/ml26fall/hw4/hw4_train.dat)
- Test set for estimating $E_{\mathrm{out}}$ — [hw4_test.dat](https://www.csie.ntu.edu.tw/~htlin/course/ml26fall/hw4/hw4_test.dat)

Call the training algorithm $\mathcal A_\lambda$. Use [LIBLINEAR](https://www.csie.ntu.edu.tw/~cjlin/liblinear/) with solver option `-s 0`, tolerance `-e 0.000001`, and `-c C` with $C$ derived from the LIBLINEAR README and the objective above. Keep the other training options at their defaults. Treat this as binary classification and evaluate every error below with 0/1 loss.

For every selection, consider $\log_{10}\lambda \in \{-4,-2,0,2,4\}$ and break ties by selecting the largest $\lambda$. For each question, choose the closest option and upload the source code.

## Q12 — Select using test error

Train each candidate on all of $\mathcal D$. Select $\lambda^*$ by minimizing $E_{\mathrm{out}}(\mathbf w_\lambda)$ on the test set. What is $\log_{10}\lambda^*$?

**Options** — −4, −2, 0, 2, 4

## Q13 — Select using training error

Train each candidate on all of $\mathcal D$. Select $\lambda^*$ by minimizing $E_{\mathrm{in}}(\mathbf w_\lambda)$ on that same set. What is $\log_{10}\lambda^*$?

**Options** — −4, −2, 0, 2, 4

## Q14 — Select using a validation set

The training examples are already randomly permuted. Keep their order and use the first 120 rows as $\mathcal D_{\mathrm{train}}$ and the last 80 as $\mathcal D_{\mathrm{val}}$. Train each $\mathbf w_\lambda^-$ on $\mathcal D_{\mathrm{train}}$ only. Select $\lambda^*$ by minimizing $E_{\mathrm{val}}(\mathbf w_\lambda^-)$ on the validation set. What is the test estimate of $E_{\mathrm{out}}(\mathbf w_{\lambda^*}^-)$ for the selected model trained on 120 rows?

**Options** — 0.10, 0.11, 0.12, 0.13, 0.14

## Q15 — Refit after validation

Use the $\lambda^*$ selected in Q14 and run $\mathcal A_{\lambda^*}$ on the full training set $\mathcal D$. What is the test estimate of $E_{\mathrm{out}}(\mathbf w_{\lambda^*})$ for the refitted model?

**Options** — 0.10, 0.11, 0.12, 0.13, 0.14

## Q16 — Select using five-fold cross-validation

Use five consecutive folds of 40 examples each, preserving the input order. Select $\lambda^*$ by minimizing $E_{\mathrm{cv}}(\mathcal A_\lambda)$. What is the selected $E_{\mathrm{cv}}(\mathcal A_{\lambda^*})$?

**Options** — 0.10, 0.11, 0.12, 0.13, 0.14
