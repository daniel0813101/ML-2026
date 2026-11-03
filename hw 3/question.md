# Q11

**總分：30 分**

## Experiments with Linear Models

Next, we will play with linear regression, logistic regression, and their use for binary classification. Please use the following set for training:

https://www.csie.ntu.edu.tw/~htlin/course/ml26fall/hw3/hw3_train.dat

and the following set for testing (estimating $E_{\mathrm{out}}$):

https://www.csie.ntu.edu.tw/~htlin/course/ml26fall/hw3/hw3_test.dat

The files are of the same format as the one you received in Homework 1, except that the $\mathbf{x}$'s are now 10-dimensional instead of 12.

## Q11.1

**分數：10 分**

(*) Add $x_{n,0} = 1$ to each $\mathbf{x}_n$. Then, implement the linear regression algorithm on page 11 of Lecture 9. What is $E_{\mathrm{in}}^{\mathrm{sqr}}(\mathbf{w}_{\mathrm{lin}})$, where $E_{\mathrm{in}}^{\mathrm{sqr}}$ denotes the averaged squared error over $N$ examples? Choose the closest value; upload your source code.

### 選項

1. 0.76
2. 0.78
3. 0.80
4. 0.82
5. 0.84

# Q12

**總分：30 分**

## Q12.1

**分數：10 分**

(*) Add $x_{n,0} = 1$ to each $\mathbf{x}_n$. Then, implement the SGD algorithm for linear regression using a similar derivation like page 12 of Lecture 11. Pick one example uniformly at random in each iteration, take $\eta = 0.001$ and initialize $\mathbf{w}$ with $\mathbf{w}_0 = \mathbf{0}$. Run the algorithm for 800 iterations. Repeat the experiment 1000 times, each with a different random seed. What is the average $E_{\mathrm{in}}^{\mathrm{sqr}}(\mathbf{w}_{800})$ over the 1000 experiments, where $E_{\mathrm{in}}^{\mathrm{sqr}}$ denotes the averaged squared error over $N$ examples? Choose the closest value; upload your source code.

### 選項

1. 0.76
2. 0.78
3. 0.80
4. 0.82
5. 0.84

# Q13

**總分：30 分**

## Q13.1

**分數：10 分**

(*) Add $x_{n,0} = 1$ to each $\mathbf{x}_n$. Then, implement the SGD algorithm for logistic regression by replacing the SGD update step in the previous problem with the one for logistic regression. Pick one example uniformly at random in each iteration, take $\eta = 0.001$ and initialize $\mathbf{w}$ with $\mathbf{w}_0 = \mathbf{0}$. Run the algorithm for 800 iterations. Repeat the experiment 1000 times, each with a different random seed. What is the average $E_{\mathrm{in}}^{\mathrm{ce}}(\mathbf{w}_{800})$ over the 1000 experiments, where $E_{\mathrm{in}}^{\mathrm{ce}}$ denotes the averaged cross-entropy error over $N$ examples? Choose the closest value; upload your source code.

### 選項

1. 0.60
2. 0.63
3. 0.66
4. 0.69
5. 0.72

# Q14

**總分：30 分**

## Q14.1

**分數：10 分**

(*) Repeat the previous problem, but with $\mathbf{w}$ initialized by $\mathbf{w}_0 = \mathbf{w}_{\mathrm{lin}}$ of Problem 11 instead. Repeat the experiment 1000 times, each with a different random seed. What is the average $E_{\mathrm{in}}^{\mathrm{ce}}(\mathbf{w}_{800})$ over the 1000 experiments? Choose the closest value; upload your source code.

### 選項

1. 0.60
2. 0.63
3. 0.66
4. 0.69
5. 0.72

# Q15

**總分：30 分**

## Q15.1

**分數：10 分**

(*) Following the previous problem, what is the average $\left|E_{\mathrm{in}}^{0/1}(\mathbf{w}_{800}) - E_{\mathrm{out}}^{0/1}(\mathbf{w}_{800})\right|$ over 1000 experiments, where $E_{\mathrm{in}}^{0/1}$ denotes the averaged 0/1 error (i.e. using $\mathbf{w}_{800}$ for binary classification), and $E_{\mathrm{out}}^{(0/1)}$ is estimated using the test set provided above? Choose the closest value; upload your source code.

### 選項

1. 0.03
2. 0.04
3. 0.05
4. 0.06
5. 0.07

# Q16

**總分：30 分**

## Q16.1

**分數：10 分**

(*) Following Problem 11, what is $\left|E_{\mathrm{in}}^{0/1}(\mathbf{w}_{\mathrm{lin}}) - E_{\mathrm{out}}^{0/1}(\mathbf{w}_{\mathrm{lin}})\right|$, where $E_{\mathrm{in}}^{0/1}$ denotes the averaged 0/1 error (i.e. using $\mathbf{w}_{\mathrm{lin}}$ for binary classification), and $E_{\mathrm{out}}^{(0/1)}$ is estimated using the test set provided above? Choose the closest value; upload your source code.

### 選項

1. 0.03
2. 0.04
3. 0.05
4. 0.06
5. 0.07
