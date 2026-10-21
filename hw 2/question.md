# Q11

**總分：30 分**

In page 22 of the Lecture 5 slides (the Fun Time that you should play by yourself), and also in class this year, we taught about the learning model of "positive and negative rays" (which is simply one-dimensional perceptron). The model contains hypotheses of the form:

$$
h_{s,\theta}(x) = s \cdot \operatorname{sign}(x - \theta),
$$

where $s \in \{-1, +1\}$ is the "direction" of the ray and $\theta \in \mathbb{R}$ is the threshold. You can take $\operatorname{sign}(0) = -1$ for simplicity. The model is frequently named the "decision stump" model and is one of the simplest learning models. As shown in class, the growth function of the model is $2N$ and the VC Dimension is $2$.

In fact, the decision stump model is one of the few models that we could minimize $E_{\mathrm{in}}$ efficiently by enumerating all possible thresholds. In particular, for $N$ examples, there are at most $2N$ dichotomies (see page 22 of the Lecture 5 slides), and thus at most $2N$ different $E_{\mathrm{in}}$ values. We can then easily choose the hypothesis that leads to the lowest $E_{\mathrm{in}}$ by the following decision stump learning algorithm.

1. Sort all $N$ examples $x_n$ to a sorted sequence $x'_1, x'_2, \ldots, x'_N$ such that $x'_1 \le x'_2 \le x'_3 \le \cdots \le x'_N$.
2. For each

   $$
   \theta \in \{-1\} \cup \left\{\frac{x'_i + x'_{i+1}}{2} : 1 \le i \le N-1 \text{ and } x'_i \ne x'_{i+1}\right\}
   $$

   and $s \in \{-1, +1\}$, calculate $E_{\mathrm{in}}(h_{s,\theta})$.
3. Return the $h_{s,\theta}$ with the minimum $E_{\mathrm{in}}$ as $g$; if multiple hypotheses reach the minimum $E_{\mathrm{in}}$, return the one with the smallest $s + \theta$.

(Hint: CS-majored students are encouraged to think about whether the second step can be carried out efficiently, i.e. $O(N)$, using dxxxxxc pxxxxxxxxxg instead of the naive implementation of $O(N^2)$.)

Next, you are asked to implement such an algorithm and run your program on an artificial data set. We shall start by generating $(x,y)$ with the following procedure. We will take the target function $f(x) = \operatorname{sign}(x)$:

1. Generate $x$ by a uniform distribution in $[-1, +1]$.
2. Generate $y$ from $x$ by $y = f(x)$ and then flip $y$ to $-y$ with $\tau$ probability independently.

## Q11.1

**分數：10 分**

For $\theta \in [-1, +1]$, what is $E_{\mathrm{out}}(h_{+1,\theta}, 0)$, where $E_{\mathrm{out}}(h, \tau)$ is defined in Problem 7? Choose the correct answer; prove your answer.

### 選項

1. |θ|
2. (1/2)|θ|
3. 2|θ|
4. 1 − |θ|
5. 1 − (1/2)|θ|

# Q12

**總分：30 分**

## Q12.1

**分數：10 分**

(*) For $\tau = 0$, which means that your data is noiseless. Generate a data set of size $3$ by the procedure above and run the decision stump algorithm on the data set to get $g$. Repeat the experiment $10000$ times, each with a different data set. What is the median of $E_{\mathrm{out}}(g, \tau) - E_{\mathrm{in}}(g)$ within the $10000$ results? Choose the closest value; upload your source code.

(By extending the results in Problem 7 and Problem 11, you can actually compute any $E_{\mathrm{out}}(h_{s,\theta}, \tau)$ analytically. But if you do not trust your math derivation, you can get a very accurate estimate of $E_{\mathrm{out}}(g)$ by evaluating $g$ on a separate test data set of size $100000$, as guaranteed by Hoeffding's inequality.)

### 選項

1. 0.10
2. 0.15
3. 0.20
4. 0.25
5. 0.30

# Q13

**總分：30 分**

## Q13.1

**分數：10 分**

(*) For $\tau = 0$, generate a data set of size $30$ by the procedure above and run the decision stump algorithm on the data set to get $g$. Repeat the experiment $10000$ times, each with a different data set. What is the median of $E_{\mathrm{out}}(g, \tau) - E_{\mathrm{in}}(g)$ within the $10000$ results? Choose the closest value; upload your source code.

### 選項

1. 0.00
2. 0.01
3. 0.02
4. 0.03
5. 0.04

# Q14

**總分：30 分**

## Q14.1

**分數：10 分**

(*) For $\tau = 0.1$, generate a data set of size $3$ by the procedure above and run the decision stump algorithm on the data set to get $g$. Repeat the experiment $10000$ times, each with a different data set. What is the median of $E_{\mathrm{out}}(g, \tau) - E_{\mathrm{in}}(g)$ within the $10000$ results? Choose the closest value; upload your source code.

### 選項

1. 0.10
2. 0.15
3. 0.20
4. 0.25
5. 0.30

# Q15

**總分：30 分**

## Q15.1

**分數：10 分**

(*) For $\tau = 0.1$, generate a data set of size $30$ by the procedure above and run the decision stump algorithm on the data set to get $g$. Repeat the experiment $10000$ times, each with a different data set. What is the median of $E_{\mathrm{out}}(g, \tau) - E_{\mathrm{in}}(g)$ within the $10000$ results? Choose the closest value; upload your source code.

### 選項

1. 0.00
2. 0.01
3. 0.02
4. 0.03
5. 0.04

# Q16

**總分：30 分**

## Q16.1

**分數：10 分**

(*) For $\tau = 0.1$, generate a data set of size $300$ by the procedure above and run the decision stump algorithm on the data set to get $g$. Repeat the experiment $10000$ times, each with a different data set. What is the median of $E_{\mathrm{out}}(g, \tau) - E_{\mathrm{in}}(g)$ within the $10000$ results? Choose the closest value; upload your source code.

### 選項

1. 0.000
2. 0.002
3. 0.004
4. 0.006
5. 0.008

