# Homework 1

## Q11（30 分）

### Q11.1（10 分）

Next, we use an artificial data set to study PLA. The data set with $N = 256$ examples is in:

http://www.csie.ntu.edu.tw/~htlin/course/ml26fall/hw1/hw1_train.dat

Each line of the data set contains one $(\mathbf{x}_n, y_n)$ with $\mathbf{x}_n \in \mathbb{R}^{12}$. The first 12 numbers of the line contains the components of $\mathbf{x}_n$ orderly, the last number is $y_n$. Please initialize your algorithm with $\mathbf{w} = \mathbf{0}$ and take $\operatorname{sign}(0)$ as $+1$.

(\*) Please first follow Page 4 of Lecture 2, and add $x_0 = 1$ to every $\mathbf{x}_n$. Implement a version of PLA that randomly picks an example $(\mathbf{x}_n, y_n)$ in every iteration, and updates $\mathbf{w}_t$ if and only if $\mathbf{w}_t$ is incorrect on the example. Note that the random picking can be simply implemented with replacement — that is, the same example can be picked multiple times, even consecutively. Stop updating and return $\mathbf{w}_t$ as $\mathbf{w}_{\mathrm{PLA}}$ if $\mathbf{w}_t$ is correct consecutively after checking $2N$ randomly-picked examples.

**Hint:**

1. The update procedure described above is equivalent to the procedure of gathering all the incorrect examples first and then randomly picking an example among the incorrect ones. But the description above is usually much easier to implement.
2. The stopping criterion above is a randomized, more efficient implementation of checking whether $\mathbf{w}_t$ makes no mistakes on the data set.

Repeat your experiment for 1000 times, each with a different random seed. Recalculate $E_{\mathrm{in}}(\mathbf{w}_{\mathrm{PLA}})$ after you get $\mathbf{w}_{\mathrm{PLA}}$ in each experiment. What is the percentage of $E_{\mathrm{in}}(\mathbf{w}_{\mathrm{PLA}}) = 0$? Choose the closest value; **upload your source code**.

#### 選項

1. 0%
2. 2%
3. 4%
4. 6%
5. 8%
## Q12（30 分）

### Q12.1（10 分）

(\*) Following the previous problem, when running PLA, record the number of updates (i.e. step 2 of the PLA algorithm, only when encountering an incorrect example). Repeat your experiment for 1000 times, each with a different random seed. What is the median number of updates? Choose the closest value; **upload your source code**.

#### 選項

1. 600
2. 850
3. 1100
4. 1350
5. 1600

## Q13（30 分）

### Q13.1（10 分）

(\*) Among the $\mathbf{w}_{\mathrm{PLA}}$ obtained from the 1000 experiments above, what is the median of $\|\mathbf{w}_{\mathrm{PLA}}\|$? Choose the closest value; **upload your source code**.

#### 選項

1. 65
2. 75
3. 85
4. 95
5. 105

## Q14（30 分）

### Q14.1（10 分）

(\*) Scale down each $\mathbf{x}_n$ by 2, including scaling each $x_0$ from 1 to 0.5. Then, run PLA on the scaled examples for 1000 experiments, each with a different random seed. What is the median number of updates? Choose the closest value; **upload your source code**.

#### 選項

1. 600
2. 850
3. 1100
4. 1350
5. 1600

## Q15（30 分）

### Q15.1（10 分）

(\*) Set $x_0 = 0.5$ to every $\mathbf{x}_n$ instead of $x_0 = 1$, and do not do any scaling. Then, run PLA on the scaled examples for 1000 experiments, each with a different random seed. What is the median number of updates? Choose the closest value; **upload your source code**.

#### 選項

1. 2200
2. 2400
3. 2600
4. 2800
5. 3000

## Q16（30 分）

### Q16.1（10 分）

(\*) Set $x_0 = 0$ to every $\mathbf{x}_n$ instead of $x_0 = 1$, and do not do any scaling. This equivalently means not adding any $x_0$, and you will get a separating hyperplane that passes the origin. Then, run PLA on the scaled examples for 1000 experiments, each with a different random seed. What is the median number of updates? Choose the closest value; **upload your source code**.

#### 選項

1. 7500
2. 10000
3. 12500
4. 15000
5. 17500
