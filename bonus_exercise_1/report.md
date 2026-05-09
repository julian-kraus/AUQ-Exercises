# Report


## Team Members

Julian Kraus - 03734087


## Assignments


### Assignment 3

| N | Monte Carlo | CV $h_1$ | CV $h_2$ | CV $h_3$ | IS Beta(5,1) | IS Beta(0.5,0.5) |
|---|---|---|---|---|---|---|
| 10 | 0.038746 | 0.005176 | 0.005176 | 0.000907 | 0.526304 | 0.020013 |
| 100 | 0.046221 | 0.003523 | 0.003523 | 0.001228 | 0.223869 | 0.083207 |
| 1000 | 0.014795 | 0.001610 | 0.001610 | 0.000572 | 0.318954 | 0.032457 |
| 10000 | 0.010349 | 0.000500 | 0.000500 | 0.000124 | 0.210658 | 0.001706 |

#### Standard Monte Carlo
Errors decrease from $0.0387$ at $N=10$ to $0.0103$ at $N=10{,}000$, roughly consistent with the theoretical $O(1/\sqrt{N})$ rate. The non-monotone decrease (error rises slightly from $N=10$ to $N=100$) reflects the stochastic nature of the method.

#### Control Variates
All three control variates outperform standard MC. Notably, **$h_1$ and $h_2$ produce identical errors** at every $N$. This is expected: since $h_2(x) = 1 + h_1(x)$, adding a constant shifts neither the covariance with $f$ nor the variance of $\phi$, so the optimal coefficient $c^* = \text{Cov}(f,\phi)/\text{Var}(\phi)$ and the correction term $c^*(E[\phi] - \bar{\phi})$ are the same for both.

$h_3$ achieves the best results because $1 + x + x^2/2$ is the 2nd-order Taylor expansion of $e^x$, making it highly correlated with $f$ and maximising variance reduction.

#### Importance Sampling
Both Beta proposals perform poorly compared to the control variate methods, and neither improves consistently over standard MC:

- **Beta(5,1)**: PDF $\propto x^4$ concentrates samples near $x=1$, assigning very small probability to $x \approx 0$. The resulting importance weights $p(x)/q(x) = 1/q(x)$ become enormous near zero, dramatically inflating variance. Errors are an order of magnitude larger than MC and fail to converge consistently.

- **Beta(0.5,0.5)**: The arcsine distribution is U-shaped, concentrating at both endpoints. This is a poor match for the smoothly increasing $e^x$. The convergence is highly erratic: the error is lower than MC at $N=10$ and $N=10{,}000$, but worse at $N=100$ and $N=1{,}000$. This non-monotone behaviour is a consequence of the high variance of the importance weights: with a poorly matched proposal, small sample sizes are subject to large random fluctuations, preventing reliable convergence.
