# Energy Distance Formulae for Probability Distributions

This file collects analytic formulae, derivations, and summary tables for energy distance calculations ($\mathbb{E}|X-Y|$) for probability distributions implemented in skpro.

---

## Summary Table (energy_report.tex)

| Distribution         | Analytic  | Monte Carlo | Abs Error |
|---------------------|-----------|-------------|-----------|
| Beta(2,3)           | 0.228571  | 0.228128    | 0.000444  |
| ChiSquared(2)       | 2.000000  | 2.000000    | 0.000000  |
| Exponential(2)      | 0.500000  | 0.500166    | 0.000166  |
| Gamma(2,3)          | 0.500000  | 0.497054    | 0.002946  |
| Logistic(0,1)       | 2.000000  | 1.997396    | 0.002604  |
| LogNormal(0,1)      | 1.716318  | 1.716318    | 0.000000  |
| Pareto(1,3)         | 0.600000  | 0.599371    | 0.000629  |
| T(0,1,5)            | 1.383983  | 1.383983    | 0.000000  |
| Weibull(1,2)        | 0.519140  | 0.521159    | 0.002019  |

## Summary Table (energy_report2.tex)

| Distribution                  | Analytic  | Monte Carlo | Abs Error |
|-------------------------------|-----------|-------------|-----------|
| InverseGamma(3,2)             | 0.750000  | 0.749457    | 0.000543  |
| InverseGaussian(2,1)          | 2.272415  | 2.278161    | 0.005746  |
| LogGamma(2)                   | 0.886294  | 0.692743    | 0.193551  |
| Poisson(3)                    | 1.907392  | 1.910830    | 0.003438  |
| TruncatedNormal(0,1,-1,2)     | 0.824430  | 0.824881    | 0.000451  |

## Example: Beta(2,3)

**PDF:**
$$
f(x) = \frac{x^{\alpha-1}(1-x)^{\beta-1}}{B(\alpha, \beta)}, \quad x \in (0,1)
$$

**Energy:**
$$
\mathbb{E}|X-Y| = 2 \int_0^1 F(t)(1-F(t)) dt
$$
where $F$ is the Beta CDF.

**Derivation:**
Let $F$ be the CDF of Beta$(\alpha,\beta)$. The energy is:
$$
\mathbb{E}|X-Y| = 2 \int_{-\infty}^{\infty} F(t)(1-F(t)) dt
$$
For Beta, the support is $(0,1)$, so:

## Example: Weibull(1,2)

**PDF:**
$$
f(x) = 2x e^{-x^2}, \quad x > 0
$$

**Energy:**
$$
\mathbb{E}|X-Y| = 2 \int_0^\infty F(t)(1-F(t)) dt
$$
where $F$ is the Weibull CDF.

**Derivation:**
Let $F$ be the CDF of Weibull$(\lambda, k)$ ($\lambda=1$, $k=2$):
$$
F(t) = 1 - e^{-t^2}
$$
Plug into the general formula.

## Example: Inverse Gaussian(2,1)

**PDF:**
$$
f(x) = \left(\frac{1}{2\pi x^3}\right)^{1/2} \exp\left(-\frac{(x-2)^2}{2 \cdot 2^2 x}\right), \quad x > 0
$$

**Energy:**
$$
\mathbb{E}|X-Y| = 2 \int_0^\infty F(t)(1-F(t)) dt
$$
where $F$ is the Inverse Gaussian CDF.

**Derivation:**
Plug the CDF of Inverse Gaussian into the general formula.

## Example: LogGamma(2)

**PDF:**
$$
f(x) = \frac{1}{\Gamma(2)} e^{2x - e^x}, \quad x \in \mathbb{R}
$$

**Energy:**
$$
\mathbb{E}|X-Y| = 2 \int_{-\infty}^{\infty} F(t)(1-F(t)) dt
$$
where $F$ is the LogGamma CDF.

**Derivation:**
Plug the CDF of LogGamma into the general formula.

## Example: Poisson(3)

**PMF:**
$$
P(X = k) = \frac{3^k e^{-3}}{k!}, \quad k = 0,1,2,...
$$

**Energy:**
$$
\mathbb{E}|X-Y| = \sum_{k=0}^\infty \sum_{l=0}^\infty |k-l| P(X=k)P(Y=l)
$$
where $X, Y \sim \text{Poisson}(3)$ i.i.d.

**Derivation:**
For discrete distributions, the energy is the expected absolute difference between two independent samples.


## Example: Truncated Normal(0,1,-1,2)

**PDF:**
$$
f(x) = \frac{\phi(x)}{\Phi(2) - \Phi(-1)}, \quad x \in (-1,2)
$$
where $\phi$ is the standard normal PDF and $\Phi$ is the CDF.

**Energy:**
$$
\mathbb{E}|X-Y| = 2 \int_{-1}^{2} F(t)(1-F(t)) dt
$$
where $F$ is the CDF of the truncated normal.

**Derivation:**
Plug the CDF of the truncated normal into the general formula, integrating over the truncated support.

## Example: Exponential(2)

**PDF:**
$$
f(x) = 2 e^{-2x}, \quad x > 0
$$

**Energy:**
$$
\mathbb{E}|X-Y| = \frac{1}{\lambda}
$$
For Exponential$(\lambda)$, $\lambda=2$, so $\mathbb{E}|X-Y| = 0.5$.

**Derivation:**
For $X, Y \sim \text{Exp}(\lambda)$ i.i.d.,
$$
\mathbb{E}|X-Y| = \frac{1}{\lambda}
$$
This follows from integrating the absolute difference of two independent exponentials.

## Example: Gamma(2,3)

**PDF:**
$$
f(x) = \frac{3^2}{\Gamma(2)} x^{2-1} e^{-3x} = 9x e^{-3x}, \quad x > 0
$$

**Energy:**
$$
\mathbb{E}|X-Y| = 2 \int_0^\infty F(t)(1-F(t)) dt
$$
where $F$ is the Gamma CDF.

**Derivation:**
Let $F$ be the CDF of Gamma$(k,\theta)$ (here $k=2$, $\theta=1/3$). The general formula applies:
$$
\mathbb{E}|X-Y| = 2 \int_0^\infty F(t)(1-F(t)) dt
$$i

## Example: Logistic(0,1)

**PDF:**
$$
f(x) = \frac{e^{-x}}{(1+e^{-x})^2}, \quad x \in \mathbb{R}
$$

**Energy:**
$$
\mathbb{E}|X-Y| = 2 \int_{-\infty}^{\infty} F(t)(1-F(t)) dt = 2
$$
where $F$ is the Logistic CDF.

**Derivation:**
For standard Logistic, the integral evaluates to 1, so $2 \times 1 = 2$.


## Example: Pareto(1,3)

**PDF:**
$$
f(x) = 3 x^{-4}, \quad x > 1
$$

**Energy:**
$$
\mathbb{E}|X-Y| = 2 \int_1^\infty F(t)(1-F(t)) dt
$$
where $F$ is the Pareto CDF.

**Derivation:**
Let $F$ be the CDF of Pareto$(x_m,\alpha)$ ($x_m=1$, $\alpha=3$):
$$
F(t) = 1 - t^{-3}, \quad t \geq 1
$$
Plug into the general formula.

## Example: Inverse Gamma(3,2)

**PDF:**
$$
f(x) = \frac{2^3 x^{-4} \exp\left(-\frac{2}{x}\right)}{\Gamma(3)}, \quad x > 0
$$

**Energy:**
$$
\mathbb{E}|X-Y| = 2 \int_0^\infty F(t)(1-F(t)) dt
$$
where $F$ is the Inverse Gamma CDF.

**Derivation:**
Let $F$ be the CDF of Inverse Gamma$(\alpha,\beta)$. The energy is:
$$
\mathbb{E}|X-Y| = 2 \int_{0}^{\infty} F(t)(1-F(t)) dt
$$
This follows from the general result for continuous distributions:
$$
\mathbb{E}|X-Y| = 2 \int_{-\infty}^{\infty} F(t)(1-F(t)) dt
$$
where the support is $(0,\infty)$ for Inverse Gamma.

## Example: LogNormal(0,1)

**PDF:**
$$
f(x) = \frac{1}{x \sqrt{2\pi}} \exp\left(-\frac{(\log x)^2}{2}\right), \quad x > 0
$$

**Energy:**
$$
\mathbb{E}|X-Y| = 2 \int_0^\infty F(t)(1-F(t)) dt
$$
where $F$ is the LogNormal CDF with $\mu=0$, $\sigma=1$.

**Derivation:**
For LogNormal$(\mu, \sigma)$, the energy is computed using numerical integration of the CDF:
$$
\mathbb{E}|X-Y| = 2 \int_{0}^{\infty} \Phi\left(\frac{\log t - \mu}{\sigma}\right) \left(1 - \Phi\left(\frac{\log t - \mu}{\sigma}\right)\right) dt
$$

## Example: ChiSquared(2)

**PDF:**
$$
f(x) = \frac{1}{2} e^{-x/2}, \quad x > 0
$$

**Energy:**
$$
\mathbb{E}|X-Y| = 2
$$

**Derivation:**
ChiSquared$(k)$ with $k=2$ is equivalent to Exponential$(1/2)$, and Exponential$(\lambda)$ has energy $1/\lambda = 2$.

For general ChiSquared$(k)$, the energy is computed using numerical integration.

## Example: T(0,1,5)

**PDF:**
$$
f(x) = \frac{\Gamma(3)}{\sqrt{5\pi} \Gamma(2.5)} \left(1 + \frac{x^2}{5}\right)^{-3}, \quad x \in \mathbb{R}
$$

**Energy:**
$$
\mathbb{E}|X-Y| = 2 \int_{-\infty}^{\infty} F(t)(1-F(t)) dt
$$
where $F$ is the t-distribution CDF with 5 degrees of freedom.

**Derivation:**
For Student's t-distribution with $\nu$ degrees of freedom, the energy is computed using numerical integration of the CDF.

---

## General Formula

For a continuous distribution with CDF $F$ and support $S$:
$$
\mathbb{E}|X-Y| = 2 \int_S F(t)(1-F(t)) dt
$$

---

*This file is a temporary collection of analytic energy distance formulae and derivations for skpro distributions, until a more permanent documentation solution is implemented (see #689).*
