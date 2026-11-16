# How to Estimate a Stochastic Volatility Model by Gibbs Sampling

*Code: [`guide.py`](guide.py), with
[`ssm.ksc_ar1_mean`](../../src/statespace_toolkit/_ksc_ar1_mean.py),
[`ssm.sv_params`](../../src/statespace_toolkit/_sv_params.py),
[`ssm.inefficiency_factor`](../../src/statespace_toolkit/_inefficiency_factor.py),
[`ssm.mcse`](../../src/statespace_toolkit/_mcse.py),
[`ssm.geweke`](../../src/statespace_toolkit/_geweke.py) and
[`ssm.armh`](../../src/statespace_toolkit/_armh.py). Method:
[Kim, Shephard and Chib (1998)](https://github.com/joshuaccchan/statespace-toolkit/blob/main/CITING.md#kim-shephard-and-chib-1998),
[Del Negro and Primiceri (2015)](https://github.com/joshuaccchan/statespace-toolkit/blob/main/CITING.md#del-negro-and-primiceri-2015),
[Chan and Grant (2016)](https://github.com/joshuaccchan/statespace-toolkit/blob/main/CITING.md#chan-and-grant-2016b),
[Geweke (1992)](https://github.com/joshuaccchan/statespace-toolkit/blob/main/CITING.md#geweke-1992)
and [Chan (2027)](https://github.com/joshuaccchan/statespace-toolkit/blob/main/CITING.md#chan-2027).
MATLAB version:
[the same guide in statespace-toolkit](https://github.com/joshuaccchan/statespace-toolkit/tree/main/guides/sv_gibbs).*

We estimate the stochastic volatility (SV) model of
[How to Draw the Log-Volatilities of a Stochastic Volatility Model](../sv_states/) by Gibbs
sampling, on the data of that guide. That guide holds the mean $`\mu`$ and the parameters
$`\mu_h`$, $`\phi`$ and $`\sigma_h^2`$ of the log-volatility at their true values. Here we
estimate them along with the log-volatilities. Each iteration of the sampler draws the
log-volatilities by the auxiliary mixture sampler of the first guide. It then draws three
parameters from standard distributions and $`\phi`$ by a Metropolis-Hastings step.

To print the numbers and draw the figure first, run from the root of the repository, after
`pip install -e .`:

```
python guides/sv_gibbs/guide.py
```

It takes about 70 seconds on the machine that produced this page.

## The Model and Its Priors

The model is that of the first guide,

```math
y_t = \mu + \mathrm{e}^{h_t/2}\varepsilon_t, \qquad \varepsilon_t \sim \mathcal{N}(0, 1), \qquad\qquad h_t = \mu_h + \phi(h_{t-1} - \mu_h) + u_t, \qquad u_t \sim \mathcal{N}(0, \sigma_h^2),
```

with $`h_1 \sim \mathcal{N}(\mu_h, \sigma_h^2/(1 - \phi^2))`$ and $`\mu`$, $`\mu_h`$, $`\phi`$ and
$`\sigma_h^2`$ unknown. We follow Chan and Grant (2016) to set the priors:

```math
\mu \sim \mathcal{N}(0, 10), \qquad \mu_h \sim \mathcal{N}(1, 10), \qquad \phi \sim \mathcal{TN}_{(-1,1)}(0.97, 0.1^2), \qquad \sigma_h^2 \sim \mathcal{IG}(5, 0.16),
```

where $`\mathcal{TN}_{(-1,1)}`$ is the normal distribution truncated to $`(-1, 1)`$ and
$`\mathcal{IG}(\nu, S)`$ is the inverse-gamma distribution with shape $`\nu`$ and scale $`S`$.
The prior mean of $`\sigma_h^2`$ is $`0.16/4 = 0.04`$. In general notation the priors are
$`\mu \sim \mathcal{N}(\mu_0, V_\mu)`$, $`\mu_h \sim \mathcal{N}(\mu_{h0}, V_{\mu_h})`$,
$`\phi \sim \mathcal{TN}_{(-1,1)}(\phi_0, V_\phi)`$ and
$`\sigma_h^2 \sim \mathcal{IG}(\nu_h, S_h)`$. The data are those of the first guide: $`T = 1000`$
observations in percent, generated with $`\mu = 0.1`$, $`\mu_h = 0`$, $`\phi = 0.97`$ and
$`\sigma_h^2 = 0.04`$.

## Step 1: Split the Unknowns into Blocks

The unknowns are $`\mu`$, the log-volatilities $`\mathbf{h} = (h_1, \ldots, h_T)'`$, the mixture
indicators $`\mathbf{s}`$ of the first guide, and $`\mu_h`$, $`\phi`$ and $`\sigma_h^2`$. The
sampler cycles through five blocks:

1. $`(\mu \mid \mathbf{y}, \mathbf{h})`$, a normal distribution;
2. $`(\mathbf{s} \mid \mathbf{y}, \mu, \mathbf{h})`$ and then
   $`(\mathbf{h} \mid \mathbf{y}, \mu, \mathbf{s}, \mu_h, \phi, \sigma_h^2)`$, the mixture step of
   the first guide;
3. $`(\sigma_h^2 \mid \mathbf{h}, \mu_h, \phi)`$, an inverse-gamma distribution;
4. $`(\phi \mid \mathbf{h}, \mu_h, \sigma_h^2)`$, by a Metropolis-Hastings step;
5. $`(\mu_h \mid \mathbf{h}, \phi, \sigma_h^2)`$, a normal distribution.

Block 1 draws $`\mu`$ given $`\mathbf{h}`$ and not given the indicators. Given $`h_t`$, the
observation $`y_t`$ is normal with mean $`\mu`$ and known variance $`\mathrm{e}^{h_t}`$, so the
distribution of $`\mu`$ is normal. Given the indicators as well, $`\mu`$ enters through
$`y_t^* = \log\big((y_t - \mu)^2 + c\big)`$, and its distribution has no standard form. Block 2
then draws the indicators afresh, from the $`y_t^*`$ that the new $`\mu`$ gives. Drawing the
indicators immediately before $`\mathbf{h}`$ follows Del Negro and Primiceri (2015). The
parameters of the log-volatility enter only the state equation, so given $`\mathbf{h}`$ they do
not depend on $`\mathbf{y}`$ or on the indicators.

## Step 2: Derive the Conditional Distributions

**The mean.** Given $`\mathbf{h}`$, the model is a regression of $`y_t`$ on a constant with known
variances $`\mathrm{e}^{h_t}`$. The normal prior is conjugate:

```math
(\mu \mid \mathbf{y}, \mathbf{h}) \sim \mathcal{N}\big(K_\mu^{-1}c_\mu,\; K_\mu^{-1}\big), \qquad K_\mu = \frac{1}{V_\mu} + \sum_{t=1}^{T}\mathrm{e}^{-h_t}, \qquad c_\mu = \frac{\mu_0}{V_\mu} + \sum_{t=1}^{T}\mathrm{e}^{-h_t}y_t.
```

**The variance of the log-volatility.** Given $`\mathbf{h}`$, $`\mu_h`$ and $`\phi`$, the state
errors are known: $`u_1 = \sqrt{1 - \phi^2}(h_1 - \mu_h)`$, scaled to variance $`\sigma_h^2`$, and
$`u_t = h_t - \mu_h - \phi(h_{t-1} - \mu_h)`$ for $`t > 1`$. The inverse-gamma prior is conjugate:

```math
(\sigma_h^2 \mid \mathbf{h}, \mu_h, \phi) \sim \mathcal{IG}\Big(\nu_h + \frac{T}{2},\; S_h + \frac{1}{2}\sum_{t=1}^{T}u_t^2\Big).
```

**The persistence.** The state equations for $`t > 1`$ form a regression of $`h_t - \mu_h`$ on
$`h_{t-1} - \mu_h`$ with coefficient $`\phi`$. Combined with the normal part of the prior, they
give the normal density $`q(\phi)`$ of $`\mathcal{N}(\hat{\phi}, K_\phi^{-1})`$, with

```math
K_\phi = \frac{1}{V_\phi} + \frac{1}{\sigma_h^2}\sum_{t=2}^{T}(h_{t-1} - \mu_h)^2, \qquad \hat{\phi} = K_\phi^{-1}\Big(\frac{\phi_0}{V_\phi} + \frac{1}{\sigma_h^2}\sum_{t=2}^{T}(h_{t-1} - \mu_h)(h_t - \mu_h)\Big).
```

The first state equation also depends on $`\phi`$, through the variance
$`\sigma_h^2/(1 - \phi^2)`$ of $`h_1`$. The conditional density of $`\phi`$ is therefore
proportional to $`g(\phi)q(\phi)`$ on $`(-1, 1)`$, with

```math
g(\phi) = \sqrt{1 - \phi^2}\,\exp\Big(-\frac{(1 - \phi^2)(h_1 - \mu_h)^2}{2\sigma_h^2}\Big).
```

An independence-chain Metropolis-Hastings step draws a candidate $`\phi^*`$ from $`q`$. It rejects
the candidate if $`|\phi^*| \geq 1`$, and otherwise accepts it with probability
$`\min\{1, g(\phi^*)/g(\phi)\}`$. The factor $`g`$ comes from one period only, so the acceptance
rate is high.

**The mean of the log-volatility.** Written as
$`h_1 = \mu_h + \sqrt{\sigma_h^2/(1 - \phi^2)}\,e_1`$ with $`e_1 \sim \mathcal{N}(0, 1)`$, and
$`h_t - \phi h_{t-1} = (1 - \phi)\mu_h + u_t`$, the state equations form a regression in $`\mu_h`$
with known variances. With the normal prior,

```math
(\mu_h \mid \mathbf{h}, \phi, \sigma_h^2) \sim \mathcal{N}\big(K_{\mu_h}^{-1}c_{\mu_h},\; K_{\mu_h}^{-1}\big), \qquad K_{\mu_h} = \frac{1}{V_{\mu_h}} + \frac{1 - \phi^2 + (T - 1)(1 - \phi)^2}{\sigma_h^2},
```

```math
c_{\mu_h} = \frac{\mu_{h0}}{V_{\mu_h}} + \frac{1}{\sigma_h^2}\Big((1 - \phi^2)h_1 + (1 - \phi)\sum_{t=2}^{T}(h_t - \phi h_{t-1})\Big).
```

## Step 3: Write the Loop

```python
import numpy as np
import statespace_toolkit as ssm

mu0, Vmu = 0.0, 10.0                                    # mu ~ N(mu0, Vmu)
muh0, Vmuh = 1.0, 10.0                                  # mu_h ~ N(muh0, Vmuh)
phi0, Vphi = .97, .1**2                                 # phi ~ TN(phi0, Vphi) on (-1, 1)
nuh, Sh = 5.0, .16                                      # sigma_h^2 ~ IG(nuh, Sh)
# the priors of sigma_h^2, phi and mu_h, with the keys of ssm.sv_params
hyp = dict(nuh=nuh, Sh=Sh, phi0=phi0, Vphi=Vphi, mu0=muh0, Vmu=Vmuh)
offset = 1e-4
nsim, burnin = 20000, 1000
muh, phi, sig2h = np.log(y.var(ddof=1)), .98, .04       # starting values
h = muh * np.ones(T)
store_theta = np.zeros((nsim, 4))
nacc = 0
store_h = np.zeros((T, nsim // 10))
for isim in range(burnin + nsim):
    Kmu = 1 / Vmu + np.sum(np.exp(-h))                  # K_mu = 1/V_mu + sum_t e^{-h_t}
    # mu ~ N(c_mu/K_mu, 1/K_mu)
    mu = (mu0 / Vmu + np.exp(-h) @ y) / Kmu + rng.standard_normal() / np.sqrt(Kmu)

    ystar = np.log((y - mu)**2 + offset)                # s, then h: guides/sv_states
    h, _ = ssm.ksc_ar1_mean(ystar, h, muh, phi, sig2h, rng=rng)

    # sigma_h^2, phi, then mu_h; each output holds one value per series
    [muh], [phi], [sig2h], [acc] = ssm.sv_params(h, muh, phi, hyp, 1, rng=rng)
    nacc += acc * (isim >= burnin)

    if isim >= burnin:
        store_theta[isim - burnin] = mu, muh, phi, sig2h
        if (isim - burnin + 1) % 10 == 0:
            store_h[:, (isim - burnin + 1) // 10 - 1] = h
```

The chain starts from the starting values of Chan and Grant (2016): $`\mu_h`$ equal to the log of
the sample variance of $`y_t`$, $`\phi = 0.98`$ and $`\sigma_h^2 = 0.04`$. The path of the
log-volatilities starts flat, at $`\mu_h`$.
[`ssm.sv_params`](../../src/statespace_toolkit/_sv_params.py) takes blocks 3 to 5, in that order.
Its fifth argument is the bound on $`|\phi|`$, here 1, the edge of the support of the prior. Each
of its outputs is a vector with one value per series, here of length 1, which the brackets unpack.
The first 1,000 iterations are discarded as burn-in.

## Step 4: Check the Chain

Figure 1 shows the draws of the four parameters after the burn-in. The draws of $`\mu`$ and
$`\mu_h`$ are nearly independent, while those of $`\phi`$ and $`\sigma_h^2`$ are strongly
correlated. The draws of $`\mu_h`$ have spikes. The three farthest from 0, at 5.92, $`-3.52`$ and
$`-3.02`$, occur with $`\phi`$ between 0.9966 and 0.9991. There the state equation carries little
information about the mean of the log-volatility. Draws of $`\phi`$ above 0.995 make up 28 of the
20,000. The sampler draws $`\phi`$ and $`\sigma_h^2`$ given the path of the log-volatilities, and
the path given $`\phi`$ and $`\sigma_h^2`$. The two depend strongly on each other in the
posterior. [`ssm.inefficiency_factor`](../../src/statespace_toolkit/_inefficiency_factor.py)
estimates the long-run variance of the draws with Bartlett weights up to a truncation lag $`L`$.
That lag must reach past the lag at which the autocorrelations die out. For $`\phi`$ and
$`\sigma_h^2`$ the inefficiency factors grow up to $`L = 200`$ and then stay put:

| Truncation lag $`L`$ | 50 | 100 | 200 | 400 |
|---|---|---|---|---|
| $`\phi`$ | 13.9 | 19.3 | 22.4 | 22.4 |
| $`\sigma_h^2`$ | 32.8 | 46.8 | 56.3 | 59.2 |

With $`L = 200`$, the 20,000 draws of $`\sigma_h^2`$ carry about as much information as 360
independent draws. The Monte Carlo standard errors from
[`ssm.mcse`](../../src/statespace_toolkit/_mcse.py) measure how far each posterior mean can be
from its value in an infinitely long chain.

| | True value | Posterior mean | 90% interval | Inefficiency factor | Monte Carlo standard error |
|---|---|---|---|---|---|
| $`\mu`$ | 0.1 | 0.1227 | (0.0786, 0.1681) | 1.6 | 0.0002 |
| $`\mu_h`$ | 0 | -0.0402 | (-0.4804, 0.3963) | 1.5 | 0.0025 |
| $`\phi`$ | 0.97 | 0.9678 | (0.9481, 0.9848) | 22.4 | 0.0004 |
| $`\sigma_h^2`$ | 0.04 | 0.0594 | (0.0375, 0.0878) | 56.3 | 0.0008 |

All four true values lie inside their 90 percent intervals. The candidate $`\phi^*`$ is accepted
in 94.5 percent of the iterations.

Geweke's (1992) diagnostic compares the mean of the first 10 percent of the draws with the mean of
the last half. It divides the difference by its standard error.
[`ssm.geweke`](../../src/statespace_toolkit/_geweke.py) returns it, and it is approximately
standard normal if the chain has converged. Its standard error needs the same truncation lag. With
$`L = 200`$ the four values are $`-1.74`$, 0.87, 1.31 and $`-1.27`$, all inside $`\pm 1.96`$.

![The draws of the four parameters and their histograms](fig_chain.png)

*Figure 1: Top, the 20,000 draws of $`\mu`$, $`\mu_h`$, $`\phi`$ and $`\sigma_h^2`$ kept after the
burn-in, in the order the sampler makes them. Bottom, their histograms over the central 99.9
percent of the draws, the posterior densities from the sampler of How We Check It, and the true
values.*

The volatility estimated with the parameters unknown is close to that of the first guide. The 90
percent intervals of $`\mathrm{e}^{h_t/2}`$ are 0.76 wide on average, against 0.68 with the
parameters known. They contain the true volatility in 922 of the 1,000 periods.

## How We Check It

<details>
<summary>The posterior from a second sampler, which draws the log-volatilities from their exact
conditional distribution</summary>

The sampler of Chan and Grant (2016) has the same blocks, except for $`\mathbf{h}`$. It draws
$`\mathbf{h}`$ from its exact conditional distribution, by the accept-reject Metropolis-Hastings
step of [How to Draw the Log-Volatilities of a Stochastic Volatility Model](../sv_states/). It
does not use the mixture. `guide.py` runs it on the same data with the same priors, for 20,000
draws after a burn-in of 1,000, starting the log-volatilities at the mode. Its Metropolis-Hastings
step takes the candidate path in 54.9 percent of the iterations.

The posterior means of the two samplers differ by 0.2, $`-2.4`$, $`-3.4`$ and 3.0 Monte
Carlo standard errors for $`\mu`$, $`\mu_h`$, $`\phi`$ and $`\sigma_h^2`$. Their posterior
standard deviations differ by at most 12.2 percent.

</details>

## Adapt It Yourself

The book *Bayesian Macroeconometrics* (Chan, 2027) lets the log-volatility follow a random walk,
$`h_t = h_{t-1} + u_t`$ with $`u_t \sim \mathcal{N}(0, \sigma_h^2)`$, from an unknown initial
value $`h_0 \sim \mathcal{N}(a_0, b_0)`$. Which blocks change?

<details>
<summary>Answer</summary>

The blocks of $`\phi`$ and $`\mu_h`$ go, and $`h_0`$ gets a block of its own. Given $`h_0`$, the
mixture step uses the random-walk state equation of the Adapt It Yourself answer in
[How to Draw the Log-Volatilities of a Stochastic Volatility Model](../sv_states/), which
[`ssm.ksc_rw_h0`](../../src/statespace_toolkit/_ksc_rw_h0.py) implements. The state errors become
$`u_t = h_t - h_{t-1}`$ with the current $`h_0`$, and $`\sigma_h^2`$ keeps its inverse-gamma form.
The initial value enters only the first state equation, $`h_1 = h_0 + u_1`$, so

$`\displaystyle (h_0 \mid h_1, \sigma_h^2) \sim \mathcal{N}\big(K_{h_0}^{-1}c_{h_0},\; K_{h_0}^{-1}\big), \qquad K_{h_0} = \frac{1}{b_0} + \frac{1}{\sigma_h^2}, \qquad c_{h_0} = \frac{a_0}{b_0} + \frac{h_1}{\sigma_h^2}.`$

```python
a0, b0 = 0.0, 10.0           # h_0 ~ N(a0, b0)
nuh_rw, Sh_rw = 5.0, .04     # sigma_h^2 ~ IG(nuh_rw, Sh_rw)
H = ssm.diffmat(T)
h0, sig2h = np.log(y_rw.var(ddof=1)), .01
h = h0 * np.ones(T)
store_rw = np.zeros((nsim, 4))
for isim in range(burnin + nsim):
    Kmu = 1 / Vmu + np.sum(np.exp(-h))
    mu = (mu0 / Vmu + np.exp(-h) @ y_rw) / Kmu + rng.standard_normal() / np.sqrt(Kmu)

    ystar = np.log((y_rw - mu)**2 + offset)
    h = ssm.ksc_rw_h0(ystar, h, sig2h, h0, rng=rng)      # s, then h, given h_0

    u = H @ h - np.r_[h0, np.zeros(T - 1)]               # h_t - h_{t-1}, with h_0
    sig2h = 1 / rng.gamma(nuh_rw + T / 2, 1 / (Sh_rw + u @ u / 2))

    Kh0 = 1 / b0 + 1 / sig2h                             # h_0 given h_1 and sigma_h^2
    h0 = (a0 / b0 + h[0] / sig2h) / Kh0 + rng.standard_normal() / np.sqrt(Kh0)

    if isim >= burnin:
        store_rw[isim - burnin] = mu, sig2h, h0, h[0]
```

`guide.py` runs this sampler on data generated from the random walk, with $`T = 1000`$,
$`\mu = 0.1`$, $`h_0 = 0`$ and $`\sigma_h^2 = 0.01`$. The posterior mean of $`\sigma_h^2`$ is
0.0093, with 90 percent interval (0.0058, 0.0141). The sampler with the exact conditional
distribution of $`\mathbf{h}`$ gives the same posterior. Under the two samplers, the posterior
means of $`\mu`$, $`\sigma_h^2`$, $`h_0`$ and $`h_1`$ differ by at most 0.4 Monte Carlo standard
errors.

</details>

## Next

[How to Add Student-t Errors to a Stochastic Volatility Model](../sv_t/) lets the errors have
heavier tails than the normal distribution. Among the examples of the MATLAB toolkit,
[ex08](https://github.com/joshuaccchan/statespace-toolkit/blob/main/examples/ex08_sv_ma_t.m) runs
a sampler with the same blocks for the log-volatility on daily silver returns.

## Using This Method in Your Research

The sampler draws the log-volatilities by the auxiliary mixture sampler of Kim, Shephard and Chib
(1998), with the model and priors of Chan and Grant (2016). The inefficiency factors and the
convergence diagnostic are those of Geweke (1992), and Chan (2027) gives the textbook treatment of
the sampler. [`CITING.md`](https://github.com/joshuaccchan/statespace-toolkit/blob/main/CITING.md)
lists the paper behind each function of the toolkit and how to cite the toolkit itself.

## References

Chan, J. C. C. (2027). *Bayesian Macroeconometrics: Methods and Applications*. Chapman &
Hall/CRC. [Code repository](https://github.com/joshuaccchan/bayesian-macroeconometrics)

Chan, J. C. C. and Grant, A. L. (2016). Modeling Energy Price Dynamics: GARCH versus Stochastic
Volatility. *Energy Economics*, 54: 182-189.
[doi:10.1016/j.eneco.2015.12.003](https://doi.org/10.1016/j.eneco.2015.12.003)

Geweke, J. (1992). Evaluating the Accuracy of Sampling-Based Approaches to the Calculation of
Posterior Moments. In J. M. Bernardo, J. O. Berger, A. P. Dawid and A. F. M. Smith (Eds),
*Bayesian Statistics 4*, 169-193. Oxford University Press.
[doi:10.1093/oso/9780198522669.003.0010](https://doi.org/10.1093/oso/9780198522669.003.0010)

Kim, S., Shephard, N. and Chib, S. (1998). Stochastic Volatility: Likelihood Inference and
Comparison with ARCH Models. *Review of Economic Studies*, 65(3): 361-393.
[doi:10.1111/1467-937X.00050](https://doi.org/10.1111/1467-937X.00050)
