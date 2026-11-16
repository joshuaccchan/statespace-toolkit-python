# statespace-toolkit for Python

[statespace-toolkit](https://github.com/joshuaccchan/statespace-toolkit) is a MATLAB library for
Bayesian state space models. It is built around the precision sampler of Chan and Jeliazkov
(2009), which draws the whole path of the states at once, with samplers for unobserved
components, time-varying parameter and stochastic volatility models. This repository is its
Python version: the same functions under the same names, each tested against its MATLAB twin.

It covers 26 of the 31 core functions so far, all but the five model functions, and the ten
[how-to guides](guides/) of the MATLAB toolkit. [PARITY.md](PARITY.md) lists the functions and
how each is checked.

## Installation

From a clone:

```
git clone https://github.com/joshuaccchan/statespace-toolkit-python.git
cd statespace-toolkit-python
pip install -e .
```

## A First Example

The trend of a local level model, drawn given its parameters:

```python
import numpy as np
import scipy.sparse as sp
import statespace_toolkit as ssm

T, sig2, omega2 = 200, 1.0, .05
rng = np.random.default_rng(1)
y = np.cumsum(np.sqrt(omega2) * rng.standard_normal(T)) + np.sqrt(sig2) * rng.standard_normal(T)
H = ssm.diffmat(T)                                   # first differences
K = H.T @ H / omega2 + sp.eye_array(T) / sig2        # the precision of the trend given y
draws, tauhat, C = ssm.simulate_states(K, y / sig2, 1000, rng=rng)
```

[How to Draw the States of an Unobserved Components Model](guides/uc_states/) explains each line.

## Citing

[`CITATION.cff`](CITATION.cff) gives the citation of this package, and
[`CITING.md`](https://github.com/joshuaccchan/statespace-toolkit/blob/main/CITING.md) of the
MATLAB toolkit lists the paper behind each function.

## License

MIT; see [LICENSE](LICENSE).
