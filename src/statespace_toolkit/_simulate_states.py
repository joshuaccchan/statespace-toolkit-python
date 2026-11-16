import numpy as np

from ._chol import chol
from ._rng import as_rng


def simulate_states(K, c, ndraws=None, rng=None):
    """ssm.simulate_states - draws from N(alphahat, K^{-1}), where K is the precision matrix and
    alphahat solves K @ alphahat = c: Algorithm 1 of Chan and Jeliazkov (2009), which obtains the
    mean and the draws from one Cholesky factor of K. It is a more efficient implementation of
    Algorithm 9.1 of Chan (2027), which takes the mean as an input and so needs a separate solve
    with K to compute it. Written for this toolkit.

      alpha, alphahat, C = ssm.simulate_states(K, c)
      alpha, alphahat, C = ssm.simulate_states(K, c, ndraws)
      alpha, alphahat, C = ssm.simulate_states(K, c, ndraws, rng)

      K        : n x n symmetric positive definite precision matrix. For the states of a
                 linear Gaussian state space model (Theorem 9.1 of the book),
                 K = P + Z' inv(R) Z and c = P b + Z' inv(R) y, with the prior precision
                 P = G' inv(Q) G and prior mean b. K is banded, so a sparse K makes each
                 draw O(n).
      c        : n-vector
      ndraws   : number of independent draws; None (the default) gives one draw as an
                 n-vector, and an integer gives an n x ndraws array
      rng      : None, an integer seed, a numpy.random.Generator or a ssm.Stream
      alpha    : n x ndraws array of draws
      alphahat : n-vector, the mean K^{-1} c
      C        : lower Cholesky factor of K, with the methods solve, solve_t and logdet, so
                 that a caller needing a further solve with K, such as ssm.restrict, which
                 conditions the draw on a linear restriction, does not factorize it a second time

    See:
    Chan, J.C.C. (2027). Bayesian Macroeconometrics: Methods and Applications, Chapman &
    Hall/CRC, Section 9.1.1 and Algorithm 9.1.
    Chan, J.C.C. and Jeliazkov, I. (2009). Efficient Simulation and Integrated Likelihood
    Estimation in State Space Models, International Journal of Mathematical Modelling and
    Numerical Optimisation, 1(1/2): 101-120, Algorithm 1.
    """
    r = as_rng(rng)
    C = chol(K)
    alphahat = C.solve_t(C.solve(np.asarray(c, dtype=float).ravel()))
    n = alphahat.size
    if ndraws is None:
        alpha = alphahat + C.solve_t(r.randn(n))
    else:
        alpha = alphahat[:, None] + C.solve_t(r.randn(n, int(ndraws)))
    return alpha, alphahat, C
