import numpy as np
import scipy.sparse as sp
from scipy.special import gammaln

from ._intlike_sv import _check, _newton, _prior, _weights
from ._linalg import BandChol
from ._rng import as_rng


def intlike_sv_t(s2, muh, phih, omegah2, nu, ht, M, tol=1e-4, max_iter=500, rng=None):
    """ssm.intlike_sv_t - integrated log likelihood of the stochastic volatility model with
    Student-t errors, the log-volatilities integrated out by importance sampling:
      y_t - mu = exp(h_t/2)*eps_t,  eps_t ~ Student-t with nu degrees of freedom,
      h_t = muh + phih*(h_{t-1} - muh) + v_t,  v_t ~ N(0,omegah2),
      h_1 ~ N(muh, omegah2/(1-phih^2)),  with omegah2 a VARIANCE.

      llike, nse = ssm.intlike_sv_t(s2, muh, phih, omegah2, nu, ht, M)
      llike, nse = ssm.intlike_sv_t(..., tol=1e-4, max_iter=500, rng=rng)

      s2        : T-vector of squared demeaned data, (y - mu)**2, finite and nonnegative
      muh       : the mean of h, finite
      phih      : the persistence of h, |phih| < 1
      omegah2   : the innovation variance of h, positive and finite
      nu        : degrees of freedom, positive and finite
      ht        : T-vector, the finite starting point of the search for the mode of
                  p(h | y), for example the posterior mean of h
      M         : number of importance draws, a positive integer
      tol       : stop the EM algorithm, and each of its Newton-Raphson searches, once
                  max|step| <= tol (default 1e-4)
      max_iter  : raise if the EM algorithm has not stopped after this many iterations, or
                  one of its Newton-Raphson searches after this many steps (default 500)
      rng       : None, an integer seed, a numpy.random.Generator or a ssm.Stream
      llike     : log p(y | mu, muh, phih, omegah2, nu), the log of the mean of the M
                  importance weights
      nse       : the numerical standard error of llike by the delta method, the standard
                  deviation of the weights over sqrt(M) times their mean

    The mode hhat of p(h | y) is found by EM, with the scales of the Student-t errors as the
    missing data; each M-step is a Newton-Raphson search. The importance density is
    N(hhat, Kh^{-1}), where Kh is the negative Hessian of log p(h | y) at hhat, and the
    weights use the exact Student-t likelihood. nse approximates the standard deviation of
    llike over repeated calls: at T = 137 and 1000, its median over 200 calls was 0.90 to
    1.10 of that standard deviation. Invalid inputs raise ValueError; a Newton-Raphson step
    that returns NaN or Inf, or reaching max_iter in either loop, raises RuntimeError.
    Random numbers per call: randn(T, M), drawn at once, so memory grows with T*M.

    See:
    Chan, J.C.C. and Grant, A.L. (2016). Modeling Energy Price Dynamics: GARCH versus
    Stochastic Volatility, Energy Economics, 54: 182-189.
    Chan, J.C.C. and Jeliazkov, I. (2009). Efficient Simulation and Integrated Likelihood
    Estimation in State Space Models, International Journal of Mathematical Modelling and
    Numerical Optimisation, 1(1/2): 101-120.
    """
    s2, ht = _check(s2, muh, phih, omegah2, ht, M, nu)
    T, M = s2.size, int(M)
    band, HiSH, deltah, HiSHdeltah = _prior(T, muh, phih, omegah2)
    errh_out, iter_out = 1.0, 0
    while errh_out > tol:
        if iter_out >= max_iter:
            raise RuntimeError(f'the EM algorithm for the mode of h: no convergence to '
                               f'tol = {tol:g} in {int(max_iter)} iterations '
                               f'(last max|step| = {errh_out:g})')
        iter_out += 1
        Eilam = (nu + 1) / (nu + s2 / np.exp(ht))                 # E-step
        htt, _ = _newton(band, HiSHdeltah, s2 * Eilam, ht, tol, max_iter,      # M-step
                         f'the Newton-Raphson search of EM iteration {iter_out}')
        errh_out = np.max(np.abs(ht - htt))
        ht = htt

    # the negative Hessian at the mode
    Gh = (nu + 1) / (2 * nu) * (s2 * np.exp(ht)) / ((np.exp(ht) + s2 / nu)**2)
    Kb = band.copy()
    Kb[0] += Gh
    CKh = BandChol.from_band(Kb if T > 1 else Kb[:1])
    Kh = sp.diags_array([Kb[1, :-1], Kb[0], Kb[1, :-1]], offsets=[-1, 0, 1], format='csr')
    c = -.5 * (T * np.log(omegah2) - np.log(1 - phih**2)) - CKh.logdet() / 2
    bigh = ht[:, None] + CKh.solve_t(as_rng(rng).randn(T, M))
    D = bigh - deltah[:, None]
    E = bigh - ht[:, None]
    cy = T * (gammaln((nu + 1) / 2) - gammaln(nu / 2) - .5 * np.log(nu * np.pi))
    deny_h = (cy - .5 * np.sum(bigh, axis=0)
              - (nu + 1) / 2 * np.sum(np.log(1 + (s2[:, None] / np.exp(bigh)) / nu), axis=0))
    store_llike = (c - .5 * np.sum(D * (HiSH @ D), axis=0) + deny_h
                   + .5 * np.sum(E * (Kh @ E), axis=0))
    return _weights(store_llike)
