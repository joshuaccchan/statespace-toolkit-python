import numpy as np
import scipy.sparse as sp
from scipy.signal import lfilter

from ._linalg import BandChol
from ._rng import as_rng


def _check(s2, muh, phih, omegah2, ht, M, nu=None):
    """the input checks of ssm.intlike_sv and ssm.intlike_sv_t; s2 and ht as float vectors"""
    def scalar(v):
        return np.ndim(v) == 0 and np.isrealobj(v) and isinstance(np.asarray(v).item(),
                                                                  (int, float))
    s2 = np.asarray(s2)
    if not (s2.size > 0 and np.isrealobj(s2) and np.issubdtype(s2.dtype, np.number)
            and np.all(np.isfinite(s2)) and np.all(s2 >= 0)):
        raise ValueError('s2 must be a nonempty vector of finite, nonnegative values')
    if not (scalar(muh) and np.isfinite(muh)):
        raise ValueError('muh must be a finite real scalar')
    if not (scalar(phih) and abs(phih) < 1):
        raise ValueError('phih must be a real scalar with |phih| < 1')
    if not (scalar(omegah2) and omegah2 > 0 and np.isfinite(omegah2)):
        raise ValueError('omegah2 must be a positive, finite scalar')
    if nu is not None and not (scalar(nu) and nu > 0 and np.isfinite(nu)):
        raise ValueError('nu must be a positive, finite scalar')
    ht = np.asarray(ht)
    if not (np.isrealobj(ht) and ht.size == s2.size and np.all(np.isfinite(ht))):
        raise ValueError(f'ht must be a finite vector with as many elements as s2 ({s2.size})')
    if not (scalar(M) and M >= 1 and M == int(M) and np.isfinite(M)):
        raise ValueError('M must be a positive integer')
    return np.asarray(s2, dtype=float).ravel(), np.asarray(ht, dtype=float).ravel()


def _prior(T, muh, phih, omegah2):
    """the stationary AR(1) prior of h: the lower band of its precision HiSH = Hphi' S^{-1}
    Hphi, the sparse HiSH, its mean deltah and HiSH @ deltah"""
    s = np.r_[(1 - phih**2) / omegah2, np.ones(T - 1) / omegah2]
    band = np.zeros((2, T))
    band[0] = s
    band[0, :-1] += phih**2 * s[1:]
    band[1, :-1] = -phih * s[1:]
    HiSH = sp.diags_array([band[1, :-1], band[0], band[1, :-1]], offsets=[-1, 0, 1],
                          format='csr')
    deltah = lfilter([1.0], [1.0, -phih], np.r_[muh, muh * (1 - phih) * np.ones(T - 1)])
    return band, HiSH, deltah, HiSH @ deltah


def _newton(band, HiSHdeltah, s2, ht, tol, max_iter, where):
    """the Newton-Raphson search for the mode of h given s2; returns the mode and the band of
    the negative Hessian of the last step, Kh"""
    T = s2.size
    errh, it = 1.0, 0
    while errh > tol:
        if it >= max_iter:
            raise RuntimeError(f'{where}: no convergence to tol = {tol:g} in {int(max_iter)} '
                               f'steps (last max|step| = {errh:g})')
        it += 1
        sinvexpht = s2 / np.exp(ht)
        fh = -.5 + .5 * sinvexpht
        Gh = .5 * sinvexpht
        if not np.all(np.isfinite(fh) & np.isfinite(Gh)):
            raise RuntimeError(f'{where}: step {it} produced NaN or Inf')
        Kb = band.copy()
        Kb[0] += Gh
        C = BandChol.from_band(Kb if T > 1 else Kb[:1])
        newht = C.solve_t(C.solve(fh + Gh * ht + HiSHdeltah))
        if not np.all(np.isfinite(newht)):
            raise RuntimeError(f'{where}: step {it} produced NaN or Inf')
        errh = np.max(np.abs(newht - ht))
        ht = newht
    return ht, Kb


def _weights(store_llike):
    """the log of the mean of the importance weights and its numerical standard error"""
    M = store_llike.size
    maxllike = np.max(store_llike)
    w = np.exp(store_llike - maxllike)              # the weights, scaled by their largest
    llike = maxllike + np.log(np.mean(w))
    nse = np.std(w, ddof=1) / (np.sqrt(M) * np.mean(w)) if M > 1 else np.nan
    return llike, nse


def intlike_sv(s2, muh, phih, omegah2, ht, M, tol=1e-4, max_iter=500, rng=None):
    """ssm.intlike_sv - integrated log likelihood of the stochastic volatility model, the
    log-volatilities integrated out by importance sampling:
      y_t - mu = exp(h_t/2)*eps_t,  eps_t ~ N(0,1),
      h_t = muh + phih*(h_{t-1} - muh) + v_t,  v_t ~ N(0,omegah2),
      h_1 ~ N(muh, omegah2/(1-phih^2)),  with omegah2 a VARIANCE.

      llike, nse = ssm.intlike_sv(s2, muh, phih, omegah2, ht, M)
      llike, nse = ssm.intlike_sv(..., tol=1e-4, max_iter=500, rng=rng)

      s2        : T-vector of squared demeaned data, (y - mu)**2, finite and nonnegative
      muh       : the mean of h, finite
      phih      : the persistence of h, |phih| < 1
      omegah2   : the innovation variance of h, positive and finite
      ht        : T-vector, the finite starting point of the Newton-Raphson search for the
                  mode of p(h | y), for example the posterior mean of h
      M         : number of importance draws, a positive integer
      tol       : stop the search once max|step| <= tol (default 1e-4)
      max_iter  : raise if that has not happened after this many steps (default 500)
      rng       : None, an integer seed, a numpy.random.Generator or a ssm.Stream
      llike     : log p(y | mu, muh, phih, omegah2), the log of the mean of the M importance
                  weights
      nse       : the numerical standard error of llike by the delta method, the standard
                  deviation of the weights over sqrt(M) times their mean

    The importance density is N(hhat, Kh^{-1}), where hhat is the mode of p(h | y) and Kh the
    negative Hessian of log p(h | y) at the iterate before hhat. The weights use the exact
    Gaussian likelihood, so no mixture approximation enters. nse approximates the standard
    deviation of llike over repeated calls. With heavy-tailed weights it understates it: at
    T = 1000, its median over 200 calls was 0.58 to 0.83 of that standard deviation. Invalid
    inputs raise ValueError; a step that returns NaN or Inf, or reaching max_iter, raises
    RuntimeError. Random numbers per call: randn(T, M), drawn at once, so memory grows with
    T*M.

    See:
    Chan, J.C.C. and Grant, A.L. (2016). Modeling Energy Price Dynamics: GARCH versus
    Stochastic Volatility, Energy Economics, 54: 182-189.
    Chan, J.C.C. and Jeliazkov, I. (2009). Efficient Simulation and Integrated Likelihood
    Estimation in State Space Models, International Journal of Mathematical Modelling and
    Numerical Optimisation, 1(1/2): 101-120.
    """
    s2, ht = _check(s2, muh, phih, omegah2, ht, M)
    T, M = s2.size, int(M)
    band, HiSH, deltah, HiSHdeltah = _prior(T, muh, phih, omegah2)
    ht, Kb = _newton(band, HiSHdeltah, s2, ht, tol, max_iter,
                     'the Newton-Raphson search for the mode of h')
    CKh = BandChol.from_band(Kb if T > 1 else Kb[:1])
    Kh = sp.diags_array([Kb[1, :-1], Kb[0], Kb[1, :-1]], offsets=[-1, 0, 1], format='csr')
    c = -T / 2 * np.log(2 * np.pi) - .5 * (T * np.log(omegah2) - np.log(1 - phih**2)) \
        - CKh.logdet() / 2
    bigh = ht[:, None] + CKh.solve_t(as_rng(rng).randn(T, M))
    D = bigh - deltah[:, None]
    E = bigh - ht[:, None]
    store_llike = (c - .5 * np.sum(D * (HiSH @ D), axis=0) - .5 * np.sum(bigh, axis=0)
                   - .5 * (s2 @ np.exp(-bigh)) + .5 * np.sum(E * (Kh @ E), axis=0))
    return _weights(store_llike)
