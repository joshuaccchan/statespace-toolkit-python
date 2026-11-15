import numpy as np
from scipy.signal import lfilter

from ._rng import as_rng


def ma_params(x, w, psi, Hyper, rng=None):
    """ssm.ma_params - posterior draw of the coefficient psi of MA(1) errors with time-varying
    variances,
      x_t = u_t + psi*u_{t-1},  u_t ~ N(0, 1/w_t),  u_0 = 0,
    given x and the precisions w, under the prior psi ~ N(psi0, Vpsi) truncated to (-1, 1).

      psi, u, flag = ssm.ma_params(x, w, psi, Hyper)
      psi, u, flag = ssm.ma_params(x, w, psi, Hyper, rng)

      x     : T-vector of errors
      w     : T-vector of precisions of u_t, such as exp(-h) or 1/(lam*exp(h))
      psi   : the current draw; the new draw is returned
      Hyper : Struct or dict with psi0 and Vpsi
      rng   : None, an integer seed, a numpy.random.Generator or a ssm.Stream
      u     : T-vector of errors u_t under the returned psi, lfilter([1], [1, psi], x)
      flag  : 1 if the candidate was accepted

    Stacked over t, x = H_psi u with H_psi lower bidiagonal, and H_psi has determinant one, so
    the conditional density of psi is exact. The draw is an independence Metropolis-Hastings
    step whose proposal is N(psihat, Dpsi): psihat is the mode of the conditional density and
    Dpsi is minus the inverse of its second derivative there. Newton steps find the mode,
    starting from the best of 40 points on [-0.975, 0.975], so the search does not depend on
    the current draw. Random numbers per call: randn, then rand.

    Trap: the conditional density has thicker tails than the normal proposal, so a chain
    started far from the mode can keep its starting value for many sweeps. From psi = 0, with
    the mode at 0.62 and a conditional standard deviation of 0.057, no candidate is accepted
    in 20,000 draws. Start psi near its mode.

    Written for this toolkit. ssm.arma_params calls it.

    See:
    Chan, J.C.C. (2013). Moving Average Stochastic Volatility Models with Application to
    Inflation Forecast, Journal of Econometrics, 176(2): 162-172.
    """
    r = as_rng(rng)
    x = np.asarray(x, dtype=float).ravel()
    w = np.asarray(w, dtype=float).ravel()
    psi0, Vpsi = Hyper['psi0'], Hyper['Vpsi']
    psig = np.linspace(-.975, .975, 40)
    fg = _logcond(psig, x, w, psi0, Vpsi)
    pm, Dp = _psi_mode(psig[np.argmax(fg)], x, w, psi0, Vpsi)
    pc = pm + np.sqrt(Dp) * r.randn()
    ua = r.rand()
    flag = 0
    if abs(pc) < 1:
        la = (_logcond(pc, x, w, psi0, Vpsi)[0] - _logcond(psi, x, w, psi0, Vpsi)[0]
              + .5 * (pc - pm)**2 / Dp - .5 * (psi - pm)**2 / Dp)
        if np.log(ua) < la:
            psi, flag = pc, 1
    return float(psi), lfilter([1.0], [1.0, psi], x), flag


def _logcond(q, x, w, psi0, Vpsi):
    """log p(psi | x, w) up to a constant, at each element of q"""
    q = np.atleast_1d(np.asarray(q, dtype=float))
    f = np.zeros(q.size)
    for k in range(q.size):
        u = lfilter([1.0], [1.0, q[k]], x)
        f[k] = -.5 * (w @ u**2) - .5 * (q[k] - psi0)**2 / Vpsi
    return f


def _psi_mode(p, x, w, psi0, Vpsi):
    """Newton steps on the log conditional of psi, with the derivatives v = du/dpsi and
    w2 = d2u/dpsi2 from the same recursion as u"""
    def derivs(q):
        u = lfilter([1.0], [1.0, q], x)
        v = lfilter([1.0], [1.0, q], -np.r_[0.0, u[:-1]])
        w2 = lfilter([1.0], [1.0, q], -2 * np.r_[0.0, v[:-1]])
        d1 = -np.sum(w * u * v) - (q - psi0) / Vpsi
        d2 = -np.sum(w * (v**2 + u * w2)) - 1 / Vpsi
        return d1, d2

    for _ in range(100):
        d1, d2 = derivs(p)
        if d2 >= 0:
            break
        pn = min(max(p - d1 / d2, -.999), .999)
        if abs(pn - p) < 1e-10:
            p = pn
            break
        p = pn
    _, d2 = derivs(p)
    return p, (-1 / d2 if d2 < 0 else .05**2)
