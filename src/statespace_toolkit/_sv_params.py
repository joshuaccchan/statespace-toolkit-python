import numpy as np

from ._rng import as_rng


def sv_params(h, mu, phi, Hyper, phi_bnd=.999, rng=None):
    """ssm.sv_params - posterior draw of the parameters (mu, phi, sig2) of stationary AR(1)
    log-volatilities with mean mu,
      h_t = mu + phi*(h_{t-1} - mu) + u_t,  u_t ~ N(0, sig2),  h_1 ~ N(mu, sig2/(1-phi^2)),
    given their paths, under the priors sig2 ~ IG(nuh, Sh), phi ~ N(phi0, Vphi) truncated to
    (-phi_bnd, phi_bnd) and mu ~ N(mu0, Vmu).

      mu, phi, sig2, flag_phi = ssm.sv_params(h, mu, phi, Hyper)
      mu, phi, sig2, flag_phi = ssm.sv_params(h, mu, phi, Hyper, phi_bnd)
      mu, phi, sig2, flag_phi = ssm.sv_params(h, mu, phi, Hyper, phi_bnd, rng)

      h        : T x (n+r) log-volatility paths; the first n columns have mean mu (n = len(mu)),
                 the last r have mean zero and get no mean draw
      mu       : n-vector of current means; the new draw is returned
      phi      : (n+r)-vector of current AR(1) coefficients; the new draw is returned
      Hyper    : Struct or dict with nuh, Sh, phi0, Vphi, mu0 and Vmu
      phi_bnd  : bound on |phi|; default .999, and 1 gives the prior support (-1, 1)
      rng      : None, an integer seed, a numpy.random.Generator or a ssm.Stream
      sig2     : (n+r)-vector of innovation variances
      flag_phi : (n+r)-vector, 1 where the phi candidate was accepted

    Draws sig2, then phi, then mu. The phi candidate comes from N(phi_hat, 1/Kphi), the conditional
    without the density of h_1. It is rejected unless |phi| < phi_bnd, and is otherwise accepted
    with the Metropolis-Hastings probability that adds that density.

    Trap: the mu block runs only when every element of mu is nonzero; a single exact zero skips the
    draw of all of mu.

    Code-identical to bvar_toolkit.sv.sv_params of the Python bvar-toolkit, and the two must stay
    so.

    See:
    Chan, J.C.C., Koop, G. and Yu, X. (2024). Large Order-Invariant Bayesian VARs with Stochastic
    Volatility, Journal of Business and Economic Statistics, 42(2): 825-837.
    """
    if phi_bnd is None:
        phi_bnd = .999     # OISV canonical truncation bound
    r = as_rng(rng)
    h = np.asarray(h, dtype=float)
    if h.ndim == 1:
        h = h[:, None]
    mu = np.array(mu, dtype=float).ravel()
    n = mu.size
    T, npr = h.shape
    phi = np.array(phi, dtype=float).ravel() * np.ones(npr)

        # sample sig2
    tmp_h = np.column_stack([h[:, :n] - mu, h[:, n:]])
    e_h = np.vstack([tmp_h[0] * np.sqrt(1 - phi**2), tmp_h[1:] - phi * tmp_h[:-1]])
    sig2 = 1 / r.gamrnd(Hyper['nuh'] + T / 2, 1 / (Hyper['Sh'] + np.sum(e_h**2, axis=0) / 2))
    sig2 = np.ravel(sig2) * np.ones(npr)

        # sample phi
    Kphi = 1 / Hyper['Vphi'] + np.sum(tmp_h[:T - 1]**2, axis=0) / sig2
    phi_hat = (Hyper['phi0'] / Hyper['Vphi'] + np.sum(tmp_h[:T - 1] * tmp_h[1:T], axis=0) / sig2) \
        / Kphi
    phic = phi_hat + 1 / np.sqrt(Kphi) * r.randn(npr)
    flag_phi = np.zeros(npr)
    for ii in range(npr):
        if abs(phic[ii]) < phi_bnd:
            alpMH = np.exp(_g_phi(phic[ii], sig2[ii], tmp_h[0, ii])
                           - _g_phi(phi[ii], sig2[ii], tmp_h[0, ii]))
            if alpMH > r.rand():
                phi[ii] = phic[ii]
                flag_phi[ii] = 1

        # sample mu
    if n > 0 and np.all(mu != 0):
        p1, s1 = phi[:n], sig2[:n]
        Kmu = 1 / Hyper['Vmu'] + ((1 - p1**2) + (T - 1) * (1 - p1)**2) / s1
        mu_hat = (Hyper['mu0'] / Hyper['Vmu'] + (1 - p1**2) / s1 * h[0, :n]
                  + (1 - p1) / s1 * np.sum(h[1:, :n] - p1 * h[:-1, :n], axis=0)) / Kmu
        mu = mu_hat + 1 / np.sqrt(Kmu) * r.randn(n)
    return mu, phi, sig2, flag_phi


def _g_phi(x, sig2, h1):
    return .5 * np.log(1 - x**2) - .5 * (1 - x**2) / sig2 * h1**2
