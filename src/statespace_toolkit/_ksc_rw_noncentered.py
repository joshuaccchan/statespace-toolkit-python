import numpy as np
from scipy.linalg import solve_triangular

from ._ksc_ar1_mean import M_N, P_N, SIG2_N
from ._linalg import BandChol
from ._rng import as_rng


def ksc_rw_noncentered(Ystar, htilde, h0, omegah, b0, Vh0, Vh, Vomegah, rng=None):
    """ssm.ksc_rw_noncentered - KSC auxiliary-mixture sampler for a random-walk log-volatility
    in the noncentered parameterization h_t = h0 + omegah*htilde_t:
      Ystar_t = h0 + omegah*htilde_t + eps_t,  eps_t = log of a chi^2_1 draw, approximated by
                the Kim-Shephard-Chib (1998) 7-component mixture,
      htilde_t = htilde_{t-1} + v_t,  v_t ~ N(0,1),  htilde_1 ~ N(0,Vh),
    with priors h0 ~ N(b0,Vh0) and omegah ~ N(0,Vomegah) (all VARIANCES).

      htilde, h0, omegah, omegahhat, Domegah = \\
          ssm.ksc_rw_noncentered(Ystar, htilde, h0, omegah, b0, Vh0, Vh, Vomegah)
      ... = ssm.ksc_rw_noncentered(..., rng=rng)

      Ystar and htilde are T-vectors. h0 is the level of h, not a value at time 0 as in
      ssm.ksc_rw_h0. One call draws the mixture indicators, then htilde, then (h0, omegah)
      jointly, then flips the signs of htilde and omegah together with probability 1/2; the
      flip leaves h unchanged, and the posterior of omegah is symmetric about 0. omegahhat
      and Domegah are the mean and variance of the Gaussian conditional posterior of omegah
      given Ystar, the indicators and htilde, with h0 integrated out, computed before the
      flip. Their density at 0 does not depend on the flip; averaged over draws, it estimates
      the posterior density of omegah at 0 in the Savage-Dickey ratio for omegah = 0 (Chan
      2018). rng is None, an integer seed, a numpy.random.Generator or a ssm.Stream. Random
      numbers per call: rand(T), randn(T), randn(2), rand.

    The posterior means of htilde and of (h0, omegah) are solved with the Cholesky factor that
    the draw uses.

    See:
    Chan, J.C.C. (2027). Bayesian Macroeconometrics: Methods and Applications, Chapman &
    Hall/CRC, Section 10.1.1.
    Chan, J.C.C. (2018). Specification Tests for Time-Varying Parameter Models with
    Stochastic Volatility, Econometric Reviews, 37(8): 807-823.
    Fruhwirth-Schnatter, S. and Wagner, H. (2010). Stochastic Model Specification Search for
    Gaussian and Partial Non-Gaussian State Space Models, Journal of Econometrics, 154(1):
    85-100.
    Kim, S., Shephard, N. and Chib, S. (1998). Stochastic Volatility: Likelihood Inference
    and Comparison with ARCH Models, Review of Economic Studies, 65(3): 361-393.
    """
    r = as_rng(rng)
    Ystar = np.asarray(Ystar, dtype=float).ravel()
    htilde = np.asarray(htilde, dtype=float).ravel()
    T = htilde.size

    # the mixture indicators, in 0, ..., 6
    temprand = r.rand(T)
    sd = np.sqrt(SIG2_N)
    q = P_N * (np.exp(-0.5 * ((Ystar[:, None] - ((h0 + omegah * htilde)[:, None] + M_N)) / sd)**2)
               / (np.sqrt(2 * np.pi) * sd))
    q = q / q.sum(axis=1, keepdims=True)
    S = 7 - np.sum(temprand[:, None] < np.cumsum(q, axis=1), axis=1)

    # htilde: Ystar = h0 + omegah*htilde + d + e with e ~ N(0, Omega), where d_t and Omega_tt
    # are the mean and variance of mixture component S_t, and Hh*htilde = v with
    # v ~ N(0, diag(Vh, 1, ..., 1)); Kh = Hh' diag(dv) Hh + omegah^2 Omega^{-1} is tridiagonal
    dconst = M_N[S]
    iOmega = 1 / SIG2_N[S]
    dv = np.r_[1 / Vh, np.ones(T - 1)]
    ab = np.zeros((2, T))
    ab[0] = dv + iOmega * omegah**2
    ab[0, :-1] += dv[1:]
    ab[1, :-1] = -dv[1:]
    CKh = BandChol.from_band(ab if T > 1 else ab[:1])
    htildehat = CKh.solve_t(CKh.solve(iOmega * omegah * (Ystar - dconst - h0)))
    htilde = htildehat + CKh.solve_t(r.randn(T))

    # h0 and omegah, jointly
    Xbeta = np.column_stack([np.ones(T), htilde])
    invVbeta = np.diag([1 / Vh0, 1 / Vomegah])
    XbetainvOmega = Xbeta.T * iOmega
    invDbeta = invVbeta + XbetainvOmega @ Xbeta
    CDbeta = np.linalg.cholesky(invDbeta)
    rhs = invVbeta @ np.array([b0, 0.0]) + XbetainvOmega @ (Ystar - dconst)
    betahat = solve_triangular(CDbeta.T, solve_triangular(CDbeta, rhs, lower=True), lower=False)
    beta = betahat + solve_triangular(CDbeta.T, r.randn(2), lower=False)
    h0, omegah = float(beta[0]), float(beta[1])

    U = -1 + 2 * (r.rand() > 0.5)
    htilde = U * htilde
    omegah = U * omegah

    # the mean and variance of the conditional posterior of omegah
    Xbeta = np.column_stack([np.ones(T), htilde])
    Dbeta = np.linalg.solve(invVbeta + (Xbeta.T * iOmega) @ Xbeta, np.eye(2))
    return htilde, h0, omegah, float(betahat[1]), float(Dbeta[1, 1])
