import numpy as np
from scipy.special import digamma, gammaln, polygamma

from ._rng import as_rng


def nu_studentt(lam, nu, nu_ub, rng=None):
    """ssm.nu_studentt - independence Metropolis-Hastings draw of the degrees of freedom nu
    of Student-t errors, given the latent scales lam_t ~ IG(nu/2, nu/2), under the prior
    nu ~ U(2, nu_ub). The proposal is N(nut, -1/H): nut is the mode of the log target, found by
    Newton-Raphson, and H its second derivative there.

      nu, flag, f_nu = ssm.nu_studentt(lam, nu, nu_ub)
      nu, flag, f_nu = ssm.nu_studentt(lam, nu, nu_ub, rng)

      lam   : T-vector of latent scales
      nu    : the current draw; the new draw is returned
      nu_ub : upper bound of the uniform prior
      rng   : None, an integer seed, a numpy.random.Generator or a ssm.Stream
      flag  : 1 if the candidate was accepted
      f_nu  : the log target of nu, up to a constant, as a function

    nu comes back unchanged, with flag = 0, when the candidate is rejected and when it falls
    outside (2, nu_ub). If a Newton-Raphson step falls below 2, the proposal is centered at 5.

    Code-identical to bvar_toolkit.sv.nu_studentt of the Python bvar-toolkit, and the two must stay
    so.

    See:
    Chan, J.C.C. (2020). Large Bayesian Vector Autoregressions. In: P. Fuleky (Eds), Macroeconomic
    Forecasting in the Era of Big Data, 95-125, Springer, Cham.
    Chan, J.C.C. (2020). Large Bayesian VARs: A flexible Kronecker error covariance structure,
    Journal of Business and Economic Statistics, 38(1), 68-79.
    """
    r = as_rng(rng)
    lam = np.asarray(lam, dtype=float).ravel()
    flag = 0
    T = lam.size
    sum1 = np.sum(np.log(lam))
    sum2 = np.sum(1 / lam)

    def f_nu(x):
        return T * (x / 2 * np.log(x / 2) - gammaln(x / 2)) - (x / 2 + 1) * sum1 - x / 2 * sum2

    def df_nu(x):
        return T / 2 * (np.log(x / 2) + 1 - digamma(x / 2)) - .5 * (sum1 + sum2)

    def d2f_nu(x):
        return T / (2 * x) - T / 4 * polygamma(1, x / 2)

    S_nu = 1.0
    nut = nu
    while abs(S_nu) > 10**(-5):                          # stopping criteria
        S_nu = df_nu(nut)
        H_nu = d2f_nu(nut)
        nut = nut - S_nu / H_nu
        if nut < 2:
            nut = 5
            H_nu = d2f_nu(nut)
            break
    Dnu = -1 / H_nu
    nuc = nut + np.sqrt(Dnu) * r.randn()
    if 2 < nuc < nu_ub:
        lalp_MH = f_nu(nuc) - f_nu(nu) - .5 * (nu - nut)**2 / Dnu + .5 * (nuc - nut)**2 / Dnu
        if np.exp(lalp_MH) > r.rand():
            nu = nuc
            flag = 1
    return nu, flag, f_nu
