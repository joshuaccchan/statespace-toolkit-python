import numpy as np

from ._chol import chol
from ._mode_newton import mode_newton
from ._rng import as_rng


def armh(alpha, logf, gradK, c_reject=3, force_accept=False, start=None, tol=1e-4,
         max_iter_mode=100, max_iter_ar=1000, rng=None):
    """ssm.armh - one accept-reject Metropolis-Hastings update of a state vector, with a
    Gaussian proposal at the mode of the target: the algorithm of Chan (2017).

      alpha, accept, ntry = ssm.armh(alpha, logf, gradK)
      alpha, accept, ntry = ssm.armh(alpha, logf, gradK, c_reject=3, force_accept=False,
                                     rng=rng)

      alpha         : n-vector, the current state; the new state is returned
      logf          : function; logf(alpha) is the log target density, up to a constant
      gradK         : function; g, K = gradK(alpha) returns the gradient of logf and its
                      negative Hessian, sparse and banded (see ssm.mode_newton)
      c_reject      : envelope constant c = c_reject*f(alphahat)/g(alphahat) (default 3)
      force_accept  : take the candidate whatever the MH ratio (default False), for the
                      first sweeps of a chain; the uniform is drawn regardless
      start         : starting point of the mode search (default: the current state)
      tol, max_iter_mode : passed to ssm.mode_newton (defaults 1e-4 and 100). A tol above
                      1e-3 is refused: by default the search starts from the current state,
                      and a loose tolerance lets the proposal depend on it.
      max_iter_ar   : raise after this many rejected candidates (default 1000)
      rng           : None, an integer seed, a numpy.random.Generator or a ssm.Stream
      accept        : True if the MH step took the candidate
      ntry          : number of candidates drawn in the accept-reject step

    The proposal is g = N(alphahat, K^{-1}) at the mode alphahat. Candidates from g are kept
    with probability min{f/(c g), 1}, and the survivor passes an MH step. For a textbook
    discussion, see Chan (2027), Sections 6.2.3, 10.2.2 and 10.3.3. Written for this toolkit.

    See:
    Chan, J.C.C. (2027). Bayesian Macroeconometrics: Methods and Applications, Chapman &
    Hall/CRC, Sections 6.2.3, 10.2.2 and 10.3.3.
    Chan, J.C.C. (2017). The Stochastic Volatility in Mean Model with Time-Varying
    Parameters: An Application to Inflation Modeling, Journal of Business and Economic
    Statistics, 35(1): 17-28.
    """
    if not (np.isscalar(c_reject) and 0 < c_reject < np.inf):
        raise ValueError('c_reject must be a positive, finite scalar')
    if not (np.isscalar(tol) and tol > 0):
        raise ValueError('tol must be a positive scalar')
    for name, v in (('max_iter_mode', max_iter_mode), ('max_iter_ar', max_iter_ar)):
        if not (np.isscalar(v) and v >= 1 and v == int(v)):
            raise ValueError(f'{name} must be a positive integer')
    if tol > 1e-3:
        raise ValueError(f'tol = {tol:g} is looser than 1e-3. The mode search starts from the '
                         'current state, so a loose tolerance lets the proposal depend on it.')
    r = as_rng(rng)
    alpha = np.array(alpha, dtype=float).ravel()
    start = alpha if start is None else np.asarray(start, dtype=float).ravel()
    ahat, K, _ = mode_newton(start, gradK, tol, max_iter_mode)
    C = chol(K)
    logc = np.log(c_reject) + logf(ahat)        # log c; the kernel of g is 1 at ahat
    n = alpha.size

    # accept-reject step: screen candidates from g
    accepted = False
    it = 0
    while not accepted:
        it += 1
        if it > max_iter_ar:
            raise RuntimeError(f'no candidate accepted in {int(max_iter_ar)} draws at '
                               f'c_reject = {c_reject:g}')
        ac = ahat + C.solve_t(r.randn(n))
        dc = ac - ahat
        lac = logf(ac) + .5 * dc @ (K @ dc) - logc      # log f/(c g) at the candidate
        accepted = lac > np.log(r.rand())

    # MH step
    d = alpha - ahat
    la = logf(alpha) + .5 * d @ (K @ d) - logc          # log f/(c g) at the current state
    if la < 0:
        lalpha = 0.0
    elif lac < 0:
        lalpha = -la
    else:
        lalpha = lac - la
    accept = bool(lalpha > np.log(r.rand())) or bool(force_accept)
    if accept:
        alpha = ac
    return alpha, accept, it
