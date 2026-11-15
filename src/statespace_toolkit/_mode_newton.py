import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve


def _step(K, g):
    """K^{-1} g for a sparse or dense square K; NaN where the solve fails"""
    try:
        if sp.issparse(K):
            return np.asarray(spsolve(sp.csc_array(K), g), dtype=float).ravel()
        return np.linalg.solve(np.asarray(K, dtype=float), g)
    except (np.linalg.LinAlgError, RuntimeError):
        return np.full(g.size, np.nan)


def mode_newton(alpha0, gradK, tol=1e-4, max_iter=100):
    """ssm.mode_newton - the mode of a concave log density by Newton-Raphson, each step solved
    with the banded negative Hessian. See Chan (2027), Section 10.2.2.

      alphahat, K, iter = ssm.mode_newton(alpha0, gradK)
      alphahat, K, iter = ssm.mode_newton(alpha0, gradK, tol, max_iter)

      alpha0   : n-vector, the starting point
      gradK    : function; g, K = gradK(alpha) returns the gradient g of the log density at
                 alpha and its negative Hessian K, n x n, sparse and banded
      tol      : stop once max|step| <= tol (default 1e-4)
      max_iter : raise if that has not happened after this many steps (default 100)
      alphahat : the mode
      K        : the negative Hessian of the last step, evaluated at the iterate before
                 alphahat
      iter     : number of steps taken

    Each step is alpha + K^{-1} g. A NaN step runs to the cap and raises. Written for this
    toolkit.

    See:
    Chan, J.C.C. (2027). Bayesian Macroeconometrics: Methods and Applications, Chapman &
    Hall/CRC, Sections 10.2.2 and 10.3.3.
    """
    if not (np.isscalar(tol) and tol > 0):
        raise ValueError('tol must be a positive scalar')
    if not (np.isscalar(max_iter) and max_iter >= 1 and max_iter == int(max_iter)):
        raise ValueError('max_iter must be a positive integer')
    alphahat = np.array(alpha0, dtype=float).ravel()
    err = np.inf
    it = 0
    K = None
    while not err <= tol:
        if it >= max_iter:
            raise RuntimeError(f'no convergence to tol = {tol:g} in {int(max_iter)} steps '
                               f'(last max|step| = {err:g})')
        it += 1
        g, K = gradK(alphahat)
        alphanew = alphahat + _step(K, np.asarray(g, dtype=float).ravel())
        err = np.max(np.abs(alphanew - alphahat))
        alphahat = alphanew
    return alphahat, K, it
