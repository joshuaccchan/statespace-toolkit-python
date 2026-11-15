import numpy as np
import scipy.sparse as sp


def diffmat(T, a=1.0):
    """ssm.diffmat - the sparse first-order difference matrix of a state equation,
    H = I_T - a*L, where L has ones on the first subdiagonal and zeros elsewhere (the lag operator
    on a stacked path).

      H = ssm.diffmat(T)        # a = 1: the random-walk difference matrix
      H = ssm.diffmat(T, a)

      T : path length, a positive integer
      a : scalar autoregressive coefficient (default 1)
      H : T x T sparse lower bidiagonal matrix (csc_array)

    For x_t = a*x_{t-1} + u_t with u ~ N(0, S), H @ x = u + a*x_0*e_1, so the path has precision
    H.T @ inv(S) @ H, which is banded. Sign conventions, both of which appear in the papers:
      AR(1) / random-walk state:  H_rho  = I - rho*L   ->  diffmat(T, rho)
      MA(1) error transform:      H_psi  = I + psi*L   ->  diffmat(T, -psi)

    Code-identical to bvar_toolkit.util.diffmat of the Python bvar-toolkit, and the two must stay
    so.

    See:
    Chan, J.C.C. (2027). Bayesian Macroeconometrics: Methods and Applications, Chapman &
    Hall/CRC, Section 9.1.1.
    """
    if not np.isscalar(T) or T < 1 or T != int(T):
        raise ValueError('T must be a positive integer')
    if not np.isscalar(a):
        raise ValueError('a must be a scalar')
    T = int(T)
    if T == 1:
        return sp.eye_array(1, format='csc')
    L = sp.csc_array((np.ones(T - 1), (np.arange(1, T), np.arange(T - 1))), shape=(T, T))
    return (sp.eye_array(T, format='csc') - a * L).tocsc()
