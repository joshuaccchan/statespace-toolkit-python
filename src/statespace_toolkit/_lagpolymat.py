import numpy as np
import scipy.sparse as sp


def lagpolymat(T, phi):
    """ssm.lagpolymat - the sparse matrix of the lag polynomial 1 - phi_1*L - ... - phi_p*L^p
    applied to a stacked path, where L is the lag operator.

      H = ssm.lagpolymat(T, phi)

      T   : path length, a positive integer
      phi : vector of lag coefficients (phi_1, ..., phi_p); empty gives the identity
      H   : T x T sparse lower-banded matrix (csc_array), 1 on the diagonal and -phi_j on the
            jth subdiagonal

    For x_t = phi_1*x_{t-1} + ... + phi_p*x_{t-p} + u_t with the pre-sample values set to zero,
    H @ x = u; nonzero pre-sample values enter through a vector the caller adds. Second
    differences are phi = [2, -1], and for a scalar a, ssm.lagpolymat(T, a) equals
    ssm.diffmat(T, a). Written for this toolkit.

    See:
    Chan, J.C.C. (2027). Bayesian Macroeconometrics: Methods and Applications, Chapman &
    Hall/CRC, Section 9.1.3.
    """
    if not np.isscalar(T) or T < 1 or T != int(T):
        raise ValueError('T must be a positive integer')
    try:
        phi = np.asarray(phi, dtype=float)
    except (TypeError, ValueError):
        raise ValueError('phi must be a vector of lag coefficients') from None
    if phi.ndim > 1 and phi.size != max(phi.shape):
        raise ValueError('phi must be a vector of lag coefficients')
    phi = phi.ravel()
    T = int(T)
    H = sp.eye_array(T, format='csc')
    for j in range(1, min(phi.size, T - 1) + 1):
        L = sp.csc_array((np.ones(T - j), (np.arange(j, T), np.arange(T - j))), shape=(T, T))
        H = H - phi[j - 1] * L
    return H.tocsc()
