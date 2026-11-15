import numpy as np
import scipy.sparse as sp


def surform(X):
    """ssm.surform - T x Tk block-diagonal sparse SUR expansion, the design matrix of a
    regression whose coefficients are stacked over time.

      Xout = ssm.surform(X)

      X    : T x k matrix of regressors
      Xout : T x T*k sparse matrix (csc_array) whose row t holds X[t] in columns t*k to
             t*k + k - 1, so that Xout @ [b_1; b_2; ...; b_T] returns the fitted series of the
             TVP regression y_t = X[t] @ b_t

    Xout is the matrix Z = diag(x_1', ..., x_T') of the book's TVP regression. Code-identical to
    bvar_toolkit.util.surform of the Python bvar-toolkit, and the two must stay so.

    See:
    Chan, J.C.C. (2027). Bayesian Macroeconometrics: Methods and Applications, Chapman &
    Hall/CRC, Section 9.3.
    """
    X = np.asarray(X, dtype=float)
    r, c = X.shape
    Xout = sp.csc_array((X.reshape(-1), (np.repeat(np.arange(r), c), np.arange(r * c))),
                        shape=(r, r * c))
    Xout.eliminate_zeros()
    return Xout
