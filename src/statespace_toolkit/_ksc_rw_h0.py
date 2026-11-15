import numpy as np

from ._rng import as_rng
from ._ksc_ar1_mean import M_N, SIG2_N
from ._ksc_rw_diffuse import _draw, _indicators


def ksc_rw_h0(Ystar, h, sig, h0, rng=None):
    """ssm.ksc_rw_h0 - KSC auxiliary-mixture sampler for the log-volatility path,
    random-walk state equation with KNOWN initial value h0 (h_1 ~ N(h0,sig)):
      ystar_t = h_t + eps_t,  eps_t = log of a chi^2_1 draw, approximated by the
                Kim-Shephard-Chib (1998) 7-component normal mixture,
      h_t = h_{t-1} + v_t,    v_t ~ N(0,sig),   with sig a VARIANCE.

      h = ssm.ksc_rw_h0(Ystar, h, sig, h0)
      h = ssm.ksc_rw_h0(Ystar, h, sig, h0, rng)

      Ystar is a T-vector; h is the current path, and the new draw is returned; h0 is the known
      log-volatility at time 0. Only the path is returned: the mixture indicators are drawn
      internally and discarded. rng is None, an integer seed, a numpy.random.Generator or a
      ssm.Stream.

    The path is drawn in one block from its banded precision (Chan and Jeliazkov 2009).
    Code-identical to bvar_toolkit.sv.ksc_rw_h0 of the Python bvar-toolkit, and the two must stay
    so.

    See:
    Chan, J.C.C. (2027). Bayesian Macroeconometrics: Methods and Applications, Chapman &
    Hall/CRC, Section 10.1.1.
    Chan, J.C.C. and Jeliazkov, I. (2009). Efficient Simulation and Integrated Likelihood
    Estimation in State Space Models, International Journal of Mathematical Modelling and
    Numerical Optimisation, 1(1/2): 101-120.
    Kim, S., Shephard, N. and Chib, S. (1998). Stochastic Volatility: Likelihood Inference
    and Comparison with ARCH Models, Review of Economic Studies, 65(3): 361-393.
    """
    r = as_rng(rng)
    Ystar = np.asarray(Ystar, dtype=float).ravel()
    h = np.asarray(h, dtype=float).ravel()
    T = h.size
    S = _indicators(Ystar, h, r)
    # Given the indicators S: Ystar = h + d + e with e ~ N(0,Omega), where d_t and
    # Omega_tt are the mean and variance of mixture component S_t; and
    # Hh*h = [h0;0] + v with v ~ N(0,sig*I_T), so h has prior mean alph = h0 1_T
    # and precision Kh, and Kh @ alph is h0/sig in its first element and zero after
    dv = 1 / sig * np.ones(T)
    iOmega = 1 / SIG2_N[S]
    rhs = iOmega * (Ystar - M_N[S])
    rhs[0] = 1 / sig * h0 + rhs[0]
    return _draw(dv, iOmega, rhs, r)
