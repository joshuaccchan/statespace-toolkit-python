import numpy as np


def _days(a):
    """dates as float days: datetime64 values since 1970-01-01, numbers unchanged"""
    a = np.asarray(a)
    if np.issubdtype(a.dtype, np.datetime64):
        return (a - np.datetime64('1970-01-01')) / np.timedelta64(1, 'D')
    return a.astype(float)


def midas_basis(d, x, wstart, wend, scale='mean', release=None):
    """ssm.midas_basis - the regressors and restriction rows of one high-frequency predictor in
    the MIDAS regressions of Chan, Poon and Zhu (2026), with the Fourier weighting function,
    over the window of each low-frequency period. Written for this toolkit.

      W, A, m = ssm.midas_basis(d, x, wstart, wend)
      W, A, m = ssm.midas_basis(d, x, wstart, wend, scale='mean', release=r)

      d, x         : N-vectors of the dates and values of the predictor; NaN values are left
                     out
      wstart, wend : T-vectors of the ends of the windows; the window of period t is
                     (wstart[t], wend[t]]
      scale        : 'mean' (the default) or 'sum'; see below
      release      : N-vector of the release dates of the observations, default d; an
                     observation enters the window of period t only if it is released by
                     wend[t]
      W            : T x 5 array whose row t is the mean of v(s)*x over the window of period
                     t, the regressors w_t' of ssm.models.tvp_midas; with 'sum', the sum
      A            : T x 5 array whose row t is the mean of v(s) over that window, the
                     restriction rows a_t'; with 'sum', the sum
      m            : T-vector of the numbers of observations in the windows

    An observation dated d sits at s = (wend - d)/(wend - wstart), 0 at the end of its window
    and 1 at the start, and v(s) = [1, cos(2 pi s), sin(2 pi s), cos(4 pi s), sin(4 pi s)]' is
    the Fourier basis. The weights theta'v(s)/m[t] of period t then give the weighted average
    theta'W[t] and sum to A[t] @ theta. With 'sum', the scaling of Chan, Poon and Zhu (2026),
    the weights are theta'v(s), so theta is m[t] times smaller for the same weights, and the
    same prior variances on theta allow far larger weights. Dates are numpy datetime64
    values, or numbers measured in days. A window without an observation is an error.

    See:
    Chan, J.C.C., Poon, A. and Zhu, D. (2026). Time-Varying Parameter MIDAS Models:
    Application to Nowcasting US Real GDP, Journal of Econometrics, forthcoming.
    """
    scale = str(scale).lower()
    if scale not in ('sum', 'mean'):
        raise ValueError("scale must be 'sum' or 'mean'")
    d = _days(d).ravel()
    x = np.asarray(x, dtype=float).ravel()
    wstart = _days(wstart).ravel()
    wend = _days(wend).ravel()
    r = d if release is None else _days(release).ravel()
    if d.size != x.size or r.size != d.size:
        raise ValueError('d, x and the release dates must have the same number of entries')
    if wstart.size != wend.size:
        raise ValueError('wstart and wend must have the same number of entries')
    ok = ~np.isnan(x)
    T = wend.size
    W = np.zeros((T, 5))
    A = np.zeros((T, 5))
    m = np.zeros(T)
    for t in range(T):
        inside = ok & (d > wstart[t]) & (d <= wend[t]) & (r <= wend[t])
        if not inside.any():
            raise ValueError(f'the window of period {t} holds no observation')
        s = (wend[t] - d[inside]) / (wend[t] - wstart[t])
        V = np.column_stack([np.ones(s.size), np.cos(2 * np.pi * s), np.sin(2 * np.pi * s),
                             np.cos(4 * np.pi * s), np.sin(4 * np.pi * s)])
        if scale == 'sum':
            W[t] = np.sum(V * x[inside][:, None], axis=0)
            A[t] = np.sum(V, axis=0)
        else:
            W[t] = np.mean(V * x[inside][:, None], axis=0)
            A[t] = np.mean(V, axis=0)
        m[t] = np.count_nonzero(inside)
    return W, A, m
