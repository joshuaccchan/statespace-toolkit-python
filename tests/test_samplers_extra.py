"""Checks that do not rest on the MATLAB twins. (1) Invariance: one update of ssm.armh or
ssm.nu_studentt, applied to exact draws from its target, leaves their distribution unchanged.
(2) Draws far in the tails: ssm.tnormrnd where both normal cdf values round to 1, and the KSC
samplers when every mixture density of some periods underflows. (3) The inputs of
ssm.simulate_states."""

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.special import gammaln
from scipy.stats import norm

import statespace_toolkit as ssm


def _exact_draws(logf, lo, hi, n, rng, m=200001):
    """inverse-cdf draws from the density proportional to exp(logf) on (lo, hi), on a fine grid"""
    x = np.linspace(lo, hi, m)
    lf = logf(x)
    w = np.exp(lf - lf.max())
    cdf = np.cumsum(w)
    cdf /= cdf[-1]
    return np.interp(rng.random(n), cdf, x)


def test_armh_invariance():
    """a smooth, strictly concave target: the probabilities of two intervals stay within four
    standard errors after one update"""
    rng = np.random.default_rng(0)
    n = 4000

    def logf(a):
        return float(-a[0]**4 / 4 - a[0]**2 / 2)

    def gradK(a):
        return np.array([-a[0]**3 - a[0]]), sp.csc_array([[3 * a[0]**2 + 1]])

    a0 = _exact_draws(lambda x: -x**4 / 4 - x**2 / 2, -6, 6, n, rng)
    a1 = np.array([ssm.armh(np.array([v]), logf, gradK, rng=rng)[0][0] for v in a0])
    for c in (.5, 1.0):
        p0, p1 = np.mean(np.abs(a0) < c), np.mean(np.abs(a1) < c)
        assert abs(p1 - p0) < 4 * np.sqrt(2 * p0 * (1 - p0) / n)


def test_nu_studentt_invariance():
    """1,000 latent scales from IG(5/2, 5/2): the mean and standard deviation of the draws of nu
    stay within four standard errors after one update"""
    rng = np.random.default_rng(1)
    lam = 1 / rng.gamma(2.5, 1 / 2.5, 1000)
    T, s1, s2 = lam.size, np.sum(np.log(lam)), np.sum(1 / lam)

    def f(x):
        return T * (x / 2 * np.log(x / 2) - gammaln(x / 2)) - (x / 2 + 1) * s1 - x / 2 * s2

    n = 4000
    nu0 = _exact_draws(f, 2, 100, n, rng)
    nu1 = np.array([ssm.nu_studentt(lam, v, 100, rng=rng)[0] for v in nu0])
    d = nu1 - nu0
    assert abs(d.mean()) < 4 * d.std(ddof=1) / np.sqrt(n)
    assert abs(nu1.std() / nu0.std() - 1) < 4 * np.sqrt(2 / n)


def test_tnormrnd_tails():
    """on (9, 10) and (-10, -9) the draws are finite, lie in the bounds and have the mean of the
    truncated normal"""
    N = 2000
    m = (norm.pdf(9) - norm.pdf(10)) / (norm.sf(9) - norm.sf(10))   # the mean of N(0,1) on (9, 10)
    t = ssm.tnormrnd(0, 1, 9, 10, N, rng=5)
    assert np.all(np.isfinite(t) & (t >= 9) & (t <= 10))
    assert abs(t.mean() - m) < 4 * t.std() / np.sqrt(N)
    t = ssm.tnormrnd(0, 1, -10, -9, N, rng=5)
    assert np.all(np.isfinite(t) & (t >= -10) & (t <= -9))
    assert abs(t.mean() + m) < 4 * t.std() / np.sqrt(N)


def test_ksc_underflow():
    """the draws stay finite, and the periods whose densities all underflow take the widest
    component, whose density is largest there by thousands of log points"""
    r = np.random.default_rng(3)
    T = 60
    ystar = np.log(r.standard_normal(T)**2 + 1e-4)
    h = np.zeros(T)
    h[9:12] = 2000.0                     # every density underflows in periods 10 to 12
    h1, S1 = ssm.ksc_ar1_mean(ystar, h, 0, .9, .1, rng=1)
    h2, S2 = ssm.ksc_rw_diffuse(ystar, h, .1, 10, rng=1)
    h3 = ssm.ksc_rw_h0(ystar, h, .1, 0, rng=1)
    h4, h0, om = ssm.ksc_rw_noncentered(ystar, h / .4, 0, .4, 0, 10, 10, .2, rng=1)[:3]
    assert np.all(np.isfinite(np.r_[h1, h2, h3, h4, h0, om]))
    assert np.all(np.isin(np.r_[S1, S2], np.arange(7)))
    assert np.all(S1[9:12] == 0) and np.all(S2[9:12] == 0)


def test_simulate_states_inputs():
    """a K that is not square, or a c whose length is not the order of K, raises ValueError;
    zero draws give an empty n x 0 array, also over repeated calls"""
    K = sp.diags_array([np.full(4, -1.0), np.full(5, 3.0), np.full(4, -1.0)], offsets=[-1, 0, 1])
    with pytest.raises(ValueError):
        ssm.simulate_states(2 * np.eye(2), [8, 12, 17], 2, rng=1)
    with pytest.raises(ValueError):
        ssm.simulate_states(np.ones((2, 3)), [1, 2], rng=1)
    with pytest.raises(ValueError):
        ssm.simulate_states(K, np.ones(5), -1, rng=1)
    for _ in range(50):
        alpha, alphahat, _ = ssm.simulate_states(K.tocsc(), np.ones(5), 0, rng=1)
        assert alpha.shape == (5, 0) and alphahat.shape == (5,)
