"""The core functions against their MATLAB twins, on the cases of tests/fixtures/core.mat that
tools/matlab/export_fixtures.m exports, with the random numbers replayed; then checks that need
no MATLAB: dense algebra, the Kalman filter, and the drawing of a band."""

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.signal import lfilter

import statespace_toolkit as ssm
from conftest import cases, check, close, dense, fixtures, grid, inputs, outputs, run

FX = fixtures('core')


def _int_arg(i):
    """an adapt function that turns argument i, a double in the MAT-file, into an integer"""
    def adapt(args, kwargs):
        if len(args) > i:
            args[i] = int(args[i])
        return args, kwargs
    return adapt


@pytest.mark.parametrize('case', cases(FX, 'diffmat'))
def test_diffmat(case):
    check(ssm.diffmat, case)


@pytest.mark.parametrize('case', cases(FX, 'lagpolymat'))
def test_lagpolymat(case):
    check(ssm.lagpolymat, case)


@pytest.mark.parametrize('case', cases(FX, 'surform'))
def test_surform(case):
    check(ssm.surform, case)


@pytest.mark.parametrize('case', cases(FX, 'select_obs'))
def test_select_obs(case):
    check(ssm.select_obs, case)


@pytest.mark.parametrize('case', cases(FX, 'tnormrnd'))
def test_tnormrnd(case):
    check(ssm.tnormrnd, case, _int_arg(4))


@pytest.mark.parametrize('case', cases(FX, 'specvar0'))
def test_specvar0(case):
    check(ssm.specvar0, case)


@pytest.mark.parametrize('name', ['inefficiency_factor', 'mcse'])
def test_inefficiency_factor_mcse(name):
    for case in cases(FX, name):
        check(getattr(ssm, name), case)


@pytest.mark.parametrize('case', cases(FX, 'geweke'))
def test_geweke(case):
    Z, pval, info = run(ssm.geweke, case, _int_arg(3))
    Zm, pm, im = outputs(case)
    close(Z, Zm)
    close(pval, pm)
    close(info['A'], np.asarray(im['A']) - 1)       # row indices count from 0 here
    close(info['B'], np.asarray(im['B']) - 1)
    assert (info['LA'], info['LB'], info['a'], info['b']) == (im['LA'], im['LB'], im['a'], im['b'])


@pytest.mark.parametrize('case', cases(FX, 'simulate_states'))
def test_simulate_states(case):
    alpha, alphahat, C = run(ssm.simulate_states, case, _int_arg(2))
    a_m, ahat_m, C_m = outputs(case)
    close(alpha, a_m)
    close(alphahat, ahat_m)
    n = alphahat.size
    close(C._dense(0, n, 0, n), C_m)                # MATLAB's chol(K, 'lower')


@pytest.mark.parametrize('case', cases(FX, 'intlike'))
def test_intlike(case):
    check(ssm.intlike, case)


@pytest.mark.parametrize('case', cases(FX, 'restrict'))
def test_restrict(case):
    x, _, M, z = inputs(case)
    out = ssm.restrict(x, ssm.BandChol(case['K']), M, z)
    for o, e in zip(out, outputs(case)):
        close(o, e)


@pytest.mark.parametrize('case', cases(FX, 'ksc_ar1_mean') + cases(FX, 'ksc_rw_diffuse'),
                         ids=lambda c: f'seed{c["seed"]}')
def test_ksc_with_indicators(case):
    fn = ssm.ksc_ar1_mean if case['nin'] == 5 else ssm.ksc_rw_diffuse
    h, S = run(fn, case)
    h_m, S_m = outputs(case)
    close(h, h_m)
    close(S + 1, S_m)                               # the mixture indicators count from 0 here


@pytest.mark.parametrize('case', cases(FX, 'ksc_rw_h0'))
def test_ksc_rw_h0(case):
    check(ssm.ksc_rw_h0, case)


@pytest.mark.parametrize('case', cases(FX, 'sv_params'))
def test_sv_params(case):
    check(ssm.sv_params, case)


@pytest.mark.parametrize('case', cases(FX, 'nu_studentt'))
def test_nu_studentt(case):
    check(grid(ssm.nu_studentt), case)


# ---- checks that need no MATLAB ----

def _local_level(T=200, sig2=1.0, omega2=.05, tau0=2.0, seed=3):
    rng = np.random.default_rng(seed)
    tau = tau0 + np.cumsum(np.sqrt(omega2) * rng.standard_normal(T))
    y = tau + np.sqrt(sig2) * rng.standard_normal(T)
    H = ssm.diffmat(T)
    e1 = np.zeros(T)
    e1[0] = tau0
    K = H.T @ H / omega2 + sp.eye_array(T) / sig2
    c = H.T @ e1 / omega2 + y / sig2
    return y, H, K, c


def test_simulate_states_algebra():
    """the mean solves K alphahat = c, C C' = K, and the draws are alphahat + C'^{-1} z for the
    normals z of the stream, filled column by column"""
    y, H, K, c = _local_level()
    T = y.size
    r = ssm.Replay(5)
    alpha, alphahat, C = ssm.simulate_states(K, c, 3, rng=r)
    Kd = K.toarray()
    close(alphahat, np.linalg.solve(Kd, c))
    Cd = C._dense(0, T, 0, T)
    close(Cd @ Cd.T, Kd)
    z = ssm.Replay(5).randn(T, 3)
    close(Cd.T @ (alpha - alphahat[:, None]), z)
    assert r.count == 3 * T
    a1, _, _ = ssm.simulate_states(K, c, rng=ssm.Replay(5))
    close(a1, alpha[:, 0])                          # one draw is the first column of several


def test_simulate_states_moments():
    """20,000 draws have the mean and the covariance K^{-1}, within Monte Carlo error"""
    y, H, K, c = _local_level(T=30)
    alpha, alphahat, _ = ssm.simulate_states(K, c, 20000, rng=11)
    V = np.linalg.inv(K.toarray())
    sd = np.sqrt(np.diag(V))
    assert np.max(np.abs(alpha.mean(axis=1) - alphahat) / sd) < 5 / np.sqrt(20000) * 1.5
    assert np.max(np.abs(np.cov(alpha) - V)) < .05 * np.max(V)


def test_intlike_kalman():
    """the integrated likelihood of the local level model equals the prediction-error
    decomposition of the Kalman filter"""
    sig2, omega2, tau0 = 1.0, .05, 2.0
    y, H, K, c = _local_level(sig2=sig2, omega2=omega2, tau0=tau0)
    T = y.size
    ll, alphahat, K2 = ssm.intlike(y, sp.eye_array(T), sp.eye_array(T) / sig2, H.T @ H / omega2,
                                   tau0 * np.ones(T))
    m, P, ll_kf = tau0, 0.0, 0.0
    for t in range(T):
        P = P + omega2                              # predict tau_t
        F = P + sig2
        v = y[t] - m
        ll_kf += -.5 * (np.log(2 * np.pi * F) + v**2 / F)
        m, P = m + P / F * v, P - P**2 / F
    assert abs(ll - ll_kf) < 1e-9 * abs(ll_kf)
    close(alphahat, np.linalg.solve(K.toarray(), c))
    close(K2, K)


def test_intlike_bad_iR():
    with pytest.raises(ValueError, match='iR must be 3 x 3'):
        ssm.intlike(np.zeros(3), sp.eye_array(3), sp.eye_array(2), sp.eye_array(3), np.zeros(3))


def test_restrict_dense():
    """the restricted mean is the mean of the Gaussian conditional, and every restricted draw
    satisfies M x = z"""
    y, H, K, c = _local_level(T=40)
    T = y.size
    alpha, alphahat, C = ssm.simulate_states(K, c, 5, rng=2)
    M = sp.csr_array(([1., 1., 1., -1.], ([0, 0, 1, 1], [4, 5, 19, 29])), shape=(2, T))
    z = np.array([.5, 0.])
    x, U, MU = ssm.restrict(alpha, C, M, z)
    close(M @ x, np.tile(z[:, None], (1, 5)))
    V = np.linalg.inv(K.toarray())
    Md = M.toarray()
    mu = alphahat + V @ Md.T @ np.linalg.solve(Md @ V @ Md.T, z - Md @ alphahat)
    xm, _, _ = ssm.restrict(alphahat, C, M, z)
    close(xm, mu)
    close(U, V @ Md.T)
    close(MU, Md @ V @ Md.T)


def test_lagpolymat_algebra():
    """an AR(1) polynomial is ssm.diffmat, H @ x filters x, lags beyond the path drop out, and
    malformed input raises"""
    T = 9
    close(ssm.lagpolymat(T, .7), ssm.diffmat(T, .7))
    close(ssm.lagpolymat(T, [1.0]), ssm.diffmat(T))
    x = np.random.default_rng(4).standard_normal(T)
    phi = [.5, -.3, .2]
    close(ssm.lagpolymat(T, phi) @ x, lfilter(np.r_[1, -np.array(phi)], 1, x))
    close(ssm.lagpolymat(3, [.1, .2, .3, .4]), ssm.lagpolymat(3, [.1, .2]))
    close(ssm.lagpolymat(4, []), np.eye(4))
    for T_bad in (0, 2.5, [3]):
        with pytest.raises(ValueError, match='T must be a positive integer'):
            ssm.lagpolymat(T_bad, .5)
    with pytest.raises(ValueError, match='phi must be a vector'):
        ssm.lagpolymat(5, np.ones((2, 2)))
    with pytest.raises(ValueError, match='phi must be a vector'):
        ssm.lagpolymat(5, 'abc')


def test_shaded_band():
    """the band runs along lo and back along hi, in the gray level given"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    x = np.arange(5.0)
    h = ssm.shaded_band(x, x - 1, x + 2, .7, ax)
    verts = {tuple(v) for v in h.get_paths()[0].vertices}
    assert {(a, a - 1) for a in x} <= verts and {(a, a + 2) for a in x} <= verts
    close(h.get_facecolor()[0][:3], [.7, .7, .7])
    plt.close(fig)
