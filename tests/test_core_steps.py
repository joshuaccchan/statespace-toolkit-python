"""The sampler steps, the integrated likelihoods of the SV models and the MIDAS basis against
their MATLAB twins, on the cases of tests/fixtures/core.mat, with the random numbers replayed;
then checks that need no MATLAB."""

import numpy as np
import pytest
import scipy.sparse as sp
from scipy import integrate
from scipy.special import gammaln

import statespace_toolkit as ssm
from conftest import cases, check, close, fixtures, inputs, outputs, run

FX = fixtures('core')


def sv_target(s2, muh, phih, omegah2):
    """the log density of the log-volatilities h of an SV model given the squared data s2, up
    to a constant, with its gradient and negative Hessian (sv_target of export_fixtures.m)"""
    s2 = np.asarray(s2, dtype=float).ravel()
    T = s2.size
    Hphi = (sp.eye_array(T) - phih * sp.eye_array(T, k=-1)).tocsc()
    HiSH = Hphi.T @ sp.diags_array(np.r_[(1 - phih**2) / omegah2, np.ones(T - 1) / omegah2]) @ Hphi

    def logf(h):
        return -.5 * np.sum(h) - .5 * np.sum(s2 * np.exp(-h)) - .5 * (h - muh) @ (HiSH @ (h - muh))

    def gradK(h):
        return (-.5 + .5 * s2 * np.exp(-h) - HiSH @ (h - muh),
                HiSH + sp.diags_array(.5 * s2 * np.exp(-h)))
    return logf, gradK


def _options(args, kwargs):
    """MATLAB's name-value options under their Python names, and M as an integer"""
    names = {'Tol': 'tol', 'MaxIter': 'max_iter'}
    kwargs = {names.get(k, k): v for k, v in kwargs.items()}
    args[-1] = int(args[-1])
    return args, kwargs


@pytest.mark.parametrize('case', cases(FX, 'mode_newton'))
def test_mode_newton(case):
    alpha0, s2, muh, phih, omegah2, tol = inputs(case)
    _, gradK = sv_target(s2, muh, phih, omegah2)
    out = ssm.mode_newton(alpha0, gradK, tol)
    for o, e in zip(out, outputs(case)):
        close(o, e)


@pytest.mark.parametrize('case', cases(FX, 'armh'), ids=lambda c: f'seed{c["seed"]}')
def test_armh(case):
    alpha, s2, muh, phih, omegah2, c_reject, force, start = inputs(case)
    logf, gradK = sv_target(s2, muh, phih, omegah2)
    r = ssm.Replay(case['seed'])
    out = ssm.armh(alpha, logf, gradK, c_reject=c_reject, force_accept=bool(force),
                   start=None if np.size(start) == 0 else start, rng=r)
    assert r.count == case['count']
    for o, e in zip(out, outputs(case)):
        close(o, e)


@pytest.mark.parametrize('case', cases(FX, 'ksc_rw_noncentered'))
def test_ksc_rw_noncentered(case):
    check(ssm.ksc_rw_noncentered, case)


@pytest.mark.parametrize('name', ['intlike_sv', 'intlike_sv_t'])
def test_intlike_sv(name):
    for case in cases(FX, name):
        check(getattr(ssm, name), case, _options)


@pytest.mark.parametrize('case', cases(FX, 'ma_params'), ids=lambda c: f'seed{c["seed"]}')
def test_ma_params(case):
    check(ssm.ma_params, case)


@pytest.mark.parametrize('case', cases(FX, 'arma_params'), ids=lambda c: f'seed{c["seed"]}')
def test_arma_params(case):
    check(ssm.arma_params, case)


@pytest.mark.parametrize('case', cases(FX, 'midas_basis'))
def test_midas_basis(case):
    check(ssm.midas_basis, case)


# ---- checks that need no MATLAB ----

def test_mode_newton_gaussian_and_errors():
    """one step reaches the mean of a Gaussian; a NaN step runs to the cap and raises"""
    rng = np.random.default_rng(0)
    A = rng.standard_normal((4, 4))
    K = A @ A.T + 4 * np.eye(4)
    c = rng.standard_normal(4)
    ahat, K2, it = ssm.mode_newton(np.zeros(4), lambda a: (c - K @ a, K))
    close(ahat, np.linalg.solve(K, c))
    assert it == 2                                  # the second step has length zero
    # a Hessian twice too large halves each step, so the number of steps follows tol
    _, _, it = ssm.mode_newton(np.zeros(1), lambda a: (1 - a, np.array([[2.0]])), tol=1e-4)
    assert it == 14
    with pytest.raises(RuntimeError, match='no convergence to tol'):
        ssm.mode_newton(np.zeros(4), lambda a: (np.full(4, np.nan), K), max_iter=5)
    with pytest.raises(RuntimeError, match='in 1 steps'):
        ssm.mode_newton(np.zeros(4), lambda a: (c - K @ a, K), max_iter=1)
    with pytest.raises(ValueError, match='tol must be'):
        ssm.mode_newton(np.zeros(4), lambda a: (c, K), tol=0)


def test_armh_gaussian_target():
    """on a Gaussian target the proposal is exact: every survivor is accepted, the draws have
    the target's moments, and candidates are kept with probability 1/c_reject"""
    rng = np.random.default_rng(1)
    A = rng.standard_normal((3, 3))
    K = A @ A.T + 3 * np.eye(3)
    mu = np.array([1., -1., .5])
    logf = lambda a: -.5 * (a - mu) @ K @ (a - mu)                     # noqa: E731
    gradK = lambda a: (-K @ (a - mu), K)                               # noqa: E731
    x, draws, tries = np.zeros(3), [], 0
    for _ in range(4000):
        x, acc, n = ssm.armh(x, logf, gradK, rng=rng)
        assert acc
        draws.append(x)
        tries += n
    draws = np.array(draws)
    V = np.linalg.inv(K)
    assert np.max(np.abs(draws.mean(axis=0) - mu) / np.sqrt(np.diag(V))) < .1
    assert np.max(np.abs(np.cov(draws.T) - V)) < .1 * np.max(V)
    assert abs(4000 / tries - 1 / 3) < .03
    with pytest.raises(ValueError, match='looser than 1e-3'):
        ssm.armh(x, logf, gradK, tol=.01)


def _sv_loglik_quadrature(y, muh, phih, omegah2, nu=None):
    """log p(y) of an SV model with T = 2 by quadrature over (h_1, h_2)"""
    var1 = omegah2 / (1 - phih**2)

    def dens(h2, h1):
        if nu is None:
            py = np.exp(-.5 * (np.log(2 * np.pi) + h1 + y[0]**2 / np.exp(h1))
                        - .5 * (np.log(2 * np.pi) + h2 + y[1]**2 / np.exp(h2)))
        else:
            lc = gammaln((nu + 1) / 2) - gammaln(nu / 2) - .5 * np.log(nu * np.pi)
            lt = sum(lc - .5 * h - (nu + 1) / 2 * np.log1p(yy**2 / (nu * np.exp(h)))
                     for yy, h in ((y[0], h1), (y[1], h2)))
            py = np.exp(lt)
        p1 = np.exp(-.5 * (h1 - muh)**2 / var1) / np.sqrt(2 * np.pi * var1)
        m2 = muh + phih * (h1 - muh)
        p2 = np.exp(-.5 * (h2 - m2)**2 / omegah2) / np.sqrt(2 * np.pi * omegah2)
        return py * p1 * p2
    val, _ = integrate.dblquad(dens, -12, 12, -12, 12, epsabs=1e-13, epsrel=1e-10)
    return np.log(val)


@pytest.mark.parametrize('nu', [None, 6.0])
def test_intlike_sv_quadrature(nu):
    """with T = 2, the importance sampling estimate lies within four of its standard errors
    of the exact log likelihood"""
    y = np.array([.8, -1.9])
    muh, phih, omegah2 = .2, .8, .3
    if nu is None:
        ll, nse = ssm.intlike_sv(y**2, muh, phih, omegah2, np.zeros(2), 20000, rng=3)
    else:
        ll, nse = ssm.intlike_sv_t(y**2, muh, phih, omegah2, nu, np.zeros(2), 20000, rng=3)
    exact = _sv_loglik_quadrature(y, muh, phih, omegah2, nu)
    assert abs(ll - exact) < 4 * nse + 1e-6


def test_intlike_sv_input_errors():
    s2 = np.ones(5)
    for args, msg in (((-s2, 0, .9, .1, np.zeros(5), 10), 's2 must be'),
                      ((s2, np.nan, .9, .1, np.zeros(5), 10), 'muh must be'),
                      ((s2, 0, 1.0, .1, np.zeros(5), 10), 'phih must be'),
                      ((s2, 0, .9, -.1, np.zeros(5), 10), 'omegah2 must be'),
                      ((s2, 0, .9, .1, np.zeros(4), 10), 'ht must be'),
                      ((s2, 0, .9, .1, np.zeros(5), 2.5), 'M must be')):
        with pytest.raises(ValueError, match=msg):
            ssm.intlike_sv(*args)
    with pytest.raises(ValueError, match='nu must be'):
        ssm.intlike_sv_t(s2, 0, .9, .1, 0, np.zeros(5), 10)
    with pytest.raises(RuntimeError, match='no convergence'):           # zeros is not the mode
        ssm.intlike_sv(4 * s2, 0, .9, .1, np.zeros(5), 10, tol=1e-12, max_iter=1)


def test_ma_params_conditional():
    """a chain of the step alone, with x and w fixed, has the mean and standard deviation of
    the conditional density of psi computed on a grid"""
    rng = np.random.default_rng(5)
    T = 80
    u = rng.standard_normal(T)
    x = u + .4 * np.r_[0, u[:-1]]
    w = np.ones(T)
    Hy = {'psi0': 0.0, 'Vpsi': 1.0}
    grid = np.linspace(-.999, .999, 4001)
    from statespace_toolkit._ma_params import _logcond
    lg = _logcond(grid, x, w, 0.0, 1.0)
    p = np.exp(lg - lg.max())
    p /= p.sum()
    m = p @ grid
    sd = np.sqrt(p @ (grid - m)**2)
    psi, draws = .4, []
    for _ in range(3000):
        psi, _, _ = ssm.ma_params(x, w, psi, Hy, rng=rng)
        draws.append(psi)
    draws = np.array(draws)
    assert abs(draws.mean() - m) < .1 * sd
    assert abs(draws.std() / sd - 1) < .1


def test_midas_basis_dates_and_errors():
    """datetime64 dates give the same rows as dates in days; an empty window and a bad option
    raise"""
    d = np.arange(7, 7 * 40 + 1, 7).astype(float)
    x = np.random.default_rng(2).standard_normal(40)
    ws, we = np.array([0., 91., 182.]), np.array([91., 182., 273.])
    epoch = np.datetime64('2000-01-01')
    as_dates = lambda a: epoch + a.astype(int).astype('timedelta64[D]')   # noqa: E731
    W1, A1, m1 = ssm.midas_basis(d, x, ws, we)
    W2, A2, m2 = ssm.midas_basis(as_dates(d), x, as_dates(ws), as_dates(we))
    close(W2, W1)
    close(A2, A1)
    close(m2, m1)
    with pytest.raises(ValueError, match='holds no observation'):
        ssm.midas_basis(d, x, np.array([300.]), np.array([310.]))
    with pytest.raises(ValueError, match="scale must be"):
        ssm.midas_basis(d, x, ws, we, scale='median')
