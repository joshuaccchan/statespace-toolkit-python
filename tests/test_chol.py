"""The factor of the precision samplers: a banded K gives the band factor of ssm.BandChol
unchanged, and a K that is banded apart from a few dense trailing rows gives the Cholesky
factor of the whole matrix, held as a band and a dense border."""

import numpy as np
import pytest
import scipy.sparse as sp

import statespace_toolkit as ssm
from conftest import close
from statespace_toolkit._chol import BorderedChol, chol


def _arrow(n, kd, m, seed=0):
    """a symmetric positive definite K = A A' of order n whose leading block has bandwidth kd
    and whose last m rows and columns are dense: A is lower triangular, banded apart from its
    last m rows"""
    rng = np.random.default_rng(seed)
    A = np.diag(3 + rng.random(n))
    for d in range(1, kd + 1):
        A += np.diag(rng.standard_normal(n - d), -d)
    A[n - m:, :n - m] = rng.standard_normal((m, n - m))
    return A @ A.T


@pytest.mark.parametrize('fmt', ['csc', 'csr', 'coo', 'dense'])
@pytest.mark.parametrize('kd, m', [(1, 1), (2, 3), (4, 2)])
def test_bordered_factor(fmt, kd, m):
    """the factor, the solves and the log determinant equal those of the dense Cholesky factor"""
    n = 120
    Kd = _arrow(n, kd, m)
    K = Kd if fmt == 'dense' else sp.csc_array(Kd).asformat(fmt)
    C = chol(K)
    assert isinstance(C, BorderedChol) and C.m == m
    L = np.linalg.cholesky(Kd)
    close(C._dense(0, n, 0, n), L)
    rng = np.random.default_rng(1)
    b, B = rng.standard_normal(n), rng.standard_normal((n, 10))
    close(C.solve(b), np.linalg.solve(L, b))
    close(C.solve_t(B), np.linalg.solve(L.T, B))
    close(C.logdet(), np.linalg.slogdet(Kd)[1])


def test_band_unchanged():
    """a banded K gives the band factor of ssm.BandChol, bit for bit, in every format"""
    H = ssm.diffmat(300)
    K = (H.T @ H * 20 + sp.eye_array(300)).tocsc()
    ref = ssm.BandChol(K).ab
    for KK in (K, K.tocsr(), K.tocoo(), K.todia(), K.toarray()):
        C = chol(KK)
        assert type(C) is ssm.BandChol
        assert np.array_equal(C.ab, ref)
    D = sp.eye_array(50) * 3.0                              # a diagonal K from sp.eye_array
    C = chol(D)
    assert C.ab.shape == (1, 50)
    close(C.logdet(), 50 * np.log(3.0))


def test_duplicates_and_explicit_zeros():
    """duplicate entries are summed, and explicit zeros are dropped, as MATLAB's sparse
    matrices never store them, so a stored zero far below the diagonal does not widen the band"""
    rows, cols = np.r_[np.arange(6), 1], np.r_[np.arange(6), 1]
    K = sp.coo_array((np.r_[np.full(6, 2.0), 1.0], (rows, cols)), shape=(6, 6))
    close(chol(K).logdet(), np.log(2.0**5 * 3.0))
    Kz = sp.csc_array((np.r_[np.ones(6), 0.0], (np.r_[np.arange(6), 5], np.r_[np.arange(6), 0])),
                      shape=(6, 6))
    assert ssm.BandChol(Kz).ab.shape[0] == 6                # the stored zero widens its band
    C = chol(Kz)
    assert type(C) is ssm.BandChol and C.ab.shape[0] == 1


def test_simulate_states_and_intlike_with_a_constant_coefficient():
    """a TVP regression with a constant slope last among the states: the draws, the mean and
    the integrated likelihood equal those from dense algebra"""
    rng = np.random.default_rng(3)
    T = 150
    X = np.column_stack([np.ones(T), rng.standard_normal(T)])
    y = rng.standard_normal(T)
    G = ssm.diffmat(T + 1)
    Z = sp.hstack([sp.csc_array((T, 1)), sp.eye_array(T), sp.csc_array(X[:, [1]])], format='csc')
    P = sp.block_diag([G.T @ sp.diags_array(np.r_[.01, np.full(T, 200.0)]) @ G,
                       sp.csc_array([[.01]])], format='csc')
    b = np.zeros(T + 2)
    K = P + Z.T @ Z
    c = Z.T @ y
    r = ssm.Replay(7)
    alpha, alphahat, C = ssm.simulate_states(K, c, 3, rng=r)
    assert isinstance(C, BorderedChol)
    Kd = K.toarray()
    L = np.linalg.cholesky(Kd)
    close(alphahat, np.linalg.solve(Kd, c))
    close(alpha - alphahat[:, None], np.linalg.solve(L.T, ssm.Replay(7).randn(T + 2, 3)))
    ll, ahat, _ = ssm.intlike(y, Z, sp.eye_array(T), P, b)
    V = np.linalg.inv(P.toarray())
    S = Z.toarray() @ V @ Z.toarray().T + np.eye(T)
    exact = -.5 * (T * np.log(2 * np.pi) + np.linalg.slogdet(S)[1] + y @ np.linalg.solve(S, y))
    assert abs(ll - exact) < 1e-9 * abs(exact)
    x, _, _ = ssm.restrict(alphahat, C, sp.csr_array(np.r_[np.zeros(T + 1), 1.0][None, :]),
                           [.5])
    assert abs(x[-1] - .5) < 1e-12
