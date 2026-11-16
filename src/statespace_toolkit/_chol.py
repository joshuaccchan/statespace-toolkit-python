"""The lower Cholesky factor of a precision matrix K: in band storage when K is banded, and with
its last rows dense when the last states enter every period, as constant coefficients do."""

import numpy as np
import scipy.sparse as sp
from scipy.linalg import solve_triangular

from ._linalg import BandChol

_MAX_BORDER = 32          # trailing rows considered for dense storage


class BorderedChol:
    """The lower Cholesky factor C = [[CA, 0], [W', CD]] of K = [[A, B], [B', D]], with CA the
    banded factor of A, W = CA^{-1} B and CD the dense factor of D - W'W. It has the methods of
    ssm.BandChol: solve(b) = C^{-1} b, solve_t(b) = C'^{-1} b and logdet() = log det K."""

    def __init__(self, CA, W, CD):
        self.CA, self.W, self.CD = CA, W, CD
        self.n1, self.m = W.shape

    def solve(self, b):
        """C \\ b, for a vector or a matrix b."""
        b = np.asarray(b, dtype=float)
        y1 = self.CA.solve(b[:self.n1])
        y2 = solve_triangular(self.CD, b[self.n1:] - self.W.T @ y1, lower=True)
        return np.concatenate([y1, y2])

    def solve_t(self, b):
        """C.T \\ b, for a vector or a matrix b."""
        b = np.asarray(b, dtype=float)
        x2 = solve_triangular(self.CD, b[self.n1:], lower=True, trans='T')
        x1 = self.CA.solve_t(b[:self.n1] - self.W @ x2)
        return np.concatenate([x1, x2])

    def logdet(self):
        """log det K."""
        return self.CA.logdet() + 2.0 * np.sum(np.log(np.diag(self.CD)))

    def _dense(self, r0, r1, c0, c1):
        """C[r0:r1, c0:c1] as a dense matrix."""
        n = self.n1 + self.m
        C = np.zeros((n, n))
        C[:self.n1, :self.n1] = self.CA._dense(0, self.n1, 0, self.n1)
        C[self.n1:, :self.n1] = self.W.T
        C[self.n1:, self.n1:] = self.CD
        return C[r0:r1, c0:c1]


def _lower_entries(K):
    """the rows, columns and values of the nonzero entries of the lower triangle of K, with
    duplicates summed"""
    if sp.issparse(K) and K.format == 'dia':            # diagonals, as sp.eye_array gives
        n = K.shape[0]
        i, j, v = [], [], []
        for k, off in enumerate(K.offsets):
            if off <= 0:
                cols = np.arange(n + off)
                i.append(cols - off)
                j.append(cols)
                v.append(K.data[k, :n + off])
        if not i:
            return n, np.zeros(0, int), np.zeros(0, int), np.zeros(0)
        i, j, v = np.concatenate(i), np.concatenate(j), np.concatenate(v).astype(float)
        keep = v != 0
        return n, i[keep], j[keep], v[keep]
    if sp.issparse(K):
        if K.format not in ('csc', 'csr'):
            K = sp.csc_array(K)
        if not K.has_canonical_format:
            K = K.copy()
            K.sum_duplicates()
        n = K.shape[0]
        outer = np.repeat(np.arange(n), np.diff(K.indptr))
        i, j = (K.indices, outer) if K.format == 'csc' else (outer, K.indices)
        v = K.data
    else:
        K = np.asarray(K, dtype=float)
        n = K.shape[0]
        i, j = np.nonzero(np.tril(K))
        v = K[i, j]
    low = (i >= j) & (v != 0)            # explicit zeros dropped, as MATLAB never stores them
    return n, i[low], j[low], np.asarray(v[low], dtype=float)


def chol(K):
    """The lower Cholesky factor of a symmetric positive definite K, sparse or dense: a
    ssm.BandChol, or a BorderedChol when keeping a few trailing rows dense narrows the band of
    the rest enough to save work. Only the lower triangle of K is read."""
    n, i, j, v = _lower_entries(K)
    reach = np.zeros(n, dtype=int)                      # i minus the first column of row i
    np.maximum.at(reach, i, i - j)
    kd_rows = np.maximum.accumulate(reach)              # band of the leading block 0..r
    kd = int(kd_rows[-1]) if n else 0
    m_best, cost_best = 0, np.inf                       # operations, roughly, by trailing rows
    for m in range(1, min(_MAX_BORDER, n - 1) + 1):
        kdm = int(kd_rows[n - m - 1])
        cost = (n - m) * (kdm + 1) * (kdm + 1 + m) + m**3
        if cost < cost_best:
            m_best, cost_best = m, cost
    if not cost_best < n * (kd + 1)**2 / 2:             # keep the band unless it halves the work
        m_best = 0
    if m_best == 0:
        ab = np.zeros((kd + 1, n))
        ab[i - j, j] = v
        return BandChol.from_band(ab)
    n1 = n - m_best
    lead = i < n1
    kdA = int(kd_rows[n1 - 1])
    abA = np.zeros((kdA + 1, n1))
    abA[i[lead] - j[lead], j[lead]] = v[lead]
    tail = ~lead
    Bt = np.zeros((m_best, n1))                         # the trailing rows left of the corner
    D = np.zeros((m_best, m_best))
    left = tail & (j < n1)
    Bt[i[left] - n1, j[left]] = v[left]
    corner = tail & (j >= n1)
    D[i[corner] - n1, j[corner] - n1] = v[corner]
    D = D + np.tril(D, -1).T
    CA = BandChol.from_band(abA)
    W = CA.solve(Bt.T)
    W = W.reshape(n1, m_best)
    CD = np.linalg.cholesky(D - W.T @ W)
    return BorderedChol(CA, W, CD)
