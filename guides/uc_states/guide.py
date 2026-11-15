"""guide - the code of guides/uc_states/README.md: the trend of a local level model drawn by the
precision sampler, given the parameters, on generated data; then the same with a persistent
transitory component, and with a known variance that changes over time. The sections called
"the code to adapt" are all a sampler needs. The sections called "the checks" form K^{-1} and
other dense matrices, which only a small example can afford.

Saves fig_spy.png, fig_level.png and fig_ar1.png next to this file.

Usage, from the root of the repository:  python guides/uc_states/guide.py
"""

import os

# one BLAS thread, set before NumPy loads
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402
from scipy.signal import lfilter  # noqa: E402

import statespace_toolkit as ssm  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(1)


def rel(a, b):
    """the largest difference between a and b, relative to the largest entry of b"""
    a, b = np.asarray(a), np.asarray(b)
    return np.max(np.abs(a - b)) / np.max(np.abs(b))


# ---- 1. the local level model: the code to adapt ----
T, sig2, omega2, tau0 = 200, 1.0, .05, 2.0                          # the parameters, known
tau = tau0 + np.cumsum(np.sqrt(omega2) * rng.standard_normal(T))     # tau_1 ~ N(tau0, omega2)
y = tau + np.sqrt(sig2) * rng.standard_normal(T)

H = ssm.diffmat(T)                                                   # first differences
tau0_vec = np.r_[tau0, np.zeros(T - 1)]                              # (tau0, 0, ..., 0)'
K = H.T @ H / omega2 + sp.eye_array(T) / sig2                        # quadratic terms
c = H.T @ tau0_vec / omega2 + y / sig2                               # linear terms: K tauhat = c
draws, tauhat, _ = ssm.simulate_states(K, c, 5000, rng=rng)          # 5,000 independent draws

# ---- 1. the checks ----
print(f'\nthe local level model, T = {T}')
print('the top left corner of K:')
print(K.toarray()[:4, :4])
Hd = H.toarray()
V = omega2 * np.linalg.inv(Hd.T @ Hd)                                # the prior covariance of tau
S = V + sig2 * np.eye(T)
mu = tau0 + V @ np.linalg.solve(S, y - tau0)                         # the covariance form
Vpost = V - V @ np.linalg.solve(S, V)
print(f'mean and covariance against dense algebra:   {rel(tauhat, mu):.1e}, '
      f'{rel(np.linalg.inv(K.toarray()), Vpost):.1e} (relative)')
ratio = draws.var(axis=1, ddof=1) / np.diag(Vpost)
dev = np.max(np.abs(draws.mean(axis=1) - mu) / np.sqrt(np.diag(Vpost)))
print(f'5,000 draws: means within {dev:.2f} posterior sd, variances {ratio.min():.3f} to '
      f'{ratio.max():.3f} of the posterior\'s')
b1 = np.quantile(draws, [.05, .95], axis=1, method='hazen').T
print(f'this realization: the 90% intervals contain the true trend in '
      f'{np.sum((tau >= b1[:, 0]) & (tau <= b1[:, 1]))} of {T} periods')
y1, tau1, m1 = y, tau, draws.mean(axis=1)
H50 = ssm.diffmat(50)
Kspy = H50.T @ H50 / omega2 + sp.eye_array(50) / sig2               # K for T = 50

# ---- 2. a persistent transitory component: the code to adapt ----
rho = .8                                                             # eps_t = rho eps_{t-1} + u_t
tau = tau0 + np.cumsum(np.sqrt(omega2) * rng.standard_normal(T))
y = tau + lfilter([1.0], [1.0, -rho], np.sqrt(sig2) * rng.standard_normal(T))   # eps_0 = 0

Hrho = ssm.diffmat(T, rho)                                           # I - rho L
K = H.T @ H / omega2 + Hrho.T @ Hrho / sig2
c = H.T @ tau0_vec / omega2 + Hrho.T @ (Hrho @ y) / sig2
draws, tauhat, _ = ssm.simulate_states(K, c, 5000, rng=rng)

# ---- 2. the checks ----
Kc = K.tocoo()
print(f'\na persistent transitory component, rho = {rho:.1f}; K has bandwidth '
      f'{np.max(Kc.row - Kc.col)}')
Hr = Hrho.toarray()
W = sig2 * np.linalg.inv(Hr.T @ Hr)                                  # the covariance of eps
S = V + W
mu = tau0 + V @ np.linalg.solve(S, y - tau0)
Vpost = V - V @ np.linalg.solve(S, V)
print(f'mean and covariance against dense algebra:   {rel(tauhat, mu):.1e}, '
      f'{rel(np.linalg.inv(K.toarray()), Vpost):.1e} (relative)')
b2 = np.quantile(draws, [.05, .95], axis=1, method='hazen').T
print(f'this realization: the 90% intervals contain the true trend in '
      f'{np.sum((tau >= b2[:, 0]) & (tau <= b2[:, 1]))} of {T} periods')
print(f'average width of the 90% intervals: {np.mean(b2[:, 1] - b2[:, 0]):.2f}, against '
      f'{np.mean(b1[:, 1] - b1[:, 0]):.2f} with a serially uncorrelated transitory component')
y2, tau2, m2 = y, tau, draws.mean(axis=1)

# ---- 3. a known measurement variance that changes over time: the adaptation ----
sig2t = .5 + 1.5 * np.arange(1, T + 1) / T                           # from 0.5 to 2
K = H.T @ H / omega2 + sp.diags_array(1 / sig2t)
c = H.T @ tau0_vec / omega2 + y1 / sig2t
draws, tauhat, _ = ssm.simulate_states(K, c, 5000, rng=rng)
S = V + np.diag(sig2t)
print(f'\na known, time-varying measurement variance: mean against dense algebra '
      f'{rel(tauhat, tau0 + V @ np.linalg.solve(S, y1 - tau0)):.1e}')

# ---- the figures ----
fig, axes = plt.subplots(1, 2, figsize=(6.3, 3.15))
axes[0].spy(Kspy, markersize=2, color='k')
axes[0].set_title('$\\mathbf{K}$, $T = 50$', fontsize=10)
axes[1].spy(np.abs(np.linalg.inv(Kspy.toarray())) > 1e-12, markersize=2, color='k')
axes[1].set_title('$\\mathbf{K}^{-1}$, $T = 50$', fontsize=10)
for ax in axes:
    ax.tick_params(labelsize=8)
fig.tight_layout()
fig.savefig(os.path.join(HERE, 'fig_spy.png'), dpi=150)
plt.close(fig)

t = np.arange(1, T + 1)
for name, yy, tt, mm, bb in (('fig_level.png', y1, tau1, m1, b1),
                             ('fig_ar1.png', y2, tau2, m2, b2)):
    fig, ax = plt.subplots(figsize=(7.09, 2.95))
    hb = ssm.shaded_band(t, bb[:, 0], bb[:, 1], ax=ax)
    hy, = ax.plot(t, yy, '.', color=(.6, .6, .6), markersize=3)
    ht, = ax.plot(t, tt, 'k--', linewidth=1)
    hm, = ax.plot(t, mm, 'k', linewidth=1.3)
    ax.set_xlim(1, T)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(labelsize=8)
    ax.legend([hy, ht, hm, hb], ['$y_t$', 'true trend', 'posterior mean',
                                 '90% pointwise credible interval'],
              loc='best', ncols=2, frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, name), dpi=150)
    plt.close(fig)
