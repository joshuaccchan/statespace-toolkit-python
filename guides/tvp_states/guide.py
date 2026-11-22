"""guide - the code of guides/tvp_states/README.md: the coefficients of a time-varying parameter
regression, drawn by the precision sampler given the parameters. The sections called "the code to
adapt" are all a sampler needs. Saves fig_spy.png, fig_coef.png and fig_mean.png next to this
file.
"""

import os

# one BLAS thread, set before NumPy loads
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402
from scipy.linalg import block_diag  # noqa: E402

import statespace_toolkit as ssm  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(13)


def rel(a, b):
    """the largest difference between a and b, relative to the largest entry of b"""
    a, b = np.asarray(a), np.asarray(b)
    return np.max(np.abs(a - b)) / np.max(np.abs(b))


# ---- 1. the TVP regression: the code to adapt ----
T, k, sig2 = 200, 2, 1.0                                    # the parameters, known
Omega, beta0 = np.diag([.005, .02]), np.array([2.0, 1.0])
X = np.column_stack([np.ones(T), rng.standard_normal(T)])   # x_t = (1, x_t)'
# beta_t = beta_{t-1} + eta_t, a T x k array
beta = beta0 + np.cumsum(rng.standard_normal((T, k)) @ np.sqrt(Omega), axis=0)
y = np.sum(X * beta, axis=1) + np.sqrt(sig2) * rng.standard_normal(T)   # y_t = x_t' beta_t + eps_t

Z = ssm.surform(X)                                          # Z = diag(x_1', ..., x_T')
H = sp.kron(ssm.diffmat(T), sp.eye_array(k), format='csc')  # H = H_T kron I_k
iS = sp.kron(sp.eye_array(T), np.linalg.inv(Omega), format='csc')  # S^{-1} = I_T kron Omega^{-1}
btil = np.r_[beta0, np.zeros((T - 1) * k)]                  # betatilde_0 = (beta_0', 0', ..., 0')'
K = H.T @ iS @ H + Z.T @ Z / sig2                           # K = H' S^{-1} H + Z'Z/sig2
c = H.T @ iS @ btil + Z.T @ y / sig2                        # c = H' S^{-1} betatilde_0 + Z'y/sig2
draws, betahat, _ = ssm.simulate_states(K, c, 5000, rng=rng)  # 5,000 independent draws of the path
B = draws.reshape(k, T, -1, order='F')              # B[i, t, d]: coefficient i, period t, draw d

# ---- 1. the checks ----
print(f'\nthe TVP regression, T = {T}, k = {k}')
rows, cols = K.nonzero()
print(f'K is {K.shape[0]} x {K.shape[1]} with lower bandwidth {np.max(rows - cols)}')
print('the top left corner of K:')
with np.printoptions(precision=4, suppress=True):
    print(K.toarray()[:4, :4])
Hd, Zd = H.toarray(), Z.toarray()
Sd = np.kron(np.eye(T), Omega)                              # S, dense
V = np.linalg.solve(Hd, np.linalg.solve(Hd, Sd).T)          # the prior covariance of the path
mu0 = np.tile(beta0, T)                                     # its prior mean
G = Zd @ V @ Zd.T + sig2 * np.eye(T)
mu = mu0 + V @ Zd.T @ np.linalg.solve(G, y - Zd @ mu0)      # the covariance form
Vpost = V - V @ Zd.T @ np.linalg.solve(G, Zd @ V)
print(f'mean and covariance against dense algebra:   {rel(betahat, mu):.1e}, '
      f'{rel(np.linalg.inv(K.toarray()), Vpost):.1e} (relative)')
ratio = draws.var(axis=1, ddof=1) / np.diag(Vpost)
dev = np.max(np.abs(draws.mean(axis=1) - mu) / np.sqrt(np.diag(Vpost)))
print(f'5,000 draws: means within {dev:.2f} posterior sd, variances {ratio.min():.3f} to '
      f'{ratio.max():.3f} of the posterior\'s')
q1 = np.quantile(B, [.05, .95], axis=2, method='hazen')              # 2 x k x T
in1 = (beta.T >= q1[0]) & (beta.T <= q1[1])
print(f'this realization: the 90% intervals contain the true intercept in {np.sum(in1[0])} and '
      f'the true slope in {np.sum(in1[1])} of {T} periods')
print(f'average width of the 90% intervals: intercept {np.mean(q1[1, 0] - q1[0, 0]):.2f}, '
      f'slope {np.mean(q1[1, 1] - q1[0, 1]):.2f}')
beta1, m1 = beta, B.mean(axis=2)
Tspy = 25                                                            # K for T = 25, for Figure 1
Zs, Hs = ssm.surform(X[:Tspy]), sp.kron(ssm.diffmat(Tspy), sp.eye_array(k), format='csc')
Kspy = (Hs.T @ sp.kron(sp.eye_array(Tspy), np.linalg.inv(Omega), format='csc') @ Hs
        + Zs.T @ Zs / sig2)

# ---- 2. coefficients that revert to a mean: the code to adapt ----
Phi, mu = np.diag([.9, .9]), beta0                    # beta_t - mu = Phi (beta_{t-1} - mu) + eta_t
beta, b = np.zeros((T, k)), beta0                                    # beta_0 = mu
for t in range(T):
    b = mu + Phi @ (b - mu) + np.sqrt(Omega) @ rng.standard_normal(k)
    beta[t] = b
y = np.sum(X * beta, axis=1) + np.sqrt(sig2) * rng.standard_normal(T)

L = sp.eye_array(T) - ssm.diffmat(T)                      # the lag matrix: ones below the diagonal
HPhi = sp.eye_array(T * k) - sp.kron(L, Phi, format='csc')  # H_Phi = I - L kron Phi
btil = np.r_[(np.eye(k) - Phi) @ mu + Phi @ beta0, np.tile((np.eye(k) - Phi) @ mu, T - 1)]
K = HPhi.T @ iS @ HPhi + Z.T @ Z / sig2                   # K = H_Phi' S^{-1} H_Phi + Z'Z/sig2
c = HPhi.T @ iS @ btil + Z.T @ y / sig2                   # c = H_Phi' S^{-1} betatilde + Z'y/sig2
draws, betahat, _ = ssm.simulate_states(K, c, 5000, rng=rng)
B = draws.reshape(k, T, -1, order='F')

# ---- 2. the checks ----
rows, cols = K.nonzero()
print(f'\ncoefficients that revert to a mean, Phi = diag({Phi[0, 0]:.1f}, {Phi[1, 1]:.1f}); K has '
      f'lower bandwidth {np.max(rows - cols)}')
Hd = HPhi.toarray()
V = np.linalg.solve(Hd, np.linalg.solve(Hd, Sd).T)
mu0 = np.linalg.solve(Hd, btil)                                      # the prior mean of the path
G = Zd @ V @ Zd.T + sig2 * np.eye(T)
mpost = mu0 + V @ Zd.T @ np.linalg.solve(G, y - Zd @ mu0)
Vpost = V - V @ Zd.T @ np.linalg.solve(G, Zd @ V)
print(f'mean and covariance against dense algebra:   {rel(betahat, mpost):.1e}, '
      f'{rel(np.linalg.inv(K.toarray()), Vpost):.1e} (relative)')
q2 = np.quantile(B, [.05, .95], axis=2, method='hazen')
in2 = (beta.T >= q2[0]) & (beta.T <= q2[1])
print(f'this realization: the 90% intervals contain the true intercept in {np.sum(in2[0])} and '
      f'the true slope in {np.sum(in2[1])} of {T} periods')
print(f'average width of the 90% intervals: intercept {np.mean(q2[1, 0] - q2[0, 0]):.2f}, '
      f'slope {np.mean(q2[1, 1] - q2[0, 1]):.2f}')
beta2, m2 = beta, B.mean(axis=2)

# ---- 3. a prior of their own for the first coefficients: the adaptation ----
a, V1 = np.zeros(k), 10 * np.eye(k)                     # beta_1 ~ N(a, V1), beta_0 not needed
iS1 = sp.block_diag([np.linalg.inv(V1),                 # first block V1^{-1}
                     sp.kron(sp.eye_array(T - 1), np.linalg.inv(Omega), format='csc')])
btil1 = np.r_[a, np.zeros((T - 1) * k)]                 # (a', 0', ..., 0')'
# the random-walk coefficients of Section 1
y = np.sum(X * beta1, axis=1) + np.sqrt(sig2) * rng.standard_normal(T)
K = H.T @ iS1 @ H + Z.T @ Z / sig2
c = H.T @ iS1 @ btil1 + Z.T @ y / sig2
_, betahat, _ = ssm.simulate_states(K, c, 1, rng=rng)
Hd = H.toarray()
V = np.linalg.solve(Hd, np.linalg.solve(Hd, block_diag(V1, np.kron(np.eye(T - 1), Omega))).T)
mu0 = np.linalg.solve(Hd, btil1)
G = Zd @ V @ Zd.T + sig2 * np.eye(T)
print(f'\na prior of their own for the first coefficients: mean against dense algebra '
      f'{rel(betahat, mu0 + V @ Zd.T @ np.linalg.solve(G, y - Zd @ mu0)):.1e}')

# ---- the figures ----
fig, axes = plt.subplots(1, 2, figsize=(6.3, 3.15))
axes[0].spy(Kspy, markersize=2, color='k')
axes[0].set_title(f'$\\mathbf{{K}}$, $T = {Tspy}$, $k = {k}$', fontsize=10)
axes[1].spy(np.abs(np.linalg.inv(Kspy.toarray())) > 1e-12, markersize=2, color='k')
axes[1].set_title(f'$\\mathbf{{K}}^{{-1}}$, $T = {Tspy}$, $k = {k}$', fontsize=10)
for ax in axes:
    ax.tick_params(labelsize=8)
fig.tight_layout()
fig.savefig(os.path.join(HERE, 'fig_spy.png'), dpi=150)
plt.close(fig)

t = np.arange(1, T + 1)
names = ['intercept, $\\beta_{1t}$', 'slope, $\\beta_{2t}$']
for name, tb, mm, qq in (('fig_coef.png', beta1, m1, q1), ('fig_mean.png', beta2, m2, q2)):
    fig, axes = plt.subplots(k, 1, figsize=(7.09, 4.33))
    for i, ax in enumerate(axes):
        hb = ssm.shaded_band(t, qq[0, i], qq[1, i], ax=ax)
        ht, = ax.plot(t, tb[:, i], 'k--', linewidth=1)
        hm, = ax.plot(t, mm[i], 'k', linewidth=1.3)
        ax.set_xlim(1, T)
        ax.set_title(names[i], fontsize=10)
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(labelsize=8)
    fig.legend([ht, hm, hb], ['true coefficient', 'posterior mean',
                              '90% pointwise credible interval'],
               loc='lower center', ncols=3, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, .05, 1, 1))
    fig.savefig(os.path.join(HERE, name), dpi=150)
    plt.close(fig)
