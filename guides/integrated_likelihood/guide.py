"""guide - the code of guides/integrated_likelihood/README.md: the integrated likelihood of the
local level model of guides/uc_states, from ssm.intlike, first with the initial value of the trend
known and then with it integrated out as well; then a sampler that draws the two variances with
the trend integrated out, on the data of that guide. The sections called "the code to adapt" are
all a sampler needs. The checks compare ssm.intlike with the dense normal density of y, and the
sampler with the Gibbs sampler of guides/gibbs_sampler and with the posterior on a grid. Saves
fig_acf.png next to this file.
"""

import os
import time

# one BLAS thread, set before NumPy loads
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402
from scipy.optimize import minimize  # noqa: E402
from scipy.special import gammaln  # noqa: E402

import statespace_toolkit as ssm  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(3)


def mat2str(x, d):
    """x rounded to d decimals and printed as MATLAB's mat2str(round(x, d)) prints it"""
    return '[' + ' '.join(f'{v:.15g}' for v in np.round(np.ravel(x), d) + 0.0) + ']'


# ---- the data of guides/uc_states ----
T, sig2_true, omega2_true, tau0_true = 200, 1.0, .05, 2.0
tau_true = tau0_true + np.cumsum(np.sqrt(omega2_true) * rng.standard_normal(T))
y = tau_true + np.sqrt(sig2_true) * rng.standard_normal(T)

# ---- 1. the integrated likelihood: the code to adapt ----
sig2, omega2, tau0 = 1.0, .05, 2.0                      # any parameter values
H = ssm.diffmat(T)
HH = H.T @ H
ll, _, _ = ssm.intlike(y, sp.eye_array(T), sp.eye_array(T) / sig2, HH / omega2, tau0 * np.ones(T))


# ---- 1. the checks ----
def lnorm(x, m, S):
    """the log density of N(m, S) at x, from dense matrices"""
    return (-x.size / 2 * np.log(2 * np.pi) - np.sum(np.log(np.diag(np.linalg.cholesky(S))))
            - (x - m) @ np.linalg.solve(S, x - m) / 2)


# y ~ N(tau0 1, sig2 I + omega2 (H'H)^{-1}), from dense matrices
V = np.linalg.inv(HH.toarray())
pars = np.array([[1, .05, 2], [.5, .2, 0], [2, .01, 4], [1.3, .5, -1]])
d = np.zeros(pars.shape[0])
for k in range(pars.shape[0]):
    lk, _, _ = ssm.intlike(y, sp.eye_array(T), sp.eye_array(T) / pars[k, 0], HH / pars[k, 1],
                           pars[k, 2] * np.ones(T))
    ld = lnorm(y, pars[k, 2], pars[k, 0] * np.eye(T) + pars[k, 1] * V)
    d[k] = abs(lk - ld) / abs(ld)
print(f'\nlog p(y | sig2, omega2, tau0) = {ll:.4f} at the true values')
print(f'against the dense normal density at {pars.shape[0]} parameter values: at most '
      f'{d.max():.1e} (relative)')

# ---- 2. the initial value integrated out as well: the code to adapt ----
a0, b0 = 5.0, 100.0                                     # tau0 ~ N(a0, b0)
G = ssm.diffmat(T + 1)                                  # alpha = (tau0, tau')'
Z = sp.hstack([sp.csc_array((T, 1)), sp.eye_array(T)], format='csc')
b = a0 * np.ones(T + 1)                                 # the prior mean of alpha
P = G.T @ sp.diags_array(np.r_[1 / b0, np.ones(T) / omega2]) @ G   # its prior precision
ll0, _, _ = ssm.intlike(y, Z, sp.eye_array(T) / sig2, P, b)

# ---- 2. the checks ----
ld = lnorm(y, a0, sig2 * np.eye(T) + omega2 * V + b0 * np.ones((T, T)))
print(f'\nlog p(y | sig2, omega2) = {ll0:.4f}, tau0 integrated out; dense '
      f'{abs(ll0 - ld) / abs(ld):.1e} (relative)')

# ---- 3. the variances without the states: the code to adapt ----
nu_sig, S_sig, nu_om, S_om = 3.0, 2.0, 3.0, 2 * .25**2  # as in guides/gibbs_sampler
A0 = sp.csc_array(([1 / b0], ([0], [0])), shape=(T + 1, T + 1))   # P = A0 + A1/omega2
A1 = G.T @ sp.diags_array(np.r_[0.0, np.ones(T)]) @ G


def logig(x, nu, S):
    """the log density of IG(nu, S) at x"""
    return nu * np.log(S) - gammaln(nu) - (nu + 1) * np.log(x) - S / x


def lpost(phi):
    """the log posterior of phi = (log sig2, log omega2), up to a constant"""
    ll, _, _ = ssm.intlike(y, Z, sp.eye_array(T) * np.exp(-phi[0]), A0 + A1 * np.exp(-phi[1]), b)
    return (ll + logig(np.exp(phi[0]), nu_sig, S_sig) + logig(np.exp(phi[1]), nu_om, S_om)
            + np.sum(phi))


# the proposal: a t distribution with 5 degrees of freedom, located at the mode of lpost
# and scaled by the inverse of its negative Hessian there, from finite differences
phihat = minimize(lambda phi: -lpost(phi), [0, np.log(.1)], method='Nelder-Mead').x
h = 1e-3
E = h * np.eye(2)
Hs = np.zeros((2, 2))
for i in range(2):
    for j in range(2):
        Hs[i, j] = (lpost(phihat + E[:, i] + E[:, j]) - lpost(phihat + E[:, i] - E[:, j])
                    - lpost(phihat - E[:, i] + E[:, j])
                    + lpost(phihat - E[:, i] - E[:, j])) / (4 * h**2)
C = np.linalg.cholesky(np.linalg.inv(-Hs))
nu = 5


def logq(phi):
    """the log density of the proposal at phi, up to a constant"""
    return -(nu + 2) / 2 * np.log(1 + np.sum(np.linalg.solve(C, phi - phihat)**2) / nu)


nsim, burnin = 20000, 1000
rng = np.random.default_rng(2)
t0 = time.perf_counter()
phi = phihat
lw = lpost(phi) - logq(phi)
acc = 0
store_theta = np.zeros((nsim, 3))
for isim in range(burnin + nsim):
    # a draw from the proposal
    phic = phihat + C @ rng.standard_normal(2) / np.sqrt(rng.gamma(nu / 2, 2 / nu))
    lwc = lpost(phic) - logq(phic)
    if np.log(rng.random()) < lwc - lw:                 # independence-chain MH
        phi, lw = phic, lwc
        acc += isim >= burnin
    sig2, omega2 = np.exp(phi[0]), np.exp(phi[1])
    P = A0 + A1 / omega2
    K = P + Z.T @ Z / sig2                              # Theorem 9.1
    alpha, _, _ = ssm.simulate_states(K, P @ b + Z.T @ y / sig2, rng=rng)
    if isim >= burnin:
        store_theta[isim - burnin] = sig2, omega2, alpha[0]
time_i = time.perf_counter() - t0

# ---- 3. the checks ----
# the Gibbs sampler of guides/gibbs_sampler, on the same data
rng = np.random.default_rng(2)
t0 = time.perf_counter()
sig2, omega2, tau0 = 1.0, .1, a0
store_g = np.zeros((nsim, 3))
for isim in range(burnin + nsim):
    K = HH / omega2 + sp.eye_array(T) / sig2
    c = H.T @ np.r_[tau0, np.zeros(T - 1)] / omega2 + y / sig2
    tau, _, _ = ssm.simulate_states(K, c, rng=rng)
    sig2 = 1 / rng.gamma(nu_sig + T / 2, 1 / (S_sig + (y - tau) @ (y - tau) / 2))
    eta = H @ tau - np.r_[tau0, np.zeros(T - 1)]
    omega2 = 1 / rng.gamma(nu_om + T / 2, 1 / (S_om + eta @ eta / 2))
    Ktau0 = 1 / b0 + 1 / omega2
    tau0 = (a0 / b0 + tau[0] / omega2) / Ktau0 + rng.standard_normal() / np.sqrt(Ktau0)
    if isim >= burnin:
        store_g[isim - burnin] = sig2, omega2, tau0
time_g = time.perf_counter() - t0
L = 200
IFg = ssm.inefficiency_factor(store_g, L)
IFi = ssm.inefficiency_factor(store_theta, L)
print(f'\n{nsim} draws after a burn-in of {burnin}; the MH step accepts {100 * acc / nsim:.0f}% '
      f'of proposals')
print(f'inefficiency factors of sig2, omega2, tau0: Gibbs {mat2str(IFg, 1)}, without the states '
      f'{mat2str(IFi, 1)}')
print(f'time: Gibbs {time_g:.1f} s, without the states {time_i:.1f} s '
      f'(ratio {time_i / time_g:.1f})')
print(f'effective draws per second: Gibbs {mat2str(nsim / IFg / time_g, -1)}, without the states '
      f'{mat2str(nsim / IFi / time_i, -1)}')


# the posterior of (sig2, omega2) on a grid of their logs, from lpost
def span(x, n):
    """n points spanning 8 standard deviations of x on either side of its mean"""
    return np.linspace(x.mean() - 8 * x.std(ddof=1), x.mean() + 8 * x.std(ddof=1), n)


ng = 60
ls = span(np.log(store_theta[:, 0]), ng)
lo = span(np.log(store_theta[:, 1]), ng)
lp = np.zeros((ng, ng))
m0 = np.zeros((ng, ng))
for i in range(ng):
    for j in range(ng):
        lp[i, j] = lpost(np.array([ls[i], lo[j]]))
        _, ahat, _ = ssm.intlike(y, Z, sp.eye_array(T) * np.exp(-ls[i]),
                                 A0 + A1 * np.exp(-lo[j]), b)
        m0[i, j] = ahat[0]
w = np.exp(lp - lp.max())
w = w / w.sum()
S2, O2 = np.meshgrid(np.exp(ls), np.exp(lo), indexing='ij')
gm = np.array([np.sum(w * S2), np.sum(w * O2), np.sum(w * m0)])
mcse = ssm.mcse(store_theta, L)
edge = np.r_[w[0, :], w[-1, :], w[:, 0], w[:, -1]]
print(f'largest weight on the edge of the grid, relative to the largest weight: '
      f'{edge.max() / w.max():.1e}')
print(f'posterior means {mat2str(store_theta.mean(axis=0), 3)}, Gibbs '
      f'{mat2str(store_g.mean(axis=0), 3)}, grid {mat2str(gm, 3)}')
print(f'posterior means, minus the grid, in Monte Carlo standard errors: '
      f'{mat2str((store_theta.mean(axis=0) - gm) / mcse, 1)}')

# ---- 4. an AR(1) transitory component: the adaptation ----
rho = .8
Hrho = ssm.diffmat(T, rho)
ll_ar, _, _ = ssm.intlike(y, sp.eye_array(T), Hrho.T @ Hrho / sig2_true, HH / omega2_true,
                          tau0_true * np.ones(T))
ld = lnorm(y, tau0_true, sig2_true * np.linalg.inv((Hrho.T @ Hrho).toarray()) + omega2_true * V)
print(f'\nAR(1) transitory component, rho = {rho:.1f}: against the dense normal density '
      f'{abs(ll_ar - ld) / abs(ld):.1e} (relative)')


# ---- the figure: the autocorrelations of the draws of omega2 ----
def acf(x, lag):
    """the sample autocorrelation of x at a lag"""
    dx = x - x.mean()
    return np.sum(dx[lag:] * dx[:dx.size - lag]) / np.sum(dx**2)


lags = np.arange(101)
fig, ax = plt.subplots(figsize=(5.51, 2.76))
ax.plot(lags, [acf(store_g[:, 1], lag) for lag in lags], 'k--', linewidth=1.2)
ax.plot(lags, [acf(store_theta[:, 1], lag) for lag in lags], 'k', linewidth=1.2)
ax.spines[['top', 'right']].set_visible(False)
ax.set_xlim(lags[0], lags[-1])
ax.set_ylim(-.1, 1)
ax.set_xlabel('lag', fontsize=9)
ax.tick_params(labelsize=8)
ax.legend(['Gibbs sampler', 'drawn without the states'], loc='upper right', frameon=False,
          fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(HERE, 'fig_acf.png'), dpi=150)
plt.close(fig)
