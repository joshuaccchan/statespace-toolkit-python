"""guide - the code of guides/tvp_gibbs/README.md: a Gibbs sampler for the TVP regression of
guides/tvp_states. The sections called "the code to adapt" are all a sampler needs. Saves
fig_chain.png next to this file.
"""

import os

# one BLAS thread, set before NumPy loads
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402
from scipy.special import gammaln  # noqa: E402

import statespace_toolkit as ssm  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(13)


def logig(x, nu, S):
    """the log density of the inverse-gamma distribution IG(nu, S) at x"""
    return nu * np.log(S) - gammaln(nu) - (nu + 1) * np.log(x) - S / x


def span(x, n, s):
    """n points from s standard deviations below the mean of x to s above it"""
    return np.linspace(np.mean(x) - s * np.std(x, ddof=1), np.mean(x) + s * np.std(x, ddof=1), n)


def vecstr(v):
    """a vector as MATLAB's mat2str prints it, [a b c]; adding 0.0 turns -0 into 0"""
    return '[' + ' '.join(f'{x:g}' for x in np.asarray(v, dtype=float) + 0.0) + ']'


# ---- the data of guides/tvp_states ----
T, k, sig2_true = 200, 2, 1.0
Omega_true, beta0_true = np.diag([.005, .02]), np.array([2.0, 1.0])
X = np.column_stack([np.ones(T), rng.standard_normal(T)])   # x_t = (1, x_t)'
# beta_t = beta_{t-1} + eta_t, a T x k array
beta_true = beta0_true + np.cumsum(rng.standard_normal((T, k)) @ np.sqrt(Omega_true), axis=0)
y = np.sum(X * beta_true, axis=1) + np.sqrt(sig2_true) * rng.standard_normal(T)

# ---- 1. the Gibbs sampler: the code to adapt ----
a0, V0 = np.zeros(k), 100 * np.ones(k)                    # beta_0 ~ N(a0, diag(V0))
nu_sig, S_sig = 3, 2                                      # sig2 ~ IG(nu_sig, S_sig)
nu_om, S_om = 3, .02 * np.ones(k)                         # omega_i^2 ~ IG(nu_om, S_om[i])
nsim, burnin = 20000, 1000
Z = ssm.surform(X)
ZZ, Zy = Z.T @ Z, Z.T @ y                                 # computed once
H = sp.kron(ssm.diffmat(T), sp.eye_array(k), format='csc')  # H = H_T kron I_k
sig2, omega2, b0 = np.var(y, ddof=1), .01 * np.ones(k), a0  # starting values
store_theta, store_beta = np.zeros((nsim, 1 + 2 * k)), np.zeros((k * T, nsim))
for isim in range(burnin + nsim):
    # S^{-1} = I_T kron Omega^{-1}
    iS = sp.kron(sp.eye_array(T), sp.diags_array(1 / omega2), format='csc')
    btil = np.r_[b0, np.zeros((T - 1) * k)]               # betatilde_0 = (beta_0', 0', ..., 0')'
    K = H.T @ iS @ H + ZZ / sig2                          # the coefficients: guides/tvp_states
    c = H.T @ iS @ btil + Zy / sig2
    beta, _, _ = ssm.simulate_states(K, c, rng=rng)

    e = y - Z @ beta                                      # the measurement errors
    sig2 = 1 / rng.gamma(nu_sig + T / 2, 1 / (S_sig + e @ e / 2))

    eta = (H @ beta - btil).reshape(k, T, order='F')      # the state errors, a row per coefficient
    omega2 = 1 / rng.gamma(nu_om + T / 2, 1 / (S_om + np.sum(eta**2, axis=1) / 2))

    Kb0 = 1 / V0 + 1 / omega2                             # beta_0 given beta_1 and Omega
    b0 = (a0 / V0 + beta[:k] / omega2) / Kb0 + rng.standard_normal(k) / np.sqrt(Kb0)

    if isim >= burnin:
        store_theta[isim - burnin] = np.r_[sig2, omega2, b0]
        store_beta[:, isim - burnin] = beta

# ---- 1. the checks ----
L = 200                                                              # Bartlett truncation lag
IF = ssm.inefficiency_factor(store_theta, L)
pm, psd = store_theta.mean(axis=0), store_theta.std(axis=0, ddof=1)
pq = np.quantile(store_theta, [.05, .95], axis=0, method='hazen')
names = ['sig2', 'omega2_1', 'omega2_2', 'beta0_1', 'beta0_2']
truth = np.r_[sig2_true, np.diag(Omega_true), beta0_true]
fmt = ('{:<8s} true {:.3f}, posterior mean {:.4f}, 90% interval ({:.4f}, {:.4f}), '
       'inefficiency {:.1f}')
print(f'\nthe Gibbs sampler, {nsim} draws after a burn-in of {burnin}')
for i in range(len(names)):
    print(fmt.format(names[i], truth[i], pm[i], pq[0, i], pq[1, i], IF[i]))
# The posterior of (sig2, omega2_1, omega2_2) on a grid of their logs, with the coefficients
# and beta_0 integrated out: alpha = (beta_0', beta_1', ..., beta_T')' has prior mean
# (a0', ..., a0')' and precision G' Q^{-1} G = A0 + A1/omega2_1 + A2/omega2_2, with G the
# first differences of alpha and Q = diag(V0, Omega, ..., Omega). The grid spans 6
# standard deviations of the logged draws on either side of their mean.
G = sp.kron(ssm.diffmat(T + 1), sp.eye_array(k), format='csc')
Za = sp.hstack([sp.csc_array((T, k)), Z], format='csc')
ba = np.tile(a0, T + 1)
A0 = G.T @ sp.diags_array(np.r_[1 / V0, np.zeros(k * T)]) @ G
Ai = []
for i in range(k):
    sel = np.r_[np.zeros(k), np.tile((np.arange(k) == i).astype(float), T)]
    Ai.append(G.T @ sp.diags_array(sel) @ G)
ng = 40
gs = span(np.log(store_theta[:, 0]), ng, 6)
g1 = span(np.log(store_theta[:, 1]), ng, 6)
g2 = span(np.log(store_theta[:, 2]), ng, 6)
S2, O1, O2 = np.meshgrid(np.exp(gs), np.exp(g1), np.exp(g2), indexing='ij')
lp, m0 = np.zeros((ng, ng, ng)), np.zeros((k, ng**3))
for j in range(ng**3):
    s2, o1, o2 = S2.flat[j], O1.flat[j], O2.flat[j]
    P = A0 + Ai[0] / o1 + Ai[1] / o2
    ll, ahat, _ = ssm.intlike(y, Za, sp.eye_array(T) / s2, P, ba)
    lp.flat[j] = (ll + logig(s2, nu_sig, S_sig) + logig(o1, nu_om, S_om[0])
                  + logig(o2, nu_om, S_om[1])
                  + np.log(s2) + np.log(o1) + np.log(o2))           # the densities of the logs
    m0[:, j] = ahat[:k]
w = np.exp(lp - lp.max())
w = w / w.sum()
gm = np.r_[np.sum(w * S2), np.sum(w * O1), np.sum(w * O2), m0 @ w.ravel()]
gsd = np.sqrt(np.r_[np.sum(w * S2**2), np.sum(w * O1**2), np.sum(w * O2**2)] - gm[:3]**2)
mcse = ssm.mcse(store_theta, L)
print(f'\nthe posterior on a {ng} x {ng} x {ng} grid, from ssm.intlike')
face = max(w[[0, -1]].max(), w[:, [0, -1]].max(), w[:, :, [0, -1]].max())
print(f'largest weight on a face of the grid, relative to the largest weight: '
      f'{face / w.max():.1e}')
print(f'posterior means, Gibbs minus grid, in Monte Carlo standard errors: '
      f'{vecstr(np.round((pm - gm) / mcse, 1))}')
print(f'posterior standard deviations of the variances, Gibbs over grid: '
      f'{vecstr(np.round(psd[:3] / gsd, 3))}')
mb = np.zeros(k * T)                                                 # E(beta | y) from the grid
for j in np.flatnonzero(w > 1e-12):
    P = A0 + Ai[0] / O1.flat[j] + Ai[1] / O2.flat[j]
    _, ahat, _ = ssm.intlike(y, Za, sp.eye_array(T) / S2.flat[j], P, ba)
    mb = mb + w.flat[j] * ahat[k:]
IFb = ssm.inefficiency_factor(store_beta.T, L)
zb = (store_beta.mean(axis=1) - mb) / ssm.mcse(store_beta.T, L)
print(f'coefficients: posterior means, Gibbs minus grid, at most {np.max(np.abs(zb)):.1f} Monte '
      f'Carlo standard errors')
print(f'coefficients: inefficiency factors from {IFb.min():.1f} to {IFb.max():.1f}')

# the coefficients with the parameters known, as in guides/tvp_states
iSk = sp.kron(sp.eye_array(T), np.linalg.inv(Omega_true), format='csc')
Kk = H.T @ iSk @ H + ZZ / sig2_true
ck = H.T @ iSk @ np.r_[beta0_true, np.zeros((T - 1) * k)] + Zy / sig2_true
dk, bhat_k, _ = ssm.simulate_states(Kk, ck, 5000, rng=rng)
bk = np.quantile(dk, [.05, .95], axis=1, method='hazen').T
bg = np.quantile(store_beta, [.05, .95], axis=1, method='hazen').T
bt = beta_true.ravel()                                               # stacked by period
print('\nthe coefficients, with the parameters estimated and known:')
print(f'largest difference between the posterior means: '
      f'{np.max(np.abs(store_beta.mean(axis=1) - bhat_k) / store_beta.std(axis=1, ddof=1)):.2f} '
      f'posterior sd')
for i in range(k):
    r = slice(i, k * T, k)
    print(f'coefficient {i + 1}: 90% intervals {np.mean(bg[r, 1] - bg[r, 0]):.2f} and '
          f'{np.mean(bk[r, 1] - bk[r, 0]):.2f} wide on average; they contain the truth in '
          f'{np.sum((bt[r] >= bg[r, 0]) & (bt[r] <= bg[r, 1]))} and '
          f'{np.sum((bt[r] >= bk[r, 0]) & (bt[r] <= bk[r, 1]))} of {T} periods')

# ---- 2. a constant slope: the adaptation ----
rng = np.random.default_rng(7)
b2_true, om2_true, b10_true = 1.0, Omega_true[0, 0], beta0_true[0]
x = X[:, 1]
tau_true = b10_true + np.cumsum(np.sqrt(om2_true) * rng.standard_normal(T))  # a random walk
y2 = tau_true + b2_true * x + np.sqrt(sig2_true) * rng.standard_normal(T)
as_, Vs = 0.0, 100.0                                       # b2 ~ N(as_, Vs)
H1 = ssm.diffmat(T)
HH1 = H1.T @ H1
sig2, om2, b10, b2 = np.var(y2, ddof=1), .01, a0[0], as_   # starting values
store_cs = np.zeros((nsim, 4))
for isim in range(burnin + nsim):
    K = HH1 / om2 + sp.eye_array(T) / sig2                 # the intercept: uc_states on y - b2 x
    c = H1.T @ np.r_[b10, np.zeros(T - 1)] / om2 + (y2 - b2 * x) / sig2
    tau, _, _ = ssm.simulate_states(K, c, rng=rng)

    Kb2 = 1 / Vs + x @ x / sig2                            # the constant slope: a regression
    b2 = (as_ / Vs + x @ (y2 - tau) / sig2) / Kb2 + rng.standard_normal() / np.sqrt(Kb2)

    e = y2 - tau - b2 * x
    sig2 = 1 / rng.gamma(nu_sig + T / 2, 1 / (S_sig + e @ e / 2))

    eta = H1 @ tau - np.r_[b10, np.zeros(T - 1)]
    om2 = 1 / rng.gamma(nu_om + T / 2, 1 / (S_om[0] + eta @ eta / 2))

    Kb10 = 1 / V0[0] + 1 / om2
    b10 = (a0[0] / V0[0] + tau[0] / om2) / Kb10 + rng.standard_normal() / np.sqrt(Kb10)

    if isim >= burnin:
        store_cs[isim - burnin] = [sig2, om2, b2, b10]

# ---- 2. the checks ----
IFc = ssm.inefficiency_factor(store_cs, L)
pmc, psdc = store_cs.mean(axis=0), store_cs.std(axis=0, ddof=1)
pqc = np.quantile(store_cs, [.05, .95], axis=0, method='hazen')
namesc, truthc = ['sig2', 'omega2_1', 'b2', 'beta0_1'], [sig2_true, om2_true, b2_true, b10_true]
print(f'\na constant slope, {nsim} draws after a burn-in of {burnin}')
for i in range(4):
    print(fmt.format(namesc[i], truthc[i], pmc[i], pqc[0, i], pqc[1, i], IFc[i]))
# the posterior of (sig2, omega2_1) on a grid of their logs, with the intercept, its initial
# value and the slope in the states: alpha = (beta0_1, beta_11, ..., beta_1T, b2)'
Gc = ssm.diffmat(T + 1)
Zc = sp.hstack([sp.csc_array((T, 1)), sp.eye_array(T), sp.csc_array(x[:, None])], format='csc')
bc = np.r_[a0[0] * np.ones(T + 1), as_]
Ac0 = sp.block_diag([Gc.T @ sp.diags_array(np.r_[1 / V0[0], np.zeros(T)]) @ Gc, [[1 / Vs]]],
                    format='csc')
Ac1 = sp.block_diag([Gc.T @ sp.diags_array(np.r_[0, np.ones(T)]) @ Gc, [[0.0]]], format='csc')
nc = 60
cs = span(np.log(store_cs[:, 0]), nc, 8)
co = span(np.log(store_cs[:, 1]), nc, 8)
Sc, Oc = np.meshgrid(np.exp(cs), np.exp(co), indexing='ij')
lpc, mc = np.zeros((nc, nc)), np.zeros((2, nc**2))
for j in range(nc**2):
    ll, ahat, _ = ssm.intlike(y2, Zc, sp.eye_array(T) / Sc.flat[j], Ac0 + Ac1 / Oc.flat[j], bc)
    lpc.flat[j] = (ll + logig(Sc.flat[j], nu_sig, S_sig) + logig(Oc.flat[j], nu_om, S_om[0])
                   + np.log(Sc.flat[j]) + np.log(Oc.flat[j]))
    mc[:, j] = ahat[[-1, 0]]                                         # b2 and beta0_1
wc = np.exp(lpc - lpc.max())
wc = wc / wc.sum()
gmc = np.r_[np.sum(wc * Sc), np.sum(wc * Oc), mc @ wc.ravel()]
print(f'the posterior on a {nc} x {nc} grid, from ssm.intlike')
edge = max(wc[0].max(), wc[-1].max(), wc[:, 0].max(), wc[:, -1].max())
print(f'largest weight on the edge of the grid, relative to the largest weight: '
      f'{edge / wc.max():.1e}')
print(f'posterior means, Gibbs minus grid, in Monte Carlo standard errors: '
      f'{vecstr(np.round((pmc - gmc) / ssm.mcse(store_cs, L), 1))}')

# ---- the figure: the draws of the three variances of section 1 and their histograms, with the
# grid posterior ----
fig, axes = plt.subplots(2, 3, figsize=(7.09, 3.94))
lab = ['$\\sigma^2$', '$\\omega_1^2$', '$\\omega_2^2$']
gx = [np.exp(gs), np.exp(g1), np.exp(g2)]
pw = [w.sum(axis=(1, 2)) / np.exp(gs) / (gs[1] - gs[0]),
      w.sum(axis=(0, 2)) / np.exp(g1) / (g1[1] - g1[0]),
      w.sum(axis=(0, 1)) / np.exp(g2) / (g2[1] - g2[0])]
for i in range(3):
    ax = axes[0, i]
    ax.plot(store_theta[:, i], color=(.35, .35, .35), linewidth=.3)
    ax.set_xlim(0, nsim)
    ax.set_xticks([0, nsim // 2, nsim])
    ax.set_title(lab[i], fontsize=10)
    ax = axes[1, i]
    ax.hist(store_theta[:, i], bins=60, density=True, color=(.75, .75, .75), edgecolor='none')
    keep = pw[i] > 1e-3 * pw[i].max()
    ax.plot(gx[i][keep], pw[i][keep], 'k', linewidth=1.2)
    ax.axvline(truth[i], color='k', linestyle='--', linewidth=1)
    for ax in axes[:, i]:
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(labelsize=8)
fig.tight_layout()
fig.savefig(os.path.join(HERE, 'fig_chain.png'), dpi=150)
plt.close(fig)
