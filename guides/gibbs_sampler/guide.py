"""guide - the code of guides/gibbs_sampler/README.md: a Gibbs sampler for the local level model
of guides/uc_states. The sections called "the code to adapt" are all a sampler needs. Saves
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
from scipy.signal import lfilter  # noqa: E402
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

# ---- 1. the Gibbs sampler: the code to adapt ----
a0, b0 = 5.0, 100.0                                     # tau0 ~ N(a0, b0)
nu_sig, S_sig = 3.0, 2.0                                # sig2 ~ IG(nu_sig, S_sig)
nu_om, S_om = 3.0, 2 * .25**2                           # omega2 ~ IG(nu_om, S_om)
nsim, burnin = 20000, 1000
H = ssm.diffmat(T)
HH = H.T @ H                                            # computed once
sig2, omega2, tau0 = 1.0, .1, a0                        # starting values
store_theta = np.zeros((nsim, 3))
store_tau = np.zeros((T, nsim))
for isim in range(burnin + nsim):
    K = HH / omega2 + sp.eye_array(T) / sig2            # the states: guides/uc_states
    c = H.T @ np.r_[tau0, np.zeros(T - 1)] / omega2 + y / sig2
    tau, _, _ = ssm.simulate_states(K, c, rng=rng)
    sig2 = 1 / rng.gamma(nu_sig + T / 2, 1 / (S_sig + (y - tau) @ (y - tau) / 2))
    eta = H @ tau - np.r_[tau0, np.zeros(T - 1)]        # the state equation residuals
    omega2 = 1 / rng.gamma(nu_om + T / 2, 1 / (S_om + eta @ eta / 2))
    Ktau0 = 1 / b0 + 1 / omega2                         # tau0 given tau_1 and omega2
    tau0 = (a0 / b0 + tau[0] / omega2) / Ktau0 + rng.standard_normal() / np.sqrt(Ktau0)
    if isim >= burnin:
        store_theta[isim - burnin] = sig2, omega2, tau0
        store_tau[:, isim - burnin] = tau

# ---- 1. the checks ----
L = 200                                                 # Bartlett truncation lag
IF = ssm.inefficiency_factor(store_theta, L)
pm = store_theta.mean(axis=0)
psd = store_theta.std(axis=0, ddof=1)
pq = np.quantile(store_theta, [.05, .95], axis=0, method='hazen')
names = ['sig2', 'omega2', 'tau0']
truth = np.array([sig2_true, omega2_true, tau0_true])
fmt = ('{:<6} true {:.3f}, posterior mean {:.3f}, 90% interval ({:.3f}, {:.3f}), '
       'inefficiency {:.1f}')
print(f'\nthe Gibbs sampler, {nsim} draws after a burn-in of {burnin}')
for k in range(3):
    print(fmt.format(names[k], truth[k], pm[k], pq[0, k], pq[1, k], IF[k]))
# The posterior of (sig2, omega2) on a grid of their logs, with the states and tau0
# integrated out: alpha = (tau0, tau_1, ..., tau_T)' has prior mean a0 and precision
# G' Q^{-1} G = A0 + A1/omega2, with Q = diag(b0, omega2, ..., omega2). The grid spans 8
# standard deviations of the logged draws on either side of their mean.
G = ssm.diffmat(T + 1)
Z = sp.hstack([sp.csc_array((T, 1)), sp.eye_array(T)], format='csc')
b = a0 * np.ones(T + 1)
A0 = sp.csc_array(([1 / b0], ([0], [0])), shape=(T + 1, T + 1))
A1 = G.T @ sp.diags_array(np.r_[0.0, np.ones(T)]) @ G
e1 = np.r_[1.0, np.zeros(T)]


def logig(x, nu, S):
    """the log density of IG(nu, S) at x"""
    return nu * np.log(S) - gammaln(nu) - (nu + 1) * np.log(x) - S / x


def span(x, n):
    """n points spanning 8 standard deviations of x on either side of its mean"""
    return np.linspace(x.mean() - 8 * x.std(ddof=1), x.mean() + 8 * x.std(ddof=1), n)


ng = 60
ls = span(np.log(store_theta[:, 0]), ng)
lo = span(np.log(store_theta[:, 1]), ng)
S2, O2 = np.meshgrid(np.exp(ls), np.exp(lo), indexing='ij')
lp = np.zeros((ng, ng))
m0 = np.zeros((ng, ng))
v0 = np.zeros((ng, ng))
for k in np.ndindex(ng, ng):                            # every point (i, j) of the grid
    P = A0 + A1 / O2[k]
    ll, ahat, Kg = ssm.intlike(y, Z, sp.eye_array(T) / S2[k], P, b)
    lp[k] = (ll + logig(S2[k], nu_sig, S_sig) + logig(O2[k], nu_om, S_om)
             + np.log(S2[k]) + np.log(O2[k]))           # the densities of the logs
    m0[k] = ahat[0]
    v0[k] = np.sum(ssm.BandChol(Kg).solve(e1)**2)       # (K^{-1})_{11}
w = np.exp(lp - lp.max())
w = w / w.sum()
gm = np.array([np.sum(w * S2), np.sum(w * O2), np.sum(w * m0)])
gsd = np.sqrt(np.array([np.sum(w * S2**2), np.sum(w * O2**2), np.sum(w * (v0 + m0**2))])
              - gm**2)
mcse = ssm.mcse(store_theta, L)
print(f'\nthe posterior on a {ng} x {ng} grid, from ssm.intlike')
edge = np.r_[w[0, :], w[-1, :], w[:, 0], w[:, -1]]
print(f'largest weight on the edge of the grid, relative to the largest weight: '
      f'{edge.max() / w.max():.1e}')
print(f'posterior means, Gibbs minus grid, in Monte Carlo standard errors: '
      f'{mat2str((pm - gm) / mcse, 1)}')
print(f'posterior standard deviations, Gibbs over grid: {mat2str(psd / gsd, 3)}')
mt = np.zeros(T)                                        # E(tau | y) from the grid
for i, j in np.argwhere(w > 1e-12):
    _, ahat, _ = ssm.intlike(y, Z, sp.eye_array(T) / S2[i, j], A0 + A1 / O2[i, j], b)
    mt = mt + w[i, j] * ahat[1:]
IFt = ssm.inefficiency_factor(store_tau.T, L)
zt = (store_tau.mean(axis=1) - mt) / ssm.mcse(store_tau.T, L)
print(f'trend: posterior means, Gibbs minus grid, at most {np.max(np.abs(zt)):.1f} Monte Carlo '
      f'standard errors')
print(f'trend: inefficiency factors from {IFt.min():.1f} to {IFt.max():.1f}')

# the trend with the parameters known, as in guides/uc_states
Kk = HH / omega2_true + sp.eye_array(T) / sig2_true
ck = H.T @ np.r_[tau0_true, np.zeros(T - 1)] / omega2_true + y / sig2_true
dk, tauhat_k, _ = ssm.simulate_states(Kk, ck, 5000, rng=rng)
bk = np.quantile(dk, [.05, .95], axis=1, method='hazen').T
bt = np.quantile(store_tau, [.05, .95], axis=1, method='hazen').T
print('\nthe trend, with the parameters estimated and known:')
dmax = np.max(np.abs(store_tau.mean(axis=1) - tauhat_k) / store_tau.std(axis=1, ddof=1))
print(f'largest difference between the posterior means: {dmax:.2f} posterior sd')
print(f'average width of the 90% intervals: {np.mean(bt[:, 1] - bt[:, 0]):.2f} and '
      f'{np.mean(bk[:, 1] - bk[:, 0]):.2f}')
print(f'the intervals contain the true trend in '
      f'{np.sum((tau_true >= bt[:, 0]) & (tau_true <= bt[:, 1]))} and '
      f'{np.sum((tau_true >= bk[:, 0]) & (tau_true <= bk[:, 1]))} of the {T} periods')

# ---- 2. an AR(1) transitory component: the adaptation ----
rng = np.random.default_rng(2)
rho_true = .5
tau_true2 = tau0_true + np.cumsum(np.sqrt(omega2_true) * rng.standard_normal(T))
y2 = tau_true2 + lfilter([1.0], [1.0, -rho_true], np.sqrt(sig2_true) * rng.standard_normal(T))
sig2, omega2, tau0, rho = 1.0, .1, a0, 0.0
store_ar = np.zeros((nsim, 4))
for isim in range(burnin + nsim):
    Hrho = ssm.diffmat(T, rho)
    HHrho = Hrho.T @ Hrho
    K = HH / omega2 + HHrho / sig2                      # Step 4 of guides/uc_states
    c = H.T @ np.r_[tau0, np.zeros(T - 1)] / omega2 + HHrho @ y2 / sig2
    tau, _, _ = ssm.simulate_states(K, c, rng=rng)
    e = y2 - tau                                        # the transitory component
    Krho = e[:T - 1] @ e[:T - 1] / sig2
    rhohat = (e[:T - 1] @ e[1:]) / (e[:T - 1] @ e[:T - 1])
    rho = ssm.tnormrnd(rhohat, 1 / Krho, -1, 1, rng=rng)[0]
    u = ssm.diffmat(T, rho) @ e                         # the AR(1) residuals
    sig2 = 1 / rng.gamma(nu_sig + T / 2, 1 / (S_sig + u @ u / 2))
    eta = H @ tau - np.r_[tau0, np.zeros(T - 1)]
    omega2 = 1 / rng.gamma(nu_om + T / 2, 1 / (S_om + eta @ eta / 2))
    Ktau0 = 1 / b0 + 1 / omega2
    tau0 = (a0 / b0 + tau[0] / omega2) / Ktau0 + rng.standard_normal() / np.sqrt(Ktau0)
    if isim >= burnin:
        store_ar[isim - burnin] = rho, sig2, omega2, tau0

# ---- 2. the checks ----
IFa = ssm.inefficiency_factor(store_ar, L)
pma = store_ar.mean(axis=0)
psda = store_ar.std(axis=0, ddof=1)
pqa = np.quantile(store_ar, [.05, .95], axis=0, method='hazen')
namesa = ['rho'] + names
trutha = np.r_[rho_true, truth]
print(f'\nAR(1) transitory component, {nsim} draws after a burn-in of {burnin}')
for k in range(4):
    print(fmt.format(namesa[k], trutha[k], pma[k], pqa[0, k], pqa[1, k], IFa[k]))
# the posterior of (rho, sig2, omega2) on a grid of (atanh(rho), log(sig2), log(omega2)),
# with tau0 in the states as above; H_rho' H_rho = I + rho B1 + rho^2 B2
na = 24
g1 = span(np.arctanh(store_ar[:, 0]), na)
g2 = span(np.log(store_ar[:, 1]), na)
g3 = span(np.log(store_ar[:, 2]), na)
R3, S3, O3 = np.meshgrid(np.tanh(g1), np.exp(g2), np.exp(g3), indexing='ij')
Lag = sp.csc_array((np.ones(T - 1), (np.arange(1, T), np.arange(T - 1))), shape=(T, T))
B1 = -(Lag + Lag.T)
B2 = Lag.T @ Lag
lpa = np.zeros((na, na, na))
m0a = np.zeros((na, na, na))
for k in np.ndindex(na, na, na):
    iR = (sp.eye_array(T) + R3[k] * B1 + R3[k]**2 * B2) / S3[k]
    ll, ahat, _ = ssm.intlike(y2, Z, iR, A0 + A1 / O3[k], b)
    lpa[k] = (ll + logig(S3[k], nu_sig, S_sig) + logig(O3[k], nu_om, S_om)
              + np.log((1 - R3[k]**2) / 2) + np.log(S3[k]) + np.log(O3[k]))   # rho ~ U(-1, 1)
    m0a[k] = ahat[0]
wa = np.exp(lpa - lpa.max())
wa = wa / wa.sum()
gma = np.array([np.sum(wa * R3), np.sum(wa * S3), np.sum(wa * O3), np.sum(wa * m0a)])
gsda = np.sqrt(np.array([np.sum(wa * R3**2), np.sum(wa * S3**2), np.sum(wa * O3**2)])
               - gma[:3]**2)
mcsea = ssm.mcse(store_ar, L)
print(f'the posterior on a {na} x {na} x {na} grid, from ssm.intlike')
faces = np.r_[wa[[0, -1], :, :].ravel(), wa[:, [0, -1], :].ravel(), wa[:, :, [0, -1]].ravel()]
print(f'largest weight on a face of the grid, relative to the largest weight: '
      f'{faces.max() / wa.max():.1e}')
print(f'posterior means, Gibbs minus grid, in Monte Carlo standard errors: '
      f'{mat2str((pma - gma) / mcsea, 1)}')
print(f'posterior standard deviations of rho, sig2, omega2, Gibbs over grid: '
      f'{mat2str(psda[:3] / gsda, 3)}')

# ---- the figure: the draws of section 1 and their histograms, with the grid posterior ----
fig, axes = plt.subplots(2, 3, figsize=(7.09, 3.94))
lab = ['$\\sigma^2$', '$\\omega^2$', '$\\tau_0$']
pw = [w.sum(axis=1) / np.exp(ls) / (ls[1] - ls[0]), w.sum(axis=0) / np.exp(lo) / (lo[1] - lo[0])]
gx = [np.exp(ls), np.exp(lo)]
x0 = np.linspace(store_theta[:, 2].min(), store_theta[:, 2].max(), 200)
gx.append(x0)                                           # a mixture of normals for tau0
pw.append(np.sum(w.ravel() * np.exp(-(x0[:, None] - m0.ravel())**2 / (2 * v0.ravel()))
                 / np.sqrt(2 * np.pi * v0.ravel()), axis=1))
for k in range(3):
    ax = axes[0, k]
    ax.plot(store_theta[:, k], color=(.35, .35, .35), linewidth=.3)
    ax.set_xlim(0, nsim)
    ax.set_xticks([0, 10000, 20000])
    ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0), useMathText=True)
    ax.set_title(lab[k], fontsize=10)
    ax = axes[1, k]
    ax.hist(store_theta[:, k], bins=60, density=True, color=(.75, .75, .75), alpha=.6,
            edgecolor='none')
    keep = pw[k] > 1e-3 * pw[k].max()
    ax.plot(gx[k][keep], pw[k][keep], 'k', linewidth=1.2)
    ax.axvline(truth[k], color='k', linestyle='--', linewidth=1)
for ax in axes.ravel():
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(labelsize=8)
    ax.xaxis.get_offset_text().set_fontsize(8)
fig.tight_layout()
fig.savefig(os.path.join(HERE, 'fig_chain.png'), dpi=150)
plt.close(fig)
