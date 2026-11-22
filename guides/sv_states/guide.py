"""guide - the code of guides/sv_states/README.md: the log-volatilities of a stochastic volatility
model, drawn given the parameters by the auxiliary mixture sampler and from their exact
conditional distribution. The sections called "the code to adapt" are all a sampler needs. Saves
fig_mixture.png and fig_vol.png next to this file.
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
from scipy.special import digamma  # noqa: E402
from scipy.stats import norm  # noqa: E402

import statespace_toolkit as ssm  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(146)


def mat2str(x):
    """x printed as MATLAB's mat2str prints a row vector"""
    return '[' + ' '.join(f'{v:.15g}' for v in np.ravel(x) + 0.0) + ']'


# ---- the data, in percent ----
T, mu, muh, phi, sig2h = 1000, .1, 0.0, .97, .04                    # the parameters, known
h_true = np.zeros(T)
# h_1 ~ N(mu_h, sigma_h^2/(1 - phi^2))
h_true[0] = muh + np.sqrt(sig2h / (1 - phi**2)) * rng.standard_normal()
for t in range(1, T):
    h_true[t] = muh + phi * (h_true[t - 1] - muh) + np.sqrt(sig2h) * rng.standard_normal()
y = mu + np.exp(h_true / 2) * rng.standard_normal(T)

# ---- 1. the transformed data: the code to adapt ----
offset = 1e-4
ystar = np.log((y - mu)**2 + offset)  # y*_t = log((y_t - mu)^2 + c)

# ---- 1. the checks ----
print(f'\nthe data: T = {T}, sample sd {y.std(ddof=1):.2f}, sd of the true e^{{h_t/2}} from '
      f'{np.exp(h_true / 2).min():.2f} to {np.exp(h_true / 2).max():.2f}')
h = muh * np.ones(T)             # the current path, for one draw
state = rng.bit_generator.state  # for the check against ssm.ksc_ar1_mean

# ---- 2. one draw of the indicators and of h: the code to adapt ----
pj = np.array([.0073, .10556, .00002, .04395, .34001, .24566, .2575])         # p_i
mj = np.array([-10.12999, -3.97281, -8.56686, 2.77786, .61942, 1.79518,
               -1.08819]) - 1.2704                                            # mu_i - 1.2704
s2j = np.array([5.79596, 2.61369, 5.17950, .16735, .64009, .34023, 1.26261])  # sigma_i^2

Hphi = ssm.diffmat(T, phi)                                     # H_phi = I - phi L
iS = sp.diags_array(np.r_[1 - phi**2, np.ones(T - 1)] / sig2h)  # S^{-1}
P = Hphi.T @ iS @ Hphi                                         # P = H_phi' S^{-1} H_phi

# p_i N(y*_t; h_t + mu_i - 1.2704, sigma_i^2), with t down the rows and i across the columns
q = pj * norm.pdf(ystar[:, None], h[:, None] + mj, np.sqrt(s2j))
cq = np.cumsum(q / q.sum(axis=1, keepdims=True), axis=1)  # the cumulative probabilities
s = np.sum(rng.random((T, 1)) > cq, axis=1)               # s_t from its seven probabilities
K = P + sp.diags_array(1 / s2j[s])                        # K = P + Sigma_s^{-1}
c = P @ (muh * np.ones(T)) + (ystar - mj[s]) / s2j[s]     # c = P mu_h 1 + Sigma_s^{-1}(y* - d_s)
h, _, _ = ssm.simulate_states(K, c, rng=rng)

# ---- 2. the checks ----
rng.bit_generator.state = state
h_ksc, s_ksc = ssm.ksc_ar1_mean(ystar, muh * np.ones(T), muh, phi, sig2h, rng=rng)
print('\nthe top left corner of K at this draw:')
with np.printoptions(precision=4, suppress=True):
    print(K.toarray()[:4, :4])
print(f'indicators of the first four periods: {mat2str(s[:4] + 1)}')   # numbered from 1
print(f'the steps above against ssm.ksc_ar1_mean, same random numbers: indicators equal: '
      f'{int(np.array_equal(s, s_ksc))}, largest difference in h {np.max(np.abs(h - h_ksc)):.1e}')
x = np.linspace(-30, 8, 76001)
dx = x[1] - x[0]
lf_exact = (x - np.exp(x)) / 2 - np.log(2 * np.pi) / 2       # the log density of log chi^2_1
f_exact = np.exp(lf_exact)
f_mix = norm.pdf(x[:, None], mj, np.sqrt(s2j)) @ pj
print(f'the mixture against log chi^2_1: means {pj @ mj:.4f} and {digamma(.5) + np.log(2):.4f}, '
      f'variances {pj @ (s2j + mj**2) - (pj @ mj)**2:.4f} and {np.pi**2 / 2:.4f}, '
      f'L1 distance {np.sum(np.abs(f_mix - f_exact)) * dx:.4f}')
lr = np.abs(np.log(f_mix) - lf_exact)                        # the log ratio of the two densities
print(f'log of the mixture density over the exact one: largest absolute value '
      f'{lr[(x > -8) & (x < 2)].max():.2f} on (-8, 2), {lr[(x > -15) & (x <= -8)].max():.2f} '
      f'on (-15, -8)')
tic = time.perf_counter()

# ---- 3. the chain: the code to adapt ----
nsim, burnin = 5000, 500
h = muh * np.ones(T)                                           # the starting path
store_h = np.zeros((T, nsim))
for isim in range(burnin + nsim):
    h, _ = ssm.ksc_ar1_mean(ystar, h, muh, phi, sig2h, rng=rng)  # s given h, then h given s
    if isim >= burnin:
        store_h[:, isim - burnin] = h

# ---- 3. the checks ----
t_mix = time.perf_counter() - tic
L = 100                                                             # Bartlett truncation lag
IFm = ssm.inefficiency_factor(store_h.T, L)
vol = np.exp(store_h / 2)                                    # the standard deviation e^{h_t/2}
bm = np.quantile(vol, [.05, .95], axis=1, method='hazen').T
vm = vol.mean(axis=1)
vt = np.exp(h_true / 2)
print(f'\nthe mixture sampler, {nsim} draws after a burn-in of {burnin}: '
      f'{1000 * t_mix / (nsim + burnin):.2f} ms per draw')
print(f'inefficiency factors of the h_t: median {np.median(IFm):.1f}, largest {IFm.max():.1f}')
print(f'this realization: the 90% intervals of e^{{h_t/2}} contain the truth in '
      f'{np.sum((vt >= bm[:, 0]) & (vt <= bm[:, 1]))} of {T} periods; average width '
      f'{np.mean(bm[:, 1] - bm[:, 0]):.2f}')
tic = time.perf_counter()

# ---- 4. the exact conditional by accept-reject Metropolis-Hastings: the code to adapt ----
e2 = (y - mu)**2                                                    # no offset here


def logf(x):
    """log p(h | y) + const"""
    return -.5 * (x - muh) @ (P @ (x - muh)) - .5 * np.sum(x) - .5 * np.exp(-x) @ e2


def gradK(x):
    """the gradient of logf and its negative Hessian"""
    return (-(P @ (x - muh)) - .5 + .5 * e2 * np.exp(-x),
            P + sp.diags_array(.5 * e2 * np.exp(-x)))


h, _, _ = ssm.mode_newton(muh * np.ones(T), gradK)                  # start at the mode
store_hx = np.zeros((T, nsim))
nacc = 0
for isim in range(burnin + nsim):
    h, acc, _ = ssm.armh(h, logf, gradK, force_accept=(isim == 0), rng=rng)
    if isim >= burnin:
        store_hx[:, isim - burnin] = h
        nacc += acc

# ---- 4. the checks ----
t_armh = time.perf_counter() - tic
IFx = ssm.inefficiency_factor(store_hx.T, L)
volx = np.exp(store_hx / 2)
psd = volx.std(axis=1, ddof=1)
dvol = np.abs(vm - volx.mean(axis=1))
print(f'\nthe exact conditional by ssm.armh: acceptance rate {nacc / nsim:.3f}, '
      f'{1000 * t_armh / (nsim + burnin):.2f} ms per draw')
print(f'inefficiency factors of the h_t: median {np.median(IFx):.1f}, largest {IFx.max():.1f}')
print(f'posterior means of e^{{h_t/2}}, mixture minus exact: largest difference '
      f'{dvol.max():.3f}, {np.max(dvol / psd):.2f} posterior sd')

# ---- 5. a random-walk log-volatility: the adaptation ----
h0 = 0.0                                                            # the known initial value
h = muh * np.ones(T)
state = rng.bit_generator.state
q = pj * norm.pdf(ystar[:, None], h[:, None] + mj, np.sqrt(s2j))
s = np.sum(rng.random((T, 1)) > np.cumsum(q / q.sum(axis=1, keepdims=True), axis=1), axis=1)
H = ssm.diffmat(T)                                       # first differences
K = H.T @ H / sig2h + sp.diags_array(1 / s2j[s])         # K = H'H/sigma_h^2 + Sigma_s^{-1}
c = (H.T @ np.r_[h0, np.zeros(T - 1)] / sig2h            # c = H' htilde_0/sigma_h^2
     + (ystar - mj[s]) / s2j[s])                         #   + Sigma_s^{-1}(y* - d_s)
h, _, _ = ssm.simulate_states(K, c, rng=rng)
rng.bit_generator.state = state
h_rw = ssm.ksc_rw_h0(ystar, muh * np.ones(T), sig2h, h0, rng=rng)
print(f'\na random walk: the answer against ssm.ksc_rw_h0, same random numbers: '
      f'{np.max(np.abs(h - h_rw)):.1e}')

# ---- the figures ----
fig, axes = plt.subplots(1, 2, figsize=(7.09, 2.95))
for ax, fe, fm, name in ((axes[0], f_exact, f_mix, 'density'),
                         (axes[1], lf_exact, np.log(f_mix), 'log density')):
    he, = ax.plot(x, fe, 'k', linewidth=1.3)
    hm, = ax.plot(x, fm, '--', color=(.45, .45, .45), linewidth=1.3)
    ax.set_xlim(-12, 4)
    ax.set_title(name, fontsize=10)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(labelsize=8)
axes[1].set_ylim(-12, 0)
fig.legend([he, hm], ['$\\log\\chi^2_1$', 'seven-component mixture'], loc='lower center',
           ncols=2, frameon=False, fontsize=8)
fig.tight_layout(rect=(0, .08, 1, 1))
fig.savefig(os.path.join(HERE, 'fig_mixture.png'), dpi=150)
plt.close(fig)

t = np.arange(1, T + 1)
fig, axes = plt.subplots(2, 1, figsize=(7.09, 4.72))
axes[0].plot(t, y, color=(.45, .45, .45), linewidth=.5)
axes[0].set_title('$y_t$', fontsize=10)
ax = axes[1]
hb = ssm.shaded_band(t, bm[:, 0], bm[:, 1], ax=ax)
ht, = ax.plot(t, vt, 'k--', linewidth=1)
hm, = ax.plot(t, vm, 'k', linewidth=1.2)
ax.set_title('$e^{h_t/2}$', fontsize=10)
for ax in axes:
    ax.set_xlim(1, T)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(labelsize=8)
fig.legend([ht, hm, hb], ['true value', 'posterior mean', '90% pointwise credible interval'],
           loc='lower center', ncols=3, frameon=False, fontsize=8)
fig.tight_layout(rect=(0, .05, 1, 1))
fig.savefig(os.path.join(HERE, 'fig_vol.png'), dpi=150)
plt.close(fig)
