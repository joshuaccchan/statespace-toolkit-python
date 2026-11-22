"""guide - the code of guides/sv_gibbs/README.md: a Gibbs sampler for the stochastic volatility
model of guides/sv_states. The sections called "the code to adapt" are all a sampler needs. Saves
fig_chain.png next to this file.
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
from scipy.stats import norm  # noqa: E402

import statespace_toolkit as ssm  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(146)


def mat2str(x, d):
    """x rounded to d decimals and printed as MATLAB's mat2str(round(x, d)) prints it"""
    return '[' + ' '.join(f'{v:.15g}' for v in np.round(np.ravel(x), d) + 0.0) + ']'


def ksdensity(x):
    """MATLAB's ksdensity(x): a normal kernel with the bandwidth of its default rule, at 100
    points from three bandwidths below the smallest draw to three above the largest"""
    x = np.asarray(x, dtype=float)
    bw = np.median(np.abs(x - np.median(x))) / .6745 * (4 / (3 * x.size))**(1 / 5)
    xi = np.linspace(x.min() - 3 * bw, x.max() + 3 * bw, 100)
    return norm.pdf((xi[:, None] - x) / bw).mean(axis=1) / bw, xi


# ---- the data of guides/sv_states ----
T, mu_true, muh_true, phi_true, sig2h_true = 1000, .1, 0.0, .97, .04
h_true = np.zeros(T)
h_true[0] = muh_true + np.sqrt(sig2h_true / (1 - phi_true**2)) * rng.standard_normal()
for t in range(1, T):
    h_true[t] = (muh_true + phi_true * (h_true[t - 1] - muh_true)
                 + np.sqrt(sig2h_true) * rng.standard_normal())
y = mu_true + np.exp(h_true / 2) * rng.standard_normal(T)
tic = time.perf_counter()

# ---- 1. the Gibbs sampler: the code to adapt ----
mu0, Vmu = 0.0, 10.0                                    # mu ~ N(mu0, Vmu)
muh0, Vmuh = 1.0, 10.0                                  # mu_h ~ N(muh0, Vmuh)
phi0, Vphi = .97, .1**2                                 # phi ~ TN(phi0, Vphi) on (-1, 1)
nuh, Sh = 5.0, .16                                      # sigma_h^2 ~ IG(nuh, Sh)
# the priors of sigma_h^2, phi and mu_h, with the keys of ssm.sv_params
hyp = dict(nuh=nuh, Sh=Sh, phi0=phi0, Vphi=Vphi, mu0=muh0, Vmu=Vmuh)
offset = 1e-4
nsim, burnin = 20000, 1000
muh, phi, sig2h = np.log(y.var(ddof=1)), .98, .04       # starting values
h = muh * np.ones(T)
store_theta = np.zeros((nsim, 4))
nacc = 0
store_h = np.zeros((T, nsim // 10))
for isim in range(burnin + nsim):
    Kmu = 1 / Vmu + np.sum(np.exp(-h))                  # K_mu = 1/V_mu + sum_t e^{-h_t}
    # mu ~ N(c_mu/K_mu, 1/K_mu)
    mu = (mu0 / Vmu + np.exp(-h) @ y) / Kmu + rng.standard_normal() / np.sqrt(Kmu)

    ystar = np.log((y - mu)**2 + offset)                # s, then h: guides/sv_states
    h, _ = ssm.ksc_ar1_mean(ystar, h, muh, phi, sig2h, rng=rng)

    # sigma_h^2, phi, then mu_h; each output holds one value per series
    [muh], [phi], [sig2h], [acc] = ssm.sv_params(h, muh, phi, hyp, 1, rng=rng)
    nacc += acc * (isim >= burnin)

    if isim >= burnin:
        store_theta[isim - burnin] = mu, muh, phi, sig2h
        if (isim - burnin + 1) % 10 == 0:
            store_h[:, (isim - burnin + 1) // 10 - 1] = h

# ---- 1. the checks ----
t_gibbs = time.perf_counter() - tic
names = ['mu', 'mu_h', 'phi', 'sigma_h^2']
truth = np.array([mu_true, muh_true, phi_true, sig2h_true])
Ls = [50, 100, 200, 400]
IFL = np.array([ssm.inefficiency_factor(store_theta, Lj) for Lj in Ls])
L = 200                                                 # Bartlett truncation lag
IF = ssm.inefficiency_factor(store_theta, L)
mcse = ssm.mcse(store_theta, L)
pm = store_theta.mean(axis=0)
psd = store_theta.std(axis=0, ddof=1)
pq = np.quantile(store_theta, [.05, .95], axis=0, method='hazen')
ZL, _, _ = ssm.geweke(store_theta, L=L)
print(f'\nthe Gibbs sampler, {nsim} draws after a burn-in of {burnin}: {t_gibbs:.1f} s; phi '
      f'accepted in {nacc / nsim:.3f} of the draws')
fmt = ('{:<10} true {:6.3f}, posterior mean {:7.4f} (sd {:.4f}), 90% interval ({:7.4f}, '
       '{:7.4f}), MCSE {:.4f}')
for i in range(4):
    print(fmt.format(names[i], truth[i], pm[i], psd[i], pq[0, i], pq[1, i], mcse[i]))
print(f'inefficiency factors by truncation lag L (rows L = {mat2str(Ls, 0)}):')
print(np.round(IFL, 1))
print(f"Geweke's Z with L = {L}: {mat2str(ZL, 2)}")
vol = np.exp(store_h / 2)
bv = np.quantile(vol, [.05, .95], axis=1, method='hazen').T
vt = np.exp(h_true / 2)
wm = np.exp(-h_true) @ y / np.sum(np.exp(-h_true))   # the mean of y weighted by the true e^{-h_t}
print(f'mu: the weighted mean of y with the true volatilities is {wm:.4f}, standard error '
      f'{1 / np.sqrt(np.sum(np.exp(-h_true))):.4f}')
print(f'e^{{h_t/2}}: the 90% intervals contain the truth in '
      f'{np.sum((vt >= bv[:, 0]) & (vt <= bv[:, 1]))} of {T} periods; average width '
      f'{np.mean(bv[:, 1] - bv[:, 0]):.2f}')
im = np.argsort(-np.abs(store_theta[:, 1]), kind='stable')   # the draws of mu_h farthest from 0
print(f'the three draws of mu_h farthest from 0: {mat2str(store_theta[im[:3], 1], 2)}, with phi '
      f'{mat2str(store_theta[im[:3], 2], 4)} and sigma_h^2 {mat2str(store_theta[im[:3], 3], 4)}')
print(f'phi above 0.995 in {np.sum(store_theta[:, 2] > .995)} of {nsim} draws')

# ---- 2. the check: the sampler of Chan and Grant (2016), with the exact conditional of h ----
muh, phi, sig2h, mu = np.log(y.var(ddof=1)), .98, .04, y.mean()
Hphi = ssm.diffmat(T, phi)
P = Hphi.T @ sp.diags_array(np.r_[1 - phi**2, np.ones(T - 1)] / sig2h) @ Hphi
e2 = (y - mu)**2


def grad0(x):
    """the gradient of log p(h | y) and its negative Hessian, at the starting values"""
    return (-(P @ (x - muh)) - .5 + .5 * e2 * np.exp(-x),
            P + sp.diags_array(.5 * e2 * np.exp(-x)))


h, _, _ = ssm.mode_newton(muh * np.ones(T), grad0)      # start at the mode
store_x = np.zeros((nsim, 4))
nacc_h = 0
tic = time.perf_counter()
for isim in range(burnin + nsim):
    Kmu = 1 / Vmu + np.sum(np.exp(-h))
    mu = (mu0 / Vmu + np.exp(-h) @ y) / Kmu + rng.standard_normal() / np.sqrt(Kmu)

    Hphi = ssm.diffmat(T, phi)                          # h by accept-reject MH: guides/sv_states
    P = Hphi.T @ sp.diags_array(np.r_[1 - phi**2, np.ones(T - 1)] / sig2h) @ Hphi
    e2 = (y - mu)**2

    def logf(x):
        return -.5 * (x - muh) @ (P @ (x - muh)) - .5 * np.sum(x) - .5 * np.exp(-x) @ e2

    def gradK(x):
        return (-(P @ (x - muh)) - .5 + .5 * e2 * np.exp(-x),
                P + sp.diags_array(.5 * e2 * np.exp(-x)))

    h, acc, _ = ssm.armh(h, logf, gradK, force_accept=(isim == 0), rng=rng)
    nacc_h += acc * (isim >= burnin)

    [muh], [phi], [sig2h], _ = ssm.sv_params(h, muh, phi, hyp, 1, rng=rng)

    if isim >= burnin:
        store_x[isim - burnin] = mu, muh, phi, sig2h
t_armh = time.perf_counter() - tic
mcse_x = ssm.mcse(store_x, L)
print(f'\nthe sampler of Chan and Grant (2016): {t_armh:.1f} s; h accepted in '
      f'{nacc_h / nsim:.3f} of the draws; inefficiency factors '
      f'{mat2str(ssm.inefficiency_factor(store_x, L), 1)}')
print(f'posterior means {mat2str(store_x.mean(axis=0), 4)}, sd '
      f'{mat2str(store_x.std(axis=0, ddof=1), 4)}')
print(f'posterior means, mixture minus exact, in Monte Carlo standard errors of the difference: '
      f'{mat2str((pm - store_x.mean(axis=0)) / np.sqrt(mcse**2 + mcse_x**2), 1)}')
print(f'posterior sd, mixture over exact: {mat2str(psd / store_x.std(axis=0, ddof=1), 3)}')

# ---- 3. a random-walk log-volatility: the adaptation ----
rng = np.random.default_rng(3)
h0_true, sig2h_rw = 0.0, .01
h_rw = h0_true + np.cumsum(np.sqrt(sig2h_rw) * rng.standard_normal(T))   # h_t = h_{t-1} + u_t
y_rw = mu_true + np.exp(h_rw / 2) * rng.standard_normal(T)

a0, b0 = 0.0, 10.0           # h_0 ~ N(a0, b0)
nuh_rw, Sh_rw = 5.0, .04     # sigma_h^2 ~ IG(nuh_rw, Sh_rw)
H = ssm.diffmat(T)
h0, sig2h = np.log(y_rw.var(ddof=1)), .01
h = h0 * np.ones(T)
store_rw = np.zeros((nsim, 4))
for isim in range(burnin + nsim):
    Kmu = 1 / Vmu + np.sum(np.exp(-h))
    mu = (mu0 / Vmu + np.exp(-h) @ y_rw) / Kmu + rng.standard_normal() / np.sqrt(Kmu)

    ystar = np.log((y_rw - mu)**2 + offset)
    h = ssm.ksc_rw_h0(ystar, h, sig2h, h0, rng=rng)      # s, then h, given h_0

    u = H @ h - np.r_[h0, np.zeros(T - 1)]               # h_t - h_{t-1}, with h_0
    sig2h = 1 / rng.gamma(nuh_rw + T / 2, 1 / (Sh_rw + u @ u / 2))

    Kh0 = 1 / b0 + 1 / sig2h                             # h_0 given h_1 and sigma_h^2
    h0 = (a0 / b0 + h[0] / sig2h) / Kh0 + rng.standard_normal() / np.sqrt(Kh0)

    if isim >= burnin:
        store_rw[isim - burnin] = mu, sig2h, h0, h[0]
qrw = np.quantile(store_rw[:, 1], [.05, .95], method='hazen')
print(f'\na random-walk log-volatility, on data generated from it, {nsim} draws after {burnin}')
print(f'sigma_h^2 true {sig2h_rw:.3f}, posterior mean {store_rw[:, 1].mean():.4f}, 90% interval '
      f'({qrw[0]:.4f}, {qrw[1]:.4f})')
# the same posterior with h drawn from its exact conditional distribution by ssm.armh
HH = H.T @ H
nsim_x = 10000
h0, sig2h = np.log(y_rw.var(ddof=1)), .01
e2 = (y_rw - y_rw.mean())**2


def g0(x):
    """the gradient of log p(h | y, h_0) and its negative Hessian, at the starting values"""
    return (-(HH @ (x - h0)) / sig2h - .5 + .5 * e2 * np.exp(-x),
            HH / sig2h + sp.diags_array(.5 * e2 * np.exp(-x)))


h, _, _ = ssm.mode_newton(h0 * np.ones(T), g0)          # start at the mode
store_rwx = np.zeros((nsim_x, 4))
nacc_h = 0
for isim in range(burnin + nsim_x):
    Kmu = 1 / Vmu + np.sum(np.exp(-h))
    mu = (mu0 / Vmu + np.exp(-h) @ y_rw) / Kmu + rng.standard_normal() / np.sqrt(Kmu)
    e2 = (y_rw - mu)**2

    def logf(x):
        return (-.5 * (x - h0) @ (HH @ (x - h0)) / sig2h - .5 * np.sum(x)
                - .5 * np.exp(-x) @ e2)

    def gradK(x):
        return (-(HH @ (x - h0)) / sig2h - .5 + .5 * e2 * np.exp(-x),
                HH / sig2h + sp.diags_array(.5 * e2 * np.exp(-x)))

    h, acc, _ = ssm.armh(h, logf, gradK, force_accept=(isim == 0), rng=rng)
    nacc_h += acc * (isim >= burnin)
    u = H @ h - np.r_[h0, np.zeros(T - 1)]
    sig2h = 1 / rng.gamma(nuh_rw + T / 2, 1 / (Sh_rw + u @ u / 2))
    Kh0 = 1 / b0 + 1 / sig2h
    h0 = (a0 / b0 + h[0] / sig2h) / Kh0 + rng.standard_normal() / np.sqrt(Kh0)
    if isim >= burnin:
        store_rwx[isim - burnin] = mu, sig2h, h0, h[0]
print(f'the same with h from its exact conditional, {nsim_x} draws: h accepted in '
      f'{nacc_h / nsim_x:.3f}')
d_rw = ((store_rw.mean(axis=0) - store_rwx.mean(axis=0))
        / np.sqrt(ssm.mcse(store_rw, L)**2 + ssm.mcse(store_rwx, L)**2))
print(f'posterior means, mixture minus exact, in Monte Carlo standard errors of the difference: '
      f'{mat2str(d_rw, 1)}')

# ---- the figure: the draws of the four parameters, their histograms and the check ----
fig, axes = plt.subplots(2, 4, figsize=(7.87, 4.33))
lab = ['$\\mu$', '$\\mu_h$', '$\\phi$', '$\\sigma_h^2$']
for i in range(4):
    ax = axes[0, i]
    ax.plot(np.arange(1, nsim + 1), store_theta[:, i], color=(.35, .35, .35), linewidth=.3)
    ax.set_xlim(0, nsim)
    ax.set_xticks([0, 10000, 20000])
    ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0), useMathText=True)
    ax.set_title(lab[i], fontsize=10)
for i in range(4):
    ax = axes[1, i]
    xl = np.quantile(store_theta[:, i], [.0005, .9995], method='hazen')  # the central 99.9%
    hh = ax.hist(store_theta[:, i], bins=60, range=tuple(xl), density=True,
                 color=(.75, .75, .75), alpha=.6, edgecolor='none')[2]
    fk, xk = ksdensity(store_x[:, i])
    hk, = ax.plot(xk, fk, 'k', linewidth=1.2)
    hx = ax.axvline(truth[i], color='k', linestyle='--', linewidth=1)
    ax.set_xlim(xl)
for ax in axes.ravel():
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(labelsize=8)
    ax.xaxis.get_offset_text().set_fontsize(8)
fig.legend([hh, hk, hx], ['Gibbs sampler with the mixture',
                          'sampler with the exact conditional of h', 'true value'],
           loc='lower center', ncols=3, frameon=False, fontsize=8)
fig.tight_layout(rect=(0, .06, 1, 1))
fig.savefig(os.path.join(HERE, 'fig_chain.png'), dpi=150)
plt.close(fig)
