"""guide - the code of guides/sv_t/README.md: a Gibbs sampler for the stochastic volatility model
of guides/sv_gibbs with Student-t errors, on data generated with the log-volatilities and errors
of guides/sv_states and t errors with 5 degrees of freedom; the model with normal errors on the
same data, to show what the t errors change; the same posterior by the sampler of Chan and Grant
(2016), which draws the log-volatilities from their exact conditional distribution, as a check;
then errors with an outlier component. The sections called "the code to adapt" are all a sampler
needs. Saves fig_vol.png and fig_nu.png next to this file.
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
from scipy.optimize import minimize_scalar  # noqa: E402
from scipy.special import gammaln  # noqa: E402
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


# ---- the data: the h_t and eps_t of guides/sv_states, with t errors ----
T, mu_true, muh_true, phi_true, sig2h_true, nu_true = 1000, .1, 0.0, .97, .04, 5.0
h_true = np.zeros(T)
h_true[0] = muh_true + np.sqrt(sig2h_true / (1 - phi_true**2)) * rng.standard_normal()
for t in range(1, T):
    h_true[t] = (muh_true + phi_true * (h_true[t - 1] - muh_true)
                 + np.sqrt(sig2h_true) * rng.standard_normal())
epsn = rng.standard_normal(T)                           # the eps_t of guides/sv_states
lam_true = 1 / rng.gamma(nu_true / 2, 2 / nu_true, T)   # lambda_t ~ IG(nu/2, nu/2)
y = mu_true + np.sqrt(lam_true) * np.exp(h_true / 2) * epsn
tic = time.perf_counter()

# ---- 1. the Gibbs sampler for SV-t: the code to adapt ----
mu0, Vmu = 0.0, 10.0                                # mu ~ N(mu0, Vmu)
muh0, Vmuh = 1.0, 10.0                              # mu_h ~ N(muh0, Vmuh)
phi0, Vphi = .97, .1**2                             # phi ~ TN(phi0, Vphi) on (-1, 1)
nuh, Sh = 5.0, .16                                  # sigma_h^2 ~ IG(nuh, Sh)
# the priors of sigma_h^2, phi and mu_h, with the keys of ssm.sv_params
hyp = dict(nuh=nuh, Sh=Sh, phi0=phi0, Vphi=Vphi, mu0=muh0, Vmu=Vmuh)
nuub = 100                                          # nu ~ U(2, nuub)
offset = 1e-4
nsim, burnin = 20000, 1000
muh, phi, sig2h, nu = np.log(y.var(ddof=1)), .98, .04, 5.0   # starting values
h = muh * np.ones(T)
lam = np.ones(T)
store_theta = np.zeros((nsim, 5))
nacc = 0
store_h = np.zeros((T, nsim // 10))
store_lam = np.zeros(T)
for isim in range(burnin + nsim):
    w = 1 / (lam * np.exp(h))                       # 1/(lambda_t e^{h_t})
    Kmu = 1 / Vmu + np.sum(w)                       # K_mu = 1/V_mu + sum_t 1/(lambda_t e^{h_t})
    mu = (mu0 / Vmu + w @ y) / Kmu + rng.standard_normal() / np.sqrt(Kmu)

    ystar = np.log((y - mu)**2 / lam + offset)      # y*_t = log((y_t - mu)^2/lambda_t + c)
    h, _ = ssm.ksc_ar1_mean(ystar, h, muh, phi, sig2h, rng=rng)

    e2 = (y - mu)**2 * np.exp(-h)                   # lambda_t ~ IG((nu + 1)/2, (nu + e2_t)/2)
    lam = 1 / rng.gamma((nu + 1) / 2, 2 / (nu + e2))

    nu, acc, _ = ssm.nu_studentt(lam, nu, nuub, rng=rng)   # nu given lambda
    nacc += acc * (isim >= burnin)

    [muh], [phi], [sig2h], _ = ssm.sv_params(h, muh, phi, hyp, 1, rng=rng)   # guides/sv_gibbs

    if isim >= burnin:
        store_theta[isim - burnin] = mu, muh, phi, sig2h, nu
        store_lam = store_lam + lam / nsim
        if (isim - burnin + 1) % 10 == 0:
            store_h[:, (isim - burnin + 1) // 10 - 1] = h

# ---- 1. the checks ----
t_svt = time.perf_counter() - tic
L = 400                                                 # Bartlett truncation lag
names = ['mu', 'mu_h', 'phi', 'sigma_h^2', 'nu']
truth = np.array([mu_true, muh_true, phi_true, sig2h_true, nu_true])
IF = ssm.inefficiency_factor(store_theta, L)
mcse = ssm.mcse(store_theta, L)
pm = store_theta.mean(axis=0)
psd = store_theta.std(axis=0, ddof=1)
pq = np.quantile(store_theta, [.05, .95], axis=0, method='hazen')
ZL, _, _ = ssm.geweke(store_theta, L=L)
print(f'\nSV-t, {nsim} draws after a burn-in of {burnin}: {t_svt:.1f} s; nu accepted in '
      f'{nacc / nsim:.3f} of the draws')
fmt = ('{:<10} true {:6.3f}, posterior mean {:7.4f} (sd {:.4f}), 90% interval ({:7.4f}, '
       '{:7.4f}), inefficiency {:.1f}')
for i in range(5):
    print(fmt.format(names[i], truth[i], pm[i], psd[i], pq[0, i], pq[1, i], IF[i]))
print(f"Geweke's Z with L = {L}: {mat2str(ZL, 2)}")
fk, xk = ksdensity(store_theta[:, 4])                   # the posterior mode of nu
ik = np.argmax(fk)
s1t, s2t = np.sum(np.log(lam_true)), np.sum(1 / lam_true)   # nu given the true lambda_t
nu_ml = minimize_scalar(lambda x: -(T * (x / 2 * np.log(x / 2) - gammaln(x / 2))
                                    - (x / 2 + 1) * s1t - x / 2 * s2t),
                        bounds=(2, 100), method='bounded').x
print(f'nu: posterior median {np.median(store_theta[:, 4]):.2f}, mode {xk[ik]:.2f}; the maximum '
      f'likelihood estimate from the true lambda_t is {nu_ml:.2f}')
wt = 1 / (lam_true * np.exp(h_true))                    # the true weights 1/(lambda_t e^{h_t})
print(f'mu: the mean of y weighted by the true 1/(lambda_t e^{{h_t}}) is '
      f'{wt @ y / np.sum(wt):.4f}, standard error {1 / np.sqrt(np.sum(wt)):.4f}')

# ---- 2. the model with normal errors on the same data: guides/sv_gibbs ----
muh, phi, sig2h = np.log(y.var(ddof=1)), .98, .04
h = muh * np.ones(T)
store_sv = np.zeros((nsim, 4))
store_hn = np.zeros((T, nsim // 10))
for isim in range(burnin + nsim):
    Kmu = 1 / Vmu + np.sum(np.exp(-h))
    mu = (mu0 / Vmu + np.exp(-h) @ y) / Kmu + rng.standard_normal() / np.sqrt(Kmu)
    h, _ = ssm.ksc_ar1_mean(np.log((y - mu)**2 + offset), h, muh, phi, sig2h, rng=rng)
    [muh], [phi], [sig2h], _ = ssm.sv_params(h, muh, phi, hyp, 1, rng=rng)
    if isim >= burnin:
        store_sv[isim - burnin] = mu, muh, phi, sig2h
        if (isim - burnin + 1) % 10 == 0:
            store_hn[:, (isim - burnin + 1) // 10 - 1] = h
qsv = np.quantile(store_sv, [.05, .95], axis=0, method='hazen')
print('\nSV with normal errors on the same data:')
for i in range(4):
    print(f'{names[i]:<10} posterior mean {store_sv[:, i].mean():7.4f}, 90% interval '
          f'({qsv[0, i]:7.4f}, {qsv[1, i]:7.4f})')
vt = np.exp(h_true / 2)                                 # the volatility of eps_t's scale
vm_t = np.exp(store_h / 2).mean(axis=1)
vm_n = np.exp(store_hn / 2).mean(axis=1)
bt = np.quantile(np.exp(store_h / 2), [.05, .95], axis=1, method='hazen').T
big = np.argsort(-lam_true, kind='stable')
print(f'posterior mean of e^{{h_t/2}} against the truth: RMSE '
      f'{np.sqrt(np.mean((vm_t - vt)**2)):.3f} under SV-t, '
      f'{np.sqrt(np.mean((vm_n - vt)**2)):.3f} under SV')
print(f'SV-t: 90% intervals contain the true e^{{h_t/2}} in '
      f'{np.sum((vt >= bt[:, 0]) & (vt <= bt[:, 1]))} of {T} periods')
print('the five largest lambda_t:')
for k in big[:5]:                                       # the period t is k + 1
    print(f'  period {k + 1:4d}: lambda_t {lam_true[k]:6.2f}, posterior mean {store_lam[k]:5.2f}; '
          f'|eps_t| {abs(epsn[k]):.2f}, |sqrt(lambda_t)*eps_t| '
          f'{abs(np.sqrt(lam_true[k]) * epsn[k]):.2f}; e^{{h_t/2}} true {vt[k]:.2f}, '
          f'SV-t {vm_t[k]:.2f}, SV {vm_n[k]:.2f}')

# ---- 3. the check: the sampler of Chan and Grant (2016), with the exact conditional of h ----
nsim_x = 10000
muh, phi, sig2h, nu, mu = np.log(y.var(ddof=1)), .98, .04, 5.0, y.mean()
lam = np.ones(T)
Hphi = ssm.diffmat(T, phi)
P = Hphi.T @ sp.diags_array(np.r_[1 - phi**2, np.ones(T - 1)] / sig2h) @ Hphi
e2 = (y - mu)**2 / lam


def grad0(x):
    """the gradient of log p(h | y, lambda) and its negative Hessian, at the starting values"""
    return (-(P @ (x - muh)) - .5 + .5 * e2 * np.exp(-x),
            P + sp.diags_array(.5 * e2 * np.exp(-x)))


h, _, _ = ssm.mode_newton(muh * np.ones(T), grad0)      # start at the mode
store_x = np.zeros((nsim_x, 5))
nacc_h = 0
for isim in range(burnin + nsim_x):
    w = 1 / (lam * np.exp(h))
    Kmu = 1 / Vmu + np.sum(w)
    mu = (mu0 / Vmu + w @ y) / Kmu + rng.standard_normal() / np.sqrt(Kmu)

    Hphi = ssm.diffmat(T, phi)
    P = Hphi.T @ sp.diags_array(np.r_[1 - phi**2, np.ones(T - 1)] / sig2h) @ Hphi
    e2 = (y - mu)**2 / lam

    def logf(x):
        return -.5 * (x - muh) @ (P @ (x - muh)) - .5 * np.sum(x) - .5 * np.exp(-x) @ e2

    def gradh(x):
        return (-(P @ (x - muh)) - .5 + .5 * e2 * np.exp(-x),
                P + sp.diags_array(.5 * e2 * np.exp(-x)))

    h, acc, _ = ssm.armh(h, logf, gradh, force_accept=(isim == 0), rng=rng)
    nacc_h += acc * (isim >= burnin)

    lam = 1 / rng.gamma((nu + 1) / 2, 2 / (nu + (y - mu)**2 * np.exp(-h)))
    nu, _, _ = ssm.nu_studentt(lam, nu, nuub, rng=rng)

    [muh], [phi], [sig2h], _ = ssm.sv_params(h, muh, phi, hyp, 1, rng=rng)

    if isim >= burnin:
        store_x[isim - burnin] = mu, muh, phi, sig2h, nu
mcse_x = ssm.mcse(store_x, L)
print(f'\nthe sampler of Chan and Grant (2016), {nsim_x} draws: h accepted in '
      f'{nacc_h / nsim_x:.3f} of the draws')
print(f'posterior means {mat2str(store_x.mean(axis=0), 4)}')
print(f'posterior means, mixture minus exact, in Monte Carlo standard errors of the difference: '
      f'{mat2str((pm - store_x.mean(axis=0)) / np.sqrt(mcse**2 + mcse_x**2), 1)}')

# ---- 4. an outlier component: the adaptation ----
kap = 10.0             # lambda_t = kap with probability p, else 1
ap, bp = 1.0, 19.0     # p ~ B(ap, bp)
muh, phi, sig2h, p = np.log(y.var(ddof=1)), .98, .04, .05
h = muh * np.ones(T)
lam = np.ones(T)
store_o = np.zeros((nsim, 5))
pout = np.zeros(T)
for isim in range(burnin + nsim):
    w = 1 / (lam * np.exp(h))
    Kmu = 1 / Vmu + np.sum(w)
    mu = (mu0 / Vmu + w @ y) / Kmu + rng.standard_normal() / np.sqrt(Kmu)

    h, _ = ssm.ksc_ar1_mean(np.log((y - mu)**2 / lam + offset), h, muh, phi, sig2h, rng=rng)

    e2 = (y - mu)**2 * np.exp(-h)                   # Pr(lambda_t = kap) given y_t, mu, h_t, p
    a1 = p / np.sqrt(kap) * np.exp(-e2 / (2 * kap))
    a0 = (1 - p) * np.exp(-e2 / 2)
    lam = 1 + (kap - 1) * (rng.random(T) < a1 / (a1 + a0))
    nout = np.sum(lam > 1)
    p = rng.beta(ap + nout, bp + T - nout)          # p ~ B(ap + n_out, bp + T - n_out)

    [muh], [phi], [sig2h], _ = ssm.sv_params(h, muh, phi, hyp, 1, rng=rng)

    if isim >= burnin:
        store_o[isim - burnin] = mu, muh, phi, sig2h, p
        pout = pout + (lam > 1) / nsim
# the outlier probability against the two normal densities it comes from
a1n = p * norm.pdf(y, mu, np.sqrt(kap * np.exp(h)))
a0n = (1 - p) * norm.pdf(y, mu, np.sqrt(np.exp(h)))
e2 = (y - mu)**2 * np.exp(-h)
a1 = p / np.sqrt(kap) * np.exp(-e2 / (2 * kap))
a0 = (1 - p) * np.exp(-e2 / 2)
print(f'\nan outlier component: the probability against norm.pdf, at the last draw: largest '
      f'difference {np.max(np.abs(a1 / (a1 + a0) - a1n / (a1n + a0n))):.1e}')
qo = np.quantile(store_o, [.05, .95], axis=0, method='hazen')
print(f'p: posterior mean {store_o[:, 4].mean():.4f}, 90% interval ({qo[0, 4]:.4f}, '
      f'{qo[1, 4]:.4f}); sigma_h^2: {store_o[:, 3].mean():.4f} ({qo[0, 3]:.4f}, {qo[1, 3]:.4f})')
bigz = np.argsort(-np.abs(np.sqrt(lam_true) * epsn), kind='stable')   # the largest true errors
print(f'periods with an outlier probability above 0.5: {np.sum(pout > .5)}; among the ten '
      f'largest lambda_t: {np.sum(pout[big[:10]] > .5)}; among the ten largest '
      f'|sqrt(lambda_t)*eps_t|: {np.sum(pout[bigz[:10]] > .5)}')

# ---- the figures ----
t = np.arange(1, T + 1)
fig, axes = plt.subplots(2, 1, figsize=(7.09, 4.72))
axes[0].plot(t, y, color=(.45, .45, .45), linewidth=.5)
axes[0].set_title('$y_t$', fontsize=10)
ax = axes[1]
hb = ssm.shaded_band(t, bt[:, 0], bt[:, 1], ax=ax)
ht, = ax.plot(t, vt, 'k--', linewidth=1)
hn, = ax.plot(t, vm_n, color=(.55, .55, .55), linewidth=1)
hm, = ax.plot(t, vm_t, 'k', linewidth=1.2)
ax.set_title('$e^{h_t/2}$', fontsize=10)
for ax in axes:
    ax.set_xlim(1, T)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(labelsize=8)
fig.legend([ht, hm, hb, hn], ['true value', 'posterior mean, SV-t',
                              '90% pointwise credible interval, SV-t', 'posterior mean, SV'],
           loc='lower center', ncols=2, frameon=False, fontsize=8)
fig.tight_layout(rect=(0, .08, 1, 1))
fig.savefig(os.path.join(HERE, 'fig_vol.png'), dpi=150)
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(7.09, 2.56))
ax = axes[0]
ax.plot(np.arange(1, nsim + 1), store_theta[:, 4], color=(.35, .35, .35), linewidth=.3)
ax.set_xlim(0, nsim)
ax.set_xticks([0, 5000, 10000, 15000, 20000])
ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0), useMathText=True)
ax.set_title('the draws of $\\nu$', fontsize=10)
ax = axes[1]
hh = ax.hist(store_theta[:, 4], bins=60, density=True, color=(.75, .75, .75), alpha=.6,
             edgecolor='none')[2]
hx = ax.axvline(nu_true, color='k', linestyle='--', linewidth=1)
ax.set_title('their histogram', fontsize=10)
for ax in axes:
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(labelsize=8)
    ax.xaxis.get_offset_text().set_fontsize(8)
fig.legend([hh, hx], ['posterior draws of $\\nu$', 'true value'], loc='lower center', ncols=2,
           frameon=False, fontsize=8)
fig.tight_layout(rect=(0, .1, 1, 1))
fig.savefig(os.path.join(HERE, 'fig_nu.png'), dpi=150)
plt.close(fig)
