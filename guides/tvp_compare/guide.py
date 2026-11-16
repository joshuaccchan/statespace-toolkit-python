"""guide - the code of guides/tvp_compare/README.md: four TVP regressions, which differ in the
coefficients that follow random walks, compared by their marginal likelihoods on two data sets:
the data of guides/tvp_states, whose intercept and slope both vary, and the data of
guides/tvp_gibbs whose slope is constant. For each model, ssm.intlike integrates the coefficients
out, an independence-chain Metropolis-Hastings sampler draws the variances, and importance
sampling with the cross-entropy importance density estimates the marginal likelihood. Section 2
computes the Bayes factor of a varying slope under other priors of its variance. The sections
called "the code to adapt" are all a comparison needs. The checks compute each marginal likelihood
on a grid and compare ssm.intlike with the dense normal density of y, also for the eight models of
three regressors in section 3. Saves fig_prior.png next to this file.
"""

import os

# one BLAS thread, set before NumPy loads
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402
from scipy.linalg import solve_triangular  # noqa: E402
from scipy.optimize import minimize  # noqa: E402
from scipy.special import gammaln  # noqa: E402

import statespace_toolkit as ssm  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def vecstr(v):
    """a vector as MATLAB's mat2str prints it, [a b c]; adding 0.0 turns -0 into 0"""
    return '[' + ' '.join(f'{x:g}' for x in np.asarray(v, dtype=float) + 0.0) + ']'


# ---- local functions ----
def model_matrices(X, V, V0):
    """the stacked design Z, prior mean b and prior precision P(iom) of the regression whose
    coefficients V follow random walks from beta_i0 ~ N(0, V0) and whose others are constant
    with the prior N(0, V0); the states are (beta_V0', beta_V1', ..., beta_VT', beta_C')',
    and iom holds 1/omega_i^2 for i in V"""
    T, k = X.shape
    C = [i for i in range(k) if i not in V]
    kv, kc = len(V), len(C)
    if kv > 0:
        # the first differences of the paths
        G = sp.kron(ssm.diffmat(T + 1), sp.eye_array(kv), format='csc')
        A0 = G.T @ sp.diags_array(np.r_[np.ones(kv) / V0, np.zeros(kv * T)]) @ G
        Ai = []
        for i in range(kv):
            sel = np.r_[np.zeros(kv), np.tile((np.arange(kv) == i).astype(float), T)]
            Ai.append(sp.block_diag([G.T @ sp.diags_array(sel) @ G, sp.csc_array((kc, kc))],
                                    format='csc'))
        Zv = sp.hstack([sp.csc_array((T, kv)), ssm.surform(X[:, V])], format='csc')
    else:
        A0, Ai, Zv = sp.csc_array((0, 0)), [], sp.csc_array((T, 0))
    A0 = sp.block_diag([A0, sp.eye_array(kc) / V0], format='csc')
    M = ssm.Struct()
    M.Z = sp.hstack([Zv, sp.csc_array(X[:, C])], format='csc')
    M.b = np.zeros(A0.shape[0])
    M.P = lambda iom: prior_precision(A0, Ai, iom)
    return M


def prior_precision(A0, Ai, iom):
    """A0 + sum_i Ai[i] iom[i]"""
    P = A0
    for i in range(len(Ai)):
        P = P + Ai[i] * iom[i]
    return P


def draw_variances(lp, d, nsim, burnin, rng):
    """independence-chain Metropolis-Hastings on phi, with a t proposal (5 degrees of freedom) at
    the mode of lp and scaled by the inverse of its negative Hessian there, as in
    guides/integrated_likelihood; draws holds one draw of phi per row"""
    opt = {'xatol': 1e-8, 'fatol': 1e-8, 'maxfev': 10000, 'maxiter': 10000}
    phihat = minimize(lambda phi: -lp(phi), np.r_[0.0, np.log(.01) * np.ones(d - 1)],
                      method='Nelder-Mead', options=opt).x
    h = 1e-3
    E, Hs = h * np.eye(d), np.zeros((d, d))
    for i in range(d):                                    # the Hessian by finite differences
        for j in range(d):
            Hs[i, j] = (lp(phihat + E[:, i] + E[:, j]) - lp(phihat + E[:, i] - E[:, j])
                        - lp(phihat - E[:, i] + E[:, j]) + lp(phihat - E[:, i] - E[:, j])
                        ) / (4 * h**2)
    C, nu = np.linalg.cholesky(np.linalg.inv(-Hs)), 5

    def logq(phi):
        return -(nu + d) / 2 * np.log(1 + np.sum(solve_triangular(C, phi - phihat, lower=True)**2)
                                      / nu)

    phi = phihat
    lw = lp(phi) - logq(phi)
    draws, acc = np.zeros((nsim, d)), 0
    for isim in range(burnin + nsim):
        phic = phihat + C @ rng.standard_normal(d) / np.sqrt(rng.gamma(nu / 2, 2 / nu))  # proposal
        lwc = lp(phic) - logq(phic)
        if np.log(rng.random()) < lwc - lw:
            phi, lw = phic, lwc
            acc = acc + (isim >= burnin)
        if isim >= burnin:
            draws[isim - burnin] = phi
    return draws, acc / nsim


def is_logml(lp, draws, R, rng):
    """log p(y) by importance sampling. The importance density is a t with 5 degrees of freedom
    at the mean and covariance of the posterior draws: the cross-entropy fit of a normal
    density (Chan and Eisenstat, 2015), with heavier tails. R is a multiple of 20, and the
    numerical standard error comes from 20 batches."""
    nu, d = 5, draws.shape[1]
    m, C = draws.mean(axis=0), np.linalg.cholesky(np.atleast_2d(np.cov(draws, rowvar=False)))
    z = rng.standard_normal((d, R)) / np.sqrt(rng.chisquare(nu, R) / nu)
    phi = m[:, None] + C @ z
    lconst = (gammaln((nu + d) / 2) - gammaln(nu / 2) - d / 2 * np.log(nu * np.pi)
              - np.sum(np.log(np.diag(C))))
    logw = np.zeros(R)
    for r in range(R):                                     # log p(y, theta) - log g(phi)
        logw[r] = lp(phi[:, r]) - (lconst - (nu + d) / 2 * np.log(1 + z[:, r] @ z[:, r] / nu))
    lml = logmeanexp(logw)
    lb, wb = np.zeros(20), logw.reshape(20, R // 20)       # row b of wb: batch b
    for b in range(20):
        lb[b] = logmeanexp(wb[b])
    nse = np.std(lb, ddof=1) / np.sqrt(20)
    return lml, nse


def grid_logml(lp, draws, n, s):
    """log p(y) as the sum of exp(lp) times the cell volume over a grid of n points per
    dimension that spans s standard deviations of each column of draws around its mean"""
    d = draws.shape[1]
    g = []
    for i in range(d):
        mi, si = np.mean(draws[:, i]), np.std(draws[:, i], ddof=1)
        g.append(np.linspace(mi - s * si, mi + s * si, n))
    G = np.meshgrid(*g, indexing='ij')
    lpg = np.zeros(G[0].size)
    for j in range(lpg.size):
        lpg[j] = lp(np.array([x.flat[j] for x in G]))
    return (lpg.max() + np.log(np.sum(np.exp(lpg - lpg.max())))
            + sum(np.log(x[1] - x[0]) for x in g))


def dense_loglike(y, X, V, V0, sig2, om2):
    """log p(y | sig2, omega2) from the T x T covariance matrix of y: cov(beta_it, beta_is) is
    V0 + min(t,s) om2[j] if coefficient i = V[j] varies and V0 if it is constant"""
    T = X.shape[0]
    t, s = np.meshgrid(np.arange(1, T + 1), np.arange(1, T + 1), indexing='ij')
    Sy = sig2 * np.eye(T)
    for i in range(X.shape[1]):
        Ci = V0 * np.ones((T, T))
        if i in V:
            Ci = Ci + np.minimum(t, s) * om2[V.index(i)]
        Sy = Sy + np.outer(X[:, i], X[:, i]) * Ci
    return (-T / 2 * np.log(2 * np.pi) - np.sum(np.log(np.diag(np.linalg.cholesky(Sy))))
            - .5 * y @ np.linalg.solve(Sy, y))


def logmeanexp(x):
    """log(mean(exp(x))) without overflow"""
    c = np.max(x)
    return c + np.log(np.mean(np.exp(x - c)))


# ---- the data of guides/tvp_states, and the data of guides/tvp_gibbs with a constant slope ----
rng = np.random.default_rng(13)
T, k = 200, 2
X = np.column_stack([np.ones(T), rng.standard_normal(T)])   # x_t = (1, x_t)'
beta = np.array([2.0, 1.0]) + np.cumsum(rng.standard_normal((T, k))
                                        @ np.sqrt(np.diag([.005, .02])), axis=0)
y1 = np.sum(X * beta, axis=1) + rng.standard_normal(T)      # both coefficients vary
rng = np.random.default_rng(7)
tau = 2 + np.cumsum(np.sqrt(.005) * rng.standard_normal(T))
y2 = tau + X[:, 1] + rng.standard_normal(T)                 # the intercept varies, the slope is 1
Y, dnames = [y1, y2], ['both vary', 'constant slope']
rng = np.random.default_rng(3)

# ---- 1. the four models and their marginal likelihoods: the code to adapt ----
nu_sig, S_sig = 3, 2                            # sig2 ~ IG(nu_sig, S_sig)
nu_om, S_om = 3, np.array([.02, .02])           # omega_i^2 ~ IG(nu_om, S_om[i])
V0 = 100                                        # beta_i0 ~ N(0, V0), constant beta_i too


def logig(x, nu, S):
    """the log density of the inverse-gamma distribution IG(nu, S) at x"""
    return nu * np.log(S) - gammaln(nu) - (nu + 1) * np.log(x) - S / x


vary = [[], [0], [1], [0, 1]]                   # the coefficients that vary, by model
mnames = ['none', 'intercept', 'slope', 'both']
nsim, burnin, R = 5000, 500, 10000
lml, nse, acc = np.zeros((4, 2)), np.zeros((4, 2)), np.zeros((4, 2))
draws, LP = {}, {}
for d in range(2):
    for m in range(4):
        M = model_matrices(X, vary[m], V0)

        # log p(y | theta) + log p(theta) + the Jacobian, at phi = (log sig2, log omega_i^2);
        # the defaults keep y, M and v at their values in this pass of the loop
        def lp(phi, y=Y[d], M=M, v=vary[m]):
            iR = sp.eye_array(T) * np.exp(-phi[0])
            return (ssm.intlike(y, M.Z, iR, M.P(np.exp(-phi[1:])), M.b)[0]
                    + logig(np.exp(phi[0]), nu_sig, S_sig)
                    + np.sum(logig(np.exp(phi[1:]), nu_om, S_om[v])) + np.sum(phi))

        draws[m, d], acc[m, d] = draw_variances(lp, 1 + len(vary[m]), nsim, burnin, rng)
        lml[m, d], nse[m, d] = is_logml(lp, draws[m, d], R, rng)
        LP[m, d] = lp

# ---- 1. the checks ----
print('\nlog marginal likelihoods by importance sampling (numerical standard error)')
print(f'{"varies":<10s} {dnames[0]:>24s} {dnames[1]:>24s}')
for m in range(4):
    print(f'{mnames[m]:<10s} {lml[m, 0]:16.2f} ({nse[m, 0]:.3f}) {lml[m, 1]:16.2f} '
          f'({nse[m, 1]:.3f})')
print(f'acceptance rates of the Metropolis-Hastings samplers: {acc.min():.2f} to {acc.max():.2f}')
glml, ng = np.zeros((4, 2)), [400, 60, 60, 24]
for d in range(2):
    for m in range(4):
        glml[m, d] = grid_logml(LP[m, d], draws[m, d], ng[m], 6)
print('the same on a grid, which spans 6 standard deviations of the draws')
for m in range(4):
    print(f'{mnames[m]:<10s} {glml[m, 0]:24.2f} {glml[m, 1]:24.2f}')
print(f'importance sampling minus grid, at most {np.max(np.abs(lml - glml)):.3f}, or '
      f'{np.max(np.abs(lml - glml) / nse):.1f} numerical standard errors')
for d in range(2):
    best = np.argmax(lml[:, d])
    print(f'{dnames[d]}: the best model lets {mnames[best]} vary; log Bayes factors against it: '
          f'{vecstr(np.round(lml[:, d] - lml[best, d], 1))}')
rv, rc = np.exp(draws[3, 0]), np.exp(draws[3, 1])
print(f'posterior means of omega2_2 in the model where both vary: {rv[:, 2].mean():.4f} on the '
      f'data whose slope varies, {rc[:, 2].mean():.5f} on the data whose slope is constant')
derr = 0                                                    # ssm.intlike against the dense density
for d in range(2):
    for m in range(4):
        th = np.exp(draws[m, d].mean(axis=0))               # (sig2, omega_i^2) at the mean of phi
        M = model_matrices(X, vary[m], V0)
        ll = ssm.intlike(Y[d], M.Z, sp.eye_array(T) / th[0], M.P(1 / th[1:]), M.b)[0]
        llD = dense_loglike(Y[d], X, vary[m], V0, th[0], th[1:])
        derr = max(derr, abs(ll - llD) / abs(llD))
print(f'ssm.intlike against the dense normal density of y: at most {derr:.0e} relative')
th, M = np.exp(draws[1, 0].mean(axis=0)), model_matrices(X, [0], V0)  # only the intercept varies
K = (M.P(1 / th[1:]) + M.Z.T @ M.Z / th[0]).toarray()
n = K.shape[0]
pf = np.r_[n - 1, np.arange(n - 1)]                         # the constant slope moved first
print(f'nonzero entries of the Cholesky factor of K when only the intercept varies: '
      f'{np.count_nonzero(np.linalg.cholesky(K))} with the constant slope last, '
      f'{np.count_nonzero(np.linalg.cholesky(K[np.ix_(pf, pf)]))} with it first')

# ---- 2. the prior of the slope's variance: the log Bayes factor of "both" against
# "intercept" ----
pmean = np.array([1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1])  # prior means of omega2_2, S/(nu - 1)
lbf, lbse = np.zeros((len(pmean), 2)), np.zeros((len(pmean), 2))
LP2, DR = {}, {}
M = model_matrices(X, [0, 1], V0)
for d in range(2):
    for p in range(len(pmean)):
        S2 = 2 * pmean[p]

        def lp(phi, y=Y[d], M=M, S2=S2):
            iR = sp.eye_array(T) * np.exp(-phi[0])
            return (ssm.intlike(y, M.Z, iR, M.P(np.exp(-phi[1:])), M.b)[0]
                    + logig(np.exp(phi[0]), nu_sig, S_sig)
                    + logig(np.exp(phi[1]), nu_om, S_om[0]) + logig(np.exp(phi[2]), nu_om, S2)
                    + np.sum(phi))

        dr, _ = draw_variances(lp, 3, nsim, burnin, rng)
        l3, s3 = is_logml(lp, dr, R, rng)
        lbf[p, d], lbse[p, d] = l3 - lml[1, d], np.sqrt(s3**2 + nse[1, d]**2)
        LP2[p, d], DR[p, d] = lp, dr

# ---- 2. the checks ----
print('\nlog Bayes factor of "both" against "intercept", by the prior mean of omega2_2')
print(f'{"prior mean":<12s} {dnames[0]:>24s} {dnames[1]:>24s}')
for p in range(len(pmean)):
    print(f'{pmean[p]:<12g} {lbf[p, 0]:16.2f} ({lbse[p, 0]:.3f}) {lbf[p, 1]:16.2f} '
          f'({lbse[p, 1]:.3f})')
gbf = np.zeros(lbf.shape)
for d in range(2):
    for p in range(len(pmean)):
        gbf[p, d] = grid_logml(LP2[p, d], DR[p, d], 24, 6) - glml[1, d]
print(f'importance sampling minus grid, at most {np.max(np.abs(lbf - gbf)):.3f}')

# ---- the figure: the log Bayes factor against the prior mean of the slope's variance ----
fig, ax = plt.subplots(figsize=(6.3, 3.54))
ax.axhline(0, color=(.6, .6, .6), linewidth=.5)
h1, = ax.plot(pmean, lbf[:, 0], 'k-o', linewidth=1.3, markerfacecolor='k')
h2, = ax.plot(pmean, lbf[:, 1], 'k--s', linewidth=1.3, markerfacecolor='none')
ax.set_xscale('log')
ax.set_xlim(5e-5, 2e-1)
ax.set_xlabel('prior mean of $\\omega_2^2$', fontsize=10)
ax.set_ylabel('log Bayes factor', fontsize=10)
ax.spines[['top', 'right']].set_visible(False)
ax.tick_params(labelsize=8)
ax.legend([h1, h2], ['data whose slope varies', 'data whose slope is constant'], loc='best',
          frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(HERE, 'fig_prior.png'), dpi=150)
plt.close(fig)

# ---- 3. three regressors and eight models: the adaptation ----
rng = np.random.default_rng(4)
k = 3
X = np.column_stack([np.ones(T), rng.standard_normal((T, 2))])
y = X @ np.array([2.0, 1.0, -1.0]) + rng.standard_normal(T)
vary = [[i for i in range(k) if m >> i & 1] for m in range(2**k)]
derr = 0
for m in range(len(vary)):
    om2 = .01 * np.ones(len(vary[m]))
    M = model_matrices(X, vary[m], V0)
    ll = ssm.intlike(y, M.Z, sp.eye_array(T), M.P(1 / om2), M.b)[0]
    llD = dense_loglike(y, X, vary[m], V0, 1, om2)
    derr = max(derr, abs(ll - llD) / abs(llD))
sub = ['{' + ', '.join(str(i) for i in v) + '}' for v in vary]
print(f'\nthree regressors, {len(vary)} models, which let these coefficients vary: '
      f'{" ".join(sub)}')
print(f'ssm.intlike against the dense normal density of y: at most {derr:.0e} relative')
