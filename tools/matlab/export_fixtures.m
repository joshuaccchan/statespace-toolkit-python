function export_fixtures(matlab_repo)
% export_fixtures - write tests/fixtures/core.mat and tests/fixtures/replay.mat from the
% MATLAB toolkit: for each ported function, a cell array of cases, each holding the inputs and
% outputs in the fields in1, in2, ... and out1, out2, .... A function that draws random numbers
% runs on the replay stream (replay_start), and its cases also hold the seed and the number of
% uniforms used.
%
%   export_fixtures                 % the MATLAB toolkit in ../statespace-toolkit
%   export_fixtures(matlab_repo)
%
% Usage, from the root of this repository:
%   matlab -batch "addpath('tools/matlab'); export_fixtures"
% The toolkit's core/ must have no uncommitted changes; each file records its commit.

here = fileparts(mfilename('fullpath'));
root = fileparts(fileparts(here));
if nargin < 1
    matlab_repo = fullfile(fileparts(root), 'statespace-toolkit');
end
[s1, sha] = system(sprintf('git -C "%s" rev-parse HEAD', matlab_repo));
[s2, dirty] = system(sprintf('git -C "%s" status --porcelain -- core', matlab_repo));
assert(s1 == 0 && s2 == 0, 'git failed in %s', matlab_repo);
assert(isempty(strtrim(dirty)), 'core/ of %s has uncommitted changes', matlab_repo);
run(fullfile(matlab_repo, 'setup.m'));
addpath(here);

meta = struct('matlab_commit', strtrim(sha), 'matlab_release', version('-release'), ...
    'exported', char(datetime('now', 'Format', 'yyyy-MM-dd')));
groups = {'core', @core_cases; 'replay', @replay_cases};
for g = 1:size(groups, 1)
    fx = groups{g, 2}();
    fx.meta = meta;
    save(fullfile(root, 'tests', 'fixtures', [groups{g, 1} '.mat']), '-struct', 'fx', '-v7');
    fprintf('%-10s %d functions\n', groups{g, 1}, numel(fieldnames(fx)) - 1);
end
end

%% ---- cases ----

function fx = core_cases()
    % the difference and lag-polynomial matrices, the SUR expansion and the selection
    % matrices of a missing data pattern
rng(1, 'twister');
fx.diffmat = {run_case(@ssm.diffmat, {1}, 1), run_case(@ssm.diffmat, {6}, 1), ...
    run_case(@ssm.diffmat, {6, .7}, 1), run_case(@ssm.diffmat, {6, -.4}, 1)};
fx.lagpolymat = {run_case(@ssm.lagpolymat, {6, [2 -1]}, 1), ...
    run_case(@ssm.lagpolymat, {5, .7}, 1), run_case(@ssm.lagpolymat, {4, [.5 .3 .1 .2 .9]}, 1), ...
    run_case(@ssm.lagpolymat, {3, []}, 1), run_case(@ssm.lagpolymat, {1, [.5; .2]}, 1)};
Xs = randn(5, 3);  Xs(2, 2) = 0;
fx.surform = {run_case(@ssm.surform, {Xs}, 1)};
Ys = randn(5, 3);  Ys([2 7 11]) = NaN;
fx.select_obs = {run_case(@ssm.select_obs, {Ys}, 3), run_case(@ssm.select_obs, {randn(4, 2)}, 3)};
fx.tnormrnd = {replay_case(16, @ssm.tnormrnd, {randn(5, 1), rand(5, 1) + .5, -1, 2}, 1), ...
    replay_case(17, @ssm.tnormrnd, {.5, 2, 0, Inf, 6}, 1), ...
    replay_case(18, @ssm.tnormrnd, {zeros(4, 1), ones(4, 1), [-1; -2; 0; 1], [1; 0; 3; Inf]}, 1)};

    % the MCMC diagnostics, on three AR(1) chains
R = 600;  e = randn(R, 3);  x = zeros(R, 3);
for t = 2:R, x(t, :) = [.8 .3 -.2].*x(t-1, :) + e(t, :); end
fx.specvar0 = {run_case(@ssm.specvar0, {x, 0}, 1), run_case(@ssm.specvar0, {x, 10}, 1), ...
    run_case(@ssm.specvar0, {x(:, 1)', 50}, 1)};
fx.inefficiency_factor = {run_case(@ssm.inefficiency_factor, {x, 20}, 1)};
fx.mcse = {run_case(@ssm.mcse, {x, 20}, 1)};
fx.geweke = {run_case(@ssm.geweke, {x}, 3), run_case(@ssm.geweke, {x, .2, .6, 5}, 3), ...
    run_case(@ssm.geweke, {x(:, 2)}, 3)};

    % the precision sampler: a local level model with one draw and with four, a TVP
    % regression whose K has bandwidth 3 with two draws and with ten, and a dense K
T = 50;  H = ssm.diffmat(T);
yl = 2 + cumsum(sqrt(.05)*randn(T, 1)) + randn(T, 1);
Kl = H'*H/.05 + speye(T);  cl = H'*[2; zeros(T-1, 1)]/.05 + yl;
k = 3;  Tk = 40;
Xt = [ones(Tk, 1) randn(Tk, 2)];
Hk = speye(Tk*k) - sparse(k+1:Tk*k, 1:(Tk-1)*k, 1, Tk*k, Tk*k);
Zt = ssm.surform(Xt);
beta = cumsum(.1*randn(Tk, k));
yt = sum(Xt.*beta, 2) + sqrt(.5)*randn(Tk, 1);
Pt = Hk'*Hk/.01;
Kt = Pt + Zt'*Zt/.5;  ct = Zt'*yt/.5;
A = randn(5);  Kd = A*A' + 5*eye(5);  cd = randn(5, 1);
fx.simulate_states = {replay_case(31, @ssm.simulate_states, {Kl, cl}, 3), ...
    replay_case(32, @ssm.simulate_states, {Kl, cl, 4}, 3), ...
    replay_case(33, @ssm.simulate_states, {Kt, ct, 2}, 3), ...
    replay_case(34, @ssm.simulate_states, {Kt, ct, 10}, 3), ...
    replay_case(35, @ssm.simulate_states, {Kd, cd, 3}, 3)};

    % the integrated likelihood: the local level model, the same with AR(1) measurement
    % errors, whose precision iR is banded, and the TVP regression
Hr = ssm.diffmat(T, .6);
fx.intlike = {run_case(@ssm.intlike, {yl, speye(T), speye(T), H'*H/.05, 2*ones(T, 1)}, 3), ...
    run_case(@ssm.intlike, {yl, speye(T), Hr'*Hr/.8, H'*H/.05, 2*ones(T, 1)}, 3), ...
    run_case(@ssm.intlike, {yt, Zt, speye(Tk)/.5, Pt, zeros(Tk*k, 1)}, 3)};

    % restrictions on draws and on the mean, with a sparse and a dense M; each case also
    % holds K, from which the Python test forms its own factor
[al, alhat, Cl] = ssm.simulate_states(Kl, cl, 3);
Mr = sparse([1 1 2 2], [5 6 20 30], [1 1 1 -1], 2, T);
Md = randn(1, T);
fx.restrict = {with_K(run_case(@ssm.restrict, {al, Cl, Mr, [.5; 0]}, 3), Kl), ...
    with_K(run_case(@ssm.restrict, {alhat, Cl, Md, 1}, 3), Kl)};

    % the auxiliary mixture sampler for the three state equations, the parameters of
    % stationary AR(1) log-volatilities, and the Student-t degrees of freedom
rng(8, 'twister');
T = 80;
h = zeros(T, 1);
for t = 2:T, h(t) = .95*h(t-1) + .2*randn; end
h0 = h + .1*randn(T, 1);
ystar = log((exp(h/2).*randn(T, 1)).^2 + .0001);
fx.ksc_ar1_mean = {replay_case(87, @ssm.ksc_ar1_mean, {ystar, h, 0, .95, .04}, 2), ...
    replay_case(88, @ssm.ksc_ar1_mean, {ystar + 1, h0 + 1, 1, .9, .1}, 2)};
hr = cumsum(.1*randn(T, 1));
ystar_r = log((exp(hr/2).*randn(T, 1)).^2 + .0001);
fx.ksc_rw_h0 = {replay_case(214, @ssm.ksc_rw_h0, {ystar_r, hr, .01, 0}, 1), ...
    replay_case(215, @ssm.ksc_rw_h0, {ystar_r + 1, hr + 1, .05, .5}, 1)};
fx.ksc_rw_diffuse = {replay_case(216, @ssm.ksc_rw_diffuse, {ystar_r, hr, .01, 10}, 2), ...
    replay_case(217, @ssm.ksc_rw_diffuse, {ystar_r + 1, hr + 1, .05, 100}, 2)};
H3 = zeros(T, 3);
for t = 2:T, H3(t, :) = [.9 .5 .97].*H3(t-1, :) + .2*randn(1, 3); end
Hy = struct('nuh', 3*ones(3, 1), 'Sh', .1*ones(3, 1), 'phi0', .95*ones(3, 1), ...
    'Vphi', .05^2*ones(3, 1), 'mu0', zeros(3, 1), 'Vmu', 100*ones(3, 1));
Hy2 = Hy;  Hy2.mu0 = zeros(2, 1);  Hy2.Vmu = 100*ones(2, 1);
Hm = H3 + [1 -1 .5];
Hg = struct('nuh', 5, 'Sh', .16, 'phi0', .97, 'Vphi', .1^2, 'mu0', 1, 'Vmu', 10);
fx.sv_params = {replay_case(89, @ssm.sv_params, {Hm, [.8; -.8; .4], [.9; .5; .95], Hy}, 4), ...
    replay_case(90, @ssm.sv_params, {Hm, [.8; 0; .4], [.9; .5; .95], Hy}, 4), ...
    replay_case(91, @ssm.sv_params, {[Hm(:, 1:2) H3(:, 3)], [.8; -.8], [.9; .5; .95], Hy2, .998}, 4), ...
    replay_case(92, @ssm.sv_params, {h + 1, 1, .95, Hg, 1}, 4)};
lam5 = 1./gamrnd(5/2, 2/5, T, 1);
lam30 = 1./gamrnd(15, 1/15, T, 1);
fx.nu_studentt = {replay_case(218, @nu_studentt_grid, {lam5, 6, 50, [3 5 10 20]}, 3), ...
    replay_case(219, @nu_studentt_grid, {lam30, 20, 30, [3 10 25]}, 3), ...
    replay_case(220, @nu_studentt_grid, {lam30, 20, 21, [3 10 20]}, 3)};

    % the mode search and the accept-reject Metropolis-Hastings step, on the log density of
    % the log-volatilities of an SV model (sv_target, written out in the Python test as well):
    % the default, a tight tolerance, a far start; then forced acceptance, a small c_reject
    % and a separate start of the mode search
rng(9, 'twister');
T = 60;  muh = .5;  phih = .9;  omegah2 = .1;
hs = muh + zeros(T, 1);
for t = 2:T, hs(t) = muh + phih*(hs(t-1) - muh) + sqrt(omegah2)*randn; end
s2 = (exp(hs/2).*randn(T, 1)).^2;
fx.mode_newton = {run_case(@mode_newton_sv, {zeros(T, 1), s2, muh, phih, omegah2, 1e-4}, 3), ...
    run_case(@mode_newton_sv, {zeros(T, 1), s2, muh, phih, omegah2, 1e-10}, 3), ...
    run_case(@mode_newton_sv, {5*ones(T, 1), s2, muh, phih, omegah2, 1e-6}, 3)};
fx.armh = {replay_case(101, @armh_sv, {hs, s2, muh, phih, omegah2, 3, false, []}, 3), ...
    replay_case(102, @armh_sv, {hs, s2, muh, phih, omegah2, 3, true, []}, 3), ...
    replay_case(103, @armh_sv, {hs + .3, s2, muh, phih, omegah2, .2, false, []}, 3), ...
    replay_case(104, @armh_sv, {hs + .3, s2, muh, phih, omegah2, .2, false, []}, 3), ...
    replay_case(105, @armh_sv, {hs, s2, muh, phih, omegah2, 1, false, zeros(T, 1)}, 3)};

    % the noncentered random-walk log-volatility
hr = cumsum(.1*randn(T, 1));
Ystar = log((exp(hr/2).*randn(T, 1)).^2 + .0001);
fx.ksc_rw_noncentered = {replay_case(111, @ssm.ksc_rw_noncentered, ...
        {Ystar, hr/.1, 0, .1, 0, 10, 10, 1}, 5), ...
    replay_case(112, @ssm.ksc_rw_noncentered, {Ystar + 1, -hr/.1, .5, -.2, 1, 5, 10, .1}, 5)};

    % the integrated likelihoods of the SV models, with the default and a tighter tolerance
fx.intlike_sv = {replay_case(121, @ssm.intlike_sv, {s2, muh, phih, omegah2, zeros(T, 1), 50}, 2), ...
    replay_case(122, @ssm.intlike_sv, {s2, 0, .95, .05, hs, 20}, 2, struct('Tol', 1e-8))};
fx.intlike_sv_t = {replay_case(123, @ssm.intlike_sv_t, {s2, muh, phih, omegah2, 8, zeros(T, 1), 50}, 2), ...
    replay_case(124, @ssm.intlike_sv_t, {s2, 0, .95, .05, 30, hs, 20}, 2, struct('Tol', 1e-8))};

    % the MA(1) and ARMA(1,1) coefficients, with time-varying precisions
u = randn(T, 1)./sqrt(exp(hs));
xm = u + .6*[0; u(1:end-1)];
fx.ma_params = {replay_case(131, @ssm.ma_params, {xm, exp(-hs), .5, struct('psi0', 0, 'Vpsi', 1)}, 3), ...
    replay_case(132, @ssm.ma_params, {xm, exp(-hs), 0, struct('psi0', -.4, 'Vpsi', .05)}, 3), ...
    replay_case(133, @ssm.ma_params, {randn(T, 1), ones(T, 1), .1, struct('psi0', 0, 'Vpsi', 1)}, 3)};
ea = filter([1 .6], [1 -.5], u);
Ha = struct('rho0', 0, 'Vrho', 1, 'psi0', 0, 'Vpsi', 1);
fx.arma_params = {replay_case(141, @ssm.arma_params, {ea, exp(-hs), .4, .5, Ha}, 4), ...
    replay_case(142, @ssm.arma_params, {ea, exp(-hs), -.2, .1, Ha}, 4)};

    % the Fourier basis of a weekly predictor over quarters, dates in days: the means, the
    % sums, a missing value, and release dates a week after the observations
dw = (7:7:7*60)';  xw = randn(60, 1);  xw(10) = NaN;
qs = (0:91:91*3)';  qe = qs + 91;
fx.midas_basis = {run_case(@ssm.midas_basis, {dw, xw, qs, qe}, 3), ...
    run_case(@ssm.midas_basis, {dw, xw, qs, qe}, 3, struct('scale', 'sum')), ...
    run_case(@ssm.midas_basis, {dw, xw, qs, qe}, 3, struct('release', dw + 7))};
end

function [logf, gradK] = sv_target(s2, muh, phih, omegah2)
% the log density of the log-volatilities h of an SV model given the squared data s2, up to a
% constant, with its gradient and negative Hessian
T = numel(s2);
Hphi = speye(T) - sparse(2:T, 1:T-1, phih, T, T);
HiSH = Hphi'*spdiags([(1-phih^2)/omegah2; ones(T-1, 1)/omegah2], 0, T, T)*Hphi;
logf = @(h) -.5*sum(h) - .5*sum(s2.*exp(-h)) - .5*(h - muh)'*HiSH*(h - muh);
gradK = @(h) deal(-.5 + .5*s2.*exp(-h) - HiSH*(h - muh), HiSH + spdiags(.5*s2.*exp(-h), 0, T, T));
end

function [alphahat, K, iter] = mode_newton_sv(alpha0, s2, muh, phih, omegah2, tol)
% ssm.mode_newton on sv_target
[~, gradK] = sv_target(s2, muh, phih, omegah2);
[alphahat, K, iter] = ssm.mode_newton(alpha0, gradK, 'Tol', tol);
end

function [alpha, accept, ntry] = armh_sv(alpha, s2, muh, phih, omegah2, c_reject, force, start)
% ssm.armh on sv_target
[logf, gradK] = sv_target(s2, muh, phih, omegah2);
[alpha, accept, ntry] = ssm.armh(alpha, logf, gradK, 'c_reject', c_reject, ...
    'ForceAccept', force, 'Start', start);
end

function fx = replay_cases()
% the functions of replay/ that the ported functions use, called by name so that the replay
% versions answer
named = @(name) @(varargin) feval(name, varargin{:});
fx.randn = {replay_case(71, named('randn'), {3, 2}, 1)};
fx.rand = {replay_case(72, named('rand'), {2, 3}, 1)};
fx.gamrnd = {replay_case(73, named('gamrnd'), {[.5; 2; 7], 1.5}, 1)};
end

function [nu, flag, fgrid] = nu_studentt_grid(lam, nu, nu_ub, grid)
% ssm.nu_studentt, with its log target evaluated on a grid in place of the handle
[nu, flag, f] = ssm.nu_studentt(lam, nu, nu_ub);
fgrid = arrayfun(f, grid);
end

function c = with_K(c, K)
% a case of ssm.restrict, with the precision whose factor is its second input
c.K = K;
end

%% ---- helpers ----

function c = run_case(fn, in, nout, opts)
% the outputs of fn(in{:}, name-value options), with the built-in random numbers
if nargin < 4, opts = struct(); end
nv = reshape([fieldnames(opts)'; struct2cell(opts)'], 1, []);
out = cell(1, nout);
[out{:}] = fn(in{:}, nv{:});
c = pack(in, out);
c.opts = opts;
end

function c = replay_case(seed, fn, in, nout, opts)
% the outputs of fn(in{:}, name-value options) on the replay stream of the seed
if nargin < 5, opts = struct(); end
nv = reshape([fieldnames(opts)'; struct2cell(opts)'], 1, []);
replay_start(seed);
cleanup = onCleanup(@() replay_stop());
out = cell(1, nout);
[out{:}] = fn(in{:}, nv{:});
c = pack(in, out);
c.opts = opts;
c.seed = seed;
c.count = replay_stop();
end

function c = pack(in, out)
% one field per input and output, in1, in2, ... and out1, out2, ..., which loadmat
% reads back without the ambiguities of a cell array
c.nin = numel(in);
c.nout = numel(out);
for i = 1:numel(in), c.(sprintf('in%d', i)) = in{i}; end
for i = 1:numel(out), c.(sprintf('out%d', i)) = out{i}; end
end
