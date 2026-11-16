# Parity with the MATLAB Toolkit

Each function here is the Python twin of a function of
[statespace-toolkit](https://github.com/joshuaccchan/statespace-toolkit), under the same name.

## How a Function Is Checked

`tools/matlab/export_fixtures.m` runs each MATLAB function on a set of cases and saves the inputs
and outputs in `tests/fixtures/core.mat`, with the MATLAB commit. A function that draws random
numbers runs on the replay stream: `ssm.Replay(seed)` in Python and `replay_start(seed)` in MATLAB
give the same uniforms and turn them into normal and gamma draws by their inverse distribution
functions. Each Python output must equal MATLAB's within 1e-10 of the entry plus 1e-14 of the
largest entry, and a function that draws must use as many uniforms. Measured, the outputs agree
within 7.2e-14 of the largest entry, and matrices built without arithmetic are identical.

Checks without MATLAB compare `ssm.simulate_states` with dense algebra and 20,000 draws, the
factor of the precision samplers with the dense Cholesky factor, `ssm.intlike` with the Kalman
filter, `ssm.intlike_sv` and `ssm.intlike_sv_t` with the exact likelihood by quadrature at
T = 2, `ssm.restrict` with dense Gaussian conditioning, and `ssm.armh` with a Gaussian target,
whose draws it must reproduce. A chain of `ssm.ma_params` must match the conditional density of
psi on a grid.

Of 38 one-line changes to the functions and to their factor, most by 1e-9, the tests catch 34.
Three of the four they miss change terms too small to move the output beyond the tolerance; a
change of 1e-9 to the leading term of the same three functions is caught. The fourth changes
the upper triangle of a matrix whose Cholesky factor NumPy computes from the lower triangle.

The 14 functions shared with the
[Python bvar-toolkit](https://github.com/joshuaccchan/bvar-toolkit-python) keep its code, as
their MATLAB twins keep the code of bvar-toolkit. `tests/test_twins.py` checks each against the
file pinned in `tests/fixtures`, apart from the docstring and the depth of its imports from
`_rng` and `_struct`.

## Functions

| Function | Shared with the Python bvar-toolkit | Cases from MATLAB | Other checks |
|---|---|---|---|
| `ssm.simulate_states` | | 6, replayed | dense algebra; 20,000 draws |
| `ssm.intlike` | | 4 | Kalman filter; dense algebra |
| `ssm.restrict` | | 2 | dense conditioning |
| `ssm.lagpolymat` | | 5 | `ssm.diffmat`; linear filter; input errors |
| `ssm.mode_newton` | | 3 | a Gaussian target; the number of steps; errors |
| `ssm.armh` | | 5, replayed | a Gaussian target |
| `ssm.ksc_rw_noncentered` | | 2, replayed | |
| `ssm.intlike_sv` | | 2, replayed | quadrature at T = 2; input errors |
| `ssm.intlike_sv_t` | | 2, replayed | quadrature at T = 2 |
| `ssm.ma_params` | | 3, replayed | a chain against the conditional density |
| `ssm.arma_params` | | 2, replayed | |
| `ssm.midas_basis` | | 3 | dates as datetime64; input errors |
| `ssm.select_obs` | `util.select_obs` | 2 | |
| `ssm.diffmat` | `util.diffmat` | 4 | |
| `ssm.surform` | `util.surform` | 1 | |
| `ssm.tnormrnd` | `util.tnormrnd` | 3, replayed | |
| `ssm.shaded_band` | `util.shaded_band` | | vertices and gray level of the band |
| `ssm.ksc_ar1_mean` | `sv.ksc_ar1_mean` | 2, replayed | |
| `ssm.ksc_rw_diffuse` | `sv.ksc_rw_diffuse` | 2, replayed | |
| `ssm.ksc_rw_h0` | `sv.ksc_rw_h0` | 2, replayed | |
| `ssm.sv_params` | `sv.sv_params` | 4, replayed | |
| `ssm.nu_studentt` | `sv.nu_studentt` | 3, replayed | |
| `ssm.specvar0` | `diag.specvar0` | 3 | |
| `ssm.inefficiency_factor` | `diag.inefficiency_factor` | 1 | |
| `ssm.mcse` | `diag.mcse` | 1 | |
| `ssm.geweke` | `diag.geweke` | 3 | |

Not yet ported: the five `ssm.models` functions.

## Departures from MATLAB

- Positions count from 0: the mixture indicators of the KSC samplers run from 0 to 6, the
  segments of `ssm.geweke` are row indices from 0, and the error of `ssm.midas_basis` names
  the empty window's period from 0.
- A function that draws takes an `rng` argument in place of MATLAB's global stream.
- Name-value options become keyword arguments: `'Tol'` is `tol`, `'MaxIter'` is `max_iter`,
  `'MaxIterMode'` and `'MaxIterAR'` are `max_iter_mode` and `max_iter_ar`, and `'ForceAccept'`
  and `'Start'` are `force_accept` and `start`.
- Invalid input raises `ValueError`; non-convergence and steps that return NaN or Inf raise
  `RuntimeError`. The messages are MATLAB's.
- `ssm.simulate_states` returns all three outputs. Without `ndraws` it returns one draw as a
  vector; with `ndraws` it returns an n x ndraws array. Its factor `C` holds the Cholesky
  factor of K and solves with it. For a banded K it is a `ssm.BandChol`. When the last states
  enter every period, as constant coefficients do, it keeps the band of the leading block and
  the trailing rows dense, which gives the factor of MATLAB's sparse `chol`. `ssm.restrict`
  takes either. The band comes from the nonzero values of K, so a zero that SciPy stores
  explicitly does not widen it.
- `ssm.midas_basis` takes dates as numpy datetime64 values or as numbers of days.

## Guides

A Python guide draws its own NumPy random numbers, so its page carries the numbers of its Python
run.
