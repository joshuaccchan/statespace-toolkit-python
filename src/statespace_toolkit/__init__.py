"""statespace-toolkit for Python: Bayesian state space models.

The functions are those of the MATLAB toolkit (https://github.com/joshuaccchan/statespace-toolkit),
under the same names: with import statespace_toolkit as ssm, ssm.simulate_states in MATLAB is
ssm.simulate_states here. Positions count from 0, and every function that draws random numbers
takes an rng argument (None, an integer seed, a numpy.random.Generator or a Stream).
"""

__version__ = '0.1.0.dev0'

from ._rng import Replay, Stream
from ._struct import Struct
from ._linalg import BandChol
from ._armh import armh
from ._arma_params import arma_params
from ._diffmat import diffmat
from ._geweke import geweke
from ._inefficiency_factor import inefficiency_factor
from ._intlike import intlike
from ._intlike_sv import intlike_sv
from ._intlike_sv_t import intlike_sv_t
from ._ksc_ar1_mean import ksc_ar1_mean
from ._ksc_rw_diffuse import ksc_rw_diffuse
from ._ksc_rw_h0 import ksc_rw_h0
from ._ksc_rw_noncentered import ksc_rw_noncentered
from ._lagpolymat import lagpolymat
from ._ma_params import ma_params
from ._mcse import mcse
from ._midas_basis import midas_basis
from ._mode_newton import mode_newton
from ._nu_studentt import nu_studentt
from ._restrict import restrict
from ._select_obs import select_obs
from ._shaded_band import shaded_band
from ._simulate_states import simulate_states
from ._specvar0 import specvar0
from ._surform import surform
from ._sv_params import sv_params
from ._tnormrnd import tnormrnd

__all__ = ['BandChol', 'Replay', 'Stream', 'Struct', 'armh', 'arma_params', 'diffmat', 'geweke',
           'inefficiency_factor', 'intlike', 'intlike_sv', 'intlike_sv_t', 'ksc_ar1_mean',
           'ksc_rw_diffuse', 'ksc_rw_h0', 'ksc_rw_noncentered', 'lagpolymat', 'ma_params',
           'mcse', 'midas_basis', 'mode_newton', 'nu_studentt', 'restrict', 'select_obs',
           'shaded_band', 'simulate_states', 'specvar0', 'surform', 'sv_params', 'tnormrnd']
