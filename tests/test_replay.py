"""The replay stream: each draw of ssm.Replay equals the draw of its MATLAB twin in
tools/matlab/replay on the same seed, and uses as many uniforms."""

import numpy as np
import pytest

import statespace_toolkit as ssm
from conftest import close, fixtures, inputs, outputs

FX = fixtures('replay')
CASES = [(name, c) for name in ('randn', 'rand', 'gamrnd')
         for c in (FX[name] if isinstance(FX[name], list) else [FX[name]])]


@pytest.mark.parametrize('name, case', CASES, ids=[f'{n}-{c["seed"]}' for n, c in CASES])
def test_replay(name, case):
    r = ssm.Replay(case['seed'])
    args = [int(a) if np.ndim(a) == 0 and float(a).is_integer() else a for a in inputs(case)]
    out = getattr(r, name)(*args)
    expected, = outputs(case)
    close(out, expected)
    assert r.count == case['count']
