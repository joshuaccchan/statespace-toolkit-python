"""The modules shared with the Python bvar-toolkit. Each is the code of its bvar-toolkit file,
pinned in tests/fixtures/bvar-toolkit-python at the commit that tests/fixtures/README.md names,
apart from the docstrings and the depth of its imports from _rng and _struct, which sit in the
package itself here and in its parent package there."""

import ast
import hashlib
import pathlib

import pytest

ROOT = pathlib.Path(__file__).parents[1]
SRC = ROOT / 'src' / 'statespace_toolkit'
PIN = ROOT / 'tests' / 'fixtures' / 'bvar-toolkit-python'

# module here: (file in the Python bvar-toolkit, number of imports moved up one level)
TWINS = {
    '_rng': ('_rng.py', 0), '_struct': ('_struct.py', 0), '_linalg': ('_linalg.py', 0),
    '_surform': ('util/_surform.py', 0), '_tnormrnd': ('util/_tnormrnd.py', 1),
    '_shaded_band': ('util/_shaded_band.py', 0), '_diffmat': ('util/_diffmat.py', 0),
    '_select_obs': ('util/_select_obs.py', 0), '_specvar0': ('diag/_specvar0.py', 0),
    '_inefficiency_factor': ('diag/_inefficiency_factor.py', 0), '_mcse': ('diag/_mcse.py', 0),
    '_geweke': ('diag/_geweke.py', 1), '_ksc_ar1_mean': ('sv/_ksc_ar1_mean.py', 1),
    '_ksc_rw_diffuse': ('sv/_ksc_rw_diffuse.py', 1), '_ksc_rw_h0': ('sv/_ksc_rw_h0.py', 1),
    '_sv_params': ('sv/_sv_params.py', 1), '_nu_studentt': ('sv/_nu_studentt.py', 1),
}


def _code(path):
    """the syntax tree of a module without its docstrings"""
    tree = ast.parse(path.read_text(encoding='utf-8'))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef)):
            b = node.body
            if b and isinstance(b[0], ast.Expr) and isinstance(b[0].value, ast.Constant) \
                    and isinstance(b[0].value.value, str):
                node.body = b[1:] or [ast.Pass()]
    return tree


@pytest.mark.parametrize('mod', sorted(TWINS))
def test_twin(mod):
    theirs_file, moved = TWINS[mod]
    ours = _code(SRC / f'{mod}.py')
    theirs = _code(PIN / 'src' / 'bvar_toolkit' / theirs_file)
    k = 0
    for node in ast.walk(theirs):
        if isinstance(node, ast.ImportFrom) and node.level == 2:
            node.level = 1                          # from .._rng import -> from ._rng import
            k += 1
    assert k == moved, f'{k} imports moved up, expected {moved}'
    assert ast.dump(ours) == ast.dump(theirs)


def test_pinned_copies():
    """the pinned files are those whose MD5 sums MD5SUMS lists, and MD5SUMS lists them all"""
    listed = {}
    for line in (PIN / 'MD5SUMS').read_text(encoding='utf-8').splitlines():
        digest, name = line.split(None, 1)
        listed[name.strip()] = digest
    files = {p.relative_to(PIN).as_posix(): p for p in PIN.rglob('*.py')}
    assert set(listed) == set(files)
    for name, p in files.items():
        assert hashlib.md5(p.read_bytes()).hexdigest() == listed[name], name
