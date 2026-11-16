# Test Fixtures

- `core.mat` and `replay.mat` hold the cases that `tools/matlab/export_fixtures.m` exports from
  the MATLAB toolkit. The `meta` field of each file records the MATLAB commit, release and date:
  statespace-toolkit `0cf78b8`, MATLAB R2025b.
- `bvar-toolkit-python/` holds the files of the Python bvar-toolkit that this package shares, as
  they stand at its commit `7fbddd9`, with their MD5 sums in `MD5SUMS`. `tests/test_twins.py`
  checks that the shared modules keep their code. The replay functions in `tools/matlab` are
  copies from the same commit.
