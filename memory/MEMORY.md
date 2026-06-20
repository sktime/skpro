# skpro Project Memory

## Running Tests on Windows (IMPORTANT)

`setup.cfg` has `-n auto` in `addopts` which spawns parallel xdist workers.
On Windows this causes a **stack overflow crash** — not an infinite loop.

**Always use this command pattern to run tests locally:**
```bash
python -m pytest <test_file> --override-ini="addopts=" --no-cov -v
```

Examples:
```bash
python -m pytest skpro/distributions/tests/test_halfcauchy.py --override-ini="addopts=" --no-cov -v
python -m pytest skpro/distributions/tests/test_all_distrs.py -k "HalfCauchy" --override-ini="addopts=" --no-cov -v
```

## HalfCauchy Distribution

- File: `skpro/distributions/halfcauchy.py`
- Test file: `skpro/distributions/tests/test_halfcauchy.py`
- Implementation wraps `scipy.stats.halfcauchy` via `_ScipyAdapter`
- Parameter: `beta` (scale), mapped to `scale=beta` in scipy
- All 103 tests pass (test_halfcauchy.py + test_all_distrs.py filtered to HalfCauchy)

## Project Structure

- Repo root: `c:/Users/Kunal Kumar/Desktop/European summer of code/skpro/`
- Distributions: `skpro/distributions/`
- Distribution tests: `skpro/distributions/tests/`
- venv: `c:/Users/Kunal Kumar/Desktop/European summer of code/.venv/`
