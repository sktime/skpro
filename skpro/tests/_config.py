"""Test configs."""

# list of str, names of estimators to exclude from testing
# WARNING: tests for these estimators will be skipped
EXCLUDE_ESTIMATORS = []


from skpro.regression.base import BaseProbaRegressor  # noqa: E402


class DummySkipped(BaseProbaRegressor):
    """Dummy regressor to test exclusion."""
    pass
