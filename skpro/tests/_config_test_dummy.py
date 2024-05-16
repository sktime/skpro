"""Test dummy for testing config skips."""


from skpro.regression.base import BaseProbaRegressor  # noqa: E402


class DummySkipped(BaseProbaRegressor):
    """Dummy regressor to test exclusion."""

    pass
