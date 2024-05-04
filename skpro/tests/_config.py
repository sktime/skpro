"""Test configs."""

from skpro.tests._config_test_dummy import DummySkipped


# list of str, names of estimators to exclude from testing
# WARNING: tests for these estimators will be skipped
EXCLUDE_ESTIMATORS = [DummySkipped]
