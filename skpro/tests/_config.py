"""Test configs."""

# list of str, names of estimators to exclude from testing
# WARNING: tests for these estimators will be skipped
EXCLUDE_ESTIMATORS = [
    "DummySkipped",
    "ClassName",  # exclude classes from extension templates
]


EXCLUDED_TESTS = {
    "GLMRegressor": ["test_online_update"],  # see 497
}
