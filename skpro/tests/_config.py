"""Test configs."""

# --------------------
# configs for test run
# --------------------

# whether to test only estimators from modules that are changed w.r.t. main
# default is False, can be set to True by pytest --only_changed_modules True flag
ONLY_CHANGED_MODULES = False


# list of str, names of estimators to exclude from testing
# WARNING: tests for these estimators will be skipped
EXCLUDE_ESTIMATORS = [
    "DummySkipped",
    "ClassName",  # exclude classes from extension templates
]


EXCLUDED_TESTS = {}
