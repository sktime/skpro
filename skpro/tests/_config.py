"""Test configs."""

# --------------------
# configs for test run
# --------------------

# whether to test only estimators from modules that are changed w.r.t. main
# default is False, can be set to True by pytest --only_changed_modules True flag
ONLY_CHANGED_MODULES = False

# whether to test only estimators that require their own VM to test,
# i.e., estimators with the tag "tests:vm" set to True
# default is False, can be set to True by pytest --only_vm_estimators True flag
ONLY_VM_ESTIMATORS = False


# list of str, names of estimators to exclude from testing
# WARNING: tests for these estimators will be skipped
EXCLUDE_ESTIMATORS = [
    "DummySkipped",
    "ClassName",  # exclude classes from extension templates
]


EXCLUDED_TESTS = {}
