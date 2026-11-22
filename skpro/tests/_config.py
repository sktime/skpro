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


EXCLUDED_TESTS = {
    # test_mean_vs_sample fails until the following fixes are merged,
    # remove each entry once its fix is on main
    "MeanScale": ["test_mean_vs_sample"],  # mean ignores sigma, #1185, PR #1188
    "Mixture": ["test_mean_vs_sample"],  # sample #974 (PR #979), weights PR #1193
    "TransformedDistribution": ["test_mean_vs_sample"],  # ppf labels, #1186, PR #1189
}
