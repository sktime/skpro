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
    # Zero-inflated distributions extend _ScipyAdapter but have different
    # statistical properties than the underlying scipy distributions that they wrap.
    # The scipy adapter tests compare against the base scipy distribution
    # which is incorrect for zero-inflated mixtures.
    "ZIPoisson": [
        "test_method_no_params",  # mean/var differ due to zero-inflation
        "test_method_with_x_params",  # cdf/ppf differ due to zero-inflation
    ],
    "ZINB": [
        "test_method_no_params",  # mean/var differ due to zero-inflation
        "test_method_with_x_params",  # cdf/ppf differ due to zero-inflation
    ],
}
