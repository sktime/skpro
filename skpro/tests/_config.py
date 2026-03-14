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
    # BUG: component estimator fitted state mutates constructor params
    # see issue #922
    "BaggingRegressor": ["test_non_state_changing_method_contract"],
    "BootstrapRegressor": ["test_non_state_changing_method_contract"],
    "ConditionUncensored": ["test_non_state_changing_method_contract"],
    "DeltaPointRegressor": ["test_non_state_changing_method_contract"],
    "EnbpiRegressor": ["test_non_state_changing_method_contract"],
    "FitUncensored": ["test_non_state_changing_method_contract"],
    "GridSearchCV": ["test_non_state_changing_method_contract"],
    "HistBinnedProbaRegressor": ["test_non_state_changing_method_contract"],
    "MultipleQuantileRegressor": ["test_non_state_changing_method_contract"],
    "OnlineDontRefit": ["test_non_state_changing_method_contract"],
    "OnlineRefit": ["test_non_state_changing_method_contract"],
    "OnlineRefitEveryN": ["test_non_state_changing_method_contract"],
    "Pipeline": ["test_non_state_changing_method_contract"],
    "RandomizedSearchCV": ["test_non_state_changing_method_contract"],
    "ResidualDouble": ["test_non_state_changing_method_contract"],
    "SklearnProbaReg": ["test_non_state_changing_method_contract"],
    "TransformedTargetRegressor": ["test_non_state_changing_method_contract"],
}
