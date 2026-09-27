# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Testing of crafting functionality."""

__author__ = ["fkiraly"]

import pytest
from skbase.utils.dependencies import _check_soft_dependencies

from skpro.registry._craft import craft, deps

simple_spec = "DummyProbaRegressor()"
simple_spec_with_dep = "XGBoostLSS(n_estimators=7)"

pipe_spec_no_deps = """
regressor = DummyProbaRegressor()
cv = KFold(n_splits=3)

return GridSearchCV(
    regressor,
    param_grid=[{"strategy": ["normal", "empirical"]}],
    cv=cv,
    )
"""

pipe_spec_with_deps = """
regressor = XGBoostLSS(n_estimators=7)
cv = KFold(n_splits=3)

return GridSearchCV(
    regressor,
    param_grid=[{"n_estimators": [7, 10]}],
    cv=cv,
    )
"""

specs = [simple_spec, pipe_spec_no_deps]


if _check_soft_dependencies(["xgboostlss"], severity="none"):
    specs += [simple_spec_with_dep, pipe_spec_with_deps]


@pytest.mark.parametrize("spec", specs)
@pytest.mark.parametrize("safe", [True, False])
def test_craft(spec, safe):
    """Check that crafting works and is inverse to str coercion."""
    # hack - among test cases, all unsafe specs contain a "return" statement
    # in general, this statement is not true, i.e., unsafe iff contains return
    spec_is_unsafe = "return" in spec

    # test that unsafe specs correctly raise an error in safe mode
    if safe and spec_is_unsafe:
        with pytest.raises(ValueError):
            craft(spec, safe=safe)
        return

    # test that crafting and re-crafting produces consistent results
    crafted_obj = craft(spec, safe=safe)

    new_spec = str(crafted_obj)

    crafted_again = craft(new_spec, safe=safe)

    # sklearn equality does not allow the comparison below
    # so estimators with sklearn components are skipped
    if "KFold" not in spec:
        assert crafted_again == crafted_obj


@pytest.mark.parametrize(
    "spec",
    [
        # Attribute access
        "NaiveForecaster().fit",
        "NaiveForecaster().foo()",
        "NaiveForecaster.__init__",
        # Indirect / arbitrary function calls
        "getattr(NaiveForecaster(), 'fit')",
        "(lambda: NaiveForecaster())()",
        # Lambdas
        "lambda: NaiveForecaster()",
        "NaiveForecaster(lam=lambda: 1)",
        # Comprehensions / generators
        "[NaiveForecaster() for _ in range(1)]",
        "{NaiveForecaster() for _ in range(1)}",
        "{i: NaiveForecaster() for i in range(1)}",
        "(NaiveForecaster() for _ in range(1))",
        # Arbitrary builtin/function names are not in the safe registry
        "eval('NaiveForecaster()')",
        "exec('x = 1')",
        "open('foo')",
        "getattr",
        # Dunder names / access
        "__import__('os')",
        "__builtins__",
        "NaiveForecaster().__class__",
        "NaiveForecaster().__dict__",
        # **kwargs expansion
        "NaiveForecaster(**{})",
        # Statements / multi-statement code
        "x = NaiveForecaster()",
        "NaiveForecaster(); NaiveForecaster()",
        "import os",
        "from os import path",
        "if True:\n    return NaiveForecaster()",
        # Unsupported expression forms
        "NaiveForecaster() if True else NaiveForecaster()",
        # Boolean operators are deliberately not part of the safe grammar
        "NaiveForecaster() and NaiveForecaster()",
        "NaiveForecaster() or NaiveForecaster()",
        # Comparisons
        "NaiveForecaster() == NaiveForecaster()",
        "NaiveForecaster() < NaiveForecaster()",
        # Chained / indirect calls
        "NaiveForecaster()()",
        "(NaiveForecaster())()",
    ],
)
def test_craft_safe_rejects_unsafe_specs(spec):
    """Test that unsafe Python constructs are rejected in safe mode."""
    with pytest.raises(ValueError, match="unsafe or invalid specification|safe mode"):
        craft(spec, safe=True)


def test_deps():
    """Check that deps retrieves the correct requirement sets."""
    # should return length 0 list since has no deps
    assert deps(simple_spec) == []
    assert deps(pipe_spec_no_deps) == []

    # should correctly find the single dependency
    assert deps(simple_spec_with_dep) == ["xgboostlss"]
    assert deps(pipe_spec_with_deps) == ["xgboostlss"]


def test_sklearn_imports():
    """Check that sklearn estimators can be crafted."""
    from skpro.registry._lookup_sklearn import _all_sklearn_estimators

    sklearn_estimators = dict(_all_sklearn_estimators())

    from sklearn.ensemble import RandomForestRegressor

    assert craft("RandomForestRegressor()").__class__ == RandomForestRegressor
    rf_instance = craft("RandomForestRegressor(n_estimators=10)")
    assert isinstance(rf_instance, RandomForestRegressor)
    assert craft("RandomForestRegressor(n_estimators=10)").n_estimators == 10

    for est_name in ["StandardScaler", "KNeighborsClassifier", "RandomForestRegressor"]:
        assert est_name in sklearn_estimators.keys()

        est_spec = f"{est_name}()"
        est_obj = craft(est_spec)

        assert est_obj.__class__.__name__ == est_name
