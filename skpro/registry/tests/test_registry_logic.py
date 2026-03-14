"""Tests for registry dynamic lookup logic."""

import pytest

from skpro.base import BaseEstimator, BaseObject
from skpro.registry.test_class_register import (
    get_test_class_registry,
    get_test_classes_for_obj,
)


def test_registry_basics():
    """Verify registry is correctly populated."""
    reg = get_test_class_registry()
    assert isinstance(reg, dict)
    assert "object" in reg


@pytest.mark.parametrize("obj", ["not_an_estimator", 123, None])
def test_get_test_classes_for_obj_non_skpro(obj):
    """Verify non-skpro objects return an empty list."""
    assert get_test_classes_for_obj(obj) == []


def test_get_test_classes_for_obj_base_types():
    """Verify BaseObject and BaseEstimator return correct test classes."""
    # BaseObject should contain at least the 'object' test class
    obj_tests = get_test_classes_for_obj(BaseObject())
    assert any(issubclass(t, object) for t in obj_tests)

    # BaseEstimator should contain 'object' and 'estimator' test classes
    est_tests = get_test_classes_for_obj(BaseEstimator())
    assert len(est_tests) >= 2


def test_get_test_classes_for_obj_scitype_lookup():
    """Verify that a registered estimator returns the correct scitype test class."""
    from skpro.registry import all_objects

    # We use 'object_types' as identified in the help() output
    # We set return_names=False so it only returns the class objects
    regressors = all_objects(object_types="regressor_proba", return_names=False)

    # If the list is a list of tuples, handle that.
    # Based on the docstring, if return_names=False, it should return the objects.
    if len(regressors) > 0:
        # If it returns a list of classes directly:
        obj = regressors[0]
        test_classes = get_test_classes_for_obj(obj)

        assert isinstance(test_classes, list)
        assert len(test_classes) > 0
        for cls in test_classes:
            assert isinstance(cls, type)
    else:
        pytest.skip("No regressors found to test lookup.")


def test_get_test_classes_for_obj_no_duplicates():
    """Verify that the registry deduplicates test classes."""
    est = BaseEstimator()
    classes = get_test_classes_for_obj(est)

    # Check that there are no duplicates in the list
    assert len(classes) == len(set(classes))
