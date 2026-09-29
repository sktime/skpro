"""Tests for the test utilities."""

import pytest
from skbase.utils.dependencies import _check_estimator_deps

from skpro.tests._config import EXCLUDE_ESTIMATORS
from skpro.tests.test_switch import run_test_for_class


def test_exclude_estimators():
    """Test that EXCLUDE_ESTIMATORS is a list of strings."""
    assert isinstance(EXCLUDE_ESTIMATORS, list)
    assert all(isinstance(estimator, str) for estimator in EXCLUDE_ESTIMATORS)


def test_run_test_for_class():
    """Test that run_test_for_class runs tests for various cases."""
    # estimator without soft deps
    from skpro.regression.bootstrap import BootstrapRegressor

    # estimator with soft deps
    from skpro.regression.mapie import MapieRegressor

    # boolean flag for whether to run tests for all estimators
    from skpro.tests._config import ONLY_CHANGED_MODULES

    # estimator on the exception list
    from skpro.tests._config_test_dummy import DummySkipped

    # shorthands
    f_on_excl_list = DummySkipped
    f_no_deps = BootstrapRegressor
    f_with_deps = MapieRegressor

    # test that assumptions on being on exception list are correct
    # if any of the below fail, switch the example
    assert f_on_excl_list.__name__ in EXCLUDE_ESTIMATORS
    assert f_no_deps.__name__ not in EXCLUDE_ESTIMATORS
    assert f_with_deps.__name__ not in EXCLUDE_ESTIMATORS

    # check result for skipped estimator
    run = run_test_for_class(f_on_excl_list)
    # run should be False, as the estimator is on the exception list
    assert isinstance(run, bool)
    assert not run
    # same with reason returned
    res = run_test_for_class(f_on_excl_list, return_reason=True)
    assert isinstance(res, tuple)
    assert len(res) == 2
    run, reason = res
    assert isinstance(run, bool)
    assert not run
    assert isinstance(reason, str)
    assert reason == "False_exclude_list"

    # check result for estimator without soft deps
    run = run_test_for_class(f_no_deps)
    assert isinstance(run, bool)
    if not ONLY_CHANGED_MODULES:  # if we run all tests, we should run this one
        assert run

    # result depends now on whether there is a change in the classes
    res = run_test_for_class(f_no_deps, return_reason=True)
    assert isinstance(res, tuple)
    assert len(res) == 2
    run_nodep, reason_nodep = res
    assert isinstance(run_nodep, bool)
    assert isinstance(reason_nodep, str)

    POS_REASONS = ["True_pyproject_change", "True_changed_class", "True_changed_tests"]

    if not ONLY_CHANGED_MODULES:
        assert run_nodep
        assert reason_nodep == "True_run_always"
    elif run_nodep:
        # otherwise, if we run, it must be due to changes in class or pyproject
        assert reason_nodep in POS_REASONS
    else:  # not run and only changed modules
        assert reason_nodep == "False_no_change"

    # now check estimator with soft deps
    run_wdep = run_test_for_class(f_with_deps)
    assert isinstance(run, bool)

    dep_present = _check_estimator_deps(f_with_deps, severity="none")
    if not dep_present:
        assert not run_wdep

    res = run_test_for_class(f_with_deps, return_reason=True)
    assert isinstance(res, tuple)
    assert len(res) == 2
    run_wdep, reason_wdep = res

    if not dep_present:
        assert not run_wdep
        assert reason_wdep == "False_required_deps_missing"
    elif not ONLY_CHANGED_MODULES:
        assert run_wdep
        assert reason_wdep == "True_run_always"
    elif run_wdep:
        assert reason_wdep in POS_REASONS
    else:  # not run and only changed modules
        assert reason_wdep == "False_no_change"

    # now a list of estimator with exception plus one estimator
    run = run_test_for_class([f_on_excl_list, f_no_deps])
    assert isinstance(run, bool)
    assert not run

    res = run_test_for_class([f_on_excl_list, f_no_deps], return_reason=True)
    assert isinstance(res, tuple)
    assert len(res) == 2
    run, reason = res
    assert isinstance(run, bool)
    assert not run
    assert reason == "False_exclude_list"

    # now a list of the estimator with and without soft deps
    run = run_test_for_class([f_no_deps, f_with_deps])
    assert isinstance(run, bool)

    # if deps are not present, we do not run the test
    # otherwise we run the test iff we run one of the two
    if not dep_present:
        assert not run
    else:
        assert run == run_nodep or run_wdep

    res = run_test_for_class([f_no_deps, f_with_deps], return_reason=True)
    assert isinstance(res, tuple)
    assert len(res) == 2
    run, reason = res

    if not dep_present:
        assert not run
        assert reason == "False_required_deps_missing"
    elif not ONLY_CHANGED_MODULES:
        assert run
        assert reason == "True_run_always"
    elif run:
        assert reason in POS_REASONS
        assert reason_wdep == reason or reason_nodep == reason
    else:
        assert reason == "False_no_change"
        assert reason_wdep == "False_no_change"
        assert reason_nodep == "False_no_change"


def _make_dummy_with_tag(tag_value):
    """Create a dummy object class with ``tests:specific`` set to ``tag_value``."""
    from skpro.base import BaseObject

    class DummyWithSpecificTests(BaseObject):
        _tags = {"tests:specific": tag_value}

    return DummyWithSpecificTests


@pytest.mark.parametrize(
    "tag_value",
    [None, [], ["skpro.regression.tests.test_gam"]],
    ids=["none", "empty", "populated"],
)
def test_get_estimator_specific_test_modules_valid(monkeypatch, tag_value):
    """Test that valid tests:specific tag values are returned or resolve to None."""
    import skpro.registry
    from skpro.tests._test_vm import _get_estimator_specific_test_modules

    cls = _make_dummy_with_tag(tag_value)
    monkeypatch.setattr(skpro.registry, "craft", lambda cls_name: cls)

    modules = _get_estimator_specific_test_modules("DummyWithSpecificTests")

    # None and the empty list both mean "no estimator specific test modules"
    if tag_value is None or tag_value == []:
        assert modules is None
    else:
        assert modules == tag_value


@pytest.mark.parametrize(
    "tag_value",
    [
        "skpro.regression.tests.test_gam",  # str instead of list of str
        [42],  # not a str
        ["not_skpro.tests.test_foo"],  # not an skpro module path
        ["skpro.regression.tests.test_does_not_exist"],  # module does not exist
    ],
    ids=["str_not_list", "not_str", "not_skpro_module", "missing_module"],
)
def test_get_estimator_specific_test_modules_invalid(monkeypatch, tag_value):
    """Test that invalid tests:specific tag values are rejected."""
    import skpro.registry
    from skpro.tests._test_vm import _get_estimator_specific_test_modules

    cls = _make_dummy_with_tag(tag_value)
    monkeypatch.setattr(skpro.registry, "craft", lambda cls_name: cls)

    with pytest.raises(AssertionError):
        _get_estimator_specific_test_modules("DummyWithSpecificTests")


def test_tests_specific_tag_modules_resolve():
    """Test that tests:specific tags in the package point to existing modules.

    This guards against typos in the tag, which would otherwise surface
    only in the VM based CI runs of the estimator.
    """
    from skpro.registry import all_objects
    from skpro.tests._test_vm import _get_estimator_specific_test_modules

    objs = all_objects(return_names=True)

    tagged = {
        name: obj.get_class_tag("tests:specific", None)
        for name, obj in objs
        if obj.get_class_tag("tests:specific", None)
    }

    # the mechanism is of no use if no object populates the tag
    assert len(tagged) > 0, "no object in skpro populates the tests:specific tag"

    for name in tagged:
        # raises AssertionError if any module path is invalid or does not exist
        modules = _get_estimator_specific_test_modules(name)
        assert modules == tagged[name]
