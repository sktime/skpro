"""Tests for the test utilities."""

from skpro.tests._config import EXCLUDE_ESTIMATORS
from skpro.tests.test_switch import run_test_for_class
from skpro.utils.validation._dependencies import _check_estimator_deps


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

    # estimator on the exception list
    from skpro.tests._config_test_dummy import DummySkipped

    # boolean flag for whether to run tests for all estimators
    from skpro.tests.test_all_estimators import ONLY_CHANGED_MODULES

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
    run_nodep = run_test_for_class(f_with_deps)
    assert isinstance(run, bool)

    dep_present = _check_estimator_deps(f_with_deps, severity="none")
    if not dep_present:
        assert not run_nodep

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
