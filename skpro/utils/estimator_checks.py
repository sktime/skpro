"""Estimator checker for extension."""

__author__ = ["fkiraly"]
__all__ = ["check_estimator"]

from inspect import isclass

from skpro.utils.validation._dependencies import _check_soft_dependencies


def check_estimator(
    estimator,
    raise_exceptions=False,
    tests_to_run=None,
    fixtures_to_run=None,
    verbose=True,
    tests_to_exclude=None,
    fixtures_to_exclude=None,
):
    """Run all tests on one single estimator.

    Tests that are run on estimator:
        all tests in test_all_estimators
        all interface compatibility tests from the module of estimator's scitype
            for example, test_all_regressors if estimator is a regressor

    Parameters
    ----------
    estimator : estimator class or estimator instance
    raise_exceptions : bool, optional, default=False
        whether to return exceptions/failures in the results dict, or raise them

        * if False: returns exceptions in returned `results` dict
        * if True: raises exceptions as they occur

    tests_to_run : str or list of str, optional. Default = run all tests.
        Names (test/function name string) of tests to run.
        sub-sets tests that are run to the tests given here.
    fixtures_to_run : str or list of str, optional. Default = run all tests.
        pytest test-fixture combination codes, which test-fixture combinations to run.
        sub-sets tests and fixtures to run to the list given here.
        If both tests_to_run and fixtures_to_run are provided, runs the *union*,
        i.e., all test-fixture combinations for tests in tests_to_run,
            plus all test-fixture combinations in fixtures_to_run.
    verbose : str, optional, default=True.
        whether to print out informative summary of tests run.
    tests_to_exclude : str or list of str, names of tests to exclude. default = None
        removes tests that should not be run, after subsetting via tests_to_run.
    fixtures_to_exclude : str or list of str, fixtures to exclude. default = None
        removes test-fixture combinations that should not be run.
        This is done after subsetting via fixtures_to_run.

    Returns
    -------
    results : dict of results of the tests in self
        keys are test/fixture strings, identical as in pytest, e.g., test[fixture]
        entries are the string "PASSED" if the test passed,
        or the exception raised if the test did not pass
        returned only if all tests pass, or raise_exceptions=False

    Raises
    ------
    if raise_exceptions=True,
    raises any exception produced by the tests directly

    Examples
    --------
    >>> from skpro.regression.residual import ResidualDouble
    >>> from skpro.utils import check_estimator

    Running all tests for ResidualDouble class,
    this uses all instances from get_test_params and compatible scenarios

    >>> results = check_estimator(ResidualDouble)
    All tests PASSED!

    Running all tests for a specific ResidualDouble
    this uses the instance that is passed and compatible scenarios

    >>> from sklearn.linear_model import LinearRegression
    >>> results = check_estimator(ResidualDouble(LinearRegression()))
    All tests PASSED!

    Running specific test (all fixtures) for ResidualDouble

    >>> results = check_estimator(ResidualDouble, tests_to_run="test_clone")
    All tests PASSED!

    {'test_clone[ResidualDouble-0]': 'PASSED',
    'test_clone[ResidualDouble-1]': 'PASSED'}

    Running one specific test-fixture-combination for ResidualDouble

    >>> check_estimator(
    ...    ResidualDouble, fixtures_to_run="test_clone[ResidualDouble-1]"
    ... )
    All tests PASSED!
    {'test_clone[ResidualDouble-1]': 'PASSED'}
    """
    msg = (
        "check_estimator is a testing utility for developers, and "
        "requires pytest to be present "
        "in the python environment, but pytest was not found. "
        "pytest is a developer dependency and not included in the base "
        "sktime installation. Please run: `pip install pytest` to "
        "install the pytest package. "
        "To install sktime with all developer dependencies, run:"
        " `pip install sktime[dev]`"
    )
    _check_soft_dependencies("pytest", msg=msg)

    from skpro.base import BaseEstimator
    from skpro.distributions.tests.test_all_distrs import TestAllDistributions
    from skpro.regression.tests.test_all_regressors import TestAllRegressors
    from skpro.tests.test_all_estimators import TestAllEstimators, TestAllObjects

    testclass_dict = dict()

    testclass_dict["regressor_proba"] = TestAllRegressors
    testclass_dict["distribution"] = TestAllDistributions

    results = TestAllObjects().run_tests(
        obj=estimator,
        raise_exceptions=raise_exceptions,
        tests_to_run=tests_to_run,
        fixtures_to_run=fixtures_to_run,
        tests_to_exclude=tests_to_exclude,
        fixtures_to_exclude=fixtures_to_exclude,
    )

    def is_estimator(obj):
        """Return whether obj is an estimator class or estimator object."""
        if isclass(obj):
            return issubclass(obj, BaseEstimator)
        else:
            return isinstance(obj, BaseEstimator)

    if is_estimator(estimator):
        results_estimator = TestAllEstimators().run_tests(
            obj=estimator,
            raise_exceptions=raise_exceptions,
            tests_to_run=tests_to_run,
            fixtures_to_run=fixtures_to_run,
            tests_to_exclude=tests_to_exclude,
            fixtures_to_exclude=fixtures_to_exclude,
        )
        results.update(results_estimator)

    if isclass(estimator) and hasattr(estimator, "get_class_tag"):
        scitype_of_estimator = estimator.get_class_tag("object_type", "object")
    elif hasattr(estimator, "get_tag"):
        scitype_of_estimator = estimator.get_tag("object_type", "object")
    else:
        scitype_of_estimator = ""

    if scitype_of_estimator in testclass_dict.keys():
        results_scitype = testclass_dict[scitype_of_estimator]().run_tests(
            obj=estimator,
            raise_exceptions=raise_exceptions,
            tests_to_run=tests_to_run,
            fixtures_to_run=fixtures_to_run,
            tests_to_exclude=tests_to_exclude,
            fixtures_to_exclude=fixtures_to_exclude,
        )
        results.update(results_scitype)

    failed_tests = [key for key in results.keys() if results[key] != "PASSED"]
    if len(failed_tests) > 0:
        msg = failed_tests
        msg = ["FAILED: " + x for x in msg]
        msg = "\n".join(msg)
    else:
        msg = "All tests PASSED!"

    if verbose:
        # printing is an intended feature, for console usage and interactive debugging
        print(msg)  # noqa T001

    return results
