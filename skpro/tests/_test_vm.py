"""Auxiliary script to test an estinator in its own virtual machine."""

__all__ = ["run_test_vm"]

import os
import platform
import re

from skbase.utils.dependencies import _check_estimator_deps, _check_soft_dependencies


def run_test_vm(cls_name):
    """Test an estimator in its own virtual machine.

    Takes a string which is the name of a class in the skpro registry,
    and runs ``check_estimator`` on it in a separate virtual machine,
    with deps determined by the tag ``python_dependencies`` of the class.

    Does not run the test if python and operating system versions
    are incompatible with the estimator's dependencies,
    as checked via ``_check_estimator_deps``.

    Parameters
    ----------
    cls_name : str
        Name of the estimator class to test, e.g., "ExampleForecaster".

    Raises
    ------
    Exception
        if the ``check_estimator`` fails, or if the estimator is not found.
    """
    from skpro.registry import craft
    from skpro.utils import check_estimator

    if _check_soft_dependencies("torch", severity="none"):
        # disable mps for macos runners if torch is available
        if platform.system() == "Darwin":
            import torch

            torch.backends.mps.is_available = lambda: False

    if _check_soft_dependencies("hf-xet", severity="none"):
        # to allow hf-xet to download models on macos runners on version `latest`
        if platform.system() == "Darwin":
            os.environ["HF_XET_NUM_CONCURRENT_RANGE_GETS"] = "4"

    cls = craft(cls_name)
    if _check_estimator_deps(cls, severity="none"):
        skips = cls.get_class_tag("tests:skip_by_name", None)
        check_estimator(cls, raise_exceptions=True, tests_to_exclude=skips)
    else:
        print(  # noqa: T201
            f"Skipping estimator: {cls} due to incompatibility "
            "with python or OS version."
        )  # noqa: T201


def _get_estimator_specific_test_modules(cls_name):
    """Get the list of estimator specific test modules to run for an estimator.

    Returns the content of the ``tests:specific`` tag of the class ``cls_name``,
    after validating that the entries are importable ``skpro`` module paths.

    Used in the VM based CI test runs, to execute the pytest modules
    that contain tests specific to the estimator, in addition to the
    general API conformance tests run by ``run_test_vm``.

    Parameters
    ----------
    cls_name : str
        Name of the estimator class to test, e.g., "ExampleRegressor".

    Returns
    -------
    modules_to_run : list of str, or None
        List of module paths to run for the estimator,
        or None if the estimator specifies no such modules.

    Raises
    ------
    AssertionError
        if the ``tests:specific`` tag is not a list of strings,
        or if any of its entries is not an importable ``skpro`` module path.
    """
    from importlib.util import find_spec

    from skpro.registry import craft

    cls = craft(cls_name)

    modules = cls.get_class_tag("tests:specific", None)
    if modules is None:
        return None

    msg = f"{cls.__name__}.tests:specific must be a list of strings, found: {modules}"
    assert isinstance(modules, list), msg
    assert all(isinstance(module, str) for module in modules), msg

    if len(modules) == 0:
        return None

    module_pat = re.compile(r"^skpro(?:\.[a-z_][a-z0-9_]*)*$")
    bad_modules = [module for module in modules if not module_pat.fullmatch(module)]
    msg_bad = (
        f"{cls.__name__}.tests:specific contains invalid module paths: {bad_modules}"
    )
    assert len(bad_modules) == 0, msg_bad

    missing_modules = [module for module in modules if find_spec(module) is None]
    msg_missing = (
        f"{cls.__name__}.tests:specific contains missing modules: {missing_modules}"
    )
    assert len(missing_modules) == 0, msg_missing

    return modules.copy()
