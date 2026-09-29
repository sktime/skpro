"""Main configuration file for pytest.

Contents:
adds an --only_changed_modules option to pytest
this allows to turn on/off differential testing (for shorter runtime)
"on" condition ensures that only estimators are tested that have changed,
    more precisely, only estimators whose class is in a module
    that has changed compared to the main branch
by default, this is off, including for default local runs of pytest

adds an --only_vm_estimators option to pytest
this allows to turn on/off testing of estimators that require their own VM,
    i.e., estimators with the tag "tests:vm" set to True
"on" condition ensures that only such estimators are tested, and no others
it is used in the VM based CI runs, to execute the pytest modules
    listed in the "tests:specific" tag of the estimator
by default, this is off, including for default local runs of pytest
"""
# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["fkiraly"]

import os

from skbase.utils.dependencies import _check_soft_dependencies

# used to prevent tkinter related errors in CI
if _check_soft_dependencies("matplotlib", severity="none"):
    if os.environ.get("GITHUB_ACTIONS") == "true":
        import matplotlib

        matplotlib.use("Agg")


def pytest_addoption(parser):
    """Pytest command line parser options adder."""
    parser.addoption(
        "--only_changed_modules",
        default=False,
        help="test only estimators from modules that have changed compared to main",
    )
    parser.addoption(
        "--only_vm_estimators",
        default=False,
        help="flag for test runs on VM - tests only estimators that require a VM run",
    )


def pytest_configure(config):
    """Pytest configuration preamble."""
    from skpro.tests import _config

    if config.getoption("--only_changed_modules") in [True, "True"]:
        _config.ONLY_CHANGED_MODULES = True
    if config.getoption("--only_vm_estimators") in [True, "True"]:
        _config.ONLY_VM_ESTIMATORS = True
