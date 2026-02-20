"""Main configuration file for pytest.

Contents:
adds an --only_changed_modules option to pytest
this allows to turn on/off differential testing (for shorter runtime)
"on" condition ensures that only estimators are tested that have changed,
    more precisely, only estimators whose class is in a module
    that has changed compared to the main branch
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
        "--skip_vm_tests",
        action="store_true",
        default=False,
        help="skip estimators tagged with 'tests:vm': True",
    )


def pytest_configure(config):
    """Pytest configuration preamble."""
    from skpro.tests import _config

    if config.getoption("--only_changed_modules") in [True, "True"]:
        _config.ONLY_CHANGED_MODULES = True

    skip_vm = config.getoption("--skip_vm_tests")
    if not skip_vm:
        skip_vm = os.environ.get("SKPRO_SKIP_VM_TESTS", "false").lower() == "true"

    if skip_vm:
        _config.SKIP_VM_TESTS = True
