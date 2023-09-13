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


def pytest_addoption(parser):
    """Pytest command line parser options adder."""
    parser.addoption(
        "--only_changed_modules",
        default=False,
        help="test only estimators from modules that have changed compared to main",
    )


def pytest_configure(config):
    """Pytest configuration preamble."""
    from skpro.tests import test_all_estimators

    if config.getoption("--only_changed_modules") in [True, "True"]:
        test_all_estimators.ONLY_CHANGED_MODULES = True
