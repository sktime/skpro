# -*- coding: utf-8 -*-
"""Automated tests based on the skbase test suite template."""
from skbase.testing import TestAllObjects as _TestAllObjects


class PackageConfig:
    """Contains package config variables for test classes."""

    # class variables which can be overridden by descendants
    # ------------------------------------------------------

    # package to search for objects
    # expected type: str, package/module name, relative to python environment root
    package_name = "skpro"

    # list of object types (class names) to exclude
    # expected type: list of str, str are class names
    exclude_objects = "ClassName"  # exclude classes from extension templates

    # list of valid tags
    # expected type: list of str, str are tag names
    valid_tags = ["estimator_type", "regressor_type", "transformer_type"]


class TestAllObjects(PackageConfig, _TestAllObjects):
    """Generic tests for all objects in the mini package."""
