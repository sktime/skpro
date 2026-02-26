"""Tests for tag register and tag functionality."""

import inspect
import sys

from skpro.registry._tags import OBJECT_TAG_REGISTER, _BaseTag


def test_tag_register_type():
    """Test the specification of the tag register. See _tags for specs."""
    assert isinstance(OBJECT_TAG_REGISTER, list)
    assert all(isinstance(tag, tuple) for tag in OBJECT_TAG_REGISTER)

    for tag in OBJECT_TAG_REGISTER:
        assert len(tag) == 4
        assert isinstance(tag[0], str)
        assert isinstance(tag[1], (str, list))
        if isinstance(tag[1], list):
            assert all(isinstance(x, str) for x in tag[1])
        assert isinstance(tag[2], (str, tuple))
        if isinstance(tag[2], tuple):
            assert len(tag[2]) == 2
            assert isinstance(tag[2][0], str)
            assert isinstance(tag[2][1], (list, str))
            if isinstance(tag[2][1], list):
                assert all(isinstance(x, str) for x in tag[2][1])
        assert isinstance(tag[3], str)


def test_base_tag_classes():
    """Test that all _BaseTag subclasses have required _tags keys."""
    from skpro.registry import _tags as tags_module

    tag_clses = inspect.getmembers(tags_module, inspect.isclass)

    required_keys = {"tag_name", "parent_type", "tag_type", "short_descr"}

    found_classes = 0
    for name, cl in tag_clses:
        if name == "_BaseTag" or not issubclass(cl, _BaseTag):
            continue
        found_classes += 1

        cl_tags = cl.get_class_tags()

        # check all required keys are present
        for key in required_keys:
            assert (
                key in cl_tags
            ), f"Tag class {name} is missing required _tags key '{key}'"

        # check tag_name is a non-empty string
        assert isinstance(
            cl_tags["tag_name"], str
        ), f"Tag class {name} has non-string tag_name"
        assert len(cl_tags["tag_name"]) > 0, f"Tag class {name} has empty tag_name"

        # check parent_type is str or list of str
        pt = cl_tags["parent_type"]
        assert isinstance(
            pt, (str, list)
        ), f"Tag class {name} has invalid parent_type type"
        if isinstance(pt, list):
            assert all(
                isinstance(x, str) for x in pt
            ), f"Tag class {name} has non-string elements in parent_type"

        # check short_descr is a non-empty string
        assert isinstance(
            cl_tags["short_descr"], str
        ), f"Tag class {name} has non-string short_descr"
        assert (
            len(cl_tags["short_descr"]) > 0
        ), f"Tag class {name} has empty short_descr"

    # ensure we found at least one tag class
    assert found_classes > 0, "No _BaseTag subclasses found"


def test_tag_register_matches_classes():
    """Test that the auto-built register matches the class definitions."""
    from skpro.registry import _tags as tags_module

    tag_clses = inspect.getmembers(tags_module, inspect.isclass)

    class_tag_names = set()
    for name, cl in tag_clses:
        if name == "_BaseTag" or not issubclass(cl, _BaseTag):
            continue
        cl_tags = cl.get_class_tags()
        class_tag_names.add(cl_tags["tag_name"])

    register_tag_names = {t[0] for t in OBJECT_TAG_REGISTER}

    assert class_tag_names == register_tag_names, (
        f"Mismatch between class and register tag names. "
        f"In classes but not register: {class_tag_names - register_tag_names}. "
        f"In register but not classes: {register_tag_names - class_tag_names}."
    )
