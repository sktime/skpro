"""Tests for tag register and tag functionality."""

import inspect

import pytest

from skpro.registry import _tags
from skpro.registry._base_classes import get_obj_scitype_list
from skpro.registry._tags import OBJECT_TAG_REGISTER, _BaseTag, check_tag_is_valid

TAG_CLASSES = [
    cl
    for _, cl in inspect.getmembers(_tags, inspect.isclass)
    if issubclass(cl, _BaseTag) and cl is not _BaseTag
]


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


def test_tag_names_unique():
    """Test that no two tag classes define the same tag name."""
    tag_names = [cl.get_class_tag("tag_name") for cl in TAG_CLASSES]
    duplicates = {x for x in tag_names if tag_names.count(x) > 1}
    assert not duplicates, f"duplicate tag names in tag classes: {duplicates}"


@pytest.mark.parametrize("tag_cls", TAG_CLASSES, ids=lambda cl: cl.__name__)
def test_tag_class_spec(tag_cls):
    """Test that tag classes follow the tag class specification in _tags."""
    cls_name = tag_cls.__name__
    cl_tags = tag_cls.get_class_tags()

    tag_name = cl_tags["tag_name"]
    assert isinstance(tag_name, str)
    assert tag_name != _BaseTag.get_class_tag(
        "tag_name"
    ), f"tag class {cls_name} does not set the tag_name field"

    # class name should be tag name with ":" replaced by "__"
    assert cls_name == tag_name.replace(":", "__"), (
        f"tag class name {cls_name} does not match tag name {tag_name}, "
        f"class name should be {tag_name.replace(':', '__')}"
    )

    # parent_type should be valid scitype string(s)
    parent_type = cl_tags["parent_type"]
    if isinstance(parent_type, str):
        parent_type = [parent_type]
    assert isinstance(parent_type, list)
    valid_scitypes = get_obj_scitype_list()
    for scitype in parent_type:
        assert scitype in valid_scitypes, (
            f"parent_type {scitype!r} of tag {tag_name} is not a valid scitype, "
            f"must be one of {valid_scitypes}"
        )

    short_descr = cl_tags["short_descr"]
    assert isinstance(short_descr, str)
    assert short_descr != _BaseTag.get_class_tag(
        "short_descr"
    ), f"tag class {cls_name} does not set the short_descr field"
    assert len(short_descr) <= 80, (
        f"short_descr of tag {tag_name} must be at most 80 characters, "
        f"but has {len(short_descr)}"
    )

    assert isinstance(cl_tags["user_facing"], bool)

    # docstring should document the tag, following the common pattern
    doc = inspect.getdoc(tag_cls)
    if doc is not None:
        assert f'- String name: ``"{tag_name}"``' in doc, (
            f"docstring of tag class {cls_name} must contain the string name line "
            f'- String name: ``"{tag_name}"``'
        )


def test_object_type_doc_lists_scitypes():
    """Test that the object_type docstring lists all scitypes."""
    doc = _tags.object_type.__doc__
    if doc is None:
        pytest.skip("docstrings are not available, e.g., python -OO")

    for scitype in get_obj_scitype_list():
        assert f'``"{scitype}"``' in doc


def test_check_tag_is_valid():
    """Test check_tag_is_valid on valid and invalid tag values."""
    check_tag_is_valid("capability:survival", True)
    check_tag_is_valid("object_type", "regressor_proba")
    check_tag_is_valid("object_type", ["metric", "metric_distr"])
    check_tag_is_valid("distr:measuretype", "continuous")

    with pytest.raises(KeyError):
        check_tag_is_valid("not_a_valid_tag_name", True)
    with pytest.raises(ValueError):
        check_tag_is_valid("capability:survival", "yes")
    with pytest.raises(ValueError):
        check_tag_is_valid("distr:measuretype", "not_a_measure_type")
