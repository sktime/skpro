# -*- coding: utf-8 -*-
"""Tests for tag register an tag functionality."""

from skpro.registry._tags import OBJECT_TAG_REGISTER


def test_tag_register_type():
    """Test the specification of the tag register. See _tags for specs."""
    assert isinstance(OBJECT_TAG_REGISTER, list)
    assert all(isinstance(tag, tuple) for tag in OBJECT_TAG_REGISTER)

    for tag in OBJECT_TAG_REGISTER:
        assert len(tag) == 4
        assert isinstance(tag[0], str)
        assert isinstance(tag[1], str)
        assert isinstance(tag[2], (str, list))
        if isinstance(tag[2], list):
            assert all(isinstance(t, str) for t in tag[2])
        assert isinstance(tag[3], str)
