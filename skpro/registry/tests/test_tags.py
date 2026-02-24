"""Tests for tag register an tag functionality."""

from skpro.registry._tags import OBJECT_TAG_REGISTER, OBJECT_TAG_LIST


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


def test_update_tag_presence():
    """Test that capabilities:update tag is correctly registered."""
    # Check that the tag name exists in the list
    assert "capabilities:update" in OBJECT_TAG_LIST

    # Verify properties of the specific entry for distributions
    update_tag_entry = [tag for tag in OBJECT_TAG_REGISTER if tag[0] == "capabilities:update"]
    
    # Ensure it's registered for the correct scitypes (distribution and regressor_proba)
    scitypes = [tag[1] for tag in update_tag_entry]
    assert "distribution" in scitypes
    assert "regressor_proba" in scitypes

    # Ensure the expected value type is boolean
    for entry in update_tag_entry:
        assert entry[2] == "bool"