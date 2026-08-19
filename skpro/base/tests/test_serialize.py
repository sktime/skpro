"""Tests for the serialization-node archive format.

The format is specified in `STEP 27
<https://github.com/sktime/enhancement-proposals/pull/52>`_.
"""
# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)

import json
import pickle
import tempfile
from pathlib import Path
from zipfile import ZipFile

import pytest

from skpro.base import BaseObject, load
from skpro.base._serialize import (
    COMPONENT_PERSISTENT_ID,
    FORMAT_VERSION,
    _component_dir,
)


class _Leaf(BaseObject):
    """Object with no BaseObject children."""

    def __init__(self, value=0):
        self.value = value
        super().__init__()


class _Holder(BaseObject):
    """Object holding children in containers of varying shape."""

    def __init__(self, mapping=None, nested=None, peer=None):
        self.mapping = mapping
        self.nested = nested
        self.peer = peer
        super().__init__()


def _round_trip(obj, serialization_format="pickle"):
    """Save to a zip file and load it back, returning (loaded, namelist)."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "obj.zip"
        obj.save(path, serialization_format=serialization_format)
        with ZipFile(path, "r") as zf:
            names = zf.namelist()
        return load(path), names


# --------------------------------------------------------------------------- #
# archive layout
# --------------------------------------------------------------------------- #


def test_root_is_a_node_without_wrapper():
    """The archive root is the root object's own node."""
    _, names = _round_trip(_Leaf(1))

    assert "_metadata" in names
    assert "_obj" in names
    assert not any(name.startswith("root/") for name in names)


def test_no_global_manifest():
    """STEP 27 replaces the global manifest with node-local index files."""
    _, names = _round_trip(_Holder(mapping={"a": _Leaf(1)}))

    for rejected in ("manifest.json", "_version", "_format"):
        assert rejected not in names

    assert "_components/index.json" in names


def test_minimal_node_omits_optional_directories():
    """A leaf with no artifacts is saved as _metadata plus _obj only."""
    _, names = _round_trip(_Leaf(1))

    assert not any(name.startswith("_components") for name in names)
    assert not any(name.startswith("_artifacts") for name in names)


def test_metadata_is_versioned():
    """_metadata records the format version, class, and serialization format."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "obj.zip"
        _Leaf(1).save(path)
        with ZipFile(path, "r") as zf:
            metadata = pickle.loads(zf.read("_metadata"))

    assert metadata == {
        "format_version": FORMAT_VERSION,
        "class": _Leaf,
        "serialization_format": "pickle",
    }


def test_component_index_does_not_duplicate_child_metadata():
    """A child has authority over its own _metadata; the parent never copies it."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "obj.zip"
        _Holder(mapping={"a": _Leaf(1)}).save(path)
        with ZipFile(path, "r") as zf:
            index = json.loads(zf.read("_components/index.json"))

    assert index == {"component-0000": {"path": "component-0000"}}
    for record in index.values():
        assert "class" not in record


def test_obj_references_components_by_persistent_id():
    """Children are referenced by opaque persistent ID, not by attribute path."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "obj.zip"
        _Holder(mapping={"a": _Leaf(1)}).save(path)
        with ZipFile(path, "r") as zf:
            obj_bytes = zf.read("_obj")

    assert COMPONENT_PERSISTENT_ID.encode() in obj_bytes
    # the rejected design encoded attribute paths into component IDs
    assert b"mapping__a" not in obj_bytes


def test_nested_components_recurse():
    """A child node follows the same contract, including its own _components."""
    obj = _Holder(mapping={"outer": _Holder(mapping={"inner": _Leaf(42)})})
    loaded, names = _round_trip(obj)

    assert "_components/component-0000/_components/index.json" in names
    assert loaded.mapping["outer"].mapping["inner"].value == 42


# --------------------------------------------------------------------------- #
# component discovery
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "obj,getter",
    [
        (_Holder(mapping={"a": _Leaf(7)}), lambda o: o.mapping["a"]),
        (_Holder(nested=[[_Leaf(7)]]), lambda o: o.nested[0][0]),
        (_Holder(nested=({"k": _Leaf(7)},)), lambda o: o.nested[0]["k"]),
        (_Holder(peer=_Leaf(7)), lambda o: o.peer),
    ],
)
def test_components_found_in_arbitrary_containers(obj, getter):
    """Children are discovered wherever they live, including inside dicts."""
    loaded, names = _round_trip(obj)

    assert "_components/index.json" in names
    assert getter(loaded).value == 7


def test_shared_child_is_written_once_and_stays_shared():
    """A child referenced twice in one node keeps a single, shared component."""
    shared = _Leaf(99)
    loaded, names = _round_trip(_Holder(mapping={"x": shared}, nested=[shared]))

    component_objs = [n for n in names if n.startswith("_components/component-")]
    assert len([n for n in component_objs if n.endswith("/_obj")]) == 1
    assert loaded.mapping["x"] is loaded.nested[0]


def test_third_party_objects_stay_inline():
    """Non-BaseObject children are not components; they stay in the parent _obj."""
    from sklearn.linear_model import LinearRegression

    loaded, names = _round_trip(_Holder(peer=LinearRegression()))

    assert not any(name.startswith("_components") for name in names)
    assert isinstance(loaded.peer, LinearRegression)


def test_self_reference_round_trips():
    """A self-reference is handled by the pickle memo, not treated as a cycle."""
    obj = _Holder()
    obj.peer = obj

    loaded, _ = _round_trip(obj)

    assert loaded.peer is loaded


# --------------------------------------------------------------------------- #
# unsupported graphs
# --------------------------------------------------------------------------- #


def test_ownership_cycle_raises():
    """Cyclic ownership raises a clear error rather than recursing forever."""
    first, second = _Holder(), _Holder()
    first.peer = second
    second.peer = first

    with pytest.raises(ValueError, match="Ownership cycle detected"):
        _round_trip(first)


def test_cross_branch_alias_raises():
    """An object shared by two different components raises a clear error."""
    alias = _Leaf(1)
    obj = _Holder(nested=[_Holder(peer=alias), _Holder(peer=alias)])

    with pytest.raises(ValueError, match="referenced from two different components"):
        _round_trip(obj)


# --------------------------------------------------------------------------- #
# serialization tags
# --------------------------------------------------------------------------- #


class _Skipping(BaseObject):
    """Object dropping a reconstructable cache from its serialized state."""

    _tags = {"serialization:skip": ("cache_",)}

    def __init__(self):
        self.keep = 1
        self.cache_ = "expensive"
        super().__init__()


class _DoubleTagged(BaseObject):
    """Object tagging one attribute both skip and native artifact."""

    _tags = {
        "serialization:skip": ("model_",),
        "serialization:native_artifacts": ("model_",),
    }

    def __init__(self):
        self.model_ = 1
        super().__init__()


def test_skip_tag_drops_attribute():
    """Attributes tagged serialization:skip do not reach _obj."""
    obj = _Skipping()
    loaded, _ = _round_trip(obj)

    assert loaded.keep == 1
    assert not hasattr(loaded, "cache_")


def test_save_does_not_mutate_source():
    """Saving restores every attribute it temporarily removed."""
    obj = _Skipping()
    _round_trip(obj)

    assert obj.cache_ == "expensive"


def test_save_restores_attributes_on_failure():
    """A failure part way through save still restores the source object."""

    class _Unpicklable:
        def __reduce__(self):
            raise TypeError("nope")

    obj = _Skipping()
    obj.broken = _Unpicklable()

    with pytest.raises(TypeError):
        obj.save()

    assert obj.cache_ == "expensive"


def test_conflicting_tags_raise():
    """An attribute may carry at most one of the two serialization tags."""
    with pytest.raises(ValueError, match="at most one of the two tags"):
        _DoubleTagged().save()


# --------------------------------------------------------------------------- #
# in-memory persistence
# --------------------------------------------------------------------------- #


def test_in_memory_leaf_is_lightweight():
    """A node with no artifacts and no children stays plain pickle bytes."""
    serial = _Leaf(1).save()

    assert isinstance(serial, tuple) and len(serial) == 2
    assert serial[0] is _Leaf
    assert not serial[1].startswith(b"PK")
    assert pickle.loads(serial[1]).value == 1
    assert load(serial).value == 1


def test_in_memory_composite_is_a_zip():
    """A node with children becomes an in-memory zip of the same layout."""
    serial = _Holder(mapping={"a": _Leaf(5)}).save()

    assert serial[1].startswith(b"PK")

    from io import BytesIO

    with ZipFile(BytesIO(serial[1])) as zf:
        assert "_components/index.json" in zf.namelist()

    assert load(serial).mapping["a"].value == 5


# --------------------------------------------------------------------------- #
# backward compatibility
# --------------------------------------------------------------------------- #


def test_legacy_metadata_with_bare_class_loads():
    """Archives whose _metadata is a bare pickled class remain loadable."""
    obj = _Leaf(3)

    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "legacy.zip"
        with ZipFile(path, "w") as zf:
            zf.writestr("_metadata", pickle.dumps(type(obj)))
            zf.writestr("_obj", pickle.dumps(obj))

        assert load(path).value == 3


def test_legacy_two_tuple_loads():
    """In-memory (class, pickle_bytes) tuples remain loadable."""
    obj = _Leaf(4)

    assert load((type(obj), pickle.dumps(obj))).value == 4


def test_legacy_three_tuple_loads():
    """Three-element tuples from pre-STEP-27 development versions still load."""
    obj = _Leaf(5)

    assert load((type(obj), pickle.dumps(obj), "pickle")).value == 5


# --------------------------------------------------------------------------- #
# error handling
# --------------------------------------------------------------------------- #


def test_future_format_version_is_rejected():
    """A newer format version fails up front, not part way through unpickling."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "future.zip"
        with ZipFile(path, "w") as zf:
            zf.writestr(
                "_metadata",
                pickle.dumps(
                    {
                        "format_version": FORMAT_VERSION + 1,
                        "class": _Leaf,
                        "serialization_format": "pickle",
                    }
                ),
            )
            zf.writestr("_obj", pickle.dumps(_Leaf(1)))

        with pytest.raises(ValueError, match="declares format version"):
            load(path)


@pytest.mark.parametrize("rel_path", ["../escape", "/absolute", "", "a/../../b"])
def test_component_path_outside_node_is_rejected(rel_path):
    """Component paths resolving outside their own node are refused."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        components_dir = Path(tmp_dir) / "_components"
        components_dir.mkdir()

        with pytest.raises(ValueError, match="component path|resolves outside"):
            _component_dir(components_dir, rel_path)


def test_unknown_serialization_format_is_rejected():
    """Only the registered serialization formats are accepted."""
    with pytest.raises(ValueError, match="is not yet supported"):
        _Leaf(1).save(serialization_format="joblib")


def test_unsupported_persistent_id_is_rejected():
    """An unrecognised persistent ID fails with a clear unpickling error."""
    from io import BytesIO

    from skpro.base._serialize import _ComponentUnpickler

    class _RogueP(pickle.Pickler):
        def persistent_id(self, obj):
            return ("some-other-scheme", "0") if isinstance(obj, _Leaf) else None

    buffer = BytesIO()
    _RogueP(buffer).dump({"a": _Leaf(1)})

    unpickler = _ComponentUnpickler(BytesIO(buffer.getvalue()), lambda cid: None)
    with pytest.raises(pickle.UnpicklingError, match="unsupported persistent ID"):
        unpickler.load()


def test_load_rejects_non_zip_path():
    """Loading a path that is not a zip archive fails with a clear error."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "obj.txt"
        path.write_text("not an archive")

        with pytest.raises(ValueError, match="Expected a .zip file"):
            load(path)


def test_load_rejects_unsupported_type():
    """Loading an object of unsupported type fails with a clear error."""
    with pytest.raises(TypeError, match="must be a tuple, str, or Path"):
        load(42)
