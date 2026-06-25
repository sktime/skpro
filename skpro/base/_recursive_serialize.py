"""Recursive structured serialization for composite skpro objects."""
# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)

__all__ = ["recursive_save_to_zip", "recursive_load_from_zip"]

import dataclasses
import json
import pickle
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile


@dataclasses.dataclass
class _ComponentRef:
    """Placeholder for a recursively-serialized sub-component."""

    component_id: str


MANIFEST_VERSION = 2


def recursive_save_to_zip(obj, path, serialization_format="pickle"):
    """Save an object recursively to a zip file with manifest.

    Parameters
    ----------
    obj : BaseObject
        The object to save.
    path : str or Path
        File path for the zip archive.
    serialization_format : str
        ``"pickle"`` or ``"joblib"``.
    """
    path = Path(path)
    if path.suffix != ".zip":
        path = path.with_suffix(".zip")

    components = {}
    manifest = {
        "version": MANIFEST_VERSION,
        "serialization_format": serialization_format,
        "root_class": f"{type(obj).__module__}.{type(obj).__qualname__}",
        "is_fitted": getattr(obj, "is_fitted", False),
        "components": {},
        "load_order": [],
    }

    _recursive_save_node(
        obj=obj,
        component_id="root",
        parent_id=None,
        attr_ref="",
        components=components,
        manifest=manifest,
        serialization_format=serialization_format,
    )

    manifest["load_order"] = _topological_sort(manifest["components"])

    with ZipFile(path, "w") as zf:
        zf.writestr("manifest.json", json.dumps(manifest, indent=2))
        zf.writestr("_format", serialization_format)
        zf.writestr("_version", str(MANIFEST_VERSION))

        for comp_id, (cls_bytes, obj_bytes) in components.items():
            prefix = manifest["components"][comp_id]["path"]
            zf.writestr(prefix + "_metadata", cls_bytes)
            zf.writestr(prefix + "_obj", obj_bytes)


def recursive_load_from_zip(path):
    """Load an object from a recursive zip file.

    Parameters
    ----------
    path : str or Path
        Path to the zip archive.

    Returns
    -------
    obj : BaseObject
        The deserialized object.
    """
    path = Path(path)

    with ZipFile(path, "r") as zf:
        manifest = json.loads(zf.read("manifest.json"))
        fmt = manifest["serialization_format"]

        loaded = {}

        for comp_id in manifest["load_order"]:
            comp_info = manifest["components"][comp_id]
            prefix = comp_info["path"]

            obj_bytes = zf.read(prefix + "_obj")

            if comp_info["is_leaf"]:
                loaded[comp_id] = _deserialize_blob(obj_bytes, fmt)
            else:
                state_dict = _deserialize_blob(obj_bytes, fmt)
                state_dict = _replace_placeholders(state_dict, loaded)

                cls_bytes = zf.read(prefix + "_metadata")
                cls = pickle.loads(cls_bytes)  # noqa: S301
                obj = cls.__new__(cls)
                obj.__dict__.update(state_dict)
                loaded[comp_id] = obj

        return loaded["root"]


def is_v1_zip(namelist):
    """Check if a zip file uses the v1 flat format.

    Parameters
    ----------
    namelist : list of str
        File names inside the zip archive.

    Returns
    -------
    bool
        True if the zip is v1 (flat) format.
    """
    return "manifest.json" not in namelist


def _recursive_save_node(
    obj,
    component_id,
    parent_id,
    attr_ref,
    components,
    manifest,
    serialization_format,
):
    """Recursively save one node of the object tree."""
    from skbase.base import BaseObject as _SkbaseBaseObject

    is_skpro_obj = isinstance(obj, _SkbaseBaseObject)
    is_composite = is_skpro_obj and obj.is_composite()

    if component_id == "root":
        path_prefix = "root/"
    else:
        path_prefix = f"components/{component_id}/"

    if not is_composite:
        serialized_as = "structured" if is_skpro_obj else "pickle_fallback"

        try:
            obj_bytes = _serialize_blob(obj, serialization_format)
        except Exception as e:
            raise TypeError(
                f"Component '{component_id}' ({type(obj)}) is not serializable. "
                f"Original error: {e}"
            ) from e

        cls_bytes = pickle.dumps(type(obj))
        components[component_id] = (cls_bytes, obj_bytes)
        manifest["components"][component_id] = {
            "class": f"{type(obj).__module__}.{type(obj).__qualname__}",
            "path": path_prefix,
            "is_leaf": True,
            "serialized_as": serialized_as,
            "children": [],
            "parent": parent_id,
            "attr_ref": attr_ref,
        }
        return

    state = obj.__dict__.copy()
    sub_objects = _find_base_objects_in_dict(state)

    children = []

    for dict_key, index_path, sub_obj in sub_objects:
        child_id = _make_component_id(component_id, dict_key, index_path)
        child_attr_ref = _make_attr_ref(dict_key, index_path)
        children.append(child_id)

        _recursive_save_node(
            obj=sub_obj,
            component_id=child_id,
            parent_id=component_id,
            attr_ref=child_attr_ref,
            components=components,
            manifest=manifest,
            serialization_format=serialization_format,
        )

    modified_state = _replace_with_placeholders(state, sub_objects, component_id)

    cls_bytes = pickle.dumps(type(obj))
    obj_bytes = _serialize_blob(modified_state, serialization_format)
    components[component_id] = (cls_bytes, obj_bytes)
    manifest["components"][component_id] = {
        "class": f"{type(obj).__module__}.{type(obj).__qualname__}",
        "path": path_prefix,
        "is_leaf": False,
        "serialized_as": "structured",
        "children": children,
        "parent": parent_id,
        "attr_ref": attr_ref,
    }


def _find_base_objects_in_dict(d):
    """Walk dict values to find all BaseObject instances.

    Handles direct values, lists, tuples, and nested combinations.
    Does NOT descend into dicts or numpy arrays.

    Parameters
    ----------
    d : dict
        The ``__dict__`` of an object.

    Returns
    -------
    list of (dict_key, index_path, base_object)
        Each entry is a BaseObject found in the dict, with the key
        it was found under and the index path within nested structures.
    """
    from skbase.base import BaseObject as _SkbaseBaseObject

    results = []
    for key, value in d.items():
        _find_base_objects_recursive(value, key, [], results, _SkbaseBaseObject)
    return results


def _find_base_objects_recursive(value, dict_key, index_path, results, base_cls):
    """Recursively find BaseObject instances in nested structures."""
    if isinstance(value, base_cls):
        results.append((dict_key, list(index_path), value))
    elif isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            _find_base_objects_recursive(
                item, dict_key, index_path + [i], results, base_cls
            )


def _replace_with_placeholders(state, sub_objects, component_id):
    """Deep-copy state dict, replacing BaseObject refs with _ComponentRef.

    Parameters
    ----------
    state : dict
        The ``__dict__`` of a composite object.
    sub_objects : list of (dict_key, index_path, base_object)
        The BaseObjects to replace, as returned by ``_find_base_objects_in_dict``.
    component_id : str
        The component ID of the current node (used to generate child IDs).

    Returns
    -------
    dict
        A copy of state with BaseObject refs replaced by ``_ComponentRef``.
    """
    ref_map = {}
    for dict_key, index_path, sub_obj in sub_objects:
        ref_map[id(sub_obj)] = _make_component_id(component_id, dict_key, index_path)

    return _deep_replace(state, ref_map)


def _deep_replace(value, ref_map):
    """Recursively replace BaseObject instances with _ComponentRef placeholders."""
    if id(value) in ref_map:
        return _ComponentRef(component_id=ref_map[id(value)])

    if isinstance(value, dict):
        return {k: _deep_replace(v, ref_map) for k, v in value.items()}
    elif isinstance(value, list):
        return [_deep_replace(item, ref_map) for item in value]
    elif isinstance(value, tuple):
        return tuple(_deep_replace(item, ref_map) for item in value)
    else:
        return value


def _replace_placeholders(state_dict, loaded_components):
    """Replace _ComponentRef placeholders with loaded objects.

    Parameters
    ----------
    state_dict : dict
        Deserialized ``__dict__`` containing ``_ComponentRef`` placeholders.
    loaded_components : dict
        Already-loaded components keyed by component_id.

    Returns
    -------
    dict
        The state dict with placeholders replaced by real objects.
    """
    return _deep_restore(state_dict, loaded_components)


def _deep_restore(value, loaded):
    """Recursively replace _ComponentRef with loaded objects."""
    if isinstance(value, _ComponentRef):
        return loaded[value.component_id]

    if isinstance(value, dict):
        return {k: _deep_restore(v, loaded) for k, v in value.items()}
    elif isinstance(value, list):
        return [_deep_restore(item, loaded) for item in value]
    elif isinstance(value, tuple):
        return tuple(_deep_restore(item, loaded) for item in value)
    else:
        return value


def _make_component_id(parent_id, dict_key, index_path):
    """Generate a unique component ID.

    Parameters
    ----------
    parent_id : str
        The parent's component ID.
    dict_key : str
        The attribute name in the parent's ``__dict__``.
    index_path : list of int
        Indices within nested list/tuple structures.

    Returns
    -------
    str
        A unique component ID like ``"estimators__0__1"``.
    """
    parts = [dict_key] + [str(i) for i in index_path]
    suffix = "__".join(parts)

    if parent_id == "root":
        return suffix
    else:
        return f"{parent_id}__{suffix}"


def _make_attr_ref(dict_key, index_path):
    """Generate a human-readable attribute reference.

    Parameters
    ----------
    dict_key : str
        The attribute name.
    index_path : list of int
        Indices within nested structures.

    Returns
    -------
    str
        A reference like ``"estimators_[0][1]"``.
    """
    ref = dict_key
    for idx in index_path:
        ref += f"[{idx}]"
    return ref


def _topological_sort(components):
    """Return component IDs in load order: leaves first, root last.

    Uses Kahn's algorithm.

    Parameters
    ----------
    components : dict
        The ``components`` dict from the manifest.

    Returns
    -------
    list of str
        Component IDs in topological order.
    """
    in_degree = {cid: len(info["children"]) for cid, info in components.items()}
    queue = [cid for cid, deg in in_degree.items() if deg == 0]
    order = []

    while queue:
        node = queue.pop(0)
        order.append(node)
        parent = components[node].get("parent")
        if parent is not None and parent in in_degree:
            in_degree[parent] -= 1
            if in_degree[parent] == 0:
                queue.append(parent)

    if len(order) != len(components):
        raise ValueError(
            "Cyclic dependency detected in component tree. "
            f"Sorted {len(order)} of {len(components)} components."
        )

    return order


def _serialize_blob(obj, fmt):
    """Serialize an object or dict to bytes.

    Parameters
    ----------
    obj : object
        The object to serialize.
    fmt : str
        ``"pickle"`` or ``"joblib"``.

    Returns
    -------
    bytes
        Serialized bytes.
    """
    if fmt == "pickle":
        return pickle.dumps(obj)
    elif fmt == "joblib":
        import joblib

        buffer = BytesIO()
        joblib.dump(obj, buffer)
        return buffer.getvalue()
    else:
        raise ValueError(
            f"serialization_format must be 'pickle' or 'joblib', " f"but found: {fmt!r}"
        )


def _deserialize_blob(blob_bytes, fmt):
    """Deserialize bytes to an object.

    Parameters
    ----------
    blob_bytes : bytes
        Serialized bytes.
    fmt : str
        ``"pickle"`` or ``"joblib"``.

    Returns
    -------
    object
        The deserialized object.
    """
    if fmt == "pickle":
        return pickle.loads(blob_bytes)  # noqa: S301
    elif fmt == "joblib":
        import joblib

        return joblib.load(BytesIO(blob_bytes))
    else:
        raise ValueError(
            f"serialization_format must be 'pickle' or 'joblib', " f"but found: {fmt!r}"
        )
