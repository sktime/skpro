"""Serialization and deserialization utilities for skpro objects.

This module implements the serialization-node format specified in
`STEP 27 <https://github.com/sktime/enhancement-proposals/pull/52>`_,
"Recursive composite and native serialization".

Every ``BaseObject`` is serialized as a self-contained *serialization node*::

    node/
    ├── _metadata
    ├── _obj
    ├── _artifacts/        # optional
    │   ├── index.json
    │   └── ...
    └── _components/       # optional
        ├── index.json
        └── ...

``_artifacts`` (attributes selected for framework-native serialization) and
``_components`` (child ``BaseObject`` nodes, following this same contract
recursively) are orthogonal, and both are optional. The archive root is the
root object's own node; there is no wrapper directory and no global manifest.

See :ref:`serialization_ref` for the user-facing format documentation.
"""
# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)

__all__ = ["load"]

import json
import pickle
from functools import lru_cache
from io import BytesIO
from pathlib import Path

#: version of the serialization-node format written by this module
FORMAT_VERSION = 2

#: serialization formats available for ``_metadata`` and ``_obj``
#: native artifact backends always use their own framework formats
SERIALIZATION_FORMATS = {
    "pickle",
    "cloudpickle",
}

# tag prefix of persistent IDs used for component references inside ``_obj``.
# this is deliberately the same literal used by ``sktime``, since the archive
# format is shared across packages - it is not an ``sktime`` import.
COMPONENT_PERSISTENT_ID = "sktime-component"

_METADATA = "_metadata"
_OBJ = "_obj"
_ARTIFACTS = "_artifacts"
_COMPONENTS = "_components"
_INDEX = "index.json"


# --------------------------------------------------------------------------- #
# serialization format registry
# --------------------------------------------------------------------------- #


def _get_serializer(serialization_format):
    """Return serialization module for the given serialization format.

    Parameters
    ----------
    serialization_format : str
        one of ``SERIALIZATION_FORMATS``

    Returns
    -------
    module
        module exposing ``dump``/``dumps``, i.e., ``pickle`` or ``cloudpickle``
    """
    from skbase.utils.dependencies import _check_soft_dependencies

    if serialization_format not in SERIALIZATION_FORMATS:
        raise ValueError(
            f"The provided `serialization_format`='{serialization_format}' "
            "is not yet supported. The possible formats are: "
            f"{sorted(SERIALIZATION_FORMATS)}."
        )

    if serialization_format == "cloudpickle":
        _check_soft_dependencies("cloudpickle", severity="error")
        import cloudpickle

        return cloudpickle

    return pickle


@lru_cache(maxsize=None)
def _component_pickler_cls(serialization_format):
    """Return a ``Pickler`` subclass that externalizes ``BaseObject`` children.

    Parameters
    ----------
    serialization_format : str
        one of ``SERIALIZATION_FORMATS``

    Returns
    -------
    type
        ``Pickler`` subclass taking ``(file, registry)``
    """
    serializer = _get_serializer(serialization_format)

    if serialization_format == "cloudpickle":
        base_cls = serializer.CloudPickler
    else:
        base_cls = serializer.Pickler

    class _ComponentPickler(base_cls):
        """Pickler emitting persistent IDs for child ``BaseObject`` instances."""

        def __init__(self, file, registry):
            super().__init__(file)
            self._registry = registry

        def persistent_id(self, obj):
            """Return a component reference for ``obj``, or None to pickle inline."""
            return self._registry.persistent_id(obj)

    return _ComponentPickler


class _ComponentUnpickler(pickle.Unpickler):
    """Unpickler resolving component references through a node-local resolver.

    ``cloudpickle`` only customizes the writing side, so a plain
    ``pickle.Unpickler`` reads both formats.

    Parameters
    ----------
    file : file-like
        binary stream holding a node's ``_obj``
    resolver : callable
        maps a component ID to the loaded child object
    """

    def __init__(self, file, resolver):
        super().__init__(file)
        self._resolver = resolver

    def persistent_load(self, pid):
        """Resolve a persistent ID to an already-loaded component."""
        if (
            not isinstance(pid, tuple)
            or len(pid) != 2
            or pid[0] != COMPONENT_PERSISTENT_ID
            or not isinstance(pid[1], str)
        ):
            raise pickle.UnpicklingError(
                f"unsupported persistent ID encountered while loading: {pid!r}. "
                f"Expected a tuple ({COMPONENT_PERSISTENT_ID!r}, component_id)."
            )
        return self._resolver(pid[1])


# --------------------------------------------------------------------------- #
# native artifact backends
# --------------------------------------------------------------------------- #


class _NativeArtifactBackend:
    """Base strategy for native artifact serialization."""

    backend = None

    def _load_class(self, record):
        """Load class from artifact record."""
        import importlib

        class_path = record["class"]
        parts = class_path.split(".")

        for i in range(len(parts) - 1, 0, -1):
            module_name = ".".join(parts[:i])
            qualname = parts[i:]

            try:
                obj = importlib.import_module(module_name)
            except ModuleNotFoundError:
                continue

            for attr in qualname:
                obj = getattr(obj, attr)
            return obj

        raise ModuleNotFoundError(f"Could not import class {class_path!r}.")

    def supports(self, obj):
        """Return whether backend supports the object."""
        raise NotImplementedError

    def save(self, obj, path, *, estimator, name):
        """Save object to path."""
        raise NotImplementedError

    def load(self, path, record, *, estimator, name):
        """Load object from path and artifact metadata."""
        raise NotImplementedError


class _PretrainedArtifactBackend(_NativeArtifactBackend):
    """Native artifact backend for save_pretrained/from_pretrained objects."""

    backend = "pretrained"

    _pretrained_base_classes = {
        "huggingface_hub.hub_mixin.ModelHubMixin",
        "peft.peft_model.PeftModel",
        "transformers.integrations.peft.PeftAdapterMixin",
        "transformers.modeling_utils.PreTrainedModel",
    }

    def supports(self, obj):
        """Return whether object supports pretrained-style serialization."""
        cls = type(obj)
        mro_classes = {f"{base.__module__}.{base.__qualname__}" for base in cls.__mro__}
        from_pretrained = getattr(cls, "from_pretrained", None)
        return (
            callable(getattr(obj, "save_pretrained", None))
            and callable(getattr(obj, "from_pretrained", None))
            and bool(mro_classes & self._pretrained_base_classes)
            and getattr(from_pretrained, "__self__", None) is cls
        )

    def save(self, obj, path, *, estimator, name):
        """Save an object using save_pretrained."""
        obj.save_pretrained(path)

    def load(self, path, record, *, estimator, name):
        """Load an object using from_pretrained with estimator-provided kwargs."""
        cls = self._load_class(record)
        load_kwargs = {}
        get_load_kwargs = getattr(estimator, "_get_native_artifact_load_kwargs", None)
        if callable(get_load_kwargs):
            load_kwargs = get_load_kwargs(name)
        if cls.__module__.startswith("peft.") and "model" in load_kwargs:
            model = load_kwargs.pop("model")
            return cls.from_pretrained(model, path, **load_kwargs)
        return cls.from_pretrained(path, **load_kwargs)


class _KerasArtifactBackend(_NativeArtifactBackend):
    """Native artifact backend for Keras models."""

    backend = "keras"

    def supports(self, obj):
        """Return whether object looks like a Keras model."""
        return any(
            cls.__name__ == "Model" and "keras" in cls.__module__
            for cls in type(obj).__mro__
        )

    def save(self, obj, path, *, estimator, name):
        """Save a Keras model using the native .keras format."""
        obj.save(path / "model.keras")

    def load(self, path, record, *, estimator, name):
        """Load a Keras model using keras.models.load_model."""
        from tensorflow import keras

        custom_objects = None
        get_custom_objects = getattr(estimator, "get_custom_objects", None)
        if callable(get_custom_objects):
            custom_objects = get_custom_objects()

        model = keras.models.load_model(
            path / "model.keras",
            custom_objects=custom_objects,
        )

        if hasattr(model, "optimizer"):
            estimator.optimizer_ = model.optimizer
            estimator.optimizer = model.optimizer

        return model


class _LightningCheckpointArtifactBackend(_NativeArtifactBackend):
    """Native artifact backend for Lightning checkpoint models."""

    backend = "lightning_checkpoint"

    def supports(self, obj):
        """Return whether object supports Lightning checkpoint loading."""
        return callable(getattr(type(obj), "load_from_checkpoint", None)) and any(
            cls.__name__ == "LightningModule" and "lightning" in cls.__module__
            for cls in type(obj).__mro__
        )

    def save(self, obj, path, *, estimator, name):
        """Save a Lightning model checkpoint."""
        import lightning
        import torch

        checkpoint_path = path / "model.ckpt"
        checkpoint = {
            "state_dict": obj.state_dict(),
            obj.CHECKPOINT_HYPER_PARAMS_KEY: dict(obj.hparams),
            "pytorch-lightning_version": lightning.__version__,
        }
        obj.on_save_checkpoint(checkpoint)
        torch.save(checkpoint, checkpoint_path)

    def load(self, path, record, *, estimator, name):
        """Load a Lightning model checkpoint."""
        cls = self._load_class(record)
        checkpoint_path = path / "model.ckpt"
        return cls.load_from_checkpoint(checkpoint_path)


class _TorchStateDictArtifactBackend(_NativeArtifactBackend):
    """Native artifact backend for torch modules, using state dictionaries."""

    backend = "torch_state_dict"

    def supports(self, obj):
        """Return whether object is a torch module."""
        return any(
            cls.__name__ == "Module" and cls.__module__ == "torch.nn.modules.module"
            for cls in type(obj).__mro__
        )

    def save(self, obj, path, *, estimator, name):
        """Save a torch module's state dictionary on CPU."""
        import torch

        state_dict = obj.state_dict()
        for key, value in state_dict.items():
            if isinstance(value, torch.Tensor):
                state_dict[key] = value.detach().cpu()

        torch.save(state_dict, path / "state_dict.pt")

    def load(self, path, record, *, estimator, name):
        """Construct a torch module and restore its state dictionary."""
        import torch

        create_artifact = getattr(estimator, "_create_torch_artifact", None)
        if not callable(create_artifact):
            raise TypeError(
                f"Estimator {type(estimator).__name__} must implement "
                "`_create_torch_artifact(name)` to load native torch artifact "
                f"{name!r}."
            )

        artifact = create_artifact(name)
        if not isinstance(artifact, torch.nn.Module):
            raise TypeError(
                "`_create_torch_artifact` must return a torch.nn.Module, but "
                f"returned {type(artifact)!r} for artifact {name!r}."
            )

        state_dict_path = path / "state_dict.pt"
        state_dict = torch.load(
            state_dict_path,
            map_location="cpu",
            weights_only=True,
        )
        artifact.load_state_dict(state_dict)
        return artifact


_NATIVE_ARTIFACT_BACKENDS = [
    _PretrainedArtifactBackend(),
    _KerasArtifactBackend(),
    _LightningCheckpointArtifactBackend(),
    _TorchStateDictArtifactBackend(),
]


def _save_native_artifact_backend(obj, *, name):
    """Return native artifact backend for saving object."""
    for backend in _NATIVE_ARTIFACT_BACKENDS:
        if backend.supports(obj):
            return backend

    raise TypeError(
        f"No native serialization backend is available for artifact {name!r} "
        f"of type {type(obj)!r}."
    )


def _load_native_artifact_backend(backend_name):
    """Return native artifact backend for loading by backend name."""
    for backend in _NATIVE_ARTIFACT_BACKENDS:
        if backend.backend == backend_name:
            return backend

    raise ValueError(
        f"No native artifact backend is available for backend {backend_name!r}."
    )


class _NativeArtifactStore:
    """Store native serialization artifacts inside a serialization node.

    Parameters
    ----------
    artifact_root : Path
        the node's ``_artifacts`` directory
    """

    def __init__(self, artifact_root):
        self.artifact_root = artifact_root
        self.index = {}

    def save(self, name, obj, *, estimator):
        """Save a native artifact to the store."""
        artifact_path = self.artifact_root / name
        artifact_path.mkdir(parents=True)
        backend = _save_native_artifact_backend(obj, name=name)
        backend.save(
            obj,
            artifact_path,
            estimator=estimator,
            name=name,
        )
        cls = type(obj)
        self.index[name] = {
            "backend": backend.backend,
            "class": f"{cls.__module__}.{cls.__qualname__}",
            "path": name,
        }
        return self.index[name]

    def save_index(self):
        """Save native artifact index."""
        if len(self.index) == 0:
            return

        self.artifact_root.mkdir(exist_ok=True)
        with open(self.artifact_root / _INDEX, "w", encoding="utf-8") as file:
            json.dump(self.index, file, indent=2)


# --------------------------------------------------------------------------- #
# component references
# --------------------------------------------------------------------------- #


def _is_base_object(obj):
    """Return whether ``obj`` is a base object of the shared ``skbase`` protocol.

    Component recognition deliberately uses the ``skbase`` base object rather
    than ``skpro``'s own ``BaseObject``, so that composites mixing ``skpro``
    with ``sktime`` or other ``skbase``-based packages serialize recursively.
    """
    from skbase.base import BaseObject as _SkbaseBaseObject

    return isinstance(obj, _SkbaseBaseObject)


class _ComponentRegistry:
    """Assign persistent IDs to the child components of a single node.

    Children are recorded once per identity, so a child referenced twice inside
    one node resolves to a single component that both references share.

    Parameters
    ----------
    owner : BaseObject
        the object whose node is being written; excluded from component
        discovery, since it is the node itself rather than a child of it
    """

    def __init__(self, owner):
        self._owner = owner
        self._ids = {}
        self.children = []

    def persistent_id(self, obj):
        """Return a component reference for ``obj``, or None to pickle inline."""
        if obj is self._owner or not _is_base_object(obj):
            return None

        key = id(obj)
        if key not in self._ids:
            component_id = f"component-{len(self.children):04d}"
            self._ids[key] = component_id
            # the object is retained in ``children``, which keeps ``id(obj)``
            # stable for the lifetime of the registry
            self.children.append((component_id, obj))

        return (COMPONENT_PERSISTENT_ID, self._ids[key])


class _SaveContext:
    """Track object identities across a recursive save, to reject bad graphs.

    ``STEP 27`` does not support ownership cycles or cross-branch aliases in
    this version of the format; both raise rather than looping forever or
    producing an ambiguous graph.
    """

    def __init__(self):
        self.stack = []
        self.owners = {}

    def enter(self, obj, node_path):
        """Register ``obj`` as being written at archive-relative ``node_path``."""
        key = id(obj)

        if key in self.stack:
            raise ValueError(
                f"Ownership cycle detected while serializing "
                f"{type(obj).__name__!r} at {node_path!r}: the object is "
                "already being serialized as one of its own ancestors. "
                "Cyclic ownership graphs are not supported."
            )

        if key in self.owners:
            raise ValueError(
                f"Object of type {type(obj).__name__!r} is referenced from two "
                f"different components, {self.owners[key]!r} and {node_path!r}. "
                "Cross-branch aliases are not supported; a component may only "
                "be shared within a single parent node."
            )

        self.stack.append(key)
        self.owners[key] = node_path

    def exit(self, obj):
        """Unregister ``obj`` from the ancestor stack."""
        self.stack.remove(id(obj))


# --------------------------------------------------------------------------- #
# node writing
# --------------------------------------------------------------------------- #


def _get_serialization_tag(obj, tag_name):
    """Return a serialization tag value as a tuple of attribute names."""
    value = obj.get_tag(tag_name, (), raise_error=False)
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(value)


def _validate_serialization_tags(obj):
    """Raise if the serialization tags of ``obj`` are inconsistent."""
    skip = _get_serialization_tag(obj, "serialization:skip")
    native = _get_serialization_tag(obj, "serialization:native_artifacts")

    both = set(skip) & set(native)
    if both:
        raise ValueError(
            f"Attributes {sorted(both)} of {type(obj).__name__!r} are tagged both "
            "'serialization:skip' and 'serialization:native_artifacts'. An "
            "attribute may carry at most one of the two tags."
        )

    return skip, native


def _write_node(obj, node_dir, serialization_format, context, node_path="<root>"):
    """Write ``obj`` as a serialization node at ``node_dir``.

    Attributes are classified in the order mandated by ``STEP 27``: skipped
    attributes are dropped, native artifacts are externalized, remaining
    ``BaseObject`` children become components, and everything left goes into
    ``_obj``.

    Parameters
    ----------
    obj : BaseObject
        object to serialize
    node_dir : Path
        directory to write the node into; created if it does not exist
    serialization_format : str
        format for ``_metadata`` and ``_obj``, inherited by child nodes
    context : _SaveContext
        shared identity bookkeeping for cycle and alias detection
    node_path : str
        location of this node relative to the archive root, for error messages
    """
    context.enter(obj, node_path)
    try:
        _write_node_inner(obj, node_dir, serialization_format, context, node_path)
    finally:
        context.exit(obj)


def _write_node_inner(obj, node_dir, serialization_format, context, node_path):
    """Write a single node, without ancestor bookkeeping."""
    serializer = _get_serializer(serialization_format)
    skip, native_artifacts = _validate_serialization_tags(obj)

    node_dir.mkdir(parents=True, exist_ok=True)

    # 1./2. drop skipped attributes and externalize native artifacts, so that
    # neither reaches ``_obj``. Saving must not mutate the source object, so
    # the attributes are restored unconditionally.
    removed_attrs = {}
    for name in (*skip, *native_artifacts):
        if name in obj.__dict__:
            removed_attrs[name] = obj.__dict__.pop(name)

    try:
        # 3./4. component references and remaining state
        registry = _ComponentRegistry(obj)
        buffer = BytesIO()
        pickler_cls = _component_pickler_cls(serialization_format)
        try:
            pickler_cls(buffer, registry).dump(obj)
        except Exception as e:
            raise type(e)(
                f"Failed to serialize {type(obj).__name__!r}: {e}"
            ).with_traceback(e.__traceback__) from e
        obj_bytes = buffer.getvalue()
    finally:
        obj.__dict__.update(removed_attrs)

    metadata = {
        "format_version": FORMAT_VERSION,
        "class": type(obj),
        "serialization_format": serialization_format,
    }
    with open(node_dir / _METADATA, "wb") as file:
        serializer.dump(metadata, file)

    with open(node_dir / _OBJ, "wb") as file:
        file.write(obj_bytes)

    _write_artifacts(obj, node_dir / _ARTIFACTS, native_artifacts)
    _write_components(
        registry, node_dir / _COMPONENTS, serialization_format, context, node_path
    )


def _write_artifacts(obj, artifacts_dir, native_artifacts):
    """Write the ``_artifacts`` directory of a node, if it has any content."""
    if not native_artifacts:
        return

    store = _NativeArtifactStore(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    for name in native_artifacts:
        artifact = getattr(obj, name, None)
        if artifact is not None:
            store.save(name, artifact, estimator=obj)

    store.save_index()

    if not any(artifacts_dir.iterdir()):
        artifacts_dir.rmdir()


def _write_components(
    registry, components_dir, serialization_format, context, node_path
):
    """Write the ``_components`` directory of a node, if it has any children."""
    if not registry.children:
        return

    components_dir.mkdir(parents=True, exist_ok=True)
    index = {}

    prefix = "" if node_path == "<root>" else f"{node_path}/"
    for component_id, child in registry.children:
        index[component_id] = {"path": component_id}
        _write_node(
            child,
            components_dir / component_id,
            serialization_format,
            context,
            f"{prefix}{_COMPONENTS}/{component_id}",
        )

    with open(components_dir / _INDEX, "w", encoding="utf-8") as file:
        json.dump(index, file, indent=2)


def _is_minimal_node(node_dir):
    """Return whether a node has neither artifacts nor components."""
    return (
        not (node_dir / _ARTIFACTS).exists() and not (node_dir / _COMPONENTS).exists()
    )


# --------------------------------------------------------------------------- #
# node reading
# --------------------------------------------------------------------------- #


def _read_metadata(node_dir, node_path="<root>"):
    """Read and normalize a node's ``_metadata``.

    Accepts both the versioned mapping written by this module and the legacy
    form, a bare pickled class. ``_metadata`` is always read with plain
    ``pickle``, so that a reader can determine the node's format before it
    knows which serializer wrote the node.

    Returns
    -------
    dict
        with keys ``format_version``, ``class`` and ``serialization_format``
    """
    metadata_path = node_dir / _METADATA

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Serialization node at {node_path!r} has no {_METADATA} entry, "
            "so it is not a valid skpro save archive."
        )

    with open(metadata_path, "rb") as file:
        metadata = pickle.load(file)  # noqa: S301

    # legacy nodes store the class object directly
    if isinstance(metadata, type):
        return {
            "format_version": 1,
            "class": metadata,
            "serialization_format": "pickle",
        }

    if not isinstance(metadata, dict) or "class" not in metadata:
        raise ValueError(
            f"Unexpected {_METADATA} content in serialization node at "
            f"{node_path!r}: expected a class or a mapping with a 'class' "
            f"key, but found {type(metadata)!r}."
        )

    format_version = metadata.get("format_version", 1)
    if format_version > FORMAT_VERSION:
        raise ValueError(
            f"Serialization node at {node_path!r} declares format version "
            f"{format_version}, but this version of skpro supports at most "
            f"{FORMAT_VERSION}. Please upgrade skpro to load this archive."
        )

    metadata.setdefault("format_version", format_version)
    metadata.setdefault("serialization_format", "pickle")
    return metadata


def _component_dir(components_dir, rel_path, node_path="<root>"):
    """Resolve a component path, rejecting anything escaping the node.

    A node resolves component IDs only through its own index, and a resolved
    path must stay inside that node's ``_components`` directory.
    """
    if not isinstance(rel_path, str) or not rel_path:
        raise ValueError(
            f"Invalid component path {rel_path!r} in {_COMPONENTS}/{_INDEX} "
            f"of node {node_path!r}: expected a non-empty relative path."
        )

    root = components_dir.resolve()
    target = (components_dir / rel_path).resolve()

    if target == root or not target.is_relative_to(root):
        raise ValueError(
            f"Component path {rel_path!r} in {_COMPONENTS}/{_INDEX} of node "
            f"{node_path!r} resolves outside its own node. The archive is "
            "malformed or unsafe."
        )

    return target


def _read_component_index(node_dir, node_path="<root>"):
    """Read a node's ``_components/index.json``, empty dict if it has none."""
    index_path = node_dir / _COMPONENTS / _INDEX

    if not index_path.exists():
        return {}

    with open(index_path, encoding="utf-8") as file:
        index = json.load(file)

    if not isinstance(index, dict):
        raise ValueError(
            f"Malformed {_COMPONENTS}/{_INDEX} in serialization node at "
            f"{node_path!r}: expected a mapping of component IDs to records."
        )

    return index


def _read_node(node_dir, node_path="<root>"):
    """Read a serialization node, recursively loading its components.

    Parameters
    ----------
    node_dir : Path
        directory holding the node
    node_path : str
        location of this node relative to the archive root, for error messages

    Returns
    -------
    BaseObject
        the reconstructed object
    """
    _read_metadata(node_dir, node_path)
    index = _read_component_index(node_dir, node_path)
    components_dir = node_dir / _COMPONENTS
    cache = {}

    def resolve(component_id):
        """Load the component with ``component_id`` from this node's index."""
        if component_id in cache:
            return cache[component_id]

        if component_id not in index:
            raise ValueError(
                f"Component reference {component_id!r} is not listed in "
                f"{_COMPONENTS}/{_INDEX} of the node at {node_path!r}."
            )

        record = index[component_id]
        rel_path = record.get("path") if isinstance(record, dict) else record
        prefix = "" if node_path == "<root>" else f"{node_path}/"
        child = _read_node(
            _component_dir(components_dir, rel_path, node_path),
            f"{prefix}{_COMPONENTS}/{component_id}",
        )
        cache[component_id] = child
        return child

    with open(node_dir / _OBJ, "rb") as file:
        obj = _ComponentUnpickler(file, resolve).load()

    # ``_metadata`` is read for its version gate; the class it records is used
    # by the loose ``load`` to dispatch, not to validate ``_obj``. Under
    # ``cloudpickle``, a by-value class is pickled separately in ``_metadata``
    # and in ``_obj``, so the two are equal but not identical.
    _load_native_artifacts(obj, node_dir / _ARTIFACTS)
    return obj


def _load_native_artifacts(obj, artifacts_dir):
    """Restore native artifacts onto ``obj`` from an extracted node."""
    index_path = artifacts_dir / _INDEX
    if not index_path.exists():
        return

    with open(index_path, encoding="utf-8") as file:
        index = json.load(file)

    for name, record in index.items():
        backend = _load_native_artifact_backend(record["backend"])
        artifact_path = artifacts_dir / record["path"]
        artifact = backend.load(
            artifact_path,
            record,
            estimator=obj,
            name=name,
        )
        setattr(obj, name, artifact)


# --------------------------------------------------------------------------- #
# serialization mixin
# --------------------------------------------------------------------------- #


class _SerializationMixin:
    """Mixin providing the serialization API for skpro base objects.

    Holds ``save``, ``load_from_serial`` and ``load_from_path``, which keeps
    node reading and writing, native backend dispatch, and component reference
    handling out of ``BaseObject`` and out of the loose ``load`` function.
    """

    def save(self, path=None, serialization_format="pickle"):
        """Save serialized self to a bytes-like object or to a (.zip) file.

        Behaviour:

        * if ``path`` is None, returns an in-memory serialized self
        * if ``path`` is a file location, stores self at that location as a zip file

        Saved archives are serialization nodes, with the following contents:

        * ``_metadata`` - format version, class of self, and serialization format
        * ``_obj`` - serialized self, with child components replaced by references
        * ``_artifacts/`` - optional, framework-native model artifacts
        * ``_components/`` - optional, child ``BaseObject`` nodes, recursively
          following this same layout

        Both optional directories are omitted when empty, so a plain
        non-composite object is saved as ``_metadata`` plus ``_obj``.

        See :ref:`serialization_ref` for the full format specification.

        Parameters
        ----------
        path : None or file location (str or Path), optional (default=None)
            if None, self is saved to an in-memory object.
            if file location, self is saved to that file location, as a zip file. If:

            - path="estimator" then a zip file ``estimator.zip`` will be made at cwd.
            - path="/home/stored/estimator" then a zip file ``estimator.zip`` will be
              stored in ``/home/stored/``.

        serialization_format : str, optional (default="pickle")
            Module to use for serialization.
            The available options are ``"pickle"`` and ``"cloudpickle"``.
            Note that non-default formats might require
            installation of other soft dependencies.
            This setting applies to ``_metadata`` and ``_obj``; native artifact
            backends always use their framework-specific formats.

        Returns
        -------
        if ``path`` is None - tuple ``(cls, serialized_bytes)``
            where ``cls`` is ``type(self)``, and ``serialized_bytes`` is either a
            plain pickle stream, if self has no artifacts and no components, or
            an in-memory zip archive of the same layout written to disk.
        if ``path`` is a file location - ``ZipFile``
            reference to the written zip file.

        See Also
        --------
        skpro.base.load : Load a saved object from memory or from a zip path.
        """
        import shutil
        from tempfile import TemporaryDirectory
        from zipfile import ZipFile

        # validate the format before any work is done
        _get_serializer(serialization_format)

        if path is not None and not isinstance(path, (str, Path)):
            raise TypeError(
                "`path` is expected to either be a string or a Path object "
                f"but found of type: {type(path)}."
            )

        with TemporaryDirectory() as tmp_dir:
            node_dir = Path(tmp_dir) / "node"
            _write_node(self, node_dir, serialization_format, _SaveContext())

            # a node with neither artifacts nor components round-trips through
            # its ``_obj`` alone, which is a plain pickle stream
            if path is None and _is_minimal_node(node_dir):
                return (type(self), (node_dir / _OBJ).read_bytes())

            archive_base = str(Path(tmp_dir) / "archive")
            shutil.make_archive(
                base_name=archive_base, format="zip", root_dir=str(node_dir)
            )
            archive_path = Path(archive_base + ".zip")

            if path is None:
                return (type(self), archive_path.read_bytes())

            target_base = str(path)
            if target_base.endswith(".zip"):
                target_base = target_base[: -len(".zip")]
            zip_path = Path(target_base + ".zip")

            zip_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(archive_path), str(zip_path))

        return ZipFile(zip_path)

    @classmethod
    def load_from_serial(cls, serial):
        """Load object from serialized memory container.

        Parameters
        ----------
        serial : bytes
            second element of the output of ``cls.save(None)``.
            Either a plain pickle stream, or an in-memory zip archive.

        Returns
        -------
        Deserialized self, resulting in output ``serial``, of ``cls.save(None)``.
        """
        from zipfile import is_zipfile

        if is_zipfile(BytesIO(serial)):
            return cls.load_from_path(BytesIO(serial))

        return pickle.loads(serial)  # noqa: S301

    @classmethod
    def load_from_path(cls, serial):
        """Load object from file location.

        Parameters
        ----------
        serial : str, Path, or file-like
            location of, or open handle to, a zip archive written by ``save``

        Returns
        -------
        Deserialized self, resulting in output at ``serial``, of ``cls.save(path)``.
        """
        from tempfile import TemporaryDirectory
        from zipfile import ZipFile

        with TemporaryDirectory() as tmp_dir:
            with ZipFile(serial, "r") as file:
                file.extractall(tmp_dir)

            return _read_node(Path(tmp_dir))


# --------------------------------------------------------------------------- #
# loose load function
# --------------------------------------------------------------------------- #


def load(serial):
    """Load an object from an in-memory container or from a file location.

    Deserializes an object that was saved via the ``save`` method of an
    ``skpro`` ``BaseObject`` descendant.

    This function is deliberately thin: it opens the container, reads the root
    node's ``_metadata``, and delegates to that class. No estimator-specific or
    framework-specific logic lives here.

    Parameters
    ----------
    serial : tuple, str, or Path
        If ``tuple``: in-memory serialized form ``(cls, serialized_bytes)``,
            as returned by ``obj.save()``.
        If ``str`` or ``Path``: path to a ``.zip`` file,
            as written by ``obj.save(path)``.

    Returns
    -------
    obj : BaseObject descendant
        Deserialized object.

    Notes
    -----
    Loading is pickle-based, and therefore executes code contained in the
    archive. Only load archives from sources you trust.

    Examples
    --------
    >>> from skpro.base import load  # doctest: +SKIP
    >>> serial = estimator.save()    # doctest: +SKIP
    >>> estimator_loaded = load(serial)  # doctest: +SKIP

    See Also
    --------
    skpro.base.BaseObject.save : Persist an object to memory or to a zip file.
    """
    from zipfile import ZipFile

    if isinstance(serial, tuple):
        # three-element tuples were written by pre-STEP-27 development versions,
        # which carried the serialization format alongside the payload
        if len(serial) not in (2, 3):
            raise ValueError(
                "When `serial` is a tuple it must have two elements "
                "(cls, serialized_bytes)."
            )
        cls, stored = serial[0], serial[1]
        return cls.load_from_serial(stored)

    if isinstance(serial, (str, Path)):
        path = Path(serial)
        if not path.exists():
            raise FileNotFoundError(f"No file found at path: {path}")
        if path.suffix != ".zip":
            raise ValueError(
                f"Expected a .zip file, but got: {path.suffix}. "
                "Files saved by skpro's save method have a .zip extension."
            )

        with ZipFile(path, "r") as zip_file:
            with zip_file.open(_METADATA) as file:
                metadata = pickle.load(file)  # noqa: S301

        if isinstance(metadata, type):
            cls = metadata
        elif isinstance(metadata, dict) and "class" in metadata:
            cls = metadata["class"]
        else:
            raise ValueError(
                f"Unexpected {_METADATA} content in archive {str(path)!r}: "
                f"expected a class or a mapping with a 'class' key, but found "
                f"{type(metadata)!r}."
            )

        return cls.load_from_path(path)

    raise TypeError(
        f"`serial` must be a tuple, str, or Path, but found: {type(serial)}"
    )
