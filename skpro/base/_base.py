"""Base class and template for regressors and transformers."""
from skbase.base import BaseEstimator as _BaseEstimator
from skbase.base import BaseMetaEstimator as _BaseMetaEstimator
from skbase.base import BaseObject as _BaseObject


class _CommonTags:
    """Mixin for common tag definitions to all estimator base classes."""

    # config common to all estimators
    _config = {}

    _tags = {
        "estimator_type": "estimator",
        "authors": "skpro developers",
        "maintainers": "skpro developers",
        "capability:serializable": True,
    }

    @property
    def name(self):
        """Return the name of the object or estimator."""
        return self.__class__.__name__

    def save(self, path=None, serialization_format="pickle"):
        """Save serialized self to bytes-like object or to a (.zip) file.

        Behaviour
        ---------
        * if ``path`` is None, returns an in-memory serialized
          ``(cls, bytes, serialization_format)`` tuple that can be
          deserialized with ``load``.
        * if ``path`` is a file location, writes a zip archive with the
          recursive v2 layout: each sub-component in its own subfolder,
          plus a ``manifest.json`` blueprint with topological load order
          (leaves first, root last). Non-``BaseObject`` children fall back
          to a single pickle/joblib blob
          (``serialized_as: "pickle_fallback"``).

        Saved zip contents (v2)
        -----------------------
        * ``manifest.json`` — component tree, paths, and load order
        * ``_format`` — ``"pickle"`` or ``"joblib"``
        * ``_version`` — format version (currently ``2``)
        * ``root/_metadata``, ``root/_obj`` — root class and state
        * ``components/<id>/_metadata``, ``components/<id>/_obj`` —
          each nested component

        Full format specification (in-memory tuple, zip v2, and legacy
        flat v1) is documented at :ref:`serialization_ref`.

        Parameters
        ----------
        path : None or str or Path, optional (default=None)
            if None, return the in-memory tuple
            ``(cls, bytes, serialization_format)``.
            if str or Path, file location to write the zip archive to.
            The path should end in ``.zip``; if it does not, ``.zip`` is appended.
        serialization_format : str, optional (default="pickle")
            The serialization backend to use. One of ``"pickle"`` or ``"joblib"``.

        Returns
        -------
        If ``path`` is None: tuple ``(cls, bytes, serialization_format)``
            where ``cls`` is the class of the object.
        If ``path`` is str or Path: ``ZipFile``
            reference to the written zip file.

        See Also
        --------
        skpro.base.load : Load a saved object from memory or a zip path.
        """
        if serialization_format not in ("pickle", "joblib"):
            raise ValueError(
                f"serialization_format must be 'pickle' or 'joblib', "
                f"but found: {serialization_format!r}"
            )

        if path is None:
            import pickle
            from io import BytesIO

            if serialization_format == "pickle":
                serialized = pickle.dumps(self)
            else:
                import joblib

                buffer = BytesIO()
                joblib.dump(self, buffer)
                serialized = buffer.getvalue()

            return (type(self), serialized, serialization_format)

        from pathlib import Path as _Path
        from zipfile import ZipFile

        from skpro.base._recursive_serialize import recursive_save_to_zip

        path = _Path(path)
        if path.suffix != ".zip":
            path = path.with_suffix(".zip")

        recursive_save_to_zip(self, path, serialization_format)
        return ZipFile(path, "r")

    @classmethod
    def load_from_serial(cls, serial, serialization_format="pickle"):
        """Load object from serialized memory container.

        Parameters
        ----------
        serial : bytes
            In-memory serialized bytes of the object.
        serialization_format : str, optional (default="pickle")
            The format used to serialize. One of ``"pickle"`` or ``"joblib"``.

        Returns
        -------
        Deserialized object.
        """
        if serialization_format == "joblib":
            from io import BytesIO

            import joblib

            return joblib.load(BytesIO(serial))

        import pickle

        return pickle.loads(serial)

    @classmethod
    def load_from_path(cls, path):
        """Load object from file location.

        Supports both v1 (flat) and v2 (recursive/manifest) zip formats.

        Parameters
        ----------
        path : str or Path
            Path to the zip file containing the serialized object.

        Returns
        -------
        Deserialized object.
        """
        import pickle
        from pathlib import Path as _Path
        from zipfile import ZipFile

        from skpro.base._recursive_serialize import is_v1_zip, recursive_load_from_zip

        path = _Path(path)

        with ZipFile(path, "r") as zf:
            if not is_v1_zip(zf.namelist()):
                return recursive_load_from_zip(path)

            # v1 flat format (backward compatibility)
            fmt = "pickle"
            if "_format" in zf.namelist():
                fmt = zf.read("_format").decode("utf-8")

            obj_bytes = zf.read("_obj")

        if fmt == "joblib":
            from io import BytesIO

            import joblib

            return joblib.load(BytesIO(obj_bytes))

        return pickle.loads(obj_bytes)


class BaseObject(_CommonTags, _BaseObject):
    """Base class for fittable objects."""

    def __init__(self):
        super().__init__()


class BaseEstimator(_CommonTags, _BaseEstimator):
    """Base class for fittable objects."""


class BaseMetaEstimator(_CommonTags, _BaseMetaEstimator):
    """Base class for fittable composite meta-objects."""
