"""Serialization and deserialization utilities for skpro objects."""
# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)

__all__ = ["load"]

import pickle
from pathlib import Path
from zipfile import ZipFile


def load(serial):
    """Load a serialized skpro object.

    Deserializes an object that was saved via the ``save`` method
    of a skpro ``BaseObject`` descendant.

    Supports the in-memory tuple from ``save()``, recursive zip format
    v2 (``manifest.json``), and legacy flat zip format v1. See
    :ref:`serialization_ref` for the full file format specification.

    Parameters
    ----------
    serial : tuple or str or Path
        If ``tuple``: in-memory serialized form
            ``(cls, bytes)`` or ``(cls, bytes, format)``
            as returned by ``obj.save(None)``.
        If ``str`` or ``Path``: path to a ``.zip`` file
            as written by ``obj.save(path)``.

    Returns
    -------
    obj : BaseObject descendant
        Deserialized object.

    Examples
    --------
    >>> from skpro.base import load  # doctest: +SKIP
    >>> obj = estimator.save()       # doctest: +SKIP
    >>> obj_loaded = load(obj)       # doctest: +SKIP

    See Also
    --------
    skpro.base.BaseObject.save : Persist an object to memory or zip.
    """
    if isinstance(serial, tuple):
        if len(serial) not in (2, 3):
            raise ValueError(
                "When `serial` is a tuple it must have two or three elements "
                "(cls, serialized_bytes) or (cls, serialized_bytes, format)."
            )
        cls = serial[0]
        stored = serial[1]
        fmt = serial[2] if len(serial) == 3 else "pickle"
        return cls.load_from_serial(stored, serialization_format=fmt)

    elif isinstance(serial, (str, Path)):
        path = Path(serial)
        if not path.exists():
            raise FileNotFoundError(f"No file found at path: {path}")
        if not path.suffix == ".zip":
            raise ValueError(
                f"Expected a .zip file, but got: {path.suffix}. "
                "Files saved by skpro's save method have a .zip extension."
            )
        with ZipFile(path, "r") as zf:
            if "manifest.json" in zf.namelist():
                from skpro.base._recursive_serialize import recursive_load_from_zip

                return recursive_load_from_zip(path)

            # v1 flat format
            cls = pickle.loads(zf.read("_metadata"))
        return cls.load_from_path(path)

    else:
        raise TypeError(
            f"`serial` must be a tuple, str, or Path, but found: {type(serial)}"
        )
