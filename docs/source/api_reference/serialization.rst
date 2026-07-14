.. _serialization_ref:

=========================
Serialization File Format
=========================

This page specifies how ``skpro`` objects are persisted by
:meth:`skpro.base.BaseObject.save` and restored by
:func:`skpro.base.load` (or the classmethods
``load_from_path`` / ``load_from_serial``).

There are two persistence modes:

* **In-memory** (``path=None``): a Python tuple suitable for caching or
  transfer within a process.
* **On disk** (``path`` is a file location): a ``.zip`` archive.

Disk persistence uses the recursive **zip format v2** described below.
A simpler **v1 flat** zip layout is still accepted on load for backward
compatibility.


In-memory format
----------------

When ``obj.save()`` (or ``obj.save(path=None)``) is called, the return
value is a tuple:

.. code-block:: text

    (cls, serialized_bytes, serialization_format)

where:

* ``cls`` is ``type(obj)``
* ``serialized_bytes`` is the full object blob encoded with
  ``pickle`` or ``joblib`` (see ``serialization_format``)
* ``serialization_format`` is ``"pickle"`` or ``"joblib"``

Restore with ``load(serial)`` or ``cls.load_from_serial(...)``.


Zip format v2 (recursive / manifest)
------------------------------------

When ``obj.save(path)`` is called with a file location, ``skpro`` writes
a zip archive (``.zip`` is appended if missing).

The archive stores the object **tree** recursively:

* each ``skpro`` / ``skbase`` ``BaseObject`` sub-component is written to
  its own subfolder
* a root ``manifest.json`` is the blueprint: every component, its class,
  its path inside the zip, and the **topological load order**
  (leaves first, root last)
* on load, components are read in that order and reassembled

Recursion and stop conditions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``save`` walks the composite tree. Recursion stops when:

1. A component is an ``skpro``/``skbase`` object with **no** nested
   ``BaseObject`` children (leaf) — saved as a structured entry in its
   own subfolder (``serialized_as: "structured"``).
2. A component is **not** a ``BaseObject`` (e.g. an ``sklearn`` estimator
   or other third-party object) — saved as a single pickle/joblib blob
   and marked ``serialized_as: "pickle_fallback"`` in the manifest.
3. If an object is not picklable / joblib-serializable — an error is
   raised (there is no further fallback).

Archive layout
~~~~~~~~~~~~~~

Top-level entries:

.. code-block:: text

    model.zip
    ├── manifest.json      # blueprint (required for v2)
    ├── _format            # "pickle" or "joblib"
    ├── _version           # manifest version integer (currently 2)
    ├── root/
    │   ├── _metadata      # pickled class of the root object
    │   └── _obj           # serialized root state (or full leaf blob)
    └── components/
        ├── <component_id>/
        │   ├── _metadata
        │   └── _obj
        └── ...

Example for a fitted ``VotingProbaRegressor`` with two residual
sub-estimators (names are illustrative):

.. code-block:: text

    voter_model.zip
    ├── manifest.json
    ├── _format
    ├── _version
    ├── root/
    │   ├── _metadata
    │   └── _obj
    └── components/
        ├── estimators__0__1/     # first estimator (e.g. unfitted clone)
        ├── estimators__1__1/     # second estimator
        ├── estimators___0__1/    # first fitted estimator attribute
        └── estimators___1__1/    # second fitted estimator attribute

Each component folder contains:

* ``_metadata`` — ``pickle`` of the component's class (``type(obj)``)
* ``_obj`` — serialized payload using the chosen backend
  (``pickle`` or ``joblib``):

  * for **leaf** nodes: the full object
  * for **composite** nodes: a copy of ``__dict__`` in which nested
    ``BaseObject`` references are replaced by placeholders that point
    at child component IDs (so the tree can be rebuilt on load)

``manifest.json``
~~~~~~~~~~~~~~~~~

Schema (conceptual):

.. code-block:: json

    {
      "version": 2,
      "serialization_format": "pickle",
      "root_class": "skpro.regression.ensemble._voting.VotingProbaRegressor",
      "is_fitted": true,
      "components": {
        "root": {
          "class": "skpro.regression.ensemble._voting.VotingProbaRegressor",
          "path": "root/",
          "is_leaf": false,
          "serialized_as": "structured",
          "children": ["estimators__0__1", "estimators__1__1"],
          "parent": null,
          "attr_ref": ""
        },
        "estimators__0__1": {
          "class": "skpro.regression.residual.ResidualDouble",
          "path": "components/estimators__0__1/",
          "is_leaf": true,
          "serialized_as": "structured",
          "children": [],
          "parent": "root",
          "attr_ref": "estimators_[0][1]"
        }
      },
      "load_order": ["estimators__0__1", "estimators__1__1", "root"]
    }

Field meanings:

+-----------------------+--------------------------------------------------+
| Field                 | Meaning                                          |
+=======================+==================================================+
| ``version``           | Manifest / format version (``2`` for this        |
|                       | recursive layout).                               |
+-----------------------+--------------------------------------------------+
| ``serialization_format`` | Backend for ``_obj`` blobs: ``"pickle"`` or  |
|                       | ``"joblib"``.                                    |
+-----------------------+--------------------------------------------------+
| ``root_class``        | Fully qualified class name of the saved root.    |
+-----------------------+--------------------------------------------------+
| ``is_fitted``         | Whether the root reported ``is_fitted`` at save  |
|                       | time.                                            |
+-----------------------+--------------------------------------------------+
| ``components``        | Map from component ID to metadata (path, class,  |
|                       | parent/children, leaf flag, serialization mode). |
+-----------------------+--------------------------------------------------+
| ``load_order``        | Component IDs in topological order: leaves       |
|                       | first, ``"root"`` last.                          |
+-----------------------+--------------------------------------------------+

Per-component fields:

+------------------+-------------------------------------------------------+
| Field            | Meaning                                               |
+==================+=======================================================+
| ``class``        | Fully qualified class name.                           |
+------------------+-------------------------------------------------------+
| ``path``         | Prefix inside the zip for ``_metadata`` / ``_obj``.   |
+------------------+-------------------------------------------------------+
| ``is_leaf``      | ``true`` if no nested ``BaseObject`` children.        |
+------------------+-------------------------------------------------------+
| ``serialized_as``| ``"structured"`` for ``BaseObject`` nodes, or         |
|                  | ``"pickle_fallback"`` for non-``BaseObject`` blobs.   |
+------------------+-------------------------------------------------------+
| ``children``     | List of child component IDs.                          |
+------------------+-------------------------------------------------------+
| ``parent``       | Parent component ID, or ``null`` for root.            |
+------------------+-------------------------------------------------------+
| ``attr_ref``     | Human-readable location in the parent                 |
|                  | (e.g. ``estimators_[0][1]``).                         |
+------------------+-------------------------------------------------------+

Component IDs are derived from attribute names and list/tuple indices
(joined with ``__``), e.g. ``estimators__0__1``. Nested composites
prefix the parent ID.

Reconstruction on load
~~~~~~~~~~~~~~~~~~~~~~

Components are loaded in topological order (leaves first, root last).

* **Leaf nodes** are deserialized directly from their ``_obj`` blob
  using the chosen backend (``pickle`` or ``joblib``).

* **Composite nodes** are reconstructed without calling ``__init__``:

  1. The ``_obj`` blob is deserialized to recover the saved
     ``__dict__`` (with placeholder references in place of children).
  2. Placeholders are replaced with the already-loaded child objects.
  3. The class is recovered from ``_metadata``.
  4. A bare instance is created via ``cls.__new__(cls)``.
  5. If the class defines ``__setstate__``, it is called with the
     restored state dict; otherwise ``obj.__dict__`` is updated
     directly.

This mirrors Python's ``pickle`` protocol: ``pickle`` itself uses
``__new__`` + ``__setstate__`` (falling back to ``__dict__`` assignment
when no ``__setstate__`` is defined), so both the flat and recursive
paths reconstruct objects in the same way. Classes that need
post-deserialization setup (e.g. reopening file handles or rebuilding
caches) can define ``__setstate__`` and both paths will honour it.


Zip format v1 (flat, load-only)
-------------------------------

Older flat archives (no ``manifest.json``) remain loadable. They contain:

.. code-block:: text

    model.zip
    ├── _metadata   # pickled class of the object
    ├── _obj        # full object blob (pickle or joblib)
    └── _format     # optional: "pickle" or "joblib" (default pickle)

``load`` detects v1 vs v2 by presence of ``manifest.json``.


Usage
-----

.. code-block:: python

    from skpro.base import load
    from skpro.regression.residual import ResidualDouble

    est = ResidualDouble.create_test_instance()
    # ... fit est ...

    # on disk
    est.save("my_model")          # writes my_model.zip
    est2 = load("my_model.zip")

    # in memory
    serial = est.save()
    est3 = load(serial)

Objects may opt out of serialization via the tag
``capability:serializable=False``.
