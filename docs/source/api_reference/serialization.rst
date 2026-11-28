.. _serialization_ref:

=========================
Serialization File Format
=========================

This page specifies how ``skpro`` objects are persisted by
:meth:`skpro.base.BaseObject.save` and restored by :func:`skpro.base.load`
(or the classmethods ``load_from_path`` / ``load_from_serial``).

The format is the serialization-node format specified in
`STEP 27 <https://github.com/sktime/enhancement-proposals/pull/52>`_,
"Recursive composite and native serialization". It is shared with ``sktime``
and other ``skbase``-based packages, so a composite mixing packages can be
saved and loaded without either side needing package-specific handling.

.. warning::

    Loading is pickle-based, and therefore executes code contained in the
    archive. Only load archives from sources you trust. Cross-version
    compatibility is not guaranteed.


The serialization node
======================

Every ``BaseObject`` is serialized as a self-contained **serialization node**:

.. code-block:: text

    node/
    ├── _metadata
    ├── _obj
    ├── _artifacts/        # optional
    │   ├── index.json
    │   └── ...
    └── _components/       # optional
        ├── index.json
        └── ...

The archive root is the root object's own node; there is no wrapper directory
around it, and there is no global manifest anywhere in the archive. A node's
``_metadata`` and its two index files describe that node completely, so any
subtree can be understood by looking only inside its own directory.

``_artifacts`` and ``_components`` are orthogonal, and both are optional:

* a plain leaf object has only ``_metadata`` and ``_obj``
* a leaf holding a deep-learning model adds ``_artifacts``
* a composite adds ``_components``
* a composite that also holds native state has both

Because ``_components`` entries are themselves nodes, a child may repeat any of
these possibilities recursively.


State classification
--------------------

When a node is written, the object's attributes are classified in this order:

1. attributes tagged ``serialization:skip`` are dropped entirely
2. attributes tagged ``serialization:native_artifacts`` are written to
   ``_artifacts``
3. remaining ``BaseObject`` children become component references in
   ``_components``
4. everything left goes into ``_obj``

An attribute may not carry both tags; doing so raises a ``ValueError``. Native
artifact selection applies to the whole attribute that was selected — the
serializer does not descend into it looking for further components.

Saving never mutates the object being saved. Attributes removed for
classification are restored even if saving fails part way through.


``_metadata``
-------------

``_metadata`` identifies the loader class. It holds a mapping:

.. code-block:: python

    {
        "format_version": 2,
        "class": type(obj),
        "serialization_format": "pickle",
    }

The actual class object is stored rather than a qualified name string, because
that is what preserves ``cloudpickle`` support for classes that are not
otherwise importable. ``_metadata`` is always readable with plain ``pickle``,
so a reader can determine a node's format before it knows which serializer
wrote it.

A child always has authority over its own ``_metadata``; a parent never
duplicates or overrides it.


``_obj``
--------

``_obj`` holds whatever pickle-compatible state is left after skip, native
artifact, and component extraction.

Child components are referenced by **persistent IDs** of the form
``("sktime-component", "component-0000")``, rather than by attribute paths.
Using opaque IDs means children can live inside arbitrary container structures
— lists, tuples, dicts, or nested combinations — with no path-parsing language
to maintain, and it lets a child referenced twice inside one node resolve to a
single shared component.

Third-party, non-estimator objects — including plain ``scikit-learn`` objects —
stay inside the parent's ``_obj`` unless explicitly marked as native artifacts.
If such an object cannot be pickled and has no native backend, saving raises.


``_artifacts``
--------------

``_artifacts`` holds attributes selected for framework-native serialization:

.. code-block:: text

    _artifacts/
    ├── index.json
    ├── model_/
    │   ├── config.json
    │   └── model.safetensors
    └── network_/
        └── state_dict.pt

``index.json`` maps each attribute name to its backend, class, and path:

.. code-block:: json

    {
      "model_": {
        "backend": "pretrained",
        "class": "transformers.models.bert.modeling_bert.BertModel",
        "path": "model_"
      },
      "network_": {
        "backend": "torch_state_dict",
        "class": "package.networks.Network",
        "path": "network_"
      }
    }

The supported backends are:

.. list-table::
    :header-rows: 1

    * - Backend
      - Save form
      - Load form
    * - ``pretrained``
      - ``save_pretrained(path)``
      - ``class.from_pretrained(path, **kwargs)``
    * - ``keras``
      - ``model.keras``
      - ``keras.models.load_model``
    * - ``lightning_checkpoint``
      - ``model.ckpt``
      - ``class.load_from_checkpoint``
    * - ``torch_state_dict``
      - CPU ``state_dict.pt``
      - estimator builds the module, then loads the state dict

If an artifact attribute is ``None`` or missing, nothing is written for it, and
if ``_artifacts`` ends up empty it is omitted from the archive entirely.

Native backends always use their own framework formats, regardless of the
``serialization_format`` setting.


``_components``
---------------

``_components`` holds a node's immediate children:

.. code-block:: text

    _components/
    ├── index.json
    ├── component-0000/
    │   ├── _metadata
    │   ├── _obj
    │   └── _artifacts/...
    └── component-0001/
        ├── _metadata
        ├── _obj
        └── _components/...

``index.json`` maps opaque local IDs to their directories:

.. code-block:: json

    {
      "component-0000": {"path": "component-0000"},
      "component-0001": {"path": "component-0001"}
    }

The index carries no class information — that lives in each child's own
``_metadata`` — and no load order. Because loading is recursive, and a child is
always fully loaded before it is handed back to the parent's unpickler, the
order falls out of the recursion itself.

Ownership cycles and cross-branch aliases are not supported in this version of
the format. Both raise a clear error rather than looping forever or producing
an ambiguous graph. A self-reference is not a cycle; it is handled by the
pickle memo and round-trips normally.


Complete example
----------------

A composite with a pretrained model at the root and a Keras model inside a
child:

.. code-block:: text

    composite.zip
    ├── _metadata
    ├── _obj
    ├── _artifacts/
    │   ├── index.json
    │   └── model_/
    │       ├── config.json
    │       └── model.safetensors
    └── _components/
        ├── index.json
        └── component-0000/
            ├── _metadata
            ├── _obj
            └── _artifacts/
                ├── index.json
                └── network_/
                    └── model.keras


In-memory format
================

When ``obj.save()`` is called with no path, the return value is a tuple:

.. code-block:: text

    (cls, serialized_bytes)

where ``cls`` is ``type(obj)``. If the node has neither artifacts nor
components, ``serialized_bytes`` is a plain pickle stream — the lightweight
path. Otherwise it is an in-memory zip archive with exactly the same internal
structure as an on-disk archive. Callers may treat the payload as opaque
either way.

Restore with ``load(serial)`` or ``cls.load_from_serial(serialized_bytes)``.


Serialization formats
=====================

``serialization_format`` selects how ``_metadata`` and ``_obj`` are written at
each node. The available options are ``"pickle"`` (default) and
``"cloudpickle"``; ``cloudpickle`` is a soft dependency.

The setting is independent of recursion and native serialization. Component
references and native backends never inspect it, and ``_artifacts`` is
unaffected, since native backends always use their framework formats. Children
inherit the parent's format, and each node records its own format in its
``_metadata``.


Backward compatibility
======================

The older archive format is a strict subset of this one:

.. code-block:: text

    legacy-or-minimal-node/
    ├── _metadata
    └── _obj

Readers accept all of the following:

* legacy ``_metadata`` holding a bare pickled class
* legacy in-memory ``(class, pickle_bytes)`` tuples
* minimal nodes with neither ``_artifacts`` nor ``_components``
* nodes carrying native artifacts
* recursive nodes with components

A node declaring a ``format_version`` newer than the running ``skpro`` supports
is rejected up front with a clear error, rather than failing part way through
an unpickle.


Developer interface
===================

Two tags drive attribute classification:

.. code-block:: python

    _tags = {
        "serialization:native_artifacts": ("model_",),
        "serialization:skip": ("trainer_",),
    }

Both may be set dynamically, for instance to skip a cache only when it is
reconstructable. Components need no tagging at all: any ``BaseObject`` child
remaining after tag-based extraction is picked up automatically.

A small number of hooks cover native formats that cannot reconstruct
themselves:

* ``_create_torch_artifact(name)`` — build a module before a state dict is
  loaded into it
* ``_get_native_artifact_load_kwargs(name)`` — supply arguments for
  ``from_pretrained``
* ``get_custom_objects()`` — supply custom Keras objects

New backends belong at this layer, not inside the generic traversal logic.


Usage
=====

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

Objects may opt out of serialization testing via the tag
``capability:serializable=False``.
