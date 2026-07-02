
.. _distfitters_ref:

Distribution fitters
====================

The :mod:`skpro.distfitters` module contains
distribution fitters which combine a ``pandas.DataFrame``-like API
with a ``scikit-base`` compatible object interface.

All distribution fitters in ``skpro`` can be listed using the ``skpro.registry.all_objects`` utility,
using ``object_types="distfitter"``, optionally filtered by tags.
Valid tags can be listed using ``skpro.registry.all_tags``.

Distribution fitters
--------------------

.. currentmodule:: skpro.distfitter

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    MOMFitter
    NormalFitter

Base
----

.. currentmodule:: skpro.distfitter.base

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    BaseDistFitter
