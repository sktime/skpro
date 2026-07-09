
.. _distfitter_ref:

Distribution fitters
====================

The :mod:`skpro.distfitter` module contains distribution fitters,
which estimate distribution parameters from data and return
a fitted scalar distribution object.

All distribution fitters in ``skpro`` can be listed using the
``skpro.registry.all_objects`` utility,
using ``object_types="distfitter"``, optionally filtered by tags.
Valid tags can be listed using ``skpro.registry.all_tags``.

Parametric fitters
------------------

.. currentmodule:: skpro.distfitter

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    ExponentialFitter
    LaplaceFitter
    ScipyMLEFitter
    MOMFitter
    NormalFitter
    UniformFitter

Base
----

.. currentmodule:: skpro.distfitter.base

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    BaseDistFitter
