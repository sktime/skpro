
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


Parametric fitters - many distributions
---------------------------------------

These distribution fitters allow the user to select an arbitrary ``skpro`` distribution,
or a distribution from a longer list of distributions, to fit parametrically.

.. currentmodule:: skpro.distfitter

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    ScipyMLEFitter
    MOMFitter


Parametric fitters - single distributions
-----------------------------------------

These distribution fitters fit a single type of distribution, e.g., a normal
or exponential distribution, to the data.

.. currentmodule:: skpro.distfitter

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    ExponentialFitter
    LaplaceFitter
    NormalFitter
    UniformFitter


Base
----

.. currentmodule:: skpro.distfitter.base

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    BaseDistFitter
