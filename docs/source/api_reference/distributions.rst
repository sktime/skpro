
.. _distributions_ref:

Probability distributions
=========================

The :mod:`sktime.distributions` module contains
probability distributions which combine a ``pandas.DataFrame``-like API
with a ``scikit-base`` compatible object interface.

All distributions in ``skpro`` can be listed using the ``skpro.registry.all_objects`` utility,
using ``object_types="distribution"``, optionally filtered by tags.
Valid tags can be listed using ``sktime.registry.all_tags``.

Base
----

.. currentmodule:: skpro.distributions.base

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    BaseDistribution

Parametric distributions
------------------------

Continuous support
~~~~~~~~~~~~~~~~~~

.. currentmodule:: skpro.distributions

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    Beta
    ChiSquared
    Exponential
    Fisk
    Laplace
    Logistic
    Normal
    TDistribution
    Weibull


Integer support
~~~~~~~~~~~~~~~

.. currentmodule:: skpro.distributions

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    Poisson


Non-parametric and empirical distributions
------------------------------------------

.. currentmodule:: skpro.distributions

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    Delta
    Empirical
    QPD_Empirical
    QPD_Johnson
    QPD_U
    QPD_S
    QPD_B

Composite distributions
-----------------------

.. currentmodule:: skpro.distributions

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    IID
    Mixture
