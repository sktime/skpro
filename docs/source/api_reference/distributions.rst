
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

Continuous support - full reals
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: skpro.distributions

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    Laplace
    Logistic
    Normal
    SkewNormal
    TDistribution
    TruncatedNormal


Continuous support - non-negative reals
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: skpro.distributions

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    Alpha
    Beta
    ChiSquared
    Exponential
    Erlang
    Fisk
    Gamma
    HalfCauchy
    HalfLogistic
    HalfNormal
    InverseGamma
    LogLaplace
    Pareto
    Weibull


Integer support
~~~~~~~~~~~~~~~

.. currentmodule:: skpro.distributions

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    Binomial
    Poisson

Non-parametric and empirical distributions
------------------------------------------

.. currentmodule:: skpro.distributions

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    Delta
    Empirical
    Histogram
    QPD_Empirical
    QPD_Johnson
    QPD_U
    QPD_S
    QPD_B


Composite distributions
-----------------------

Parametric families
~~~~~~~~~~~~~~~~~~~

.. currentmodule:: skpro.distributions

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    MeanScale

Mixture composition
~~~~~~~~~~~~~~~~~~~

.. currentmodule:: skpro.distributions

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    Mixture

Sampling and multivariate composition
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: skpro.distributions

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    IID
