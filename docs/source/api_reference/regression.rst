.. _regression_ref:

Time series regression
======================

The :mod:`skpro.regression` module contains algorithms and composition tools for probabilistic supervised regression,
i.e., tabular regression estimator with a probabilistic prediction mode.

All regressors in ``skpro`` can be listed using the ``skpro.registry.all_objects`` utility,
using ``object_types="regressor_proba"``, optionally filtered by tags.
Valid tags can be listed using ``skpro.registry.all_tags``.

Composition
-----------

.. currentmodule:: skpro.regression.compose

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    Pipeline

Model selection and tuning
--------------------------

.. currentmodule:: skpro.model_selection

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    GridSearchCV
    RandomizedSearchCV

.. currentmodule:: skpro.benchmarking.evaluate

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    evaluate

Reduction - adding ``predict_proba``
------------------------------------

This section lists reduction algorithms that
take one or multiple ``sklearn`` estimators and adda probabilistic prediction mode.

.. currentmodule:: skpro.regression.bootstrap

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    BootstrapRegressor

.. currentmodule:: skpro.regression.residual

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    ResidualDouble

Base
----

.. currentmodule:: skpro.regression.base

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    BaseProbaRegressor

