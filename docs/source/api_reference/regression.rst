.. _regression_ref:

Probabilistic supervised regression
===================================

The :mod:`skpro.regression` module contains algorithms and composition tools for probabilistic supervised regression,
i.e., tabular regression estimation with a probabilistic prediction mode.

This learning task is sometimes also known as conditional distribution predictions,
or conditional density estimation, if predictive distributions are continuous.

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

.. currentmodule:: skpro.regression.multiquantile

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    MultipleQuantileRegressor

.. currentmodule:: skpro.regression.enbpi

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    EnbpiRegressor

.. currentmodule:: skpro.regression.mapie

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    MapieRegressor

.. currentmodule:: skpro.regression.ensemble

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    BaggingRegressor
    NGBoostRegressor

.. currentmodule:: skpro.regression.cyclic_boosting

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    CyclicBoosting


Naive regressors and baselines
------------------------------

This section lists simple regressors which can be used as baselines.

.. currentmodule:: skpro.regression.delta

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    DeltaPointRegressor


Linear regression
-----------------

.. currentmodule:: skpro.regression.linear

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    ARDRegression
    BayesianRidge
    GLMRegressor
    PoissonRegressor

Gaussian process and kernel regression
--------------------------------------

.. currentmodule:: skpro.regression.gp

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    GaussianProcess


Adapters to other interfaces
----------------------------

.. currentmodule:: skpro.regression.adapters.sklearn

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    SklearnProbaReg

Reduction to probabilistic classification
----------------------------

.. currentmodule:: skpro.regression.binned._sklearn_bin_regressor

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    HistBinnedProbaRegressor


Base classes
------------

.. currentmodule:: skpro.regression.base

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    BaseProbaRegressor
