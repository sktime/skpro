.. _regression_ref:

Time-to-event prediction and survival prediction
================================================

The :mod:`skpro.survival` module contains algorithms and composition tools for
time-to-event prediction or survival prediction,
i.e., tabular regression estimation with a probabilistic prediction mode,
and optional right censoring.

All survival predictors in ``skpro`` can be listed using the ``skpro.registry.all_objects`` utility,
using ``object_types="regressor_proba"``,
filtering by ``capability:survival`` being ``True``,
optionally filtered by further tags.
Valid tags can be listed using ``skpro.registry.all_tags``.

Additionally, all probabilistic regressors can be used for survival
prediction, by default they will ignore the censoring information.
Note: this is different from subsetting to uncensored observations.

Composition
-----------

The regression pipeline class ``Pipeline`` can be used
to pipeline time-to-event prediction estimators.

.. currentmodule:: skpro.regression.compose

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    Pipeline

Reduction to plain probabilistic regression
-------------------------------------------

The below estimators can be used to reduce a survival predictor
to a plain probabilistic regressor, i.e., in the ``skpro.regression`` module.

These add the capability to take censoring into account.

.. currentmodule:: skpro.survival.compose

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    FitUncensored
    ConditionUncensored

Reduction - adding ``predict_proba``
------------------------------------

Simple strategies to use sklearn regressors for survival prediction
are obtained from using any of the wrappers in ``skpro.regression``,
then applying reduction to tabular supervised probabilistic regression (above).

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

Proportional hazards models
---------------------------

.. currentmodule:: skpro.survival.coxph

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    CoxPH
    CoxPHSkSurv
    CoxNet

Tree models
-----------

.. currentmodule:: skpro.survival.tree

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    SurvivalTree

Base
----

Survival predictors inherit from the same base class as
tabular probabilistic regressors.

.. currentmodule:: skpro.regression.base

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    BaseProbaRegressor
