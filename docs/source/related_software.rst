.. _related_software:

================
Related Software
================

The following is a curated list of software packages related to ``skpro``
in the probabilistic prediction and scientific Python ecosystem.

Probabilistic Prediction & Forecasting
---------------------------------------

`sktime <https://www.sktime.net>`_
    A unified framework for time series machine learning in Python.
    ``skpro`` is maintained by the same community and integrates with ``sktime``
    to enable probabilistic forecasting pipelines: an ``sktime`` probabilistic
    forecaster can be built from an ``skpro`` probabilistic regressor.

`ngboost <https://stanfordmlgroup.github.io/projects/ngboost/>`_
    Natural Gradient Boosting for probabilistic prediction.
    ``skpro`` provides a native interface to ``ngboost`` estimators via the
    ``NGBoostRegressor`` and ``NGBoostSurvival`` classes.

`cyclic-boosting <https://cyclic-boosting.readthedocs.io>`_
    A Python package for probabilistic prediction using cyclic boosting algorithms.
    ``skpro`` provides a native interface to ``cyclic-boosting`` estimators.

Uncertainty Quantification & Conformal Prediction
--------------------------------------------------

`MAPIE <https://mapie.readthedocs.io>`_
    Model Agnostic Prediction Interval Estimator.
    A library for uncertainty quantification via conformal prediction, compatible
    with ``scikit-learn``. ``skpro`` can interface with MAPIE for interval
    and quantile prediction.

Machine Learning Foundations
-----------------------------

`scikit-learn <https://scikit-learn.org>`_
    The standard Python machine learning library.
    ``skpro`` is fully ``scikit-learn``-compatible and ``scikit-base``-compliant,
    extending ``scikit-learn`` regressors with probabilistic prediction capabilities.

Survival & Time-to-Event Analysis
-----------------------------------

`lifelines <https://lifelines.readthedocs.io>`_
    A complete survival analysis library for Python, implementing a wide range
    of parametric and non-parametric survival models.

Probabilistic Programming
--------------------------

`pymc <https://www.pymc.io>`_
    A probabilistic programming library in Python for Bayesian statistical modeling
    and inference using Markov Chain Monte Carlo (MCMC) and variational inference.
