.. _outlier_ref:

Outlier Detection
=================

The :mod:`skpro.outlier` module contains outlier detection algorithms based on probabilistic regressors,
with a PyOD-compatible interface.

These algorithms reduce outlier detection to probabilistic supervised regression,
enabling the use of skpro's probabilistic regressors for anomaly detection tasks.

All outlier detectors in ``skpro`` can be listed using the ``skpro.registry.all_objects`` utility,
using ``object_types="outlier_detector"``, optionally filtered by tags.
Valid tags can be listed using ``skpro.registry.all_tags``.

Outlier Detectors
-----------------

.. currentmodule:: skpro.outlier

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    QuantileOutlierDetector
    DensityOutlierDetector
    LossOutlierDetector

Base Classes
------------

.. currentmodule:: skpro.outlier.base

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    BaseOutlierDetector
