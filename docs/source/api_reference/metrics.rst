
.. _metrics_ref:

Performance metrics
===================

The :mod:`skpro.metrics` module contains metrics for evaluating
probabilistic predictions, including survival and time-to-event prediction.s.

All metrics in ``skpro`` can be listed using the ``skpro.registry.all_objects`` utility,
using ``object_types="metric"``, optionally filtered by tags.
Valid tags can be listed using ``sktime.registry.all_tags``.


Quantile and interval prediction metrics
----------------------------------------

.. currentmodule:: skpro.metrics

.. autosummary::
    :toctree: auto_generated/
    :template: class_with_call.rst

    PinballLoss
    EmpiricalCoverage
    ConstraintViolation

Distribution prediction metrics
-------------------------------

Distribution predictions are also known as conditional distribution predictions.
(or conditional density predictions, if continuous).

.. currentmodule:: skpro.metrics

.. autosummary::
    :toctree: auto_generated/
    :template: class_with_call.rst

    CRPS
    LogLoss
    SquaredDistrLoss
    LinearizedLogLoss
    SquaredDistrLoss
