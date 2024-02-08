
.. _metrics_ref:

Performance metrics
===================

The :mod:`skpro.metrics` module contains metrics for evaluating
probabilistic predictions, including survival and time-to-event predictions.

All metrics in ``skpro`` can be listed using the ``skpro.registry.all_objects`` utility,
using ``object_types="metric"``, optionally filtered by tags.
Valid tags can be listed using ``sktime.registry.all_tags``.

Survival/time-to-event specific metrics in ``skpro`` can be listed
by filtering by ``capability:survival`` being ``True``.

All probabilistic metrics can be used for survival
prediction, by default they will ignore the censoring information.
Note: this is different from subsetting to uncensored observations.


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

Survival prediction metrics
---------------------------

Survival or time-to-event predictions are a variant of distribution predictions,
where the ground truth may be censored.
These metrics take the censoring information into account.

.. currentmodule:: skpro.metrics.survival

.. autosummary::
    :toctree: auto_generated/
    :template: class_with_call.rst

    ConcordanceHarrell
    SPLL
