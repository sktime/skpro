"""Concrete performance metrics for probabilistic supervised regression."""
# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
# adapted from sktime

import numpy as np
import pandas as pd

from skpro.metrics.base import BaseProbaMetric


class ConstraintViolation(BaseProbaMetric):
    r"""Average absolute constraint violations for interval predictions.

    Applies to interval predictions.

    Should be used together with ``EmpiricalCoverage`` if reported.

    Up to a constant, ``PinballLoss`` is a weighted sum of ``ConstraintViolation`` and
    ``EmpiricalCoverage``.

    For an interval prediction :math:`I = [a, b]` and a ground truth value :math:`y`,
    the constraint violation loss is defined as

    .. math::

        L(y, I) :=
        \begin{cases}
        a - y, & \text{if } y < a \\
        y - b, & \text{if } y > b \\
        0, & \text{otherwise}
        \end{cases}

    * ``evaluate`` computes the average test sample loss.
    * ``evaluate_by_index`` produces the loss sample by test data point.
    * ``multivariate`` controls averaging over variables.
    * ``score_average`` controls averaging over quantiles/intervals.

    Parameters
    ----------
    multioutput : {'raw_values', 'uniform_average'} or array-like of shape \
            (n_outputs,), default='uniform_average'
        Defines whether and how to aggregate metric for across variables.

        * If 'uniform_average' (default), errors are mean-averaged across variables.
        * If array-like, errors are weighted averaged across variables,
          values as weights.
        * If 'raw_values', does not average errors across variables,
          columns are retained.

    score_average : bool, optional, default = True
        specifies whether scores for each coverage value should be averaged.

        * If True, metric/loss is averaged over all coverages present in ``y_pred``.
        * If False, metric/loss is not averaged over coverages.

    coverage (optional) : float, list of float, or 1D array-like, default=None
        nominal coverage to evaluate metric at.
        Can be specified if no explicit coverages are present in the direct use of
        the metric, for instance in benchmarking via ``evaluate``, or tuning
        via ``ForecastingGridSearchCV``.
    """

    _tags = {
        "scitype:y_pred": "pred_interval",
        "lower_is_better": True,
    }

    def __init__(
        self, multioutput="uniform_average", score_average=True, coverage=None
    ):
        self.score_average = score_average
        self.multioutput = multioutput
        self.coverage = coverage
        self._coverage = self._check_coverage(coverage)
        self.metric_args = {"coverage": self._coverage}
        super().__init__(score_average=score_average, multioutput=multioutput)

    def _evaluate_by_index(self, y_true, y_pred, **kwargs):
        """Logic for finding the metric evaluated at each index.

        y_true : pd.Series, pd.DataFrame or np.array of shape (fh,) or \
            (fh, n_outputs) where fh is the forecasting horizon
            Ground truth (correct) target values.

        y_pred : pd.Series, pd.DataFrame or np.array of shape (fh,) or  \
            (fh, n_outputs)  where fh is the forecasting horizon
            Forecasted values.

        multioutput : string "uniform_average" or "raw_values" determines how \
            multioutput results will be treated.
        """
        lower = y_pred.iloc[:, y_pred.columns.get_level_values(2) == "lower"].to_numpy()
        upper = y_pred.iloc[:, y_pred.columns.get_level_values(2) == "upper"].to_numpy()

        if not isinstance(y_true, np.ndarray):
            y_true_np = y_true.to_numpy()
        else:
            y_true_np = y_true

        if y_true_np.ndim == 1:
            y_true_np = y_true_np.reshape(-1, 1)

        scores = np.unique(np.round(y_pred.columns.get_level_values(1), 7))
        no_scores = len(scores)
        vars = np.unique(y_pred.columns.get_level_values(0))

        y_true_np = np.tile(y_true_np, no_scores)

        int_distance = ((y_true_np < lower).astype(int) * abs(lower - y_true_np)) + (
            (y_true_np > upper).astype(int) * abs(y_true_np - upper)
        )

        out_df = pd.DataFrame(
            int_distance, columns=pd.MultiIndex.from_product([vars, scores])
        )

        return out_df

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Retrieve test parameters."""
        params1 = {}
        params2 = {"coverage": 0.5}
        return [params1, params2]
