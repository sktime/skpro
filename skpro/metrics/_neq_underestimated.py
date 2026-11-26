"""Concrete performance metrics for probabilistic supervised regression."""
# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
# adapted from sktime

import numpy as np
import pandas as pd

from skpro.metrics.base import BaseProbaMetric


class Underestimated(BaseProbaMetric):
    r"""Underestimate indicator for quantile predictions.

    Measures whether predictions underestimate the true values, for a sample
    returns the percentage of predictions that underestimate the true values.

    For a quantile prediction :math:`\widehat{y}
    and a ground truth value :math:`y`, the overestimate indicator is defined as
    :math:`L(y, \widehat{y}) := 1` if :math:`\widehat{y} < y`
    and :math:`L(y, \widehat{y}) := 0` otherwise, if ``strict`` is True (default).
    If ``strict`` is False, then
    :math:`L(y, \widehat{y}) := 1` if :math:`\widehat{y} \le y`
    and :math:`L(y, \widehat{y}) := 0` otherwise.

    * ``evaluate`` computes the percentage of underestimated predictions.
    * ``evaluate_by_index`` produces the indicator for each test data point.
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
        specifies whether scores for each quantile should be averaged.

        * If True, metric/loss is averaged over all quantiles present in ``y_pred``.
        * If False, metric/loss is not averaged over quantiles.

    alpha (optional) : float, list of float, or 1D array-like, default=None
        quantiles to evaluate metric at.
        Can be specified if no explicit quantiles are present in the direct use of
        the metric, for instance in benchmarking via ``evaluate``, or tuning
        via ``ForecastingGridSearchCV``.

    strict : bool, optional, default=True
        If True, only counts predictions that strictly underestimate the true values.
        If False, counts predictions that underestimate or equal the true values.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from skpro.metrics import Underestimated
    >>> y_true = pd.Series([3, -0.5, 2, 7, 2])
    >>> y_pred = pd.DataFrame({
    ...     ('Quantiles', 0.05): [1.25, 0, 1, 4, 0.625],
    ...     ('Quantiles', 0.5): [2.5, 0, 2, 8, 1.25],
    ...     ('Quantiles', 0.95): [3.75, 0, 3, 12, 1.875],
    ... })
    >>> pl = Underestimated()
    >>> pl(y_true, y_pred)  # doctest: +SKIP
    """  # noqa: E501

    _tags = {
        "scitype:y_pred": "pred_quantiles",
        "lower_is_better": True,  # neither in general, but for typical usage
    }

    def __init__(
        self,
        multioutput="uniform_average",
        score_average=True,
        alpha=None,
        strict=True,
    ):
        self.score_average = score_average
        self.alpha = alpha
        self._alpha = self._check_alpha(alpha)
        self.metric_args = {"alpha": self._alpha}
        self.strict = strict
        super().__init__(multioutput=multioutput, score_average=score_average)

    def _evaluate_by_index(self, y_true, y_pred, **kwargs):
        """Logic for finding the metric evaluated at each index.

        y_true : pd.Series, pd.DataFrame or np.array of shape (fh,) or \
            (fh, n_outputs) where fh is the forecasting horizon
            Ground truth (correct) target value`s.

        y_pred : pd.Series, pd.DataFrame or np.array of shape (fh,) or  \
            (fh, n_outputs)  where fh is the forecasting horizon
            Forecasted values.

        multioutput : string "uniform_average" or "raw_values"
            Determines how multioutput results will be treated.
        """
        alpha = self._alpha
        y_pred_alphas = self._get_alpha_from(y_pred)
        if alpha is None:
            alphas = y_pred_alphas
        else:
            # if alpha was provided, check whether  they are predicted
            #   if not all alpha are observed, raise a ValueError
            if not np.isin(alpha, y_pred_alphas).all():
                # todo: make error msg more informative
                #   which alphas are missing
                msg = "not all quantile values in alpha are available in y_pred"
                raise ValueError(msg)
            else:
                alphas = alpha

        alphas = self._check_alpha(alphas)

        alpha_preds = y_pred.iloc[
            :, [x in alphas for x in y_pred.columns.get_level_values(1)]
        ]

        alpha_preds_np = alpha_preds.to_numpy()

        y_true_np = np.asarray(y_true)
        if y_true_np.ndim == 1:
            y_true_np = y_true_np.reshape(-1, 1)
        y_true_np = np.repeat(y_true_np, axis=1, repeats=len(alphas))
        diff = y_true_np - alpha_preds_np

        if self.strict:
            sign = (diff > 0).astype(diff.dtype)
        else:
            sign = (diff >= 0).astype(diff.dtype)
        loss = sign

        out_df = pd.DataFrame(loss, columns=alpha_preds.columns)

        return out_df

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Retrieve test parameters."""
        params1 = {}
        params2 = {"alpha": [0.1, 0.5, 0.9]}
        return [params1, params2]
