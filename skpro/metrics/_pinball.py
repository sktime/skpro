"""Concrete performance metrics for probabilistic supervised regression."""
# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
# adapted from sktime

import numpy as np
import pandas as pd

from skpro.metrics.base import BaseProbaMetric


class PinballLoss(BaseProbaMetric):
    r"""Pinball loss aka quantile loss for quantile/interval predictions.

    Can be used for both quantile and interval predictions.

    For a quantile prediction :math:`\widehat{y}
    at quantile point :math:`\alpha`,
    and a ground truth value :math:`y`, the pinball loss is defined as
    :math:`L_\alpha(y, \widehat{y}) := (y - \widehat{y}) \cdot (\alpha - H(y - \widehat{y}))`,
    where :math:`H` is the Heaviside step function defined as
    :math:`H(x) = 1` if :math:`x \ge 0` and :math:`H(x) = 0` otherwise.

    For a symmetric prediction interval :math:`I = [\widehat{y}_{\alpha}, \widehat{y}_{1 - \alpha}]`,
    the pinball loss is defined as
    :math:`L_\alpha(y, I) := L_\alpha(y, \widehat{y}_{\alpha}) + L_{1 - \alpha}(y, \widehat{y}_{1 - \alpha})`,
    or, in terms of coverage :math:`c = 1 - 2\alpha`, as
    :math:`L_c(y, I) := L_{1/2 - c/2}(y, a) + L_{1/2 + c/2}(y, b)`,
    if we write :math:`I = [a, b]`.

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
        specifies whether scores for each quantile should be averaged.

        * If True, metric/loss is averaged over all quantiles present in ``y_pred``.
        * If False, metric/loss is not averaged over quantiles.

    alpha (optional) : float, list of float, or 1D array-like, default=None
        quantiles to evaluate metric at.
        Can be specified if no explicit quantiles are present in the direct use of
        the metric, for instance in benchmarking via ``evaluate``, or tuning
        via ``ForecastingGridSearchCV``.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from skpro.metrics import PinballLoss
    >>> y_true = pd.Series([3, -0.5, 2, 7, 2])
    >>> y_pred = pd.DataFrame({
    ...     ('Quantiles', 0.05): [1.25, 0, 1, 4, 0.625],
    ...     ('Quantiles', 0.5): [2.5, 0, 2, 8, 1.25],
    ...     ('Quantiles', 0.95): [3.75, 0, 3, 12, 1.875],
    ... })
    >>> pl = PinballLoss()
    >>> pl(y_true, y_pred)  # doctest: +SKIP
    np.float64(0.1791666666666667)
    >>> pl = PinballLoss(score_average=False)
    >>> pl(y_true, y_pred).to_numpy()
    array([0.16625, 0.275  , 0.09625])
    >>> y_true = pd.DataFrame({
    ...     "Quantiles1": [3, -0.5, 2, 7, 2],
    ...     "Quantiles2": [4, 0.5, 3, 8, 3],
    ... })
    >>> y_pred = pd.DataFrame({
    ...     ('Quantiles1', 0.05): [1.5, -1, 1, 4, 0.65],
    ...     ('Quantiles1', 0.5): [2.5, 0, 2, 8, 1.25],
    ...     ('Quantiles1', 0.95): [3.5, 4, 3, 12, 1.85],
    ...     ('Quantiles2', 0.05): [2.5, 0, 2, 8, 1.25],
    ...     ('Quantiles2', 0.5): [5.0, 1, 4, 16, 2.5],
    ...     ('Quantiles2', 0.95): [7.5, 2, 6, 24, 3.75],
    ... })
    >>> pl = PinballLoss(multioutput='raw_values')
    >>> pl(y_true, y_pred).to_numpy()
    array([0.16233333, 0.465     ])
    >>> pl = PinballLoss(multioutput=np.array([0.3, 0.7]))
    >>> pl(y_true, y_pred)  # doctest: +SKIP
    np.float64(0.3742000000000001)
    """  # noqa: E501

    _tags = {
        "scitype:y_pred": "pred_quantiles",
        "lower_is_better": True,
    }

    def __init__(self, multioutput="uniform_average", score_average=True, alpha=None):
        self.score_average = score_average
        self.alpha = alpha
        self._alpha = self._check_alpha(alpha)
        self.metric_args = {"alpha": self._alpha}
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
        alpha_mat = np.repeat(
            (alpha_preds.columns.get_level_values(1).to_numpy().reshape(1, -1)),
            repeats=y_true.shape[0],
            axis=0,
        )

        y_true_np = np.asarray(y_true)
        if y_true_np.ndim == 1:
            y_true_np = y_true_np.reshape(-1, 1)
        y_true_np = np.repeat(y_true_np, axis=1, repeats=len(alphas))
        diff = y_true_np - alpha_preds_np
        sign = (diff >= 0).astype(diff.dtype)
        loss = alpha_mat * sign * diff - (1 - alpha_mat) * (1 - sign) * diff

        out_df = pd.DataFrame(loss, columns=alpha_preds.columns)

        return out_df

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Retrieve test parameters."""
        params1 = {}
        params2 = {"alpha": [0.1, 0.5, 0.9]}
        return [params1, params2]
