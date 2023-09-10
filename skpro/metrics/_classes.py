"""Concrete performance metrics for probabilistic supervised regression."""
# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
# adapted from sktime

import numpy as np
import pandas as pd

from skpro.metrics.base import BaseDistrMetric, BaseProbaMetric


class PinballLoss(BaseProbaMetric):
    """Evaluate the pinball loss at all quantiles given in data.

    Parameters
    ----------
    multioutput : string "uniform_average" or "raw_values" determines how\
        multioutput results will be treated.

    score_average : bool, optional, default = True
        specifies whether scores for each quantile should be averaged.

    alpha (optional) : float, list or np.ndarray, specifies what quantiles to \
        evaluate metric at.
    """

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
        """Evaluate the desired metric on given inputs.

        y_true : pd.Series, pd.DataFrame, 1D np.array, or 2D np.ndarray
            Ground truth (correct) target values.

        y_pred : pd.Series, pd.DataFrame, 1D np.array, or 2D np.ndarray
            Predicted values.

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

        y_true_np = np.repeat(y_true, axis=1, repeats=len(alphas))
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


class EmpiricalCoverage(BaseProbaMetric):
    """Evaluate the pinball loss at all quantiles given in data.

    Parameters
    ----------
    multioutput : string "uniform_average" or "raw_values" determines how\
        multioutput results will be treated.

    score_average : bool, optional, default = True
        specifies whether scores for each quantile should be averaged.
    """

    _tags = {
        "scitype:y_pred": "pred_interval",
        "lower_is_better": False,
    }

    def __init__(self, multioutput="uniform_average", score_average=True):
        self.score_average = score_average
        self.multioutput = multioutput
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
            y_true_np = y_true.reshape(-1, 1)

        scores = np.unique(np.round(y_pred.columns.get_level_values(1), 7))
        no_scores = len(scores)
        vars = np.unique(y_pred.columns.get_level_values(0))

        y_true_np = np.tile(y_true_np, no_scores)

        truth_array = (y_true_np > lower).astype(int) * (y_true_np < upper).astype(int)

        out_df = pd.DataFrame(
            truth_array, columns=pd.MultiIndex.from_product([vars, scores])
        )

        return out_df

    @classmethod
    def get_test_params(self):
        """Retrieve test parameters."""
        params1 = {}
        return [params1]


class ConstraintViolation(BaseProbaMetric):
    """Evaluate the pinball loss at all quantiles given in data.

    Parameters
    ----------
    multioutput : string "uniform_average" or "raw_values" determines how\
        multioutput results will be treated.

    score_average : bool, optional, default = True
        specifies whether scores for each quantile should be averaged.
    """

    _tags = {
        "scitype:y_pred": "pred_interval",
        "lower_is_better": True,
    }

    def __init__(self, multioutput="uniform_average", score_average=True):
        self.score_average = score_average
        self.multioutput = multioutput
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
            y_true_np = y_true.reshape(-1, 1)

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
    def get_test_params(self):
        """Retrieve test parameters."""
        params1 = {}
        return [params1]


class LogLoss(BaseDistrMetric):
    r"""Logarithmic loss for distributional predictions.

    For a predictive distribution :math:`d` with pdf :math:`p_d`
    and a ground truth value :math:`y`, the logarithmic loss is
    defined as :math:`L(y, d) := -\log p_d(y)`.

    `evaluate` computes the average test sample loss.
    `evaluate_by_index` produces the loss sample by test data point
    `multivariate` controls averaging over variables.

    Parameters
    ----------
    multioutput : {'raw_values', 'uniform_average'} or array-like of shape \
            (n_outputs,), default='uniform_average'
        Defines whether and how to aggregate metric for across variables.
        If 'uniform_average' (default), errors are mean-averaged across variables.
        If array-like, errors are weighted averaged across variables, values as weights.
        If 'raw_values', does not average errors across variables, columns are retained.
    multivariate : bool, optional, default=False
        if True, behaves as multivariate log-loss
        log-loss is computed for entire row, results one score per row
        if False, is univariate log-loss
        log-loss is computed per variable marginal, results in many scores per row
    """

    def __init__(self, multioutput="uniform_average", multivariate=False):
        self.multivariate = multivariate
        super().__init__(multioutput=multioutput)

    def _evaluate_by_index(self, y_true, y_pred, **kwargs):
        res = -y_pred.log_pdf(y_true)
        # replace this by multivariate log_pdf once distr implements
        # i.e., pass multivariate on to log_pdf
        if self.multivariate:
            return pd.DataFrame(res.mean(axis=1), columns=["density"])
        else:
            return res


class LinearizedLogLoss(BaseDistrMetric):
    r"""Lineararized logarithmic loss for distributional predictions.

    For a predictive distribution :math:`d` with pdf :math:`p_d`
    and a ground truth value :math:`y`, the linearized logarithmic loss is
    defined as :math:`L(y, d) := -\log p_d(y)` if :math:`p_d(y) \geq r`,
    and :math:`L(y, d) := -\log p_d(r) + 1 - \frac{1}{r} p_d(r)` otherwise,
    where :math:`r` is the range of linearization parameter, `range` below.

    `evaluate` computes the average test sample loss.
    `evaluate_by_index` produces the loss sample by test data point
    `multivariate` controls averaging over variables.

    Parameters
    ----------
    range : positive float, optional, default=1
        range of linearization, i.e., where to linearize the log-loss
        for values smaller than range, the log-loss is linearized
    multioutput : {'raw_values', 'uniform_average'} or array-like of shape \
            (n_outputs,), default='uniform_average'
        Defines whether and how to aggregate metric for across variables.
        If 'uniform_average' (default), errors are mean-averaged across variables.
        If array-like, errors are weighted averaged across variables, values as weights.
        If 'raw_values', does not average errors across variables, columns are retained.
    multivariate : bool, optional, default=False
        if True, behaves as multivariate log-loss
        log-loss is computed for entire row, results one score per row
        if False, is univariate log-loss
        log-loss is computed per variable marginal, results in many scores per row
    """

    def __init__(self, range=1, multioutput="uniform_average", multivariate=False):
        self.range = range
        self.multivariate = multivariate
        super().__init__(multioutput=multioutput)

    def _evaluate_by_index(self, y_true, y_pred, **kwargs):
        range = self.range

        pdf = y_pred.pdf(y_true)
        pdf_smaller_range = pdf < range
        pdf_greater_range = pdf >= range

        logloss = -y_pred.log_pdf(y_true)
        linear = (-1 / range) * pdf - np.log(range) + 1

        res = pdf_smaller_range * linear + pdf_greater_range * logloss

        # replace this by multivariate log_pdf once distr implements
        # i.e., pass multivariate on to log_pdf
        if self.multivariate:
            return pd.DataFrame(res.mean(axis=1), columns=["density"])
        else:
            return res

    @classmethod
    def get_test_params(self):
        """Test parameter settings."""
        params1 = {}
        params2 = {"range": 0.1}
        return [params1, params2]


class SquaredDistrLoss(BaseDistrMetric):
    r"""Squared loss for distributional predictions.

    Also known as:

    * continuous Brier loss
    * Gneiting loss
    * (mean) squared error/loss, i.e., confusingly named the same as the
      point prediction loss commonly known as the mean squared error

    For a predictive distribution :math:`d`
    and a ground truth value :math:`y`, the squared (distribution) loss is
    defined as :math:`L(y, d) := -2 p_d(y) + \|p_d\|^2`,
    where :math:`\|p_d\|^2` is the (function) L2-norm of :math:`p_d`.

    `evaluate` computes the average test sample loss.
    `evaluate_by_index` produces the loss sample by test data point
    `multivariate` controls averaging over variables.

    Parameters
    ----------
    multioutput : {'raw_values', 'uniform_average'} or array-like of shape \
            (n_outputs,), default='uniform_average'
        Defines whether and how to aggregate metric for across variables.
        If 'uniform_average' (default), errors are mean-averaged across variables.
        If array-like, errors are weighted averaged across variables, values as weights.
        If 'raw_values', does not average errors across variables, columns are retained.
    multivariate : bool, optional, default=False
        if True, behaves as multivariate squared loss
        squared loss is computed for entire row, results one score per row
        if False, is univariate squared loss
        squared loss is computed per variable marginal, results in many scores per row
    """

    def __init__(self, multioutput="uniform_average", multivariate=False):
        self.multivariate = multivariate
        super().__init__(multioutput=multioutput)

    def _evaluate_by_index(self, y_true, y_pred, **kwargs):
        res = -2 * y_pred.log_pdf(y_true) + y_pred.pdfnorm(a=2)
        # replace this by multivariate log_pdf once distr implements
        # i.e., pass multivariate on to log_pdf
        if self.multivariate:
            return pd.DataFrame(res.mean(axis=1), columns=["density"])
        else:
            return res


class CRPS(BaseDistrMetric):
    r"""Continuous rank probability score for distributional predictions.

    Also known as:

    * integrated squared loss (ISL)
    * integrated Brier loss (IBL)
    * energy loss

    For a predictive distribution :math:`d` and a ground truth value :math:`y`,
    the CRPS is defined as
    :math:`L(y, d) := \mathbb{E}_{Y \sim d}|Y-y| - \frac{1}{2} \mathbb{E}_{X,Y \sim d}|X-Y|`.  # noqa: E501

    `evaluate` computes the average test sample loss.
    `evaluate_by_index` produces the loss sample by test data point
    `multivariate` controls averaging over variables.

    Parameters
    ----------
    multioutput : {'raw_values', 'uniform_average'} or array-like of shape \
            (n_outputs,), default='uniform_average'
        Defines whether and how to aggregate metric for across variables.
        If 'uniform_average' (default), errors are mean-averaged across variables.
        If array-like, errors are weighted averaged across variables, values as weights.
        If 'raw_values', does not average errors across variables, columns are retained.
    multivariate : bool, optional, default=False
        if True, behaves as multivariate CRPS (sum of scores)
        CRPS is computed for entire row, results one score per row
        if False, is univariate log-loss, per variable
        CRPS is computed per variable marginal, results in many scores per row
    """

    def __init__(self, multioutput="uniform_average", multivariate=False):
        self.multivariate = multivariate
        super().__init__(multioutput=multioutput)

    def _evaluate_by_index(self, y_true, y_pred, **kwargs):
        # CRPS(d, y) = E_X,Y as d [abs(Y-y) - 0.5 abs(X-Y)]
        return y_pred.energy(y_true) - y_pred.energy() / 2
