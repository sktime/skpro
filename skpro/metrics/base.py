"""Base classes for probabilistic metrics."""
# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
# adapted from sktime

from warnings import warn

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.utils import check_array, check_consistent_length

from skpro.base import BaseObject
from skpro.datatypes import check_is_scitype, convert, convert_to
from skpro.metrics._coerce import _coerce_to_df, _coerce_to_scalar

__author__ = ["fkiraly", "euanenticott-shell"]


class BaseProbaMetric(BaseObject):
    """Base class for probabilistic supervised error metrics in skpro.

    Parameters
    ----------
    multioutput : {'raw_values', 'uniform_average'}  or array-like of shape \
            (n_outputs,), default='uniform_average'
        Defines how to aggregate metric for multivariate (multioutput) data.
        If array-like, values used as weights to average the errors.
        If 'raw_values', returns a full set of errors in case of multioutput input.
        If 'uniform_average', errors of all outputs are averaged with uniform weight.
    score_average : bool, optional, default=True
        for interval and quantile losses only
            if True, metric/loss is averaged by upper/lower and/or quantile
            if False, metric/loss is not averaged by upper/lower and/or quantile
    """

    _tags = {
        "object_type": "metric",  # type of object
        "reserved_params": ["multioutput", "score_average"],
        "scitype:y_pred": "pred_proba",
        "lower_is_better": True,
    }

    def __init__(self, multioutput="uniform_average", score_average=True):
        self.multioutput = multioutput
        self.score_average = score_average
        super().__init__()

    def __call__(self, y_true, y_pred, **kwargs):
        """Calculate metric value using underlying metric function.

        Parameters
        ----------
        y_true : pd.Series, pd.DataFrame, 1D np.array, or 2D np.ndarray
            Ground truth (correct) target values.

        y_pred : return object of probabilistic predictition method scitype:y_pred
            must have same index and columns as y_true
            Predicted values, i-th row is prediction for i-th row of ``y_true``.

        Returns
        -------
        loss : float or 1-column pd.DataFrame with calculated metric value(s)
            metric is always averaged (arithmetic) over fh values
            if multioutput = "raw_values",
                will have a column level corresponding to variables in y_true
            if multioutput = multioutput = "uniform_average" or or array-like
                entries will be averaged over output variable column
            if score_average = False,
                will have column levels corresponding to quantiles/intervals
            if score_average = True,
                entries will be averaged over quantiles/interval column
        """
        return self.evaluate(y_true, y_pred, **kwargs)

    def evaluate(self, y_true, y_pred, **kwargs):
        """Evaluate the metric on given inputs.

        Parameters
        ----------
        y_true : pd.Series, pd.DataFrame, 1D np.array, or 2D np.ndarray
            Ground truth (correct) target values.

        y_pred : return object of probabilistic predictition method scitype:y_pred
            must have same index and columns as y_true
            Predicted values, i-th row is prediction for i-th row of ``y_true``.

        Returns
        -------
        loss : float or 1-column pd.DataFrame with calculated metric value(s)
            metric is always averaged (arithmetic) over fh values
            if multioutput = "raw_values",
                will have a column level corresponding to variables in y_true
            if multioutput = multioutput = "uniform_average" or or array-like
                entries will be averaged over output variable column
            if score_average = False,
                will have column levels corresponding to quantiles/intervals
            if score_average = True,
                entries will be averaged over quantiles/interval column
        """
        # Input checks and conversions
        y_true_inner, y_pred_inner, multioutput = self._check_ys(
            y_true, y_pred, self.multioutput
        )

        # Don't want to include scores for 0 width intervals, makes no sense
        if 0 in y_pred_inner.columns.get_level_values(1):
            y_pred_inner = y_pred_inner.drop(0, axis=1, level=1)
            warn(
                "Dropping 0 width interval, don't include 0.5 quantile"
                "for interval metrics."
            )

        # pass to inner function
        out = self._evaluate(y_true_inner, y_pred_inner, **kwargs)

        if self.score_average and multioutput == "uniform_average":
            out = out.mean(axis=1).iloc[0]  # average over all
        if self.score_average and multioutput == "raw_values":
            out = out.T.groupby(level=0).mean().T  # average over scores
        if not self.score_average and multioutput == "uniform_average":
            out = out.T.groupby(level=1).mean().T  # average over variables
        if not self.score_average and multioutput == "raw_values":
            out = out  # don't average

        if isinstance(out, pd.DataFrame):
            out = out.squeeze(axis=0)

        return out

    def _evaluate(self, y_true, y_pred, **kwargs):
        """Evaluate the metric on given inputs.

        Private _evaluate, called by public evaluate.

        Parameters
        ----------
        y_true : pd.Series, pd.DataFrame, 1D np.array, or 2D np.ndarray
            Ground truth (correct) target values.

        y_pred : return object of probabilistic predictition method scitype:y_pred
            must have same index and columns as y_true
            Predicted values, i-th row is prediction for i-th row of ``y_true``.

        Returns
        -------
        loss : pd.DataFrame of shape (, n_outputs), calculated loss metric.
        """
        # Default implementation relies on implementation of evaluate_by_index
        try:
            index_df = self._evaluate_by_index(y_true, y_pred)
            out_df = pd.DataFrame(index_df.mean(axis=0)).T
            out_df.columns = index_df.columns
            return out_df
        except RecursionError as _err:
            msg = (
                f"{type(self).__name__} must implement one of"
                "_evaluate or _evaluate_by_index, but none of the two was implemented"
            )
            raise RecursionError(msg) from _err

    def evaluate_by_index(self, y_true, y_pred, **kwargs):
        """Evaluate the metric by instance index (row).

        Parameters
        ----------
        y_true : pd.Series, pd.DataFrame, 1D np.array, or 2D np.ndarray
            Ground truth (correct) target values.

        y_pred : return object of probabilistic predictition method scitype:y_pred
            must have same index and columns as y_true
            Predicted values, i-th row is prediction for i-th row of ``y_true``.

        Returns
        -------
        loss : pd.DataFrame of length len(fh), with calculated metric value(s)
            i-th column contains metric value(s) for prediction at i-th fh element
            if multioutput = "raw_values",
                will have a column level corresponding to variables in y_true
            if multioutput = multioutput = "uniform_average" or or array-like
                entries will be averaged over output variable column
            if score_average = False,
                will have column levels corresponding to quantiles/intervals
            if score_average = True,
                entries will be averaged over quantiles/interval column
        """
        # Input checks and conversions
        y_true_inner, y_pred_inner, multioutput = self._check_ys(
            y_true, y_pred, self.multioutput
        )

        # Don't want to include scores for 0 width intervals, makes no sense
        if 0 in y_pred_inner.columns.get_level_values(1):
            y_pred_inner = y_pred_inner.drop(0, axis=1, level=1)
            warn(
                "Dropping 0 width interval, don't include 0.5 quantile"
                "for interval metrics."
            )

        # pass to inner function
        out = self._evaluate_by_index(y_true_inner, y_pred_inner, **kwargs)

        if self.score_average and multioutput == "uniform_average":
            out = out.mean(axis=1)  # average over all
        if self.score_average and multioutput == "raw_values":
            out = out.T.groupby(level=0).mean().T  # average over scores
        if not self.score_average and multioutput == "uniform_average":
            out = out.T.groupby(level=1).mean().T  # average over variables
        if not self.score_average and multioutput == "raw_values":
            out = out  # don't average

        return out

    def _evaluate_by_index(self, y_true, y_pred, **kwargs):
        """Evaluate the metric by instance index (row).

        Private _evaluate_by_index, called by public evaluate_by_index.

        By default this uses _evaluate to find jackknifed pseudosamples. This
        estimates the error at each of the time points.

        Parameters
        ----------
        y_true : pd.Series, pd.DataFrame, 1D np.array, or 2D np.ndarray
            Ground truth (correct) target values.

        y_pred : return object of probabilistic predictition method scitype:y_pred
            must have same index and columns as y_true
            Predicted values, i-th row is prediction for i-th row of ``y_true``.
        """
        n = y_true.shape[0]
        out_series = pd.Series(index=y_pred.index)
        try:
            x_bar = self.evaluate(y_true, y_pred, self.multioutput, **kwargs)
            for i in range(n):
                out_series[i] = n * x_bar - (n - 1) * self.evaluate(
                    np.vstack((y_true[:i, :], y_true[i + 1 :, :])),  # noqa
                    np.vstack((y_pred[:i, :], y_pred[i + 1 :, :])),  # noqa
                    self.multioutput,
                    **kwargs,
                )
            return out_series
        except RecursionError:
            raise RecursionError(
                "Must implement one of _evaluate or _evaluate_by_index"
            )

    def _check_consistent_input(self, y_true, y_pred, multioutput):
        check_consistent_length(y_true, y_pred)

        y_true = check_array(y_true, ensure_2d=False)

        if not isinstance(y_pred, pd.DataFrame):
            raise ValueError("y_pred should be a dataframe.")

        if not np.all([is_numeric_dtype(y_pred[c]) for c in y_pred.columns]):
            raise ValueError("Data should be numeric.")

        if y_true.ndim == 1:
            y_true = y_true.reshape((-1, 1))

        n_outputs = y_true.shape[1]

        allowed_multioutput_str = ("raw_values", "uniform_average", "variance_weighted")
        if isinstance(multioutput, str):
            if multioutput not in allowed_multioutput_str:
                raise ValueError(
                    "Allowed 'multioutput' string values are {}. "
                    "You provided multioutput={!r}".format(
                        allowed_multioutput_str, multioutput
                    )
                )
        elif multioutput is not None:
            multioutput = check_array(multioutput, ensure_2d=False)
            if n_outputs == 1:
                raise ValueError("Custom weights are useful only in multi-output case.")
            elif n_outputs != len(multioutput):
                raise ValueError(
                    "There must be equally many custom weights (%d) as outputs (%d)."
                    % (len(multioutput), n_outputs)
                )

        return y_true, y_pred, multioutput

    def _check_ys(self, y_true, y_pred, multioutput):
        if multioutput is None:
            multioutput = self.multioutput

        valid, msg, metadata = check_is_scitype(
            y_pred, scitype="Proba", return_metadata=True, var_name="y_pred"
        )

        if not valid:
            raise TypeError(msg)

        y_pred_mtype = metadata["mtype"]
        inner_y_pred_mtype = self.get_tag("scitype:y_pred")

        y_pred_inner = convert(
            y_pred,
            from_type=y_pred_mtype,
            to_type=inner_y_pred_mtype,
            as_scitype="Proba",
        )

        if inner_y_pred_mtype == "pred_interval":
            if 0.0 in y_pred_inner.columns.get_level_values(1):
                for var in y_pred_inner.columns.get_level_values(0):
                    y_pred_inner[var, 0.0, "upper"] = y_pred_inner[var, 0.0, "lower"]

        y_true, y_pred, multioutput = self._check_consistent_input(
            y_true, y_pred, multioutput
        )

        return y_true, y_pred_inner, multioutput

    def _get_alpha_from(self, y_pred):
        """Fetch the alphas present in y_pred."""
        alphas = np.unique(list(y_pred.columns.get_level_values(1)))
        if not all((alphas > 0) & (alphas < 1)):
            raise ValueError("Alpha must be between 0 and 1.")

        return alphas

    def _check_alpha(self, alpha):
        """Check alpha input and coerce to np.ndarray."""
        if alpha is None:
            return None

        if isinstance(alpha, float):
            alpha = [alpha]

        if not isinstance(alpha, np.ndarray):
            alpha = np.asarray(alpha)

        if not all((alpha > 0) & (alpha < 1)):
            raise ValueError("Alpha must be between 0 and 1.")

        return alpha

    def _handle_multioutput(self, loss, multioutput):
        """Specificies how multivariate outputs should be handled.

        Parameters
        ----------
        loss : float, np.ndarray the evaluated metric value.

        multioutput : string "uniform_average" or "raw_values" determines how \
            multioutput results will be treated.
        """
        if isinstance(multioutput, str):
            if multioutput == "raw_values":
                return loss
            elif multioutput == "uniform_average":
                # pass None as weights to np.average: uniform mean
                multioutput = None
            else:
                raise ValueError(
                    "multioutput is expected to be 'raw_values' "
                    "or 'uniform_average' but we got %r"
                    " instead." % multioutput
                )

        if loss.ndim > 1:
            out = np.average(loss, weights=multioutput, axis=1)
        else:
            out = np.average(loss, weights=multioutput)
        return out


class BaseDistrMetric(BaseProbaMetric):
    """Intermediate base class for distributional prediction metrics/scores.

    Developer note:
    Experimental and overrides public methods of BaseProbaMetric.
    This should be refactored into one base class.
    """

    _tags = {
        "scitype:y_pred": "pred_proba",
        "lower_is_better": True,
    }

    def evaluate(self, y_true, y_pred, **kwargs):
        """Evaluate the  metric on given inputs.

        Parameters
        ----------
        y_true : pd.Series, pd.DataFrame, 1D np.array, or 2D np.ndarray
            Ground truth (correct) target values.

        y_pred : return object of probabilistic predictition method scitype:y_pred
            must have same index and columns as y_true
            Predicted values, i-th row is prediction for i-th row of ``y_true``.

        Returns
        -------
        loss : float or 1-column pd.DataFrame with calculated metric value(s)
            float if multioutput = "uniform_average" or multivariate = True
            1-column df if multioutput = "raw_values" and metric is not multivariate
            metric is always averaged (arithmetic) over rows
        """
        multioutput = self.multioutput
        multivariate = self.multivariate

        index_df = self.evaluate_by_index(y_true, y_pred)
        out_df = pd.DataFrame(index_df.mean(axis=0)).T
        out_df.columns = index_df.columns

        if multioutput == "uniform_average" and not multivariate:
            out_df = out_df.mean(axis=1)
        if multioutput == "uniform_average" or multivariate:
            out = _coerce_to_scalar(out_df)
        else:
            out = _coerce_to_df(out_df)
        return out

    def _coerce_inner_df(self, obj):
        """Coerce obj to pd_DataFrame_Table, for inner method call.

        Parameters
        ----------
        obj : object
            Object to coerce

        Returns
        -------
        obj : object
            Coerced object
        """
        obj = convert_to(obj, to_type="pd_DataFrame_Table", as_scitype="Table")
        obj = _coerce_to_df(obj)
        return obj

    def evaluate_by_index(self, y_true, y_pred, **kwargs):
        """Evaluate the metric by instance index (row).

        Parameters
        ----------
        y_true : pd.Series, pd.DataFrame, 1D np.array, or 2D np.ndarray
            Ground truth (correct) target values.

        y_pred : skpro BaseDistribution of same shape as y_true
            Predictive distribution.
            Must have same index and columns as y_true.
        """
        multioutput = self.multioutput

        if hasattr(self, "multivariate"):
            multivariate = self.multivariate
        else:
            multivariate = False

        y_true = self._coerce_inner_df(y_true)

        if "C_true" in kwargs:
            C_true = kwargs["C_true"]
            C_true = self._coerce_inner_df(C_true)
            kwargs_inner = {"C_true": C_true}
        else:
            kwargs_inner = {}

        if multivariate:
            res = self._evaluate_by_index(
                y_true=y_true, y_pred=y_pred, multioutput=multioutput, **kwargs_inner
            )
            res.columns = ["score"]
            return res
        else:
            res_by_col = []
            for col in y_pred.columns:
                y_pred_col = y_pred.loc[:, [col]]
                y_true_col = y_true.loc[:, [col]]
                res_for_col = self._evaluate_by_index(
                    y_true=y_true_col,
                    y_pred=y_pred_col,
                    multioutput=multioutput,
                    **kwargs_inner,
                )
                res_for_col.columns = [col]
                res_by_col += [res_for_col]
            res = pd.concat(res_by_col, axis=1)

        return res
