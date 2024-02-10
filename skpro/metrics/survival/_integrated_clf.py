"""Integrated classification score."""
# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)

import numpy as np
import pandas as pd

from skpro.metrics.base import BaseDistrMetric


class IntegratedClfScore(BaseDistrMetric):
    r"""Integrated classification score, e.g., Brier.

    Discrete sum of binned probabilistic classification scores,
    by default the Brier classification score.

    Wraps an ``sklearn`` classification score, such as the Brier score,
    and converts it to a score for probabilistic regression or survival,
    by summing the score over binned predictions.

    Takes as main parameter a specification of bins
    :math:`B_j, j = 1 \dots J` to use for binning the prediction values,
    and a binary classification
    score :math:`S: \{0, 1\} \times \mathbb{R} \to \mathbb{R}`.

    For ground truth samples :math:`y_i, c_i, i = 1 \dots N`,
    and predictive distributions :math:`d_i, i = 1 \dots N`,
    the :math:`i`-th score is computed as

    .. math:: L(y_i, d_i) = \sum_{j=1}^J g_i S\left(y_{ij}, P_{d_i}(y_i \in B_j)\right),

    where :math:`y_{ij}` is the 0/1 indicator of the event:
    :math:`y_i \in B_j` and :math:`c_i = 0`;
    and :math:`g_i` is the 0/1 indicator of the event:
    :math:`y_{ij} = 1,` or :math:`y_i \not \in B_j`.

    The :math:`B_j` are specified via the `bins` and `binning` parameter.

    * If `binning` is 'increasing', the bins are
      :math:`B_j = (b_0, b_j]`, where :math:`b_j` = `bins[j]`.
    * If `binning` is 'disjoint', the bins are
      :math:`B_j = (b_{j-1}, b_j]`, where :math:`b_j` = `bins[j]`.
    * if `bins` is None, the bins are the unique values of the predictions.

    This metric supports multiple options for inverse risk scores,
    including any method evaluates of predictive distributions.

    The default is the predictive mean survival time.

    `evaluate` computes the concordance index.
    `evaluate_by_index` produces, for one test sample,
    the fraction of concordant pairs among all pairs with this sample as first index.
    `multivariate` controls averaging over variables.

    Parameters
    ----------
    score : ``sklearn`` classification metric, optional
        default=``brier_score_loss`` from sklearn

    bins : str, iterable of float (monotonic), optional, default=None
        The bin bounds to use for binning the prediction values.
        The default uses the unique values of the predictions.
        If ``"observed"``, the bins are the unique values of the observed values.

    binning : str, {'increasing', 'disjoint'}, default='increasing'

    multioutput : {'raw_values', 'uniform_average'} or array-like of shape
        (n_outputs,), default='uniform_average'
        Defines whether and how to aggregate metric for across variables.
        If 'uniform_average' (default), errors are mean-averaged across variables.
        If array-like, errors are weighted averaged across variables, values as weights.
        If 'raw_values', does not average errors across variables, columns are retained.
    multivariate : bool, optional, default=False
        if True, behaves as multivariate log-loss
        C-index is computed for entire row, results one score per row
        if False, is univariate log-loss
        C-index is computed per variable marginal, results in many scores per row
    """

    _tags = {
        "authors": "fkiraly",
        "capability:survival": True,
        "scitype:y_pred": "pred_proba",
        "lower_is_better": True,
    }

    def __init__(
        self,
        score=None,
        bins=None,
        binning="increasing",
        multioutput="uniform_average",
        multivariate=False,
    ):
        self.score = score
        self.bins = bins
        self.binning = binning
        self.multivariate = multivariate
        super().__init__(multioutput=multioutput)

    def _evaluate_by_index(self, y_true, y_pred, **kwargs):
        C_true = kwargs.get("C_true", None)



        if self.multivariate:
            return pd.DataFrame(res_df.mean(axis=1), columns=["C_Harrell"])
        else:
            return res_df
