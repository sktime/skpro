"""Survival Process Logarithmic Loss for distributional predictions."""
# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)

import numpy as np
import pandas as pd

from skpro.metrics.base import BaseDistrMetric


class SPLL(BaseDistrMetric):
    r"""Survival Process Logarithmic Loss for distributional predictions.

    Same as the negative log-likelihood of the survival process (see [1]_),
    and therefore a proper scoring rule for survival predictions.

    For a predictive distribution :math:`d` with pdf :math:`d.p`,
    survival function :math:`d.S`, a ground truth value :math:`y`
    and censoring indicator :math:`\Delta`, taking values 1 (censored) and
    0 (uncensored), the survival process logarithmic loss is defined as
    :math:`L((y, \Delta), d) := (\Delta - 1) (\log d.p(y)) - \Delta \log d.S(y)`.
    Logarithms are natural logarithms.

    Where :math:`d.S = 1 - d.F` for the cdf :math:`d.F` of :math:`d`.

    To obtain the loss from formula 7.1.2 in [1]_, condition on
    :math:`N(A) \le 1` (i.e., no more than one event in the interval :math:`A`),
    use that :math:`\Delta = 1` iff :math:`N(A) = 0`,
    and :math:`-\log d.S(y) = \int_A d.\lambda(t) \; dt`, with
    :math:`d.\lambda = -d.p / d.S`.

    `evaluate` computes the average test sample loss.
    `evaluate_by_index` produces the loss sample by test data point.
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

    References
    ----------
    .. [1] Daley DJ, Vere-Jones. An Introduction to the Theory of Point Processes,
        2nd Edition, 2003, Springer, New York. Formula 7.1.2.
    """

    _tags = {
        "authors": "fkiraly",
        "capability:survival": True,
        "scitype:y_pred": "pred_proba",
        "lower_is_better": True,
    }

    def __init__(self, multioutput="uniform_average", multivariate=False):
        self.multivariate = multivariate
        super().__init__(multioutput=multioutput)

    def _evaluate_by_index(self, y_true, y_pred, **kwargs):
        C_true = kwargs.get("C_true", None)

        if C_true is None:
            # then all uncensored, Delta = 0
            res = -y_pred.log_pdf(y_true)
        else:
            cont_term = -y_pred.log_pdf(y_true) * (1 - C_true.to_numpy())
            disc_term = np.log(1 - y_pred.cdf(y_true)) * C_true.to_numpy()
            res = cont_term + disc_term

        if self.multivariate:
            return pd.DataFrame(res.mean(axis=1), columns=["SPLL"])
        else:
            return res
