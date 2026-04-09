"""Concrete performance metrics for probabilistic supervised regression."""
# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
# adapted from sktime

import numpy as np
import pandas as pd

from skpro.metrics.base import BaseDistrMetric


class LinearizedLogLoss(BaseDistrMetric):
    r"""Linearized logarithmic loss for distributional predictions.

    For a predictive distribution :math:`d` with pdf :math:`p_d`
    and a ground truth value :math:`y`, the linearized logarithmic loss is
    defined as :math:`L(y, d) := -\log p_d(y)` if :math:`p_d(y) \geq r`,
    and :math:`L(y, d) := -\log p_d(r) + 1 - \frac{1}{r} p_d(r)` otherwise,
    where :math:`r` is the range of linearization parameter, `range` below.

    * ``evaluate`` computes the average test sample loss.
    * ``evaluate_by_index`` produces the loss sample by test data point.
    * ``multivariate`` controls averaging over variables.

    Parameters
    ----------
    range : positive float, optional, default=1
        range of linearization, i.e., where to linearize the log-loss
        for values smaller than range, the log-loss is linearized

    multioutput : {'raw_values', 'uniform_average'} or array-like of shape \
            (n_outputs,), default='uniform_average'
        Defines whether and how to aggregate metric for across variables.

        * If 'uniform_average' (default), errors are mean-averaged across variables.
        * If array-like, errors are weighted averaged across variables,
          values as weights.
        * If 'raw_values', does not average errors across variables,
          columns are retained.

    multivariate : bool, optional, default=False

        * if True, behaves as multivariate squared loss:
          the score is computed for entire row, results one score per row
        * if False, is univariate squared loss:
          the score is computed per variable marginal, results in many scores per row
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
    def get_test_params(cls, parameter_set="default"):
        """Test parameter settings."""
        params1 = {}
        params2 = {"range": 0.1}
        return [params1, params2]
