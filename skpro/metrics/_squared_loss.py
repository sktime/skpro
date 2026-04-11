"""Concrete performance metrics for probabilistic supervised regression."""
# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
# adapted from sktime

import pandas as pd

from skpro.metrics.base import BaseDistrMetric


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

    * ``evaluate`` computes the average test sample loss.
    * ``evaluate_by_index`` produces the loss sample by test data point.
    * ``multivariate`` controls averaging over variables.

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

    multivariate : bool, optional, default=False

        * if True, behaves as multivariate squared loss:
          the score is computed for entire row, results one score per row
        * if False, is univariate squared loss:
          the score is computed per variable marginal, results in many scores per row
    """

    def __init__(self, multioutput="uniform_average", multivariate=False):
        self.multivariate = multivariate
        super().__init__(multioutput=multioutput)

    def _evaluate_by_index(self, y_true, y_pred, **kwargs):
        res = -2 * y_pred.pdf(y_true) + y_pred.pdfnorm(a=2)
        # replace this by multivariate pdf once distr implements
        # i.e., pass multivariate on to pdf
        if self.multivariate:
            return pd.DataFrame(res.mean(axis=1), columns=["density"])
        else:
            return res

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Retrieve test parameters."""
        params1 = {}
        params2 = {"multivariate": True}
        return [params1, params2]
