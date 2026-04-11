"""Concrete performance metrics for probabilistic supervised regression."""
# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
# adapted from sktime

from skpro.metrics.base import BaseDistrMetric


class CRPS(BaseDistrMetric):
    r"""Continuous rank probability score for distributional predictions.

    Also known as:

    * integrated squared loss (ISL)
    * integrated Brier loss (IBL)
    * energy loss

    For a predictive distribution :math:`d` and a ground truth value :math:`y`,
    the CRPS is defined as
    :math:`L(y, d) := \mathbb{E}_{Y \sim d}|Y-y| - \frac{1}{2} \mathbb{E}_{X,Y \sim d}|X-Y|`.

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

        * if True, behaves as multivariate CRPS:
          the score is computed for entire row, results one score per row
        * if False, is univariate CRPS:
          the score is computed per variable marginal, results in many scores per row
    """  # noqa: E501

    def __init__(self, multioutput="uniform_average", multivariate=False):
        self.multivariate = multivariate
        super().__init__(multioutput=multioutput)

    def _evaluate_by_index(self, y_true, y_pred, **kwargs):
        # CRPS(d, y) = E_X,Y as d [abs(Y-y) - 0.5 abs(X-Y)]
        return y_pred.energy(y_true) - y_pred.energy() / 2

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Retrieve test parameters."""
        params1 = {}
        params2 = {"multivariate": True}
        return [params1, params2]
