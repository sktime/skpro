"""Concrete performance metrics for probabilistic supervised regression."""
# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
# adapted from sktime

import numpy as np
import pandas as pd

from skpro.metrics.base import BaseDistrMetric


class AUCalibration(BaseDistrMetric):
    r"""Area under the calibration curve for distributional predictions.

    Computes the unsigned area between the calibration curve and the diagonal.

    The calibration curve is the cumulative curve of the sample of
    predictive cumulative distribution functions evaluated at the true values.

    Mathematically, let :math:`d_1, \dots, d_N` be the predictive distributions,
    let :math:`y_1, \dots, y_N` be the true values, and let :math:`F_i` be the
    cumulative distribution function of :math:`d_i`.

    Define the calibration sample as :math:`c_i := F_i(y_i)`, for
    :math:`i = 1, \dots, N`. For perfect predictions, the sample of :math:`c_i` will be
    uniformly distributed on [0, 1], and i.i.d. from that uniform distribution.

    Let :math:`c_{(i)}` be the :math:`i`-th order statistic of the sample of
    :math:`c_i`, i.e., the :math:`i`-th smallest value in the sample.

    The (unsigned) area under the calibration curve - or, more precisely,
    between the diagonal and the calibration curve - is defined as

    .. math:: \frac{1}{N} \sum_{i=1}^N \left| c_{(i)} - \frac{i}{N} \right|.

    * ``evaluate`` returns the unsigned area between the calibration curve
      and the diagonal, i.e., the above quantity.
    * ``evaluate_by_index`` returns, for the :math:`i`-th test sample, the value
      :math:`\left| c_i - \frac{r_i}{N} \right|`,
      where :math:`r_i` is the rank of :math:`c_i`
      in the sample of :math:`c_i`. In case of ties, tied ranks are averaged.
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

        * if True, behaves as multivariate metric (sum of scores):
          the metric is computed for entire row, results one score per row
        * if False, is univariate metric, per variable:
          the metric is computed per variable marginal, results in many scores per row
    """  # noqa: E501

    def __init__(self, multioutput="uniform_average", multivariate=False):
        self.multivariate = multivariate
        super().__init__(multioutput=multioutput)

    def _evaluate_by_index(self, y_true, y_pred, **kwargs):
        cdfs = y_pred.cdf(y_true)
        # using the average in case of ranks is fine
        # because the absolute sums in the metric average out
        cdfs_ranked = cdfs.rank(axis=0, method="average", pct=True)
        n = cdfs.shape[0]
        diagonal = np.arange(1, n + 1).reshape(-1, 1) / n

        res = (cdfs_ranked - diagonal).abs()

        if self.multivariate:
            return pd.DataFrame(res.mean(axis=1), columns=["score"])
        return res

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Retrieve test parameters."""
        params1 = {}
        params2 = {"multivariate": True}
        return [params1, params2]
