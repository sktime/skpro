"""
Smoothers capable of extrapolating beyond the range seen in the training

The most common example is extrapolating a target profile or factor profile
into the future, i.e. the smoother can be used to extend the profile beyond the
time range seen in the training. In this case, the smoother operates on the
feature `time`.

.. note::
    Using these components requires **regular retraining**, because the
    extrapolation originates at the end of the training interval and will
    become more and more uncertain over time.

    don't use

    * :class:`cyclic_boosting.binning.ECdfTransformer`
    * :class:`cyclic_boosting.binning.BinNumberTransformer`
    * :class:`cyclic_boosting.binning.FractionalBinNumberTransformer`

    on features these smoothers operate on, because they just assign the
    largest bin number to feature values beyond the upper bound seen in the
    training which prevents any extrapolation.

    For the `time` feature, :class:`LinearBinner` is usually sufficient,
    because the amount of samples per time interval usually doesn't vary that
    much over time.

    Extrapolation is **not** needed for times within some period, e.g. for
    the weekday, the month day or the day number in the year. Please use normal
    smoothers for that.
"""

import numpy as np

from skpro.libs.cyclic_boosting.smoothing import onedim
from skpro.libs.cyclic_boosting.utils import linear_regression


class LinearExtrapolator(onedim.AbstractBinSmoother):
    r"""Smoother of one-dimensional bins that applies a linear regression
    to fit the bin values of y (as received from the profile function in
    the fit) and applies a linear interpolation in predict.

    The :func:`math_utils.linear_regression` is used for the regression.
    The uncertainties :math:`\sigma_i` of `y` are transformed to the
    weights :math:`w_i = \frac{1}{(\sigma_i)^2 + \epsilon}` needed by
    :func:`~math_utils.linear_regression` in the fit method. An epsilon
    value is added to avoid zero division.

    :param epsilon: Epsilon value to avoid zero division when transforming
        the uncertainties to the weights.
    :type epsilon: float

    **Required columns** in the ``X_for_smoother`` passed to :meth:`fit`:

    * column 0: ...
    * column 1: ignored
    * column 2: uncertainty of the average ``y`` in each bin

    These are provided by the following **profile functions**:

    * :class:`BetaMeanSigmaProfile`

    **Used column** in the :meth:`predict`:

    * column 0: ...

    **Estimated parameters**

    :param `alpha_`: axis interception
    :type `alpha_`: :obj:`float`

    :param `beta_`: slope
    :type `beta_`: :obj:`float`

    **Small data example**

    >>> np.random.seed(4)
    >>> n = 5
    >>> x = np.arange(n)
    >>> alpha = 3.0
    >>> beta = 0.5
    >>> y = alpha + beta * x
    >>> y
    array([ 3. ,  3.5,  4. ,  4.5,  5. ])
    >>> y_err = np.array([np.mean(np.random.randn(n)) for i in range(n)])
    >>> y_smeared = y + y_err
    >>> y_smeared
    array([ 2.96598022,  3.01021291,  4.02623841,  4.91211327,  5.04914922])

    >>> X = np.c_[x, np.ones(n), y_err]

    >>> from skpro.libs.cyclic_boosting.smoothing.extrapolate import LinearExtrapolator
    >>> est = LinearExtrapolator()
    >>> est = est.fit(X, y_smeared)

    >>> from decimal import Decimal
    >>> Decimal(est.alpha_).quantize(Decimal("1.000"))
    Decimal('2.971')
    >>> Decimal(est.beta_).quantize(Decimal("1.000"))
    Decimal('0.523')

    >>> x1 = np.array([-7, -5, -3, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9., 20.])
    >>> X1 = np.c_[x1]
    >>> est.predict(X1)
    array([ -0.69276543,   0.354095  ,   1.40095542,   2.44781584,
             2.97124605,   3.49467626,   4.01810647,   4.54153668,
             5.06496689,   5.5883971 ,   6.11182731,   6.63525752,
             7.15868773,   7.68211794,  13.43985026])
    """

    def __init__(self, epsilon=1e-4):
        self.epsilon = epsilon
        self.alpha_ = None
        self.beta_ = None

    def _check_fitted(self):
        """Check if a fit was performed."""
        if self.alpha_ is None or self.beta_ is None:
            raise ValueError("You have to call fit before predict.")

    def fit(self, X_for_smoother, y):
        weights = 1.0 / (np.square(np.asarray(X_for_smoother[:, 2])) + self.epsilon)
        self.alpha_, self.beta_ = linear_regression(
            np.asarray(X_for_smoother[:, 0], dtype=np.float64),
            np.asarray(y, dtype=np.float64),
            weights,
        )
        return self

    def predict(self, X):
        self._check_fitted()
        y = self.alpha_ + np.asarray(X[:, 0], dtype=np.float64) * self.beta_
        return y


__all__ = [
    "LinearExtrapolator",
]
