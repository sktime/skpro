"""
One-dimensional smoothers
"""

import warnings

import numpy as np
import scipy.interpolate
import scipy.optimize
import sklearn.isotonic
from scipy import sparse
from sklearn.linear_model import LassoLarsIC

from skpro.libs.cyclic_boosting import utils
from skpro.libs.cyclic_boosting.smoothing.base import AbstractBinSmoother
from skpro.libs.cyclic_boosting.smoothing.orthofit import (
    cy_apply_orthogonal_poly_fit_equidistant,
    cy_orthogonal_poly_fit_equidistant,
)


class PredictingBinValueMixin(object):
    """Mixin for smoothers of one-dimensional bins with :meth:`predict`
    returning the corresponding entry in the estimated parameter
    ``smoothed_y_`` which must have been calculated in :meth:`fit`.

    Please create the array ``smoothed_y_`` in your :meth:`fit` method.

    **Estimated parameters**

    :param `smoothed_y_`: the bin values for ``y``
        (as received from the profile function in the fit)
        after some smoothing.
        a pseudobin for the missing values is not supported
    :type `smoothed_y_`: :class:`numpy.ndarray` (float64, shape `(n_bins,)`)

    For examples, see the subclasses :class:`BinValuesSmoother` and
    :class:`BinKernelSmoother`.
    """

    def predict(self, X):
        if not hasattr(self, "smoothed_y_"):
            raise ValueError("The {} has not been fitted!".format(self.__class__.__name__))
        binnos = X[:, 0]
        binnos_round = np.asarray(np.rint(binnos), dtype=np.int64)
        is_valid = np.isfinite(binnos) & (binnos >= 0) & (binnos_round < (len(self.smoothed_y_)))

        pred = utils.nans(len(binnos))
        pred[is_valid] = self.smoothed_y_[binnos_round[is_valid]]
        return pred


class BinValuesSmoother(AbstractBinSmoother, PredictingBinValueMixin):
    """Smoother that does not perform any smoothing (the identity smoother so
    to speak).

    Smoother of one-dimensional bins that outputs the saved bin values of y (as
    received from the profile function in the fit) as the prediction.  This
    smoother has **no requirements** on ``X_for_smoother`` passed by the
    **profile function**, because it just saves the target values ``y`` and
    ignores ``X_for_smoother`` completely.

    **Estimated parameters**

    :param `smoothed_y_`: the bin values for ``y``
        (as received from the profile function in the fit)
        after some smoothing.
        a pseudobin for the missing values is not supported
    :type `smoothed_y_`: :class:`numpy.ndarray` (float64, shape `(n_bins,)`)

    >>> X = np.c_[[0., 1, 2, 3],
    ...           [1, 1, 1, 1]]
    >>> y = np.array([90, 80, 50, 40])

    >>> from skpro.libs.cyclic_boosting.smoothing.onedim import BinValuesSmoother
    >>> reg = BinValuesSmoother()
    >>> assert reg.fit(X, y) is reg
    >>> assert np.allclose(reg.smoothed_y_, y)
    >>> X = np.c_[[3.1, 0.4, 1, 2.9, 3.4, 4.5]]
    >>> reg.predict(X)
    array([ 40.,  90.,  80.,  40.,  40.,  nan])
    """

    elem = "smoothed_y_"

    def fit(self, X_for_smoother, y):
        self.smoothed_y_ = y
        return self

    def __getstate__(self):
        """Return state values to be pickled."""
        state = self.__dict__.copy()
        if self.elem in state and state[self.elem] is not None:
            state[self.elem] = sparse.csr_matrix(state[self.elem])
        return state

    def __setstate__(self, state):
        """Restore state from the unpickled state values."""
        if self.elem in state and state[self.elem] is not None:
            state[self.elem] = state[self.elem].toarray()[0, :]
        self.__dict__.update(state)


class RegularizeToPriorExpectationSmoother(AbstractBinSmoother, PredictingBinValueMixin):
    r"""Smoother of one-dimensional bins regularizing values with uncertainties
    to a prior expectation.

    :param prior_expectation: The prior dominate the regularized value if the
        uncertainties are large.
    :type prior_expectation: :class:`numpy.ndarray` (float64, dim=1) or float

    :param threshold: Threshold in terms of sigma. If the significance of a
        value:

        .. math::

            \text{sig}(x_i) = \frac{x_i - \text{prior\_expectation}_i}
            {\text{uncertainty}_i}

        is below the threshold, the prior expectation replaces the value.
    :type threshold: float

    **Required columns** in the ``X_for_smoother`` passed to :meth:`fit`:

    * column 0: ...
    * column 1: ignored
    * column 2: uncertainty of the average ``y`` in each bin

    **Doctests**

    >>> y = np.array([0, 1, 2, 3])
    >>> X = np.c_[
    ...     [0, 1, 2, 3],
    ...     [1]*4,
    ...     [0.1]*4]
    >>> from skpro.libs.cyclic_boosting.smoothing.onedim import RegularizeToPriorExpectationSmoother
    >>> est = RegularizeToPriorExpectationSmoother(1.)
    >>> assert est.fit(X, y) is est
    >>> y_smoothed = est.predict(X[:, [0]])
    >>> y_smoothed
    array([ 0.03175416,  1.        ,  1.96824584,  2.98431348])
    >>> np.allclose(1 - np.sqrt(((y[0] - 1) / 0.1)**2 - 2.5**2) * 0.1,
    ...     y_smoothed[0])
    True
    >>> np.allclose(1 + np.sqrt(((y[-1] - 1) / 0.1)**2 - 2.5**2) * 0.1,
    ...     y_smoothed[-1])
    True
    """

    def __init__(self, prior_expectation, threshold=2.5):
        self.prior_expectation = prior_expectation
        self.threshold = threshold

    def fit(self, X_for_smoother, y):
        self.smoothed_y_ = utils.regularize_to_prior_expectation(
            y, X_for_smoother[:, 2], self.prior_expectation, threshold=self.threshold
        )
        return self


class RegularizeToOneSmoother(RegularizeToPriorExpectationSmoother):
    """Smoother for one-dimensional bins regularizing values with uncertainties
    to the mean of the values.

    :param threshold: threshold in terms of sigma.
        If the significance of a factor is below the threshold, the global
        measurement replaces the factor.
    :type threshold: float
    """

    def __init__(self, threshold=2.5):
        RegularizeToPriorExpectationSmoother.__init__(self, prior_expectation=1, threshold=threshold)


class WeightedMeanSmoother(AbstractBinSmoother, PredictingBinValueMixin):
    r"""Smoother for one-dimensional bins regularizing values with
    uncertainties to the weighted mean or a user defined value.

    :param prior_prediction: If the `prior_prediction` is specified, all values
        are regularized with it and not with the error weighted mean.
    :type prior_prediction: float

    >>> y_for_smoother = np.array([0.9, 0.9, 0.9, 1.8, 1.8, 0.4, 0.4])
    >>> X_for_smoother = np.c_[
    ...     [1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
    ...     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ...     [0.05, 0.05, 0.05, 0.05, 0.15, 0.15, 0.05]]
    >>> from skpro.libs.cyclic_boosting.smoothing.onedim import WeightedMeanSmoother
    >>> smoother = WeightedMeanSmoother()
    >>> smoother.fit(X_for_smoother, y_for_smoother)
    >>> smoother.smoothed_y_
    array([ 0.90096366,  0.90096366,  0.90096366,  1.79077293,  1.723854  ,
            0.45467402,  0.40662518])

    >>> y_for_smoother = np.array([-0.70710678, -0.70710678])
    >>> X_for_smoother = np.c_[[0., 1.], [0.012]*2, [9.12870929]*2]
    >>> smoother = WeightedMeanSmoother()
    >>> smoother.fit(X_for_smoother, y_for_smoother)
    >>> smoother.smoothed_y_
    array([-0.70710678, -0.70710678])

    >>> y_for_smoother = np.array([-1, -1, -1, 1, 1])
    >>> X_for_smoother = np.c_[
    ...     [1.0, 1.0, 1.0, 0.0, 0.0],
    ...     [0.0, 0.0, 0.0, 0.0, 0.0],
    ...     [0.5, 0.5, 0.5, 0.5, 0.5]]
    >>> smoother = WeightedMeanSmoother(prior_prediction=0)
    >>> smoother.fit(X_for_smoother, y_for_smoother)
    >>> smoothed_abs = (4. * 1 + 1 * 0) / (4 + 1)
    >>> np.testing.assert_allclose(smoother.smoothed_y_,
    ...     y_for_smoother * smoothed_abs)

    """

    def __init__(self, prior_prediction=None):
        self.prior_prediction = prior_prediction

    def fit(self, X_for_smoother, y):
        self.smoothed_y_ = utils.regularize_to_error_weighted_mean(y, X_for_smoother[:, 2], self.prior_prediction)


class WeightedMeanSmootherNeighbors(AbstractBinSmoother, PredictingBinValueMixin):
    """
    Smoother for regularizing one-dimensional bin values with uncertainties to
    the weighted mean of a window including only its left and right neigboring
    bins.
    """

    def fit(self, X_for_smoother, y):
        if len(y) < 3:
            self.smoothed_y_ = utils.regularize_to_error_weighted_mean(y, X_for_smoother[:, 2])
        else:
            self.smoothed_y_ = utils.regularize_to_error_weighted_mean_neighbors(y, X_for_smoother[:, 2], window_size=3)


class OrthogonalPolynomialSmoother(AbstractBinSmoother):
    """A polynomial fit that uses orthogonal polynomials as basis functions.

    Ansatz, see Blobel (http://www.desy.de/~blobel/eBuch.pdf)
    """

    def fit(self, X_for_smoother, y):
        x = np.ascontiguousarray(X_for_smoother[:, 0], dtype=np.float64)

        self.minimal_x = np.nanmin(x)
        self.maximal_x = np.nanmax(x)
        self.pp_, self.n_degrees_ = cy_orthogonal_poly_fit_equidistant(
            x,
            np.ascontiguousarray(y, dtype=np.float64),
            np.ascontiguousarray(X_for_smoother[:, 2], dtype=np.float64),
        )

        return self

    def predict(self, X):
        assert X.ndim == 2
        if not hasattr(self, "pp_"):
            raise ValueError("The {} has not been fitted!".format(self.__class__.__name__))
        x = np.ascontiguousarray(X[:, 0], dtype=np.float64)
        x[x < self.minimal_x] = self.minimal_x
        x[x > self.maximal_x] = self.maximal_x
        y = cy_apply_orthogonal_poly_fit_equidistant(x, self.pp_, self.n_degrees_)
        return y


def _choose_default_fit_function(order):
    if order == 1:
        return _seasonality_first_order
    elif order == 2:
        return _seasonality_second_order
    elif order == 3:
        return _seasonality_third_order
    else:
        raise ValueError(
            "The order parameter of the SeasonalSmoother "
            "has to be 1, 2 or 3. If this is not sufficient "
            "you can specify a `custom_fit_function`."
        )


def _seasonality_first_order(x, c_const, c_sin, c_cos):
    return c_const + c_sin * np.sin(x) + c_cos * np.cos(x)


def _seasonality_second_order(x, c_const, c_sin_1x, c_cos_1x, c_sin_2x, c_cos_2x):
    return c_const + c_sin_1x * np.sin(x) + c_cos_1x * np.cos(x) + c_sin_2x * np.sin(2 * x) + c_cos_2x * np.cos(2 * x)


def _seasonality_third_order(x, c_const, c_sin_1x, c_cos_1x, c_sin_2x, c_cos_2x, c_sin_3x, c_cos_3x):
    return (
        c_const
        + c_sin_1x * np.sin(x)
        + c_cos_1x * np.cos(x)
        + c_sin_2x * np.sin(2 * x)
        + c_cos_2x * np.cos(2 * x)
        + c_sin_3x * np.sin(3 * x)
        + c_cos_3x * np.cos(3 * x)
    )


class SeasonalSmoother(AbstractBinSmoother):
    """Seasonal Smoother of one-dimensional bins that applies an sinus/cosinus
    fit to smooth the bin values of y (as received from the profile function in
    the fit) and returns the polynomial values as the prediction.

    By default, SeasonalSmoother uses the the function  ``f(x) = c_const +
    c_sin * np.sin(x) + c_cos * np.cos(x)`` for the fit. The constant offset
    ``c_const`` is ignored if ``offset_tozero`` is set ot ``True`` (default).
    With higher ``order``, terms with frequency ``2*x`` and possibly ``3*x``
    are added.

    Instead of specifying an order, a custom fit-function can be supplied via
    the ``custom_fit_function`` argument.

    Parameters
    ----------
    offset_tozero : bool
        If ``True`` sets the constant offset to zero (default=True)

    order : int
        Order k of the series sin(k * x) + cos(k * x). Per default this
        is 1 and supported for order 2 and 3.

    custom_fit_function : function
        User defined function to define a custom fit function for the
        seasonal smoother. In this case the order parameter is ignored.

    fallback_value: float
        User defined parameter that is used when the fit does not converge
    Notes
    -----
    .. code::

        def my_custom_fit_function(x, *params):
            return ...

    **Required columns** in the ``X_for_smoother`` passed to :meth:`fit`:

    * column 0: ...
    * column 1: ignored
    * column 2: uncertainty of the average ``y`` in each bin

    These are provided by the following **profile functions**:

    * :class:`BetaMeanSigmaProfile`
    * :class:`MeanProfile`

    Attributes
    ----------

    frequency_: float
        Frequency calculated from the maximum binnumber.

    par_: numpy.ndarray
        Parameters from the fit.
    """

    def __init__(self, offset_tozero=True, order=1, custom_fit_function=None, fallback_value=0.0):
        self.offset_tozero = offset_tozero
        self.order = order
        self.custom_fit_function = custom_fit_function
        if custom_fit_function is None:
            self.fit_function = _choose_default_fit_function(order)
        else:
            self.fit_function = custom_fit_function
            if order != 1:
                warnings.warn("You specified a `custom_fit_function` thus " "the `order` parameter will be ignored!")
        self.converged = None
        self.fallback_value = fallback_value

    def transform_X(self, X):
        return 2 * np.pi * X[:, 0] * self.frequency_

    def fit(self, X_for_smoother, y):
        self.frequency_ = 1.0 / len(X_for_smoother)
        xdata = self.transform_X(X_for_smoother)
        try:
            self.par_, _ = scipy.optimize.curve_fit(self.fit_function, xdata=xdata, ydata=y, sigma=X_for_smoother[:, 2])
            if self.offset_tozero:
                self.par_[0] = 0.0
        except TypeError:
            warnings.warn("Seasonal Smoother did not converge. Fallback value is used")
            self.converged = False
        else:
            self.converged = True

        return self

    def predict(self, X):
        if self.converged or self.converged is None:
            if not hasattr(self, "par_"):
                raise ValueError("The {} has not been fitted!".format(self.__class__.__name__))
            return self.fit_function(self.transform_X(X), *self.par_)
        else:
            return np.ones(len(X), dtype=np.float64) * self.fallback_value


class SeasonalLassoSmoother(AbstractBinSmoother):
    """
    Seasonal Smoother of one-dimensional bins that applies a first order sine/cosine
    fit to smooth the bin values of y and returns the smoothed values as the prediction.

    Lasso Regularization is used to reduce the influence of outliers.

    If the number of days in the year where data is available remains below the configured
    threshold, the fallback value 0 is returned.

    Parameters
    ----------
    min_days : int
        The smoother is only fitted if more than `min_days` bins of data are available.
        Otherwise, it is set to fallback_mode and no smoothing is done. (default=300)

    Notes
    -----

    **Required columns** in the ``X_for_smoother`` passed to :meth:`fit`:

    * column 0: ...
    * column 1: The number of samples in the bin
    * column 2: uncertainty of the average ``y`` in each bin (ignored)

    Attributes
    ----------

    est: sklearn.estimator
        The fitted sklearn.LassoLarsIC estimator

    """

    def __init__(self, min_days=300):
        self.min_days = min_days
        self.max_bin = None
        self.est = None
        self.fallback_mode = None

    def prepare_X(self, x):
        x1 = 2 * np.pi * x / self.max_bin
        X = np.c_[
            np.sin(x1),
            np.cos(x1),
            -np.sin(x1),
            -np.cos(x1),
        ]
        return X

    def _fallback_predict(self, X):
        return np.zeros(len(X), dtype=np.float64)

    def predict(self, X):
        if not self.fallback_mode:
            if not self.est:
                raise ValueError("The {} has not been fitted!".format(self.__class__.__name__))
            Xt = self.prepare_X(X[:, 0])
            return self.est.predict(Xt)
        else:
            return self._fallback_predict(X)

    def fit(self, X_for_smoother, y):
        self.max_bin = X_for_smoother[-1, 0]
        days_with_samples = X_for_smoother[:, 1] > 0
        X_for_smoother = X_for_smoother[days_with_samples, :]
        y = y[days_with_samples]

        if len(y) < self.min_days:
            self.fallback_mode = True
        else:
            self.fallback_mode = False
            X = self.prepare_X(X_for_smoother[:, 0].copy())
            self.est = LassoLarsIC(positive=True, fit_intercept=False)
            self.est.fit(X, y)
        return self


class PriorExpectationMetaSmoother(AbstractBinSmoother, PredictingBinValueMixin):
    """Meta-Smoother that takes another one-dimensional smoother which
    results are additionally smoothed using
    :func:`cyclic_boosting.utils.regularize_to_prior_expectation`.

    :param prior_expectation: The prior dominate the regularized value if the
        uncertainties are large.
    :type prior_expectation: :class:`numpy.ndarray` (float64, dim=1) or float

    :param threshold: Threshold in terms of sigma. If the significance of a
        value:

        .. math::

            \text{sig}(x_i) = \frac{x_i - \text{prior_{expectation}_i}
            {\text{uncertainty}_i}

        is below the threshold, the prior expectation replaces the value.
    :type threshold: float
    """

    def __init__(self, est, prior_expectation, threshold=2.5):
        self.est = est
        self.prior_expectation = prior_expectation
        self.threshold = threshold

    def fit(self, X_for_smoother, y):
        self.est.fit(X_for_smoother.copy(), y)
        self.smoothed_y_ = utils.regularize_to_prior_expectation(
            self.est.predict(X_for_smoother.copy()),
            X_for_smoother[:, 2],
            self.prior_expectation,
            threshold=self.threshold,
        )
        return self


class UnivariateSplineSmoother(AbstractBinSmoother):
    r"""Smoother of one-dimensional bins that applies a univariate spline fit to
    smooth the bin values of y (as received from the profile function in the
    fit) and returns the spline values as the prediction.

    This class delegates to :class:`scipy.interpolate.UnivariateSpline`.

    **Required columns** in the ``X_for_smoother`` passed to :meth:`fit`:

    * column 0: ...
    * column 1: weight sum in each bin

    These are provided by the following **profile functions**:

    * :class:`MeanProfile`
    * :class:`BetaMeanSigmaProfile`

    :param k: Degree of the smoothing spline, must be ``<= 5``. The number of
        the data points must be larger than the spline degree.
    :type k: int

    :param s: Positive smoothing factor used to choose the number of knots.
        The number of knots will be increased until the smoothing condition is
        satisfied:

        .. math::

            \sum_i w_i \cdot (y_i - s(x_i))^2 \le s

        If `None` (default), ``s = len(w)`` is taken which should be a good
        value if ``1 / w[i]`` is an estimate of the standard deviation of
        ``y[i]``. If 0, the spline will interpolate through all data points.

        See :class:`scipy.interpolate.UnivariateSpline`.
    :type s: :obj:`float` or `None`

    **Estimated parameters**

    :param `spline_`: spline function that can be applied to ``x`` values
    :type `spline_`: :class:`scipy.interpolate.UnivariateSpline`
    """

    def __init__(self, k=3, s=None):
        self.k = k
        self.s = s

    def fit(self, X_for_smoother, y):
        self.spline_ = scipy.interpolate.UnivariateSpline(
            np.asarray(X_for_smoother[:, 0], dtype=np.float64),
            np.asarray(y, dtype=np.float64),
            w=np.asarray(X_for_smoother[:, 1], dtype=np.float64),
            k=self.k,
            s=self.s,
        )

        return self

    def predict(self, X):
        if not hasattr(self, "spline_"):
            raise ValueError("The {} has not been fitted!".format(self.__class__.__name__))
        return self.spline_(np.asarray(X[:, 0], dtype=np.float64))


class PolynomialSmoother(AbstractBinSmoother):
    r"""Least squares polynomial fit.

    Fit a polynomial ``p(x) = p[0] * x**deg + ... + p[deg]`` of degree `deg`
    to points `(x, y)`. Returns a vector of coefficients `p` that minimises
    the squared error.

    This class delegates to :class:`numpy.polyfit`.

    **Required columns** in the ``X_for_smoother`` passed to :meth:`fit`:

    * column 0: ...
    * column 1: ignored
    * column 2: uncertainty of the average ``y`` in each bin

    These are provided by the following **profile functions**:

    * :class:`MeanProfile`
    * :class:`BetaMeanSigmaProfile`

    :param k: Degree of the polynomial.
    :type k: int

    **Estimated parameters**

    :param `coefficients_`: Array of coefficients.
    :type `coefficients_`: :class:`numpy.ndarray` of length k
    """

    def __init__(self, k):
        self.k = k

    def fit(self, X_for_smoother, y):
        sigma = np.asarray(X_for_smoother[:, 2], dtype=np.float64)
        w = 1.0 / (sigma * sigma)
        self.coefficients_ = np.polyfit(
            np.asarray(X_for_smoother[:, 0], dtype=np.float64),
            np.asarray(y, dtype=np.float64),
            self.k,
            w=w,
        )
        return self

    def predict(self, X):
        if not hasattr(self, "coefficients_"):
            raise ValueError("The {} has not been fitted!".format(self.__class__.__name__))
        x = np.asarray(X[:, 0], dtype=np.float64)
        return np.polyval(self.coefficients_, x)


class LinearSmoother(PolynomialSmoother):
    r"""Least squares linear fit (special case of PolynomialSmoother).

    Fit a linear curve ``p(x) = p[0] * x + p[1]`` to points `(x, y)`.
    Returns a vector of coefficients `p` that minimises
    the squared error.

    This class delegates to :class:`numpy.polyfit`.

    **Required columns** in the ``X_for_smoother`` passed to :meth:`fit`:

    * column 0: ...
    * column 1: ignored
    * column 2: uncertainty of the average ``y`` in each bin

    :param fallback_when_negative_slope: Enforce non negative slope of linear fit. If negative
        slope is encountered, only the intercept is fitted. (Default=False)
    :type fallback_when_negative_slope: bool

    **Estimated parameters**

    :param `coefficients_`: Array of coefficients.
    :type `coefficients_`: :class:`numpy.ndarray` of length k
    """

    def __init__(self, fallback_when_negative_slope=False):
        self.fallback_when_negative_slope = fallback_when_negative_slope

    def fit(self, X_for_smoother, y):
        sigma = np.asarray(X_for_smoother[:, 2], dtype=np.float64)
        w = 1.0 / (sigma * sigma)
        x = X_for_smoother[:, 0]
        deg = 0 if len(x) <= 1 else 1
        self.coefficients_ = np.polyfit(
            np.asarray(x, dtype=np.float64),
            np.asarray(y, dtype=np.float64),
            deg,
            w=w,
        )
        if self.fallback_when_negative_slope is True and self.coefficients_[0] < 0:
            self.coefficients_ = np.polyfit(
                np.asarray(x, dtype=np.float64),
                np.asarray(y, dtype=np.float64),
                0,
                w=w,
            )
        return self


class LSQUnivariateSpline(AbstractBinSmoother):
    r"""
    Wrapper which delegates to :class:`scipy.interpolate.LSQUnivariateSpline`.

    Parameters
    ----------

    degree : int
        Degree of the smoothing spline, must be ``1 <= k <= 5``. The number of
        data points must be greater than the spline degree.

    interior_knots : list
        Interior knots of the spline. Must be in ascending order.
    """

    def __init__(self, interior_knots, degree=3):
        self.degree = degree
        self.interior_knots = interior_knots
        self.fitted_spline = None

    def fit(self, X_for_smoother, y):
        check_spline_degree = self.degree <= 5 and self.degree < len(X_for_smoother)
        if check_spline_degree:
            X = X_for_smoother.astype(np.float64)
            x = X[:, 0]
            y = y.astype(np.float64)
            t = np.asanyarray(self.interior_knots, dtype=np.float64)
            w = X[:, 1]
            k = self.degree
            self.fitted_spline = scipy.interpolate.LSQUnivariateSpline(x, y, t, w, k=k)
        else:
            raise ValueError(
                "Spline degree equals {}. Must be <= 5 and less" " than the available data points".format(self.degree)
            )
        return self

    def predict(self, X):
        if self.fitted_spline is None:
            raise ValueError("LSQUnivariateSpline has not been fitted!")
        return self.fitted_spline(X.astype(np.float64)[:, 0])


class IsotonicRegressor(AbstractBinSmoother):
    r"""
    Wrapper which delegates to :class:`sklearn.isotonic.IsotonicRegression`.
    Use if you expect a feature to be monotonic. An example is given in
    http://scikit-learn.org/stable/modules/isotonic.html .

    Parameters
    ----------
    increasing : boolean or string, optional, default: "auto"
        If boolean, whether or not to fit the isotonic regression with y
        increasing or decreasing.

        The string value "auto" determines whether y should
        increase or decrease based on the Spearman correlation estimate's
        sign.

    Note
    ----

    X requires the following columns.

    column0 :
        ...
    column1 :
        ignored
    column2 :
        weights
    """

    def __init__(self, increasing="auto"):
        self.est_ = None
        self.increasing = increasing

    def fit(self, X_for_smoother, y):
        X = X_for_smoother.astype(np.float64)
        sigma = X[:, 2]
        sigma = np.where(sigma > 0, sigma, 1e12)
        w = 1.0 / (sigma * sigma)
        self.est_ = sklearn.isotonic.IsotonicRegression(increasing=self.increasing, out_of_bounds="clip")
        self.est_.fit(X[:, 0], y.astype(np.float64), sample_weight=w)
        return self

    def predict(self, X):
        if self.est_ is None:
            raise ValueError("IsotonicRegressor has not been fitted!")
        return self.est_.predict(X.astype(np.float64)[:, 0])


__all__ = [
    "PredictingBinValueMixin",
    "BinValuesSmoother",
    "RegularizeToPriorExpectationSmoother",
    "RegularizeToOneSmoother",
    "WeightedMeanSmoother",
    "UnivariateSplineSmoother",
    "OrthogonalPolynomialSmoother",
    "SeasonalSmoother",
    "SeasonalLassoSmoother",
    "PolynomialSmoother",
    "LSQUnivariateSpline",
    "IsotonicRegressor",
]
