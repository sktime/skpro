"""
Multidimensional smoothers
"""

import numpy as np
import pandas as pd
from scipy import sparse

from skpro.libs.cyclic_boosting import utils
from skpro.libs.cyclic_boosting.smoothing.base import AbstractBinSmoother, SetNBinsMixin


class PredictingBinValueMixin(SetNBinsMixin):
    """Mixin for smoothers of multidimensional bins with
    :meth:`predict` returning the corresponding entry in the estimated parameter
    ``smoothed_y_`` which must have been calculated in :meth:`fit`.

    Please create the array ``smoothed_y_`` in your :meth:`fit` method.

    **Estimated parameters**

    :param `smoothed_y_`: the bin values for ``y`` (as received from the profile
        function in the fit) after some smoothing; a pseudobin
        for the missing values is not supported; the values are indexed in
        lexicographical ordering using the
    :type `smoothed_y_`: :class:`numpy.ndarray` (float64, shape `(n_bins,)`)

    :param `n_bins_`: number of bins in each dimension; it is permitted to
        append additional entries to this array. They are ignored in
        :meth:`predict` anyway.

        Please use :meth:`set_n_bins`
        to initialize this estimated parameter in your implementation of
        :meth:`fit`.
    :type `n_bins_`: :class:`numpy.ndarray` (int64, shape `(n_dims + x,)`)

    For examples, see the subclass :class:`BinValuesSmoother`.
    """

    def predict(self, X):
        if not hasattr(self, "n_bins_"):
            raise ValueError(
                'Please call the method "fit" before "predict" and '
                '"set_n_bins" in your "fit" method'
            )

        if self._bin_steps is None:
            self.n_bins_ = self.n_bins_[: X.shape[1]]
            self._bin_steps = utils.bin_steps(self.n_bins_)

        binnos = X
        binnos_round = np.asarray(np.floor(X), dtype=int)
        is_valid = np.all(
            np.isfinite(binnos)
            & (binnos >= 0)
            & (binnos_round < (self.n_bins_[None, :])),
            axis=1,
        )

        pred = utils.nans(len(binnos))
        pred[is_valid] = self.smoothed_y_[
            np.dot(binnos_round[is_valid], self._bin_steps[1:])
        ]

        return pred


class BinValuesSmoother(AbstractBinSmoother, PredictingBinValueMixin):
    """Smoother of multidimensional bins that outputs the saved bin values of
    y (as received from the profile function in the fit) as the prediction.

    This smoother only considers the first ``n_dim`` columns of
    ``X_for_smoother`` passed by the **profile function**. These columns are
    supposed to contain all the coordinates of the ``n_dim``-dimensional bin
    centers.

    **Estimated parameters**

    :param `smoothed_y_`: the bin values for ``y`` (as received from the
        profile function in the fit), in this case without any smoothing;
        a pseudobin for the missing values is not supported
    :type `smoothed_y_`: :class:`numpy.ndarray` (float64, shape `(n_bins,)`)

    :param `n_bins_`: see :class:`PredictingBinValueMixin`

    >>> from skpro.libs.cyclic_boosting import smoothing
    >>> X =    np.c_[[0., 0,  1,  1],
    ...              [0,  1,  0,  1],
    ...              [1,  1,  1,  1]]  # ignored
    >>> y = np.array([90, 80, 50, 40])

    >>> reg = smoothing.multidim.BinValuesSmoother()
    >>> assert reg.fit(X, y) is reg
    >>> assert np.allclose(reg.smoothed_y_, y)
    >>> X = np.c_[[1.1, 0.4, 0.0, 0.1, 2.],
    ...           [1.2, 1.1, 0.4, 0.4, 0.]]
    >>> reg.predict(X)
    array([ 40.,  80.,  90.,  90.,  nan])
    """

    elems = ["smoothed_y_", "bin_weights_"]

    def fit(self, X_for_smoother, y):
        self.set_n_bins(X_for_smoother)
        self.smoothed_y_ = y
        return self

    def __getstate__(self):
        """Return state values to be pickled."""
        state = self.__dict__.copy()
        for elem in self.elems:
            if elem in state and state[elem] is not None:
                state[elem] = sparse.csr_matrix(state[elem])
        return state

    def __setstate__(self, state):
        """Restore state from the unpickled state values."""
        for elem in self.elems:
            if elem in state and state[elem] is not None:
                state[elem] = state[elem].toarray()[0, :]
        self.__dict__.update(state)


class RegularizeToPriorExpectationSmoother(
    AbstractBinSmoother, PredictingBinValueMixin
):
    r"""Smoother of multidimensional bins regularizing values with uncertainties
    to a prior expectation.

    For details, see :func:`cyclic_boosting.utils.regularize_to_prior_expectation`.

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

    * columns 0 to``n_dim - 1``: coordinates of the bin centers (ignored here)
    * column ``n_dim``: ignored
    * column ``n_dim + 1``, which must be the last: uncertainty of the average
      ``y`` in each bin

    **Doctests**

    >>> from skpro.libs.cyclic_boosting import smoothing
    >>> y = np.array([0, 1, 2, 3])
    >>> X = np.c_[
    ...     [0, 0, 1, 1],
    ...     [0, 1, 0, 1],
    ...     [1]*4,
    ...     [0.1]*4]
    >>> est = smoothing.multidim.RegularizeToPriorExpectationSmoother(1.)
    >>> assert est.fit(X, y) is est
    >>> y_smoothed = est.predict(X[:, :2])
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
        self.set_n_bins(X_for_smoother)
        self.smoothed_y_ = utils.regularize_to_prior_expectation(
            y, X_for_smoother[:, -1], self.prior_expectation, threshold=self.threshold
        )
        return self


class RegularizeToOneSmoother(RegularizeToPriorExpectationSmoother):
    """Smoother for multidimensional bins regularizing values with
    uncertainties to the prior expectation 1.

    For details, see the superclass
    :class:`RegularizeToPriorExpectationSmoother` and the
    underlying function
    :func:`cyclic_boosting.utils.regularize_to_prior_expectation`.

    :param threshold: threshold in terms of sigma.
        If the significance of a factor
        is below the threshold, the global measurement replaces the factor.
        Internally :func:`cyclic_boosting.utils.regularize_to_prior_expectation`
        is used.
    :type threshold: float
    """

    def __init__(self, threshold=2.5):
        RegularizeToPriorExpectationSmoother.__init__(
            self, prior_expectation=1, threshold=threshold
        )


class WeightedMeanSmoother(AbstractBinSmoother, PredictingBinValueMixin):
    r"""Smoother for multidimensional bins regularizing values with
    uncertainties to the weighted mean.

    For details see :func:`cyclic_boosting.utils.regularize_to_error_weighted_mean`.

    :param prior_prediction: If the `prior_prediction` is specified, all values
        are regularized with it and not with the error weighted mean.
    :type prior_prediction: float

    >>> from skpro.libs.cyclic_boosting import smoothing
    >>> y_for_smoother = np.array([0.9, 0.9, 0.9, 1.8, 1.8, 0.4, 0.4])
    >>> X_for_smoother = np.c_[
    ...     [1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
    ...     [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0],
    ...     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ...     [0.05, 0.05, 0.05, 0.05, 0.15, 0.15, 0.05]]
    >>> smoother = smoothing.multidim.WeightedMeanSmoother()
    >>> smoother.fit(X_for_smoother, y_for_smoother)
    >>> smoother.smoothed_y_
    array([ 0.90096366,  0.90096366,  0.90096366,  1.79077293,  1.723854  ,
            0.45467402,  0.40662518])

    """

    def __init__(self, prior_prediction=None):
        self.prior_prediction = prior_prediction

    def fit(self, X_for_smoother, y):
        self.set_n_bins(X_for_smoother)
        self.smoothed_y_ = utils.regularize_to_error_weighted_mean(
            y, X_for_smoother[:, -1], self.prior_prediction
        )


class PriorExpectationMetaSmoother(AbstractBinSmoother, PredictingBinValueMixin):
    """Meta-Smoother that takes another multi-dimensional smoother which
    results are addionally smoothed using
    :func:`cyclic_boosting.utils.regularize_to_prior_expectation`.

    :param prior_expectation: The prior dominate the regularized value if the
        uncertainties are large.
    :type prior_expectation: :class:`numpy.ndarray` (float64, dim=1) or float

    :param threshold: Threshold in terms of sigma. If the significance of a
        value:

        .. math::

            \text{sig}(x_i) = \frac{x_i - \text{prior_{expectation}_i}}
            {\text{uncertainty}_i}

        is below the threshold, the prior expectation replaces the value.
    :type threshold: float
    """

    def __init__(self, est, prior_expectation, threshold=2.5):
        self.est = est
        self.prior_expectation = prior_expectation
        self.threshold = threshold

    def fit(self, X_for_smoother, y):
        """ """
        d = X_for_smoother.shape[1] - 2
        if d < 2:
            raise ValueError("You need at least 4 columns for multidim smoothing.")
        self.est.fit(X_for_smoother.copy(), y)

        self.set_n_bins(X_for_smoother)
        self.smoothed_y_ = utils.regularize_to_prior_expectation(
            self.est.predict(X_for_smoother[:, :d]),
            X_for_smoother[:, -1],
            self.prior_expectation,
            threshold=self.threshold,
        )
        return self


def _fit_est_on_group(X, n_group_columns, est):
    est = utils.clone(est)
    est.fit(X.iloc[:, n_group_columns:-1].values, X.iloc[:, -1].values)
    return est


def _predict_groups(x, gb, n_group_columns):
    try:
        est = gb.loc(axis=0)[x.name]
    except KeyError:
        p = np.nan
    else:
        p = est.predict(np.c_[x.values])
    return p


class GroupBySmoother(AbstractBinSmoother):
    """Multidimensional smoother that groups on the *first* k-1 columns
    of a k dimensional feature and smoothes a clone of the specified 1-dimensional
    smoother on each group.

    Parameters
    ----------

    est: :class:`AbstractBinSmoother`
        One-dimensional smoother whose clones are fitted on the grouped columns

    ndim: int
        Number of dimensions of the feature.

    index_weight_col: int
       Index of weight column. If specified, rows with zero weight are removed.
       If `None`, no rows are dropped.
    """

    @property
    def n_group_columns(self):
        return self.n_dim - 1

    def __init__(self, est, n_dim, index_weight_col=None):
        self.est = est
        self.n_dim = n_dim
        self.index_weight_col = index_weight_col

    def fit(self, X_for_smoother, y):
        if self.index_weight_col is not None:
            Xp = pd.DataFrame(X_for_smoother)
            gb_cols = list(range(self.n_dim - 1))
            gb = Xp.groupby(gb_cols)[self.n_dim].sum()
            Xp = Xp[gb_cols].merge(gb.reset_index(), how="left", on=gb_cols)
            mask = Xp[self.n_dim].values > 0
            X_for_smoother = X_for_smoother[mask]
            y = y[mask]
        X = pd.DataFrame(np.c_[X_for_smoother, y])
        self.group_cols = list(range(self.n_group_columns))
        self.gb = X.groupby(self.group_cols, sort=False).apply(
            _fit_est_on_group, self.n_group_columns, self.est
        )

    def predict(self, X):
        X = pd.DataFrame(X)
        pred = X.groupby(self.group_cols, sort=False)[X.columns[-1]].transform(
            _predict_groups, self.gb, self.n_group_columns
        )
        return pred.values


class GroupBySmootherCB(GroupBySmoother):
    """GroupBySmoother for cyclic boosting.
    Samples with zero weights are dropped to save memory.
    """

    def __init__(self, est, n_dim):
        GroupBySmoother.__init__(self, est, n_dim, -2)


class Neutralize2DMetaSmoother(AbstractBinSmoother, PredictingBinValueMixin):
    """Meta-Smoother that takes another multi-dimensional smoother which
    inputs are smoothed by neutralize_one_dim_influence.

    This means, the influence of one-dimensional features on two-dimensional features
    is removed by iteratively projecting the two-dimensional factor matrix onto one dimension
    and substract this weighted by the uncertainties from the matrix.

    In a cyclic boosting model this prevents two-dimensional features to include the effect which
    should be learned by the one-dimension features.

    :func:`cyclic_boosting.utils.neutralize_one_dim_influence`.
    """

    def __init__(self, est):
        self.est = est

    def fit(self, X_for_smoother, y):
        """Fit the transformer to training samples."""
        d = X_for_smoother.shape[1] - 2
        if d < 2:
            raise ValueError("You need at least 4 columns for multidim smoothing.")

        # We get the y values as 1d array, hence we have to reshape it into
        # the correct 2d array
        new_shape = np.max(X_for_smoother[:, :2], axis=0).astype(np.int64) + 1
        values = np.reshape(y, new_shape)
        uncertainties = np.reshape(X_for_smoother[:, -1], new_shape)
        neutralized_values = utils.neutralize_one_dim_influence(values, uncertainties)

        y = np.reshape(neutralized_values, np.prod(new_shape))
        self.est.fit(X_for_smoother.copy(), y)

        self.set_n_bins(X_for_smoother)
        self.smoothed_y_ = self.est.predict(X_for_smoother[:, :d])
        return self


__all__ = [
    "PredictingBinValueMixin",
    "BinValuesSmoother",
    "RegularizeToPriorExpectationSmoother",
    "RegularizeToOneSmoother",
    "WeightedMeanSmoother",
    "Neutralize2DMetaSmoother",
    "GroupBySmoother",
    "GroupBySmootherCB",
]
