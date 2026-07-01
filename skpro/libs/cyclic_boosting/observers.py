
import copy

import numpy as np

from skpro.libs.cyclic_boosting import utils


class BaseObserver(object):
    """
    Base class for observers

    Observers are used to extract information from cyclic boosting estimators
    that might be of further interest, but are not needed by the estimator in
    order to make predictions.
    """

    def observe_iterations(self, iteration, X, y, prediction, weights, estimator_state):
        """
        Called after each iteration of the algorithm.

        Parameters
        ----------
        iteration : int
            number of the iteration

        X : :class:`pandas.DataFrame` or :class:`numpy.ndarray`
            feature matrix

        y : numpy.ndarray
            target array

        prediction : numpy.ndarray
            current target prediction

        weights : numpy.ndarray
            target weights

        estimator_state : dict
            state of the estimator. See
            :meth:`cyclic_boosting.base.CyclicBoostingBase.get_state` for
            information on what will be passed here

        """

    def observe_feature_iterations(self, iteration, feature_i, X, y, prediction, weights, estimator_state):
        """
        Called after each feature has been processed as part of a full
        iteration of the algorithm.

        Parameters
        ----------
        iteration : int
            number of the iteration

        feature_i : int
            number of the feature that was processed

        X : :class:`pandas.DataFrame` or :class:`numpy.ndarray`
            feature matrix

        y : numpy.ndarray
            target array

        prediction : numpy.ndarray
            current target prediction

        weights : numpy.ndarray
            target weights

        estimator_state : dict
            state of the estimator. See
            :meth:`~cyclic_boosting.base.CyclicBoostingBase.get_state` for
            information on what will be passed here

        local_variables : dict
            local variables

        """


class PlottingObserver(BaseObserver):
    """
    Observer retrieving all information necessary to obtain analysis plots
    based on a cyclic boosting training.

    Instances of this class are intended to be passed as elements of the
    ``observers`` parameter to a cyclic boosting estimator, where each will
    gather information on a specific iteration. Afterwards, they can be passed
    to :func:`~cyclic_boosting.plots.plot_analysis`.

    Parameters
    ----------
    iteration : int
        The observer will save all necessary information for the analysis plots
        based on the state of the internal variables of the estimator
        after the given iteration has been calculated.
        Default is `-1`, which signifies the last iteration.

    """

    def __init__(self, iteration=-1):
        if iteration == 0:
            raise ValueError("This plotting observer only makes sense with iterations >= 1.")
        self.iteration = iteration

        self.features = None
        self.link_function = None
        self.n_feature_bins = None
        self.loss = list()
        self.factor_change = list()
        self.histograms = None

        self._fitted = False

    def observe_iterations(self, iteration, X, y, prediction, weights, estimator_state, delta=None, quantile=None):
        """Observe iterations in cyclic_boosting estimator to collect information for
        necessary for plots. This function is called in each major loop and once in the
        end.

        Parameters
        ----------

        iteration: int
            current major iteration of cyclic boosting loop

        X: pd.DataFrame or numpy.ndarray shape(n, k)
            feature matrix

        y: np.ndarray
            target array

        prediction: np.ndarray
            array of current cyclic boosting prediction

        weights: np.ndarray
            array of event weights

        estimator_state: dict
            state object of cyclic_boosting estimator
        """
        features = estimator_state["features"]

        if (iteration <= self.iteration and iteration != -1) or self.iteration == -1:
            self.loss.append(estimator_state["insample_loss"])
            if iteration != 0:
                # for iteration 0 there are no old fators to compare with
                self.factor_change.append(delta)

        if iteration == self.iteration:
            self._fitted = True
            self.features = copy.deepcopy(features)
            self.n_feature_bins = {feature.feature_group: feature.n_multi_bins_finite for feature in self.features}
            self.link_function = estimator_state["link_function"]
            self.histograms = calc_in_sample_histograms(y, prediction, weights, quantile)

    def observe_feature_iterations(self, iteration, feature_i, X, y, prediction, weights, estimator_state):
        """Observe iterations in cyclic_boosting estimator to collect information for
        necessary for plots. This function is called in each feature/minor loop.

        Parameters
        ----------

        iteration: int
            current major iteration number of cyclic boosting loop

        feature_i: int
            current minor iteration number of cyclic boosting loop

        X: pd.DataFrame or numpy.ndarray shape(n, k)
            feature matrix

        y: np.ndarray
            target array

        prediction: np.ndarray
            array of current cyclic boosting prediction

        weights: np.ndarray
            array of event weights

        estimator_state: dict
            state object of cyclic_boosting estimator
        """
        pass

    def check_fitted(self):
        if not self._fitted:
            raise ValueError("Observer not filled.")


def calc_in_sample_histograms(y, pred, weights, quantile=None):
    """
    Calculates histograms for use with diagonal plot.

    Parameters
    ----------
    y : numpy.ndarray
        truth

    pred : numpy.ndarray
        prediction

    weights: np.ndarray
        array of event weights

    Returns
    -------
    result : tuple
        Tuple consisting of:

        * means
        * bin_centers
        * errors
        * counts
    """
    nbins = 100
    bin_boundaries, bin_centers = utils.calc_linear_bins(pred, nbins)
    bin_numbers = utils.digitize(pred, bin_boundaries)
    means, _, counts, errors = utils.calc_means_medians(bin_numbers, y, weights)
    if quantile is not None:
        means = utils.calc_weighted_quantile(bin_numbers, y, weights, quantile)
        errors = None
    bin_centers = bin_centers[np.where(~np.isnan(means.reindex(np.arange(1, nbins + 1))))]
    # quantiles do not work for classification mode
    if np.isin(y, [0, 1]).all():
        return means, bin_centers, None, counts
    else:
        return means, bin_centers, errors, counts


__all__ = ["PlottingObserver", "BaseObserver", "calc_in_sample_histograms"]
