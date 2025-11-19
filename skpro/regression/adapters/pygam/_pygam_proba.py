# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Adapter to pyGAM probabilistic regressors."""

__author__ = ["ravjot"]

import numpy as np
import pandas as pd
from sklearn import clone

from skpro.regression.base import BaseProbaRegressor
from skpro.utils.sklearn import prep_skl_df


class PyGAMAdapter(BaseProbaRegressor):
    """Adapter to pyGAM Generalized Additive Models for probabilistic regression.

    Wraps a pyGAM GAM model and constructs an skpro probabilistic regressor from it.

    The pyGAM GAM class allows selection of link and distribution parameters.
    This adapter maps pyGAM distributions to appropriate skpro distribution objects
    in the predict_proba method.

    Parameters
    ----------
    estimator : pygam.GAM, optional (default=None)
        pyGAM GAM estimator to wrap. If None, creates a GAM with default parameters.
        The distribution parameter in the GAM determines which skpro distribution
        will be returned in predict_proba.
    distribution : str, optional (default='normal')
        Distribution family for the GAM model. Common options include:
        - 'normal' or 'gaussian': Normal distribution
        - 'poisson': Poisson distribution
        - 'gamma': Gamma distribution
        - 'inv_gauss' or 'inverse_gaussian': Inverse Gaussian distribution
        If estimator is provided, this parameter is ignored
        (uses estimator's distribution).
    link : str, optional (default='identity')
        Link function for the GAM model. Common options include:
        - 'identity': identity link (for normal distribution)
        - 'log': log link (for poisson, gamma, inverse_gaussian)
        - 'inverse': inverse link (for gamma)
        If estimator is provided, this parameter is ignored.
    **gam_params : dict
        Additional parameters to pass to pyGAM GAM constructor.
        Only used if estimator is None.

    Attributes
    ----------
    estimator_ : pygam.GAM
        Fitted pyGAM GAM model

    Examples
    --------
    >>> from skpro.regression.adapters.pygam import PyGAMAdapter
    >>> from pygam import GAM
    >>> from sklearn.datasets import load_diabetes
    >>> from sklearn.model_selection import train_test_split
    >>>
    >>> X, y = load_diabetes(return_X_y=True, as_frame=True)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y)
    >>>
    >>> # Using GAM estimator directly
    >>> gam = GAM(distribution='normal', link='identity')
    >>> reg = PyGAMAdapter(estimator=gam)
    >>> reg.fit(X_train, y_train)
    >>> y_pred_proba = reg.predict_proba(X_test)
    >>>
    >>> # Using distribution parameter
    >>> reg_poisson = PyGAMAdapter(distribution='poisson', link='log')
    >>> reg_poisson.fit(X_train, y_train)
    >>> y_pred_proba = reg_poisson.predict_proba(X_test)
    """

    _tags = {
        "capability:multioutput": False,
        "capability:missing": False,
        "python_dependencies": "pygam",
    }

    def __init__(
        self, estimator=None, distribution="normal", link="identity", **gam_params
    ):
        self.estimator = estimator
        self.distribution = distribution
        self.link = link
        self.gam_params = gam_params
        super().__init__()

        # keep track of what distribution we're actually using
        # gets set in _fit based on the estimator or distribution param
        self._actual_distribution = None

    def _get_distribution_name(self, estimator):
        """Extract distribution name from pyGAM estimator.

        Parameters
        ----------
        estimator : pygam.GAM
            pyGAM GAM estimator

        Returns
        -------
        str
            Distribution name (normalized to common names)
        """
        # try to figure out what distribution this GAM is using
        # sometimes it's in .distribution, sometimes in ._distribution
        dist = getattr(estimator, "distribution", None)
        if dist is None:
            dist = getattr(estimator, "_distribution", None)

        # normalize it to a string we can work with
        if dist is None:
            dist_str = "normal"
        elif isinstance(dist, str):
            dist_str = dist.lower()
        else:
            # might be a class or object, try to get its name
            if hasattr(dist, "__name__"):
                dist_str = dist.__name__.lower()
            elif hasattr(dist, "__class__"):
                dist_str = dist.__class__.__name__.lower()
            else:
                dist_str = str(dist).lower()

        # map different naming variations to our standard names
        dist_mapping = {
            "normal": "normal",
            "gaussian": "normal",
            "poisson": "poisson",
            "gamma": "gamma",
            "inv_gauss": "inverse_gaussian",
            "inverse_gaussian": "inverse_gaussian",
            "inv_gaussian": "inverse_gaussian",
            # not really used for regression but keeping for completeness
            "binomial": "binomial",
        }

        # default to normal if we can't figure it out
        return dist_mapping.get(dist_str, "normal")

    def _fit(self, X, y):
        """Fit regressor to training data.

        Writes to self:
            Sets fitted model attributes ending in "_".

        Parameters
        ----------
        X : pandas DataFrame
            feature instances to fit regressor to
        y : pandas DataFrame, must be same length as X
            labels to fit regressor to

        Returns
        -------
        self : reference to self
        """
        from pygam import GAM

        # either clone the provided estimator or make a new one
        if self.estimator is not None:
            self.estimator_ = clone(self.estimator)
            self._actual_distribution = self._get_distribution_name(self.estimator_)
        else:
            # build a new GAM with the specified dist and link
            gam_params = {"distribution": self.distribution, "link": self.link}
            gam_params.update(self.gam_params)
            self.estimator_ = GAM(**gam_params)
            self._actual_distribution = self._get_distribution_name(self.estimator_)

        self._y_cols = y.columns

        # pyGAM wants numpy arrays, not DataFrames
        X_inner = prep_skl_df(X)
        if isinstance(X_inner, pd.DataFrame):
            X_inner = X_inner.values

        y_inner = prep_skl_df(y)
        if isinstance(y_inner, pd.DataFrame) and len(y_inner.columns) == 1:
            y_inner = y_inner.iloc[:, 0].values
        elif isinstance(y_inner, pd.DataFrame):
            y_inner = y_inner.values

        # now fit it
        self.estimator_.fit(X_inner, y_inner)

        return self

    def _predict(self, X):
        """Predict labels for data from features.

        State required:
            Requires state to be "fitted" = self.is_fitted=True

        Accesses in self:
            Fitted model attributes ending in "_"

        Parameters
        ----------
        X : pandas DataFrame, must have same columns as X in `fit`
            data to predict labels for

        Returns
        -------
        y : pandas DataFrame, same length as `X`, same columns as `y` in `fit`
            labels predicted for `X`
        """
        # convert to numpy for pyGAM
        X_inner = prep_skl_df(X)
        if isinstance(X_inner, pd.DataFrame):
            X_inner = X_inner.values

        # get predictions
        y_pred = self.estimator_.predict(X_inner)

        # make sure it's 2D so we can wrap it in a DataFrame
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)

        y_pred_df = pd.DataFrame(y_pred, index=X.index, columns=self._y_cols)
        return y_pred_df

    def _predict_proba(self, X):
        """Predict distribution over labels for data from features.

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_"

        Parameters
        ----------
        X : pandas DataFrame, must have same columns as X in `fit`
            data to predict labels for

        Returns
        -------
        y_pred : skpro BaseDistribution, same length as `X`
            labels predicted for `X`
        """
        # convert to numpy
        X_inner = prep_skl_df(X)
        if isinstance(X_inner, pd.DataFrame):
            X_inner = X_inner.values

        # get the mean predictions
        y_pred_mean = self.estimator_.predict(X_inner)
        if y_pred_mean.ndim == 1:
            y_pred_mean = y_pred_mean.reshape(-1, 1)

        # try to get standard errors if pyGAM provides them
        # this helps us build better distributions later
        y_pred_std = None
        try:
            if hasattr(self.estimator_, "prediction_intervals"):
                intervals = self.estimator_.prediction_intervals(X_inner, width=0.95)
                if intervals is not None:
                    # approximate std from the 95% interval
                    y_pred_std = (intervals[:, 1] - intervals[:, 0]) / (2 * 1.96)
                    if y_pred_std.ndim == 1:
                        y_pred_std = y_pred_std.reshape(-1, 1)
        except Exception:
            pass

        # figure out which distribution we need
        dist_name = self._actual_distribution

        index = X.index
        columns = self._y_cols

        # create the right skpro distribution based on what pyGAM is using
        if dist_name == "normal":
            from skpro.distributions.normal import Normal

            # try to get std if we don't have it yet
            if y_pred_std is None:
                try:
                    if hasattr(self.estimator_, "standard_error"):
                        y_pred_std = self.estimator_.standard_error(X_inner)
                        if y_pred_std.ndim == 1:
                            y_pred_std = y_pred_std.reshape(-1, 1)
                except Exception:
                    pass

            # make sure we have valid std values (no NaN/inf nonsense)
            if y_pred_std is None or (
                hasattr(y_pred_std, "__len__") and len(y_pred_std) == 0
            ):
                # default to a small positive value if we don't have std
                y_pred_std = np.full((len(index), len(columns)), 0.1)
            else:
                y_pred_std = np.array(y_pred_std)
                # replace any NaN or inf with something reasonable
                y_pred_std = np.nan_to_num(y_pred_std, nan=0.1, posinf=0.1, neginf=0.1)
                # keep it positive
                y_pred_std = np.clip(y_pred_std, a_min=1e-6, a_max=None)

            y_pred_dist = Normal(
                mu=pd.DataFrame(y_pred_mean, index=index, columns=columns),
                sigma=pd.DataFrame(y_pred_std, index=index, columns=columns),
                index=index,
                columns=columns,
            )

        elif dist_name == "poisson":
            from skpro.distributions.poisson import Poisson

            # poisson just needs the mean, nice and simple
            y_pred_dist = Poisson(
                mu=pd.DataFrame(y_pred_mean, index=index, columns=columns),
                index=index,
                columns=columns,
            )

        elif dist_name == "gamma":
            from skpro.distributions.gamma import Gamma

            # gamma needs shape and rate parameters
            # pyGAM gives us the mean, so we need to work backwards
            # if we have std, we can estimate the scale parameter
            if y_pred_std is None:
                # fallback to a default scale
                y_pred_scale = pd.DataFrame(0.1, index=index, columns=columns).values
            else:
                # approximate scale from variance
                y_pred_scale = (y_pred_std**2) / y_pred_mean
                y_pred_scale = y_pred_scale.clip(min=1e-6)

            # calculate shape parameter from mean and scale
            y_pred_shape = y_pred_mean / y_pred_scale
            y_pred_shape = y_pred_shape.clip(min=1e-6)

            # rate is just 1/scale
            y_pred_rate = 1.0 / y_pred_scale

            y_pred_dist = Gamma(
                alpha=pd.DataFrame(y_pred_shape, index=index, columns=columns),
                beta=pd.DataFrame(y_pred_rate, index=index, columns=columns),
                index=index,
                columns=columns,
            )

        elif dist_name == "inverse_gaussian":
            # inverse gaussian isn't in skpro yet,
            # so just use normal as an approximation
            from skpro.distributions.normal import Normal

            if y_pred_std is None:
                y_pred_std = np.full((len(index), len(columns)), 0.1)
            else:
                y_pred_std = np.array(y_pred_std)
                y_pred_std = np.nan_to_num(y_pred_std, nan=0.1, posinf=0.1, neginf=0.1)
                y_pred_std = np.clip(y_pred_std, a_min=1e-6, a_max=None)

            y_pred_dist = Normal(
                mu=pd.DataFrame(y_pred_mean, index=index, columns=columns),
                sigma=pd.DataFrame(y_pred_std, index=index, columns=columns),
                index=index,
                columns=columns,
            )

        else:
            # don't recognize this distribution, default to normal
            from skpro.distributions.normal import Normal

            if y_pred_std is None:
                y_pred_std = np.full((len(index), len(columns)), 0.1)
            else:
                y_pred_std = np.array(y_pred_std)
                y_pred_std = np.nan_to_num(y_pred_std, nan=0.1, posinf=0.1, neginf=0.1)
                y_pred_std = np.clip(y_pred_std, a_min=1e-6, a_max=None)

            y_pred_dist = Normal(
                mu=pd.DataFrame(y_pred_mean, index=index, columns=columns),
                sigma=pd.DataFrame(y_pred_std, index=index, columns=columns),
                index=index,
                columns=columns,
            )

        return y_pred_dist

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        from skbase.utils.dependencies import _check_soft_dependencies

        # if pygam isn't installed, return a marker so tests know to skip
        if not _check_soft_dependencies("pygam", severity="none"):
            return {"estimator": "runtests-no-pygam"}

        from pygam import GAM

        # return a few different parameter sets for testing
        param1 = {"distribution": "normal", "link": "identity"}
        param2 = {"distribution": "poisson", "link": "log"}
        param3 = {"estimator": GAM(distribution="normal", link="identity")}

        return [param1, param2, param3]
