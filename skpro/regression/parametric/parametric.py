# -*- coding: utf-8 -*-

import numpy as np
import scipy.stats

if False:
    from sklearn.externals import six

import collections

from skpro.base.old_base import ProbabilisticEstimator, vectorvalued
from skpro.regression.parametric.estimators import Constant


class EstimatorManager:
    """Helper class that simplifies the estimator management

    Parameters
    ----------
    parent: subclass of sklearn.base.BaseEstimator
        Parent estimator object
    """

    def __init__(self, parent):
        self.estimators_ = collections.OrderedDict()
        self.parent = parent

    def register(self, name, estimator, selector=None):
        """Registers an estimator

        Parameters
        ----------
        name: str
            Name of the estimator
        estimator: Estimator object | string name of a registered estimator
            Instance of subclass of sklearn.base.BaseEstimator
        selector: callable(estimator, X) (optional)
            Defines how a prediction should be retrieved from an estimator

        Returns
        -------
        bool: True on success
        """

        if estimator is None:
            return False

        fitted = None
        if isinstance(estimator, str):
            # sanity checks for linking
            if not estimator in self.estimators_:
                raise AttributeError(
                    "Estimator %s you try to link is not registered" % estimator
                )

            if not callable(selector):
                raise ValueError("Selector has to be callable")

            # make it accessible on the parent
            setattr(self.parent, name, selector)
        else:
            if isinstance(estimator, (int, float)):
                # automatically wrap constants in Constant estimator
                estimator = Constant(estimator)
            # attach estimator
            setattr(estimator, "estimator", self.parent)
            # make it accessible on the parent
            setattr(self.parent, name, estimator)

        self.estimators_[name] = {
            "name": name,
            "estimator": estimator,
            "selector": selector,
            "fitted": fitted,
        }

        return True

    def get(self, index):
        return self.estimators_[index]

    def predict(self, name, X):
        if name not in self.estimators_:
            raise AttributeError("%s is not registered" % name)

        estimator = self.estimators_[name]

        if isinstance(estimator["estimator"], str):
            # link
            selector = self.estimators_[name]["selector"]
            return selector(self[estimator["estimator"]], X)
        else:
            return estimator["estimator"].predict(X)

    def set_params(self, name, **params):
        if name not in self.estimators_:
            raise AttributeError("%s is not registered" % name)

        estimator = self.estimators_[name]

        if isinstance(estimator["estimator"], str):
            # link
            selector = self.estimators_[name]["selector"]
            return selector.set_params(**params)
        else:
            return estimator["estimator"].set_params(**params)

    def fit(self, X, y):
        for name, estimator in self.estimators_.items():
            if not isinstance(estimator["estimator"], str):
                estimator["estimator"].fit(X, y)
                estimator["fitted"] = True

    def __len__(self):
        return len(self.estimators_)

    def __iter__(self):
        for name, item in self.estimators_.items():
            yield name, item

    def __getitem__(self, item):
        return self.estimators_[item]["estimator"]

    def __setitem__(self, key, value):
        self.estimators_[key]["estimator"] = value

    def __contains__(self, item):
        return item in self.estimators_


class ParametricEstimator(ProbabilisticEstimator):
    """
    Composite parametric prediction strategy.

    Uses classical estimators to predict the defining parameters of continuous distributions.

    Read more in the :ref:`User Guide <parametric>`.
    """

    class Distribution(ProbabilisticEstimator.Distribution):
        @vectorvalued
        def std(self):
            return self.estimator.estimators.predict("std", self.X)

        @vectorvalued
        def point(self):
            return self.estimator.estimators.predict("point", self.X)

        def pdf(self, x):
            """Probability density function

            Parameters
            ----------
            x

            Returns
            -------
            mixed  Density function evaluated at x
            """
            return self.estimator.shape_.pdf(
                x, loc=self[self.index].point(), scale=self[self.index].std()
            )

        def cdf(self, x):
            return self.estimator.shape_.cdf(
                x, loc=self[self.index].point(), scale=self[self.index].std()
            )

        def ppf(self, x):
            return self.estimator.shape_.ppf(
                x, loc=self[self.index].point(), scale=self[self.index].std()
            )

        def lp2(self):
            # Analytic solutions
            if self.estimator.shape == "norm":
                return 1 / (2 * self.std()[self.index] * np.sqrt(np.pi))
            elif self.estimator.shape == "laplace":
                return 1 / (2 * self.std()[self.index])
            elif self.estimator.shape == "uniform":
                return 1
            else:
                # fallback to numerical approximation
                super().lp2()

    def __init__(self, point=None, std=None, point_std=None, shape="norm"):
        """

        Parameters
        ----------
        point: str, num, estimator
        std
        point_std
        shape
        """
        self.estimators = EstimatorManager(self)
        self.shape = shape
        self.shape_ = getattr(scipy.stats, shape, False)

        if not self.shape_:
            raise ValueError(
                str(shape)
                + " is not a valid distribution (as defined in the scipy.stats module)"
            )

        if point_std is None:
            # default to mean baseline
            if point is None:
                point = Constant("mean(y)")
            if std is None:
                std = Constant("std(y)")

            self.estimators.register("point", point)
            self.estimators.register("std", std)
        else:
            if point is None:
                # set default point extractor
                def point(estimator, X):
                    return estimator.predict(X)

            if std is None:
                # set default std extractor
                def std(estimator, X):
                    return estimator.predict(X, return_std=True)[:, 1]

            self.estimators.register("point_std", point_std)
            self.estimators.register("point", "point_std", point)
            self.estimators.register("std", "point_std", std)

    def set_params(self, **params):
        if not params:
            # Simple optimisation to gain speed (inspect is slow)
            return self

        valid_params = self.get_params(deep=True)
        for key, value in six.iteritems(params):
            split = key.split("__", 1)
            if len(split) > 1:
                # nested objects case
                name, sub_name = split
                if name not in valid_params:
                    raise ValueError(
                        "Invalid parameter %s for estimator %s. "
                        "Check the list of available parameters "
                        "with `estimator.get_params().keys()`." % (name, self)
                    )
                if name in self.estimators:
                    self.estimators.set_params(name, **{sub_name: value})
            else:
                # simple objects case
                if key not in valid_params:
                    raise ValueError(
                        "Invalid parameter %s for estimator %s. "
                        "Check the list of available parameters "
                        "with `estimator.get_params().keys()`."
                        % (key, self.__class__.__name__)
                    )
                if key in self.estimators:
                    self.estimators[key] = value

        return self

    def fit(self, X, y):
        self.estimators.fit(X, y)

        return self

    def __str__(self, describer=str):
        if "point_std" in self.estimators:
            params = "point/std=" + describer(self.point_std)
        else:
            params = "point=" + describer(self.point) + ", std=" + describer(self.std)

        return self.shape + "(" + params + ")"

    def __repr__(self):
        return self.__str__(repr)
