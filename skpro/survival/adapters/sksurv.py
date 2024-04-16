# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements adapter for scikit-survival models."""

__all__ = ["_SksurvAdapter"]
__author__ = ["fkiraly"]

import numpy as np
import pandas as pd

from skpro.distributions.empirical import Empirical
from skpro.utils.sklearn import prep_skl_df


class _SksurvAdapter:
    """Mixin adapter class for sksurv models."""

    _tags = {
        # packaging info
        # --------------
        "authors": ["fkiraly"],
        "python_dependencies": ["scikit-survival"],
        "python_dependencies_alias": {"scikit-survival": "sksurv"},
        "license_type": "copyleft",
        # capability tags
        # ---------------
        "X_inner_mtype": "pd_DataFrame_Table",
        "y_inner_mtype": "pd_DataFrame_Table",
        "C_inner_mtype": "pd_DataFrame_Table",
        "capability:multioutput": False,
    }

    # defines the name of the attribute containing the sksurv estimator
    _estimator_attr = "_estimator"

    def _get_sksurv_class(self):
        """Abstract method to get sksurv class.

        should import and return sksurv class
        """
        # from sksurv import SksurvClass
        #
        # return SksurvClass
        raise NotImplementedError("abstract method")

    def _get_sksurv_object(self):
        """Abstract method to initialize sksurv object.

        The default initializes result of _get_sksurv_class
        with self.get_params.
        """
        cls = self._get_sksurv_class()
        return cls(**self.get_params())

    def _init_sksurv_object(self):
        """Abstract method to initialize sksurv object and set to _estimator_attr.

        The default writes the return of _get_sksurv_object to
        the attribute of self with name _estimator_attr
        """
        cls = self._get_sksurv_object()
        setattr(self, self._estimator_attr, cls)
        return getattr(self, self._estimator_attr)

    def _fit(self, X, y, C=None):
        """Fit estimator training data.

        Parameters
        ----------
        X : pd.DataFrame
            Training features
        y: pd.Series
            Training labels
        C: pd.Series, optional (default=None)
            Censoring information for survival analysis.

        Returns
        -------
        self: reference to self
            Fitted estimator.
        """
        sksurv_est = self._init_sksurv_object()

        if C is None:
            C = pd.DataFrame(np.zeros(len(y)), index=y.index, columns=y.columns)

        # input conversion
        X = X.astype("float")  # sksurv insists on float dtype
        X = prep_skl_df(X)
        y_np = y.iloc[:, 0].values  # we know univariate due to tag
        C_np = C.iloc[:, 0].values
        C_np_bool = C_np == 0  # sksurv uses "delta" indicator, 0 = censored
        # this is the opposite of skpro ("censoring" indicator), where 1 = censored

        y_sksurv = list(zip(C_np_bool, y_np))
        y_sksurv = np.array(y_sksurv, dtype=[("delta", "?"), ("time", "<f8")])

        self._y_cols = y.columns  # remember column names for later

        # fit sksurv estimator
        sksurv_est.fit(X, y_sksurv)

        # write fitted params to self
        EXCEPTED_FITTED_PARAMS = ["n_features_in", "feature_names_in"]
        sksurv_fitted_params = self._get_fitted_params_default_safe(sksurv_est)
        for k, v in sksurv_fitted_params.items():
            if k not in EXCEPTED_FITTED_PARAMS:
                setattr(self, f"{k}_", v)

        return self

    def _get_fitted_params_default_safe(self, obj=None):
        """Obtain fitted params of object, per sklearn convention.

        Same as _get_fitted_params_default, but with exception handling.

        This is since in sksurv, feature_importances_ is a property
        and may raise an exception if the estimator does not have it.

        Parameters
        ----------
        obj : any object, optional, default=self

        Returns
        -------
        fitted_params : dict with str keys
            fitted parameters, keyed by names of fitted parameter
        """
        obj = obj if obj else self

        # default retrieves all self attributes ending in "_"
        # and returns them with keys that have the "_" removed
        #
        # get all attributes ending in "_", exclude any that start with "_" (private)
        fitted_params = [
            attr for attr in dir(obj) if attr.endswith("_") and not attr.startswith("_")
        ]

        def hasattr_safe(obj, attr):
            try:
                if hasattr(obj, attr):
                    getattr(obj, attr)
                    return True
            except Exception:
                return False

        # remove the "_" at the end
        fitted_param_dict = {
            p[:-1]: getattr(obj, p) for p in fitted_params if hasattr_safe(obj, p)
        }

        return fitted_param_dict

    def _predict_proba(self, X):
        """Predict_proba method adapter.

        Parameters
        ----------
        X : pd.DataFrame
            Features to predict on.

        Returns
        -------
        skpro Empirical distribution
        """
        sksurv_est = getattr(self, self._estimator_attr)

        # input conversion
        X = X.astype("float")  # sksurv insists on float dtype
        X = prep_skl_df(X)

        # predict on X
        sksurv_survf = sksurv_est.predict_survival_function(X, return_array=True)

        times = sksurv_est.unique_times_[:-1]

        nt = len(times)
        mi = pd.MultiIndex.from_product([X.index, range(nt)]).swaplevel()

        times_val = np.repeat(times, repeats=len(X))
        times_df = pd.DataFrame(times_val, index=mi, columns=self._y_cols)

        weights = -np.diff(sksurv_survf, axis=1).flatten()
        weights_df = pd.Series(weights, index=mi)

        dist = Empirical(
            spl=times_df, weights=weights_df, index=X.index, columns=self._y_cols
        )

        return dist
