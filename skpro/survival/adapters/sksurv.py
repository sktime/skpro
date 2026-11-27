# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Adapter for scikit-survival models."""

__all__ = ["_SksurvAdapter"]
__author__ = ["fkiraly"]

import numpy as np
import pandas as pd

from skpro.distributions.empirical import Empirical
from skpro.survival.adapters._common import _get_fitted_params_default_safe
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
        # CI and test flags
        # -----------------
        "tests:vm": True,  # requires its own test VM to run
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
            Convention: C=0 (uncensored), C=1 (censored).
            Internally converted to scikit-survival convention:
            delta=True (uncensored), delta=False (censored).

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
        # some fitted parameters are properties and may raise exceptions
        # for example, AIC_ and AIC_partial_ of CoxPHFitter
        # to avoid this, we use a safe getter
        EXCEPTED_FITTED_PARAMS = ["n_features_in", "feature_names_in"]
        sksurv_fitted_params = _get_fitted_params_default_safe(sksurv_est)
        for k, v in sksurv_fitted_params.items():
            if k not in EXCEPTED_FITTED_PARAMS:
                setattr(self, f"{k}_", v)

        return self

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

        # predict on X - shape (n_samples, n_times)
        sksurv_survf = sksurv_est.predict_survival_function(X, return_array=True)
        times = sksurv_est.unique_times_

        # 1. Handle Initial Mass (S(0) = 1.0)
        # We prepend 1.0 to the survival curves to capture the first drop
        ones = np.ones((sksurv_survf.shape[0], 1))
        surv_extended = np.hstack([ones, sksurv_survf])
        
        # 2. Calculate Weights via negative difference
        # -np.diff captures the 'drop' in survival, which is the probability mass
        weights = -np.diff(surv_extended, axis=1)
        
        # 3. Handle Tail Mass (Censoring/Remaining mass)
        # If the survival function doesn't reach 0, the remaining mass 
        # is assigned to infinity (representing 'at some point in the future')
        tail_mass = sksurv_survf[:, -1:]
        final_weights = np.hstack([weights, tail_mass])
        
        # 4. Align Times
        # We append np.inf as the timestamp for the tail mass
        final_times = np.append(times, np.inf)
        
        # 5. Reshape for Empirical distribution
        # The Empirical distribution expects (n_samples * n_points) format for spl
        n_samples = len(X)
        n_points = len(final_times)
        
        # Create a MultiIndex for the weights and times
        mi = pd.MultiIndex.from_product([X.index, range(n_points)]).swaplevel()
        
        # Flatten weights and repeat times for each sample
        weights_flat = final_weights.flatten()
        times_repeated = np.tile(final_times, n_samples)
        
        times_df = pd.DataFrame(times_repeated, index=mi, columns=self._y_cols)
        weights_ser = pd.Series(weights_flat, index=mi)

        dist = Empirical(
            spl=times_df, 
            weights=weights_ser, 
            index=X.index, 
            columns=self._y_cols
        )

        return dist
