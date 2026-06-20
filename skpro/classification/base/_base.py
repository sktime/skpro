"""Base class for probabilistic classification."""
# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)

import numpy as np
import pandas as pd
from skbase.utils.dependencies import _check_estimator_deps, _check_soft_dependencies

from skpro.base import BaseEstimator
from skpro.datatypes import check_is_error_msg, check_is_mtype, convert

# allowed input mtypes
# include mtypes that are core dependencies
ALLOWED_MTYPES = [
    "pd_DataFrame_Table",
    "pd_Series_Table",
    "numpy1D",
    "numpy2D",
]
# include polars eager table if the soft dependency is installed
if _check_soft_dependencies(["polars", "pyarrow"], severity="none"):
    ALLOWED_MTYPES.append("polars_eager_table")


class BaseProbaClassifier(BaseEstimator):
    """Base class for probabilistic supervised classifiers."""

    _tags = {
        "object_type": "classifier_proba",  # type of object, e.g., "classifier_proba"
        "estimator_type": "classifier_proba",  # type of estimator, e.g., "classifier_proba"
        "capability:survival": False,
        "capability:multioutput": False,
        "capability:missing": True,
        "capability:update": False,
        "X_inner_mtype": "pd_DataFrame_Table",
        "y_inner_mtype": "pd_DataFrame_Table",
        "C_inner_mtype": "pd_DataFrame_Table",
    }

    def __init__(self):
        super().__init__()
        _check_estimator_deps(self)

        self._X_converter_store = {}
        self._y_converter_store = {}
        self._C_converter_store = {}

    def __rmul__(self, other):
        """Magic * method, return (left) concatenated Pipeline.

        Implemented for `other` being a transformer, otherwise returns `NotImplemented`.
        """
        from skpro.classification.compose._pipeline import Pipeline

        # we wrap self in a pipeline, and concatenate with the other
        if hasattr(other, "transform"):
            return other * Pipeline([self])
        else:
            return NotImplemented

    def fit(self, X, y, C=None):
        """Fit classifier to training data."""
        capa_surv = self.get_tag("capability:survival")

        check_ret = self._check_X_y(X, y, C, return_metadata=True)

        # get inner X, y, C
        X_inner = check_ret["X_inner"]
        y_inner = check_ret["y_inner"]
        if capa_surv:
            C_inner = check_ret["C_inner"]

        # remember metadata
        self._X_metadata = check_ret["X_metadata"]
        self._y_metadata = check_ret["y_metadata"]
        if capa_surv:
            self._C_metadata = check_ret["C_metadata"]

        # set fitted flag to True
        self._is_fitted = True

        if not capa_surv:
            return self._fit(X_inner, y_inner)
        else:
            return self._fit(X_inner, y_inner, C=C_inner)

    def _fit(self, X, y, C=None):
        """Fit classifier to training data."""
        raise NotImplementedError

    def update(self, X, y, C=None):
        """Update classifier with a new batch of training data."""
        capa_online = self.get_tag("capability:update")
        capa_surv = self.get_tag("capability:survival")

        if not capa_online:
            return self

        check_ret = self._check_X_y(X, y, C, return_metadata=True)

        X_inner = check_ret["X_inner"]
        y_inner = check_ret["y_inner"]
        if capa_surv:
            C_inner = check_ret["C_inner"]

        if not capa_surv:
            return self._update(X_inner, y_inner)
        else:
            return self._update(X_inner, y_inner, C=C_inner)

    def _update(self, X, y, C=None):
        """Update classifier with a new batch of training data."""
        raise NotImplementedError

    def predict(self, X):
        """Predict labels for data from features."""
        X = self._check_X(X)

        y_pred = self._predict(X)

        # output conversion - back to mtype seen in fit
        y_pred = convert(
            y_pred,
            from_type=self.get_tag("y_inner_mtype"),
            to_type=self._y_metadata["mtype"],
            as_scitype="Table",
            store=self._y_converter_store,
        )

        return y_pred

    def _predict(self, X):
        """Predict labels for data from features."""
        implements_proba = self._has_implementation_of("_predict_proba")

        if not implements_proba:
            raise NotImplementedError

        pred_proba = self._predict_proba(X=X)
        if hasattr(pred_proba, "mode"):
            return pred_proba.mode()
        else:
            raise NotImplementedError(
                "Default classification predict requires predict_proba to return "
                "a distribution with a 'mode' method."
            )

    def predict_proba(self, X):
        """Predict distribution over labels for data from features."""
        X = self._check_X(X)

        y_pred = self._predict_proba(X)
        return y_pred

    def _predict_proba(self, X):
        """Predict distribution over labels for data from features."""
        raise NotImplementedError

    def predict_interval(self, X=None, coverage=0.90):
        """Compute/return interval predictions."""
        self.check_is_fitted()

        coverage = self._check_alpha(coverage, name="coverage")
        X_inner = self._check_X(X=X)

        pred_int = self._predict_interval(X=X_inner, coverage=coverage)
        return pred_int

    def _predict_interval(self, X, coverage):
        """Compute/return interval predictions."""
        raise NotImplementedError

    def predict_quantiles(self, X=None, alpha=None):
        """Compute/return quantile predictions."""
        self.check_is_fitted()

        if alpha is None:
            alpha = [0.05, 0.95]
        alpha = self._check_alpha(alpha, name="alpha")
        X_inner = self._check_X(X=X)

        quantiles = self._predict_quantiles(X=X_inner, alpha=alpha)
        return quantiles

    def _predict_quantiles(self, X, alpha):
        """Compute/return quantile predictions."""
        raise NotImplementedError

    def predict_var(self, X=None):
        """Compute/return variance predictions."""
        self.check_is_fitted()

        X_inner = self._check_X(X=X)

        pred_var = self._predict_var(X=X_inner)
        return pred_var

    def _predict_var(self, X):
        """Compute/return variance predictions."""
        raise NotImplementedError

    def _check_X_y(self, X, y, C=None, return_metadata=False):
        X_inner, X_metadata = self._check_X(X, return_metadata=True)
        y_inner, y_metadata = self._check_y(y)

        len_X = X_metadata["n_instances"]
        len_y = y_metadata["n_instances"]

        if len_X != "NA" and len_y != "NA" and not len_X == len_y:
            raise ValueError(
                f"X and y in fit of {self} must have same number of rows, "
                f"but X had {len_X} rows, and y had {len_y} rows"
            )

        capa_surv = self.get_tag("capability:survival")

        if capa_surv and C is not None:
            C_inner, C_metadata = self._check_C(C)
            len_C = C_metadata["n_instances"]
            if len_C != "NA" and not len_C == len_y:
                raise ValueError(
                    f"X, y, C in fit of {self} must have same number of rows, "
                    f"but C had {len_C} rows, and y had {len_y} rows"
                )
        else:
            C_inner = None
            C_metadata = None

        if hasattr(X_inner, "index") and not hasattr(y, "index"):
            if isinstance(y_inner, (pd.DataFrame, pd.Series)):
                y_inner.index = X_inner.index

        if hasattr(X_inner, "index") and C is not None and not hasattr(C, "index"):
            if isinstance(C_inner, (pd.DataFrame, pd.Series)):
                C_inner.index = X_inner.index

        ret_dict = {
            "X_inner": X_inner,
            "y_inner": y_inner,
        }

        if return_metadata:
            ret_dict["X_metadata"] = X_metadata
            ret_dict["y_metadata"] = y_metadata
        if capa_surv:
            ret_dict["C_inner"] = C_inner
        if return_metadata and capa_surv:
            ret_dict["C_metadata"] = C_metadata

        return ret_dict

    def _check_X(self, X, return_metadata=False):
        if return_metadata:
            req_metadata = ["n_instances", "feature_names"]
        else:
            req_metadata = ["feature_names"]
        valid, msg, X_metadata = check_is_mtype(
            X,
            ALLOWED_MTYPES,
            "Table",
            return_metadata=req_metadata,
            var_name="X",
            msg_return_dict="list",
        )
        X_feature_names = X_metadata["feature_names"]
        if not isinstance(X_feature_names, np.ndarray):
            X_feature_names = np.array(X_feature_names)

        if not valid:
            check_is_error_msg(msg, var_name="X", raise_exception=True)

        if hasattr(self, "feature_names_in_"):
            msg_feat = (
                f"Error in {type(self).__name__}: "
                "X in predict methods must have same columns as X in fit, "
                f"columns in fit were {self.feature_names_in_}, "
                f"but in predict found X feature names = {X_feature_names}"
            )
            if not len(X_feature_names) == len(self.feature_names_in_):
                raise ValueError(msg_feat)
            if not (X_feature_names == self.feature_names_in_).all():
                raise ValueError(msg_feat)
        else:
            self.feature_names_in_ = X_feature_names
            self.n_features_in_ = len(X_feature_names)

        X_inner_mtype = self.get_tag("X_inner_mtype")
        X_inner = convert(
            X,
            from_type=X_metadata["mtype"],
            to_type=X_inner_mtype,
            as_scitype="Table",
            store=self._X_converter_store,
        )

        if return_metadata:
            return X_inner, X_metadata
        else:
            return X_inner

    def _check_y(self, y):
        req_metadata = ["n_instances", "mtype"]
        valid, msg, y_metadata = check_is_mtype(
            y,
            ALLOWED_MTYPES,
            "Table",
            return_metadata=req_metadata,
            var_name="y",
            msg_return_dict="list",
        )
        if not valid:
            check_is_error_msg(msg, var_name="y", raise_exception=True)

        y_inner_mtype = self.get_tag("y_inner_mtype")
        y_inner = convert(
            y,
            from_type=y_metadata["mtype"],
            to_type=y_inner_mtype,
            as_scitype="Table",
            store=self._y_converter_store,
        )
        return y_inner, y_metadata

    def _check_C(self, C):
        req_metadata = ["n_instances", "mtype"]
        valid, msg, C_metadata = check_is_mtype(
            C,
            ALLOWED_MTYPES,
            "Table",
            return_metadata=req_metadata,
            var_name="C",
            msg_return_dict="list",
        )
        if not valid:
            check_is_error_msg(msg, var_name="C", raise_exception=True)

        C_inner_mtype = self.get_tag("C_inner_mtype")
        C_inner = convert(
            C,
            from_type=C_metadata["mtype"],
            to_type=C_inner_mtype,
            as_scitype="Table",
            store=self._C_converter_store,
        )
        return C_inner, C_metadata

    def _check_alpha(self, alpha, name="alpha"):
        if isinstance(alpha, (int, float)):
            alpha = [alpha]
        elif isinstance(alpha, list):
            pass
        else:
            raise TypeError(
                f"{name} must be a float or a list of floats, "
                f"but found {type(alpha)}"
            )

        for a in alpha:
            if not isinstance(a, (float, int)):
                raise TypeError(
                    f"{name} must be a float or a list of floats, "
                    f"but found {type(a)} in list"
                )
            if not 0 < a < 1:
                raise ValueError(
                    f"{name} must be strictly between 0 and 1, "
                    f"but found {a}"
                )

        return alpha
