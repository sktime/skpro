"""Base class for distribution fitters."""
# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)

from skbase.utils.dependencies import _check_estimator_deps, _check_soft_dependencies

from skpro.base import BaseEstimator
from skpro.datatypes import check_is_error_msg, check_is_mtype, convert

__author__ = ["patelchaitany"]

__all__ = ["BaseDistFitter"]

ALLOWED_MTYPES = [
    "pd_DataFrame_Table",
    "pd_Series_Table",
    "numpy1D",
    "numpy2D",
]
if _check_soft_dependencies(["polars", "pyarrow"], severity="none"):
    ALLOWED_MTYPES.append("polars_eager_table")


class BaseDistFitter(BaseEstimator):
    """Base class for distribution fitters.

    A distribution fitter produces a single scalar distribution for a
    tabular dataset. It estimates distribution parameters from data
    and returns a fitted scalar distribution object.

    Concrete implementations must implement ``_fit`` and ``_proba``.
    """

    _tags = {
        "object_type": "distfitter",
        "estimator_type": "distfitter",
        "capability:survival": False,
        "X_inner_mtype": "pd_DataFrame_Table",
        "C_inner_mtype": "pd_DataFrame_Table",
    }

    def __init__(self):
        super().__init__()
        _check_estimator_deps(self)

        self._X_converter_store = {}
        self._C_converter_store = {}

    def fit(self, X, C=None):
        """Fit distribution to data.

        Writes to self:
            Sets fitted model attributes ending in "_".

        Changes state to "fitted" = sets is_fitted flag to True.

        Parameters
        ----------
        X : pandas DataFrame, pandas Series, or numpy array
            Data to fit the distribution to.
        C : pandas DataFrame, optional (default=None)
            Censoring indicator for survival analysis.
            Ignored unless ``capability:survival`` tag is True.

        Returns
        -------
        self : reference to self
        """
        capa_surv = self.get_tag("capability:survival")

        X_inner, X_metadata = self._check_X(X, return_metadata=True)

        self._X_metadata = X_metadata

        if capa_surv and C is not None:
            C_inner, C_metadata = self._check_C(C)
            self._C_metadata = C_metadata
        else:
            C_inner = None

        self._is_fitted = True

        if not capa_surv or C_inner is None:
            return self._fit(X_inner)
        else:
            return self._fit(X_inner, C=C_inner)

    def _fit(self, X, C=None):
        """Fit distribution to data (inner method).

        Writes to self:
            Sets fitted model attributes ending in "_".

        Parameters
        ----------
        X : pandas DataFrame
            Data to fit the distribution to.
        C : pandas DataFrame, optional (default=None)
            Censoring indicator.

        Returns
        -------
        self : reference to self
        """
        raise NotImplementedError

    def proba(self):
        """Return scalar distribution fitted to data seen in ``fit``.

        State required:
            Requires state to be "fitted".

        Returns
        -------
        dist : skpro BaseDistribution
            Scalar distribution fitted to data passed in ``fit``.
        """
        self.check_is_fitted()
        return self._proba()

    def _proba(self):
        """Return scalar distribution fitted to data (inner method).

        Returns
        -------
        dist : skpro BaseDistribution
            Scalar distribution fitted to data passed in ``fit``.
        """
        raise NotImplementedError

    def _check_X(self, X, return_metadata=False):
        """Check and convert X to inner mtype.

        Parameters
        ----------
        X : object
            Data to check and convert.
        return_metadata : bool, optional, default=False
            Whether to return metadata.

        Returns
        -------
        X_inner : pandas DataFrame
            X converted to X_inner_mtype.
        X_metadata : dict, only returned if return_metadata=True
            Metadata of X.
        """
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

        if not valid:
            check_is_error_msg(msg, var_name="X", raise_exception=True)

        X_inner_mtype = self.get_tag("X_inner_mtype")
        X_inner = convert(
            obj=X,
            from_type=X_metadata["mtype"],
            to_type=X_inner_mtype,
            as_scitype="Table",
            store=self._X_converter_store,
        )

        if return_metadata:
            return X_inner, X_metadata
        else:
            return X_inner

    def _check_C(self, C):
        """Check and convert censoring indicator C to inner mtype.

        Parameters
        ----------
        C : object
            Censoring indicator to check and convert.

        Returns
        -------
        C_inner : pandas DataFrame
            C converted to C_inner_mtype.
        C_metadata : dict
            Metadata of C.
        """
        valid, msg, metadata = check_is_mtype(
            C,
            ALLOWED_MTYPES,
            "Table",
            return_metadata=["n_instances"],
            var_name="C",
            msg_return_dict="list",
        )

        if not valid:
            check_is_error_msg(msg, var_name="C", raise_exception=True)

        C_inner_mtype = self.get_tag("C_inner_mtype")
        C_inner = convert(
            obj=C,
            from_type=metadata["mtype"],
            to_type=C_inner_mtype,
            as_scitype="Table",
            store=self._C_converter_store,
        )

        return C_inner, metadata
