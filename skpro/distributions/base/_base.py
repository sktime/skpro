# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Base classes for probability distribution objects."""

__author__ = ["fkiraly"]

__all__ = ["BaseDistribution"]

from warnings import warn

import numpy as np
import pandas as pd

from skpro.base import BaseObject
from skpro.utils.validation._dependencies import (
    _check_estimator_deps,
    _check_soft_dependencies,
)


class BaseDistribution(BaseObject):
    """Base probability distribution."""

    # default tag values - these typically make the "safest" assumption
    _tags = {
        "object_type": "distribution",  # type of object, e.g., 'distribution'
        "python_version": None,  # PEP 440 python version specifier to limit versions
        "python_dependencies": None,  # string or str list of pkg soft dependencies
        # property tags
        # -------------
        "distr:measuretype": "mixed",  # distribution type, mixed, continuous, discrete
        "distr:paramtype": "general",
        # parameterization type - parametric, nonparametric, composite
        #
        # default parameter settings for MC estimates
        # -------------------------------------------
        # these are used in default implementations of mean, var, energy, pdfnorm, ppf
        "approx_mean_spl": 1000,  # sample size used in MC estimates of mean
        "approx_var_spl": 1000,  # sample size used in MC estimates of var
        "approx_energy_spl": 1000,  # sample size used in MC estimates of energy
        "approx_spl": 1000,  # sample size used in other MC estimates
        "bisect_iter": 1000,  # max iters for bisection method in ppf
        # which methods are approximate (not numerically exact) should be listed here
        "capabilities:approx": ["energy", "mean", "var", "pdfnorm"],
        # broadcasting and parameter settings
        # -----------------------------------
        # used to control broadcasting of parameters
        "reserved_params": ["index", "columns"],
        "broadcast_params": None,  # list of params to broadcast
        "broadcast_init": "off",  # whether to auto-broadcast params in __init__
        "broadcast_inner": "array",  # whether inner args are array or scalar-like
        # if "scalar", assumes scalar, and broadcasts in boilerplate
    }

    def __init__(self, index=None, columns=None):
        self.index = _coerce_to_pd_index_or_none(index)
        self.columns = _coerce_to_pd_index_or_none(columns)

        super().__init__()
        _check_estimator_deps(self)

        self._init_shape_bc(index=index, columns=columns)

    def _init_shape_bc(self, index=None, columns=None):
        """Initialize shape and broadcasting of distribution parameters.

        Subclasses may choose to override this, if
        default broadcasting and pre-initialization is not desired or applicable,
        e.g., distribution parameters are not array-like.

        If overridden, must set ``self._shape``: this should be an empty tuple
        if the distribution is scalar, or a pair of integers otherwise.
        """
        if self.get_tags()["broadcast_init"] == "off":
            if index is None and columns is None:
                self._shape = ()
            else:
                self._shape = (len(index), len(columns))
            return None

        # if broadcast_init os on or other, run this auto-init
        bc_params, shape, is_scalar = self._get_bc_params_dict(return_shape=True)
        self._bc_params = bc_params
        self._is_scalar = is_scalar
        self._shape = shape

        if index is None and self.ndim > 0:
            self.index = pd.RangeIndex(shape[0])

        if columns is None and self.ndim > 0:
            self.columns = pd.RangeIndex(shape[1])

    @property
    def loc(self):
        """Location indexer, for groups of indices.

        Use ``my_distribution.loc[index]`` for ``pandas``-like row/column subsetting
        of ``BaseDistribution`` descendants.

        ``index`` can be any ``pandas`` ``iloc`` compatible index subsetter.

        ``my_distribution.loc[index]``
        or ``my_distribution.loc[row_index, col_index]``
        subset ``my_distribution`` to rows selected
        by ``row_index``, cols by ``col_index``,
        to exactly the same cols/rows as ``pandas`` ``loc`` would subset
        rows in ``my_distribution.index`` and columns in ``my_distribution.columns``.
        """
        return _Indexer(ref=self, method="_loc")

    @property
    def iloc(self):
        """Integer location indexer, for groups of indices.

        Use ``my_distribution.iloc[index]`` for ``pandas``-like row/column subsetting
        of ``BaseDistribution`` descendants.

        ``index`` can be any ``pandas`` ``iloc`` compatible index subsetter.

        ``my_distribution.iloc[index]``
        or ``my_distribution.iloc[row_index, col_index]``
        subset ``my_distribution`` to rows selected
        by ``row_index``, cols by ``col_index``,
        to exactly the same cols/rows as ``pandas`` ``iloc`` would subset
        rows in ``my_distribution.index`` and columns in ``my_distribution.columns``.
        """
        return _Indexer(ref=self, method="_iloc")

    @property
    def iat(self):
        """Integer location indexer, for single index.

        Use ``my_distribution.iat[index]`` for ``pandas``-like row/column subsetting
        of ``BaseDistribution`` descendants.

        ``index`` can be any ``pandas`` ``iat`` compatible index subsetter.

        ``my_distribution.iat[index]``
        or ``my_distribution.iat[row_index, col_index]``
        subset ``my_distribution`` to the row selected
        by ``row_index``, col by ``col_index``,
        to exactly the same col/rows as ``pandas`` ``iat`` would subset
        rows in ``my_distribution.index`` and columns in ``my_distribution.columns``.
        """
        return _Indexer(ref=self, method="_iat")

    @property
    def at(self):
        """Integer location indexer, for single index.

        Use ``my_distribution.at[index]`` for ``pandas``-like row/column subsetting
        of ``BaseDistribution`` descendants.

        ``index`` can be any ``pandas`` ``at`` compatible index subsetter.

        ``my_distribution.at[index]``
        or ``my_distribution.at[row_index, col_index]``
        subset ``my_distribution`` to the row selected
        by ``row_index``, col by ``col_index``,
        to exactly the same col/rows as ``pandas`` ``at`` would subset
        rows in ``my_distribution.index`` and columns in ``my_distribution.columns``.
        """
        return _Indexer(ref=self, method="_at")

    @property
    def shape(self):
        """Shape of self, a pair (2-tuple)."""
        return self._shape

    @property
    def ndim(self):
        """Number of dimensions of self. 2 if array, 0 if scalar."""
        return len(self._shape)

    def __len__(self):
        """Length of self, number of rows."""
        shape = self._shape
        if len(shape) == 0:
            return 1
        return shape[0]

    def head(self, n=5):
        """Return the first n rows.

        If there are less than n rows in ``self``, returns clone of ``self``.

        For negative n, returns all rows except the last n.

        Parameters
        ----------
        n : int, default=5
            Number of rows to return.

        Returns
        -------
        ``self`` subset to the first n rows, i.e., ``self.iloc[0:min(n, len(self))]``
        """
        if self.ndim < 2:
            return self
        assert isinstance(n, int)
        N = len(self)
        if n < 0:
            n = N - n
        n = min(n, N)
        return self.iloc[range(n)]

    def tail(self, n=5):
        """Return the last n rows.

        If there are less than n rows in ``self``, returns clone of ``self``.

        For negative n, returns all rows except the first n.

        Parameters
        ----------
        n : int, default=5
            Number of rows to return.

        Returns
        -------
        ``self`` subset to the last n rows, i.e., ``self.iloc[max(len(self) - n, 0):]``
        """
        if self.ndim < 2:
            return self
        assert isinstance(n, int)
        N = len(self)
        if n < 0:
            start = n
        else:
            start = N - n
        start = max(0, start)
        return self.iloc[range(start, N)]

    def _loc(self, rowidx=None, colidx=None):
        if is_scalar_notnone(rowidx) and is_scalar_notnone(colidx):
            return self._at(rowidx, colidx)
        if is_scalar_notnone(rowidx):
            rowidx = pd.Index([rowidx])
        if is_scalar_notnone(colidx):
            colidx = pd.Index([colidx])

        if rowidx is not None:
            row_iloc = self.index.get_indexer_for(rowidx)
        else:
            row_iloc = None
        if colidx is not None:
            col_iloc = self.columns.get_indexer_for(colidx)
        else:
            col_iloc = None
        return self._iloc(rowidx=row_iloc, colidx=col_iloc)

    def _at(self, rowidx=None, colidx=None):
        if rowidx is not None:
            row_iloc = self.index.get_indexer_for([rowidx])[0]
        else:
            row_iloc = None
        if colidx is not None:
            col_iloc = self.columns.get_indexer_for([colidx])[0]
        else:
            col_iloc = None
        return self._iat(rowidx=row_iloc, colidx=col_iloc)

    def _subset_params(self, rowidx, colidx, coerce_scalar=False):
        """Subset distribution parameters to given rows and columns.

        Parameters
        ----------
        rowidx : None, numpy index/slice coercible, or int
            Rows to subset to. If None, no subsetting is done.
        colidx : None, numpy index/slice coercible, or int
            Columns to subset to. If None, no subsetting is done.
        coerce_scalar : bool, optional, default=False
            If True, and the subsetted parameter is a scalar, coerce it to a scalar.

        Returns
        -------
        dict
            Dictionary with subsetted distribution parameters.
            Keys are parameter names of ``self``, values are the subsetted parameters.
        """
        params = self._get_dist_params()

        subset_param_dict = {}
        for param, val in params.items():
            subset_param_dict[param] = self._subset_param(
                val=val,
                rowidx=rowidx,
                colidx=colidx,
                coerce_scalar=coerce_scalar,
            )
        return subset_param_dict

    def _subset_param(self, val, rowidx, colidx, coerce_scalar=False):
        """Subset a single distribution parameter value to given rows and columns.

        Parameters
        ----------
        val : scalar, 1D, 2D, array-like, or None
            Distribution parameter that is to be subsetted.
        rowidx : None, numpy index/slice coercible, or int
            Rows to subset to. If None, no subsetting is done.
        colidx : None, numpy index/slice coercible, or int
            Columns to subset to. If None, no subsetting is done.
        coerce_scalar : bool, optional, default=False
            If True, and the subsetted parameter is a scalar, coerce it to a scalar.

        Returns
        -------
        scalar, 1D, 2D, array-like, or None
            Subsetted distribution parameter.
        """
        if val is None:
            return None
        arr = np.array(val)
        # if len(arr.shape) == 0:
        # do nothing with arr
        if len(arr.shape) == 2 and rowidx is not None:
            arr = arr[rowidx, :]
        if len(arr.shape) == 1 and colidx is not None:
            arr = arr[colidx]
        if len(arr.shape) >= 2 and colidx is not None:
            arr = arr[:, colidx]
        if np.issubdtype(arr.dtype, np.integer):
            arr = arr.astype("float")
        if coerce_scalar:
            arr = arr[(0,) * len(arr.shape)]
        return arr

    def _iat(self, rowidx=None, colidx=None):
        if rowidx is None or colidx is None:
            raise ValueError("iat method requires both row and column index")
        subset_params = self._subset_params(
            rowidx=rowidx, colidx=colidx, coerce_scalar=True
        )
        return type(self)(**subset_params)

    def _iloc(self, rowidx=None, colidx=None):
        if is_scalar_notnone(rowidx) and is_scalar_notnone(colidx):
            return self._iat(rowidx, colidx)
        if is_scalar_notnone(rowidx):
            rowidx = pd.Index([rowidx])
        if is_scalar_notnone(colidx):
            colidx = pd.Index([colidx])

        subset_params = self._subset_params(rowidx=rowidx, colidx=colidx)

        def subset_not_none(idx, subs):
            if subs is not None:
                return idx.take(subs)
            else:
                return idx

        index_subset = subset_not_none(self.index, rowidx)
        columns_subset = subset_not_none(self.columns, colidx)

        sk_distr_type = type(self)
        return sk_distr_type(
            index=index_subset,
            columns=columns_subset,
            **subset_params,
        )

    def _get_dist_params(self):
        params = self.get_params(deep=False)
        paramnames = params.keys()
        reserved_names = ["index", "columns"]
        paramnames = [x for x in paramnames if x not in reserved_names]

        return {k: params[k] for k in paramnames}

    def get_params_df(self):
        """Return distribution parameters in a dict of DataFrame.

        Available only for simple parametric distributions,
        i.e., distributions with tag "distr:paramtype" having value "parametric".

        Returns
        -------
        dict of pd.DataFrame
            Dictionary with all distribution parameters, as ``pd.DataFrame``.
            Keys are the parameter names, values are the ``pd.DataFrame``.
            Each ``DataFrame`` has the same index as ``self`` and columns as ``self``.
            Entries are the values of the distribution parameters.
        """
        is_parametric = self.get_class_tag("distr:paramtype") == "parametric"

        if not is_parametric:
            raise RuntimeError(
                f"Error in call of {type(self).__name__}.get_params_df, "
                "DataFrame representation of parameters via get_params_df or to_df "
                "is only available for parametric distributions, i.e., "
                "distributions with tag 'distr:paramtype' being 'parametric'"
            )

        if hasattr(self, "_bc_params"):
            bc_params = self._bc_params
        else:
            bc_params = self._get_bc_params_dict()

        paramnames = list(bc_params.keys())

        def to_df(x):
            if self.ndim > 0:
                return pd.DataFrame(x, index=self.index, columns=self.columns)
            return pd.DataFrame([[x]])

        params_df = {k: to_df(bc_params[k]) for k in paramnames}
        drop_keys = ["index", "columns"]
        params_df = {k: params_df[k] for k in params_df if k not in drop_keys}
        return params_df

    def to_df(self):
        """Return distribution parameters as a single DataFrame.

        Available only for simple parametric distributions,
        i.e., distributions with tag "distr:paramtype" having value "parametric".

        Returns
        -------
        pd.DataFrame
            DataFrame with all distribution parameters.
            column is a MultiIndex (paramname, varname).
            row index is the index of the distribution.
            Entries are the values of the distribution parameters.
        """
        params_df = self.get_params_df()
        paramnames = list(params_df.keys())
        vals_df = [params_df[k] for k in paramnames]

        param_df = pd.concat(vals_df, axis=1, keys=paramnames)
        param_df.columns = param_df.columns.swaplevel()

        # sorting for consistency with columns of self
        if self.columns is not None:
            param_df = param_df.loc[:, self.columns]
        else:
            param_df = param_df.sort_index(axis=1)

        if self.ndim == 0:
            # first level is superfluous in scalar case (always 0)
            # and inconsistent with MultiIndex handling, so we remove it
            param_df = param_df.droplevel(0, axis=1)
        return param_df

    def to_str(self):
        """Return string representation of self."""
        params = self._get_dist_params()

        prt = f"{self.__class__.__name__}("
        for paramname, val in params.items():
            prt += f"{paramname}={val}, "
        prt = prt[:-2] + ")"

        return prt

    def _method_error_msg(self, method="this method", severity="warn", fill_in=None):
        msg = (
            f"{type(self)} does not have an implementation of the '{method}' method, "
            "via numerically exact implementation or fill-in approximation."
        )
        if fill_in is None:
            fill_in = "by an approximation via other methods"
        msg_approx = (
            f"{type(self)} does not have a numerically exact implementation of "
            f"the '{method}' method, it is "
            f"filled in {fill_in}."
        )
        if severity == "warn":
            return msg_approx
        else:
            return msg

    def _get_bc_params(self, *args, dtype=None, oned_as="row", return_shape=False):
        """Fully broadcast tuple of parameters given param shapes and index, columns.

        Parameters
        ----------
        args : float, int, array of floats, or array of ints (1D or 2D)
            Distribution parameters that are to be made broadcastable. If no positional
            arguments are provided, all parameters of `self` are used except for `index`
            and `columns`.
        dtype : str, optional
            broadcasted arrays are cast to all have datatype `dtype`. If None, then no
            datatype casting is done.
        oned_as : str, optional, "row" (default) or "col"
            If 'row', then 1D arrays are treated as row vectors. If 'column', then 1D
            arrays are treated as column vectors.
        return_shape : bool, optional, default=False
            If True, return shape tuple, and a boolean tuple
            indicating which parameters are scalar.

        Returns
        -------
        Tuple of float or integer arrays
            Each element of the tuple represents a different broadcastable distribution
            parameter.
        shape : Tuple, only returned if ``return_shape`` is True
            Shape of the broadcasted parameters.
            Pair of row/column if not scalar, empty tuple if scalar.
        is_scalar : Tuple of bools, only returned if ``return_is_scalar`` is True
            Each element of the tuple is True if the corresponding parameter is scalar.
        """
        number_of_params = len(args)
        if number_of_params == 0:
            # Handle case where no positional arguments are provided
            params = self._get_dist_params()
            args = tuple(params.values())
            number_of_params = len(args)

        def row_to_col(arr):
            """Convert 1D arrays to 2D col arrays, leave 2D arrays unchanged."""
            if arr.ndim == 1:
                return arr.reshape(-1, 1)
            return arr

        args_as_np = [np.array(arg) for arg in args]
        if oned_as == "col":
            args_as_np = [row_to_col(arg) for arg in args_as_np]

        if hasattr(self, "index") and self.index is not None:
            args_as_np += (self.index.to_numpy().reshape(-1, 1),)
        if hasattr(self, "columns") and self.columns is not None:
            args_as_np += (self.columns.to_numpy(),)
        bc = np.broadcast_arrays(*args_as_np)
        if dtype is not None:
            bc = [array.astype(dtype) for array in bc]
        bc = bc[:number_of_params]
        if return_shape:
            shape = bc[0].shape
            is_scalar = tuple([arr.ndim == 0 for arr in bc])
            return bc, shape, is_scalar
        return bc

    def _get_bc_params_dict(
        self, dtype=None, oned_as="row", return_shape=False, **kwargs
    ):
        """Fully broadcast dict of parameters given param shapes and index, columns.

        Parameters
        ----------
        kwargs : float, int, array of floats, or array of ints (1D or 2D)
            Distribution parameters that are to be made broadcastable. If no positional
            arguments are provided, all parameters of `self` are used except for `index`
            and `columns`.
        dtype : str, optional
            broadcasted arrays are cast to all have datatype `dtype`. If None, then no
            datatype casting is done.
        oned_as : str, optional, "row" (default) or "col"
            If 'row', then 1D arrays are treated as row vectors. If 'column', then 1D
            arrays are treated as column vectors.
        return_shape : bool, optional, default=False
            If True, return shape tuple, and a boolean tuple
            indicating which parameters are scalar.

        Returns
        -------
        dict of float or integer arrays
            Each element of the tuple represents a different broadcastable distribution
            parameter.
        shape : Tuple, only returned if ``return_shape`` is True
            Shape of the broadcasted parameters.
            Pair of row/column if not scalar, empty tuple if scalar.
        is_scalar : Tuple of bools, only returned if ``return_is_scalar`` is True
            Each element of the tuple is True if the corresponding parameter is scalar.
        """
        number_of_params = len(kwargs)
        if number_of_params == 0:
            # Handle case where no positional arguments are provided
            kwargs = self._get_dist_params()
            number_of_params = len(kwargs)

        def row_to_col(arr):
            """Convert 1D arrays to 2D col arrays, leave 2D arrays unchanged."""
            if arr.ndim == 1 and oned_as == "col":
                return arr.reshape(-1, 1)
            return arr

        kwargs_as_np = {k: row_to_col(np.array(v)) for k, v in kwargs.items()}

        if hasattr(self, "index") and self.index is not None:
            kwargs_as_np["index"] = self.index.to_numpy().reshape(-1, 1)
        if hasattr(self, "columns") and self.columns is not None:
            kwargs_as_np["columns"] = self.columns.to_numpy()

        bc_params = self.get_tags()["broadcast_params"]
        if bc_params is None:
            bc_params = kwargs_as_np.keys()
        else:
            bc_params = bc_params.copy()
            if "index" in kwargs_as_np:
                bc_params.append("index")
            if "columns" in kwargs_as_np:
                bc_params.append("columns")

        args_as_np = [kwargs_as_np[k] for k in bc_params]
        bc = np.broadcast_arrays(*args_as_np)
        if dtype is not None:
            bc = [array.astype(dtype) for array in bc]

        shape = ()
        for i, k in enumerate(bc_params):
            kwargs_as_np[k] = row_to_col(bc[i])
            if bc[i].ndim > 0:
                shape = bc[i].shape

        # special case: user provided iterables so it broadcasts to 1D
        # this is interpreted as a row vector, i.e., one multivariate distr
        if len(shape) == 1:
            shape = (1, shape[0])
            for k, v in kwargs_as_np.items():
                kwargs_as_np[k] = np.expand_dims(v, 0)

        if return_shape:
            is_scalar = tuple([arr.ndim == 0 for arr in bc])
            return kwargs_as_np, shape, is_scalar
        return kwargs_as_np

    def _boilerplate(self, method, columns=None, **kwargs):
        """Broadcasting boilerplate for distribution methods.

        Used to link public methods to private methods,
        handles coercion, broadcasting, and checks.

        Parameters
        ----------
        method : str
            Name of the method to be called, e.g., '_pdf'
        columns : None (default) or pd.Index coercible
            If not None, set return columns to this value
        kwargs : dict
            Keyword arguments to the method
            Checks and broadcasts are applied to all values in kwargs

        Examples
        --------
        >>> self._boilerplate('_pdf', x=x)  # doctest: +SKIP
        >>> # calls self._pdf(x=x_inner), broadcasting x to self's shape in x_inner
        """
        kwargs_inner = kwargs.copy()
        d = self

        for k, x in kwargs.items():
            # if x is a DataFrame, subset and reorder distribution to match it
            if isinstance(x, pd.DataFrame):
                d = self.loc[x.index, x.columns]
                x_inner = x.values
            # else, coerce to a numpy array if needed
            # then, broadcast it to the shape of self
            else:
                x_inner = self._coerce_to_self_index_np(x, flatten=False)
            kwargs_inner[k] = x_inner

        # pass the broadcasted values to the private method
        res = getattr(d, method)(**kwargs_inner)

        # if the result is not a DataFrame, coerce it to one
        # ensur the index and columns are the same as d
        if not isinstance(res, pd.DataFrame) and self.ndim > 1:
            if columns is not None:
                res_cols = pd.Index(columns)
            else:
                res_cols = d.columns
            res = pd.DataFrame(res, index=d.index, columns=res_cols)
        # if numpy scalar, convert to python scalar, e.g., float
        if isinstance(res, np.ndarray) and self.ndim == 0:
            res = res[()]
        return res

    def pdf(self, x):
        r"""Probability density function.

        Let :math:`X` be a random variables with the distribution of ``self``,
        taking values in ``(N, n)`` ``DataFrame``-s
        Let :math:`x\in \mathbb{R}^{N\times n}`.
        By :math:`p_{X_{ij}}`, denote the marginal pdf of :math:`X` at the
        :math:`(i,j)`-th entry.

        The output of this method, for input ``x`` representing :math:`x`,
        is a ``DataFrame`` with same columns and indices as ``self``,
        and entries :math:`p_{X_{ij}}(x_{ij})`.

        If ``self`` has a mixed or discrete distribution, this returns
        the weighted continuous part of `self`'s distribution instead of the pdf,
        i.e., the marginal pdf integrate to the weight of the continuous part.

        Parameters
        ----------
        x : ``pandas.DataFrame`` or 2D ``np.ndarray``
            representing :math:`x`, as above

        Returns
        -------
        ``pd.DataFrame`` with same columns and index as ``self``
            containing :math:`p_{X_{ij}}(x_{ij})`, as above
        """
        distr_type = self.get_tag("distr:measuretype", "mixed", raise_error=False)
        if distr_type == "discrete":
            return self._coerce_to_self_index_df(0, flatten=False)

        return self._boilerplate("_pdf", x=x)

    def _pdf(self, x):
        """Probability density function.

        Private method, to be implemented by subclasses.
        """
        self_has_logpdf = self._has_implementation_of("log_pdf")
        self_has_logpdf = self_has_logpdf or self._has_implementation_of("_log_pdf")
        if self_has_logpdf:
            approx_method = (
                "by exponentiating the output returned by the log_pdf method, "
                "this may be numerically unstable"
            )
            warn(self._method_error_msg("pdf", fill_in=approx_method))

            x = self._coerce_to_self_index_df(x, flatten=False)
            res = self.log_pdf(x=x)
            if isinstance(res, pd.DataFrame):
                res = res.values
            return np.exp(res)

        raise NotImplementedError(self._method_error_msg("pdf", "error"))

    def log_pdf(self, x):
        r"""Logarithmic probability density function.

        Numerically more stable than calling pdf and then taking logartihms.

        Let :math:`X` be a random variables with the distribution of ``self``,
        taking values in `(N, n)` ``DataFrame``-s
        Let :math:`x\in \mathbb{R}^{N\times n}`.
        By :math:`p_{X_{ij}}`, denote the marginal pdf of :math:`X` at the
        :math:`(i,j)`-th entry.

        The output of this method, for input ``x`` representing :math:`x`,
        is a ``DataFrame`` with same columns and indices as ``self``,
        and entries :math:`\log p_{X_{ij}}(x_{ij})`.

        If ``self`` has a mixed or discrete distribution, this returns
        the weighted continuous part of `self`'s distribution instead of the pdf,
        i.e., the marginal pdf integrate to the weight of the continuous part.

        Parameters
        ----------
        x : ``pandas.DataFrame`` or 2D ``np.ndarray``
            representing :math:`x`, as above

        Returns
        -------
        ``pd.DataFrame`` with same columns and index as ``self``
            containing :math:`\log p_{X_{ij}}(x_{ij})`, as above
        """
        distr_type = self.get_tag("distr:measuretype", "mixed", raise_error=False)
        if distr_type == "discrete":
            return self._coerce_to_self_index_df(-np.inf, flatten=False)

        return self._boilerplate("_log_pdf", x=x)

    def _log_pdf(self, x):
        """Logarithmic probability density function.

        Private method, to be implemented by subclasses.
        """
        if self._has_implementation_of("pdf") or self._has_implementation_of("_pdf"):
            approx_method = (
                "by taking the logarithm of the output returned by the pdf method, "
                "this may be numerically unstable"
            )
            warn(self._method_error_msg("log_pdf", fill_in=approx_method))

            x = self._coerce_to_self_index_df(x, flatten=False)
            res = self.pdf(x=x)
            if isinstance(res, pd.DataFrame):
                res = res.values
            return np.log(res)

        raise NotImplementedError(self._method_error_msg("log_pdf", "error"))

    def pmf(self, x):
        r"""Probability mass function.

        Let :math:`X` be a random variables with the distribution of ``self``,
        taking values in ``(N, n)`` ``DataFrame``-s
        Let :math:`x\in \mathbb{R}^{N\times n}`.
        By :math:`m_{X_{ij}}`, denote the marginal mass of :math:`X` at the
        :math:`(i,j)`-th entry, i.e.,
        :math:`m_{X_{ij}}(x_{ij}) = \mathbb{P}(X_{ij} = x_{ij})`.

        The output of this method, for input ``x`` representing :math:`x`,
        is a ``DataFrame`` with same columns and indices as ``self``,
        and entries :math:`m_{X_{ij}}(x_{ij})`.

        If ``self`` has a mixed or discrete distribution, this returns
        the weighted continuous part of `self`'s distribution instead of the pdf,
        i.e., the marginal pdf integrate to the weight of the continuous part.

        Parameters
        ----------
        x : ``pandas.DataFrame`` or 2D ``np.ndarray``
            representing :math:`x`, as above

        Returns
        -------
        ``pd.DataFrame`` with same columns and index as ``self``
            containing :math:`p_{X_{ij}}(x_{ij})`, as above
        """
        distr_type = self.get_tag("distr:measuretype", "mixed", raise_error=False)
        if distr_type == "continuous":
            return self._coerce_to_self_index_df(0, flatten=False)

        return self._boilerplate("_pmf", x=x)

    def _pmf(self, x):
        """Probability mass function.

        Private method, to be implemented by subclasses.
        """
        self_has_logpmf = self._has_implementation_of("log_pmf")
        self_has_logpmf = self_has_logpmf or self._has_implementation_of("_log_pmf")
        if self_has_logpmf:
            approx_method = (
                "by exponentiating the output returned by the log_pmf method, "
                "this may be numerically unstable"
            )
            warn(self._method_error_msg("pmf", fill_in=approx_method))

            x = self._coerce_to_self_index_df(x, flatten=False)
            res = self.log_pmf(x=x)
            if isinstance(res, pd.DataFrame):
                res = res.values
            return np.exp(res)

        raise NotImplementedError(self._method_error_msg("pmf", "error"))

    def log_pmf(self, x):
        r"""Logarithmic probability mass function.

        Numerically more stable than calling pmf and then taking logartihms.

        Let :math:`X` be a random variables with the distribution of ``self``,
        taking values in `(N, n)` ``DataFrame``-s
        Let :math:`x\in \mathbb{R}^{N\times n}`.
        By :math:`m_{X_{ij}}`, denote the marginal pdf of :math:`X` at the
        :math:`(i,j)`-th entry, i.e.,
        :math:`m_{X_{ij}}(x_{ij}) = \mathbb{P}(X_{ij} = x_{ij})`.

        The output of this method, for input ``x`` representing :math:`x`,
        is a ``DataFrame`` with same columns and indices as ``self``,
        and entries :math:`\log m_{X_{ij}}(x_{ij})`.

        If ``self`` has a mixed or discrete distribution, this returns
        the weighted continuous part of `self`'s distribution instead of the pdf,
        i.e., the marginal pdf integrate to the weight of the continuous part.

        Parameters
        ----------
        x : ``pandas.DataFrame`` or 2D ``np.ndarray``
            representing :math:`x`, as above

        Returns
        -------
        ``pd.DataFrame`` with same columns and index as ``self``
            containing :math:`\log m_{X_{ij}}(x_{ij})`, as above
        """
        distr_type = self.get_tag("distr:measuretype", "mixed", raise_error=False)
        if distr_type == "continuous":
            return self._coerce_to_self_index_df(-np.inf, flatten=False)

        return self._boilerplate("_log_pmf", x=x)

    def _log_pmf(self, x):
        """Logarithmic probability mass function.

        Private method, to be implemented by subclasses.
        """
        if self._has_implementation_of("pmf") or self._has_implementation_of("_pmf"):
            approx_method = (
                "by taking the logarithm of the output returned by the pdf method, "
                "this may be numerically unstable"
            )
            warn(self._method_error_msg("log_pmf", fill_in=approx_method))

            x = self._coerce_to_self_index_df(x, flatten=False)
            res = self.pmf(x=x)
            if isinstance(res, pd.DataFrame):
                res = res.values
            return np.log(res)

        raise NotImplementedError(self._method_error_msg("log_pmf", "error"))

    def cdf(self, x):
        r"""Cumulative distribution function.

        Let :math:`X` be a random variables with the distribution of ``self``,
        taking values in ``(N, n)`` ``DataFrame``-s
        Let :math:`x\in \mathbb{R}^{N\times n}`.
        By :math:`F_{X_{ij}}`, denote the marginal cdf of :math:`X` at the
        :math:`(i,j)`-th entry,
        i.e., :math:`F_{X_{ij}}(t) = \mathbb{P}(X_{ij} \leq t)`.

        The output of this method, for input ``x`` representing :math:`x`,
        is a ``DataFrame`` with same columns and indices as ``self``,
        and entries :math:`F_{X_{ij}}(x_{ij})`.

        Parameters
        ----------
        x : ``pandas.DataFrame`` or 2D ``np.ndarray``
            representing :math:`x`, as above

        Returns
        -------
        ``pd.DataFrame`` with same columns and index as ``self``
            containing :math:`F_{X_{ij}}(x_{ij})`, as above
        """
        return self._boilerplate("_cdf", x=x)

    def _cdf(self, x):
        """Cumulative distribution function.

        Private method, to be implemented by subclasses.
        """
        N = self.get_tag("approx_spl")
        approx_method = (
            "by approximating the expected value by the indicator function on "
            f"{N} samples"
        )
        warn(self._method_error_msg("mean", fill_in=approx_method))

        splx = self._sample_multiply(x, N)
        sply = self.sample(N)
        spl = splx <= sply
        return self._sample_mean(spl)

    def surv(self, x):
        r"""Survival function.

        Let :math:`X` be a random variables with the distribution of ``self``,
        taking values in ``(N, n)`` ``DataFrame``-s
        Let :math:`x\in \mathbb{R}^{N\times n}`.
        By :math:`S_{X_{ij}}`, denote the marginal survival of :math:`X` at the
        :math:`(i,j)`-th entry,
        i.e., :math:`S_{X_{ij}}(t) = \mathbb{P}(X_{ij} \gneq t)`.

        The output of this method, for input ``x`` representing :math:`x`,
        is a ``DataFrame`` with same columns and indices as ``self``,
        and entries :math:`F_{X_{ij}}(x_{ij})`.

        Parameters
        ----------
        x : ``pandas.DataFrame`` or 2D ``np.ndarray``
            representing :math:`x`, as above

        Returns
        -------
        ``pd.DataFrame`` with same columns and index as ``self``
            containing :math:`S_{X_{ij}}(x_{ij})`, as above
        """
        return self._boilerplate("_surv", x=x)

    def _surv(self, x):
        """Survival function.

        Private method, to be implemented by subclasses.
        """
        x = self._coerce_to_self_index_df(x, flatten=False)
        return 1 - self.cdf(x)

    def haz(self, x):
        r"""Hazard function.

        Let :math:`X` be a random variables with the distribution of ``self``,
        taking values in ``(N, n)`` ``DataFrame``-s
        Let :math:`x\in \mathbb{R}^{N\times n}`.
        By :math:`h_{X_{ij}}`, denote the marginal hazard of :math:`X` at the
        :math:`(i,j)`-th entry,
        i.e., :math:`h_{X_{ij}}(t) = \frac{f_{X_{ij}}(t)}{S_{X_{ij}}(t)}`,
        where :math:`f_{X_{ij}}` is the marginal pdf, and :math:`S_{X_{ij}}`
        is the marginal survival function at the :math:`(i,j)`-th entry.

        The output of this method, for input ``x`` representing :math:`x`,
        is a ``DataFrame`` with same columns and indices as ``self``,
        and entries :math:`h_{X_{ij}}(x_{ij})`.

        Parameters
        ----------
        x : ``pandas.DataFrame`` or 2D ``np.ndarray``
            representing :math:`x`, as above

        Returns
        -------
        ``pd.DataFrame`` with same columns and index as ``self``
            containing :math:`h_{X_{ij}}(x_{ij})`, as above
        """
        return self._boilerplate("_haz", x=x)

    def _haz(self, x):
        """Hazard function.

        Private method, to be implemented by subclasses.
        """
        x = self._coerce_to_self_index_df(x, flatten=False)
        return self.pdf(x) / self.surv(x)

    def ppf(self, p):
        r"""Quantile function = percent point function = inverse cdf.

        Let :math:`X` be a random variables with the distribution of ``self``,
        taking values in ``(N, n)`` ``DataFrame``-s
        Let :math:`x\in \mathbb{R}^{N\times n}`.
        By :math:`F_{X_{ij}}`, denote the marginal cdf of :math:`X` at the
        :math:`(i,j)`-th entry.

        The output of this method, for input ``p`` representing :math:`p`,
        is a ``DataFrame`` with same columns and indices as ``self``,
        and entries :math:`F^{-1}_{X_{ij}}(p_{ij})`.

        Parameters
        ----------
        p : ``pandas.DataFrame`` or 2D np.ndarray
            representing :math:`p`, as above

        Returns
        -------
        ``pd.DataFrame`` with same columns and index as ``self``
            containing :math:`F_{X_{ij}}(x_{ij})`, as above
        """
        return self._boilerplate("_ppf", p=p)

    def _ppf(self, p):
        """Quantile function = percent point function = inverse cdf.

        Private method, to be implemented by subclasses.

        Parameters
        ----------
        p : 2D np.ndarray, same shape as ``self``
            values to evaluate the ppf at

        Returns
        -------
        2D np.ndarray, same shape as ``self``
            ppf values at the given points
        """
        if self._has_implementation_of("cdf") or self._has_implementation_of("_cdf"):
            from scipy.optimize import bisect

            max_iter = self.get_tag("bisect_iter")
            approx_method = (
                "by using the bisection method (scipy.optimize.bisect) on "
                f"the cdf, at {max_iter} maximum iterations"
            )
            warn(self._method_error_msg("ppf", fill_in=approx_method))

            def bisect_unb(opt_fun, **kwargs):
                """Unbound version of bisect."""
                left_bd = -1e6
                right_bd = 1e6
                while opt_fun(left_bd) > 0:
                    left_bd *= 10
                while opt_fun(right_bd) < 0:
                    right_bd *= 10
                return bisect(opt_fun, left_bd, right_bd, maxiter=max_iter, **kwargs)

            shape = self.shape

            # TODO: remove duplications in the code below
            # requires cdf to accept numpy, or allow subsetting to produce scalar
            if len(shape) == 0:

                def opt_fun(x):
                    """Optimization function, to find x s.t. cdf(x) = p_ix."""
                    return self.cdf(x) - p  # noqa: B023

                result = bisect_unb(opt_fun)
                return result

            n_row, n_col = self.shape
            result = np.array([[0.0] * n_col] * n_row, dtype=float)

            for i in range(n_row):
                for j in range(n_col):
                    ix = self.index[i]
                    col = self.columns[j]
                    d_ix = self.loc[[ix], [col]]
                    p_ix = p[i, j]

                    def opt_fun(x):
                        """Optimization function, to find x s.t. cdf(x) = p_ix."""
                        x = pd.DataFrame(x, index=[ix], columns=[col])  # noqa: B023
                        return d_ix.cdf(x).values[0][0] - p_ix  # noqa: B023

                    result[i, j] = bisect_unb(opt_fun)
            return result

        raise NotImplementedError(self._method_error_msg("ppf", "error"))

    def energy(self, x=None):
        r"""Energy of self, w.r.t. self or a constant frame x.

        Let :math:`X, Y` be i.i.d. random variables with the distribution of ``self``.

        If ``x`` is ``None``, returns :math:`\mathbb{E}[|X-Y|]` (per row),
        "self-energy".
        If ``x`` is passed, returns :math:`\mathbb{E}[|X-x|]` (per row), "energy wrt x".

        The CRPS is related to energy:
        it holds that
        :math:`\mbox{CRPS}(\mbox{self}, y)` = `self.energy(y) - 0.5 * self.energy()`.

        Parameters
        ----------
        x : None or pd.DataFrame, optional, default=None
            if ``pd.DataFrame``, must have same rows and columns as ``self``

        Returns
        -------
        ``pd.DataFrame`` with same rows as ``self``, single column ``"energy"``
            each row contains one float, self-energy/energy as described above.
        """
        if x is None:
            return self._boilerplate("_energy_self", columns=["energy"])
        return self._boilerplate("_energy_x", x=x, columns=["energy"])

    def _energy_self(self):
        r"""Energy of self, w.r.t. self.

        :math:`\mathbb{E}[|X-Y|]`, where :math:`X, Y` are i.i.d. copies of self.

        Private method, to be implemented by subclasses.

        Returns
        -------
        2D np.ndarray, same shape as ``self``
            energy values w.r.t. the given points
        """
        return self._energy_default()

    def _energy_x(self, x):
        r"""Energy of self, w.r.t. a constant frame x.

        :math:`\mathbb{E}[|X-x|]`, where :math:`X` is a copy of self,
        and :math:`x` is a constant.

        Private method, to be implemented by subclasses.

        Parameters
        ----------
        x : 2D np.ndarray, same shape as ``self``
            values to compute energy w.r.t. to

        Returns
        -------
        2D np.ndarray, same shape as ``self``
            energy values w.r.t. the given points
        """
        return self._energy_default(x)

    def _energy_default(self, x=None):
        """Energy of self, w.r.t. self or a constant frame x.

        Default implementation, using Monte Carlo estimates.

        Parameters
        ----------
        x : None or 2D np.ndarray, same shape as ``self``
            values to compute energy w.r.t. to

        Returns
        -------
        2D np.ndarray, same shape as ``self``
            energy values w.r.t. the given points
        """
        approx_spl_size = self.get_tag("approx_energy_spl")
        if x is not None and self._has_implementation_of("_ppf"):
            approx_method = (
                "by approximating the energy expectation by the integral "
                "of the absolute difference of x to the ppf,"
                f"with {approx_spl_size} equidistant nodes"
            )
            warn(self._method_error_msg("energy", fill_in=approx_method))

            ps = np.linspace(0, 1, approx_spl_size + 2)[1:-1]
            qs = [np.abs(self.ppf(p) - x) for p in ps]
            en3D = np.array(qs)
            energy = np.mean(en3D, axis=0)
            if self.ndim > 0:
                energy = np.sum(energy, axis=1)
            return energy

        # we want to approximate E[abs(X-Y)]
        # if x = None, X,Y are i.i.d. copies of self
        # if x is not None, X=x (constant), Y=self
        approx_spl_size = self.get_tag("approx_energy_spl")
        approx_method = (
            "by approximating the energy expectation by the arithmetic mean of "
            f"{approx_spl_size} samples"
        )
        warn(self._method_error_msg("energy", fill_in=approx_method))

        # splx, sply = i.i.d. samples of X - Y of size N = approx_spl_size
        N = approx_spl_size
        if x is None:
            splx = self.sample(N)
            sply = self.sample(N)
        else:  # if x is not None
            splx = self._sample_multiply(x, N)
            sply = self.sample(N)

        # approx E[abs(X-Y)] via mean of samples of abs(X-Y) obtained from splx, sply
        spl = splx - sply
        energy = spl.apply(np.linalg.norm, axis=1, ord=1)

        # todo: check if can use self._sample_mean
        if self.ndim > 0:
            energy = energy.groupby(level=1, sort=False)
        energy = energy.mean()
        if self.ndim > 0:
            energy = pd.DataFrame(energy, index=self.index, columns=["energy"])
        return energy

    def _sample_multiply(self, x, N):
        """Generate N copies of x, in a format as returned by self.sample.

        Auxiliary function used in defaults for private methods.

        Parameters
        ----------
        x : same format as output of sample(), without N,
            or np.ndarray of same shape (2D or 0D)
        N :int

        Returns
        -------
        same format as output of sample(N), containing N copies of x
        """
        if self.ndim > 0:  # and x is not None
            if not isinstance(x, pd.DataFrame):
                x = pd.DataFrame(x, index=self.index, columns=self.columns)
            spl = pd.concat([x] * N, keys=range(N))
        else:  # if self.ndim == 0 and x is not None
            spl = pd.DataFrame([x] * N)
        return spl

    def _sample_mean(self, spl):
        """Take mean of sample as returned by self.sample, respecting shape.

        Auxiliary function used in defaults for private methods.

        Parameters
        ----------
        x : same format as output of sample(N), with N:int,

        Returns
        -------
        mean of sample:
        if ``self`` is array: ``pd.DataFrame`` with same ``index`` and ``columns``
        as ``self;
        if ``self`` is scalar: scalar
        """
        if self.ndim > 0:
            levels = list(range(1, spl.index.nlevels))
            return spl.groupby(level=levels, sort=False).mean()
        else:
            return spl.mean().iloc[0]

    def mean(self):
        r"""Return expected value of the distribution.

        Let :math:`X` be a random variable with the distribution of ``self``.
        Returns the expectation :math:`\mathbb{E}[X]`

        Returns
        -------
        ``pd.DataFrame`` with same rows, columns as ``self``
            expected value of distribution (entry-wise)
        """
        return self._boilerplate("_mean")

    def _mean(self):
        """Return expected value of the distribution.

        Private method, to be implemented by subclasses.
        """
        approx_spl_size = self.get_tag("approx_mean_spl")
        if self._has_implementation_of("_ppf"):
            approx_method = (
                "by approximating the expected value by the integral of the ppf, "
                f"with {approx_spl_size} equidistant nodes"
            )
            warn(self._method_error_msg("mean", fill_in=approx_method))

            ps = np.linspace(0, 1, approx_spl_size + 2)[1:-1]
            qs = [self.ppf(p) for p in ps]
            np3D = np.array(qs)
            means = np.mean(np3D, axis=0)
            return means

        # else we have to rely on samples
        approx_method = (
            "by approximating the expected value by the arithmetic mean of "
            f"{approx_spl_size} samples"
        )
        warn(self._method_error_msg("mean", fill_in=approx_method))

        spl = self.sample(approx_spl_size)
        return self._sample_mean(spl)

    def var(self):
        r"""Return element/entry-wise variance of the distribution.

        Let :math:`X` be a random variable with the distribution of ``self``.
        Returns :math:`\mathbb{V}[X] = \mathbb{E}\left(X - \mathbb{E}[X]\right)^2`,
        where the square is element-wise.

        Returns
        -------
        ``pd.DataFrame`` with same rows, columns as ``self``
            variance of distribution (entry-wise)
        """
        return self._boilerplate("_var")

    def _var(self):
        """Return element/entry-wise variance of the distribution.

        Private method, to be implemented by subclasses.
        """
        approx_spl_size = self.get_tag("approx_var_spl")
        if self._has_implementation_of("_ppf"):
            approx_method = (
                "by approximating the variancee integrals of the ppf, "
                "integral of ppf-squared minus square of integral of ppf, "
                f"each with {approx_spl_size} equidistant nodes"
            )
            warn(self._method_error_msg("var", fill_in=approx_method))

            ps = np.linspace(0, 1, approx_spl_size + 2)[1:-1]
            qs = [self.ppf(p) for p in ps]
            qsq = [q**2 for q in qs]

            mean3D = np.array(qs)
            means = np.mean(mean3D, axis=0)

            mom2s3D = np.array(qsq)
            mom2s = np.mean(mom2s3D, axis=0)

            return mom2s - means**2

        approx_method = (
            "by approximating the variance by the arithmetic mean of "
            f"{approx_spl_size} samples of squared differences"
        )
        warn(self._method_error_msg("var", fill_in=approx_method))

        spl1 = self.sample(approx_spl_size)
        spl2 = self.sample(approx_spl_size)
        spl = (spl1 - spl2) ** 2
        return self._sample_mean(spl) / 2

    def pdfnorm(self, a=2):
        r"""a-norm of pdf, defaults to 2-norm.

        computes a-norm of the entry marginal pdf, i.e.,
        :math:`\mathbb{E}[p_X(X)^{a-1}] = \int p(x)^a dx`,
        where :math:`X` is a random variable distributed according to the entry marginal
        of `self`, and :math:`p_X` is its pdf

        Parameters
        ----------
        a: int or float, optional, default=2

        Returns
        -------
        pd.DataFrame with same rows and columns as `self`
        each entry is :math:`\mathbb{E}[p_X(X)^{a-1}] = \int p(x)^a dx`, see above
        """
        # special case: if a == 1, this is just the integral of the pdf, which is 1
        if a == 1:
            return pd.DataFrame(1.0, index=self.index, columns=self.columns)

        approx_spl_size = self.get_tag("approx_spl")
        approx_method = (
            f"by approximating the {a}-norm of the pdf by the arithmetic mean of "
            f"{approx_spl_size} samples"
        )
        warn(self._method_error_msg("pdfnorm", fill_in=approx_method))

        # uses formula int p(x)^a dx = E[p(X)^{a-1}], and MC approximates the RHS
        spl = [self.pdf(self.sample()) ** (a - 1) for _ in range(approx_spl_size)]
        spl_df = pd.concat(spl, keys=range(approx_spl_size))
        return spl_df.groupby(level=1, sort=False).mean()

    def _coerce_to_self_index_df(self, x, flatten=True):
        """Coerce input to type similar to self.

        If self is not scalar with index and columns,
        coerces x to a pd.DataFrame with index and columns as self.

        If self is scalar, coerces x to a scalar (0D) np.ndarray.
        """
        x = np.array(x)
        if flatten:
            x = x.reshape(1, -1)
        df_shape = self.shape
        x = np.broadcast_to(x, df_shape)
        if self.ndim != 0:
            df = pd.DataFrame(x, index=self.index, columns=self.columns)
            return df
        return x

    def _coerce_to_self_index_np(self, x, flatten=False):
        """Coerce input to type similar to self.

        Coerces x to a np.ndarray with same shape as self.
        Broadcasts x to self.shape, if necessary, via np.broadcast_to.

        Parameters
        ----------
        x : array-like, np.ndarray coercible
            input to be coerced to self
        flatten : bool, optional, default=True
            if True, flattens x before broadcasting
            if False, broadcasts x as is
        """
        x = np.array(x)
        if flatten:
            x = x.reshape(1, -1)
        df_shape = self.shape
        x = np.broadcast_to(x, df_shape)
        return x

    def quantile(self, alpha):
        """Return entry-wise quantiles, in Proba/pred_quantiles mtype format.

        This method broadcasts as follows:
        for a scalar `alpha`, computes the `alpha`-quantile entry-wise,
        and returns as a `pd.DataFrame` with same index, and columns as in return.
        If `alpha` is iterable, multiple quantiles will be calculated,
        and the result will be concatenated column-wise (axis=1).

        The `ppf` method also computes quantiles, but broadcasts differently, in
        `numpy` style closer to `tensorflow`.
        In contrast, this `quantile` method broadcasts
        as ``sktime`` forecaster `predict_quantiles`, i.e., columns first.

        Parameters
        ----------
        alpha : float or list of float of unique values
            A probability or list of, at which quantiles are computed.

        Returns
        -------
        quantiles : pd.DataFrame
            Column has multi-index: first level is variable name from `self.columns`,
            second level being the values of `alpha` passed to the function.
            Row index is `self.index`.
            Entries in the i-th row, (j, p)-the column is
            the p-th quantile of the marginal of `self` at index (i, j).
        """
        if not isinstance(alpha, list):
            alpha = [alpha]

        qdfs = []
        for p in alpha:
            p = self._coerce_to_self_index_df(p, flatten=self.ndim > 0)
            qdf = self.ppf(p)
            qdfs += [qdf]

        if self.ndim > 0:
            qres = pd.concat(qdfs, axis=1, keys=alpha)
            qres = qres.reorder_levels([1, 0], axis=1)
        else:
            qdfs = np.expand_dims(np.array(qdfs), 0)
            qres = pd.DataFrame(qdfs, columns=alpha)

        if self.ndim > 0:
            cols = pd.MultiIndex.from_product([self.columns, alpha])
        else:
            clsname = self.__class__.__name__
            cols = pd.MultiIndex.from_product([[clsname], alpha])
            qres.columns = cols

        quantiles = qres.loc[:, cols]
        return quantiles

    def sample(self, n_samples=None):
        """Sample from the distribution.

        Parameters
        ----------
        n_samples : int, optional, default = None
            number of samples to draw from the distribution

        Returns
        -------
        pd.DataFrame
            samples from the distribution

            * if ``n_samples`` is ``None``:
            returns a sample that contains a single sample from ``self``,
            in ``pd.DataFrame`` mtype format convention, with ``index`` and ``columns``
            as ``self``
            * if n_samples is ``int``:
            returns a ``pd.DataFrame`` that contains ``n_samples`` i.i.d.
            samples from ``self``, in ``pd-multiindex`` mtype format convention,
            with same ``columns`` as ``self``, and row ``MultiIndex`` that is product
            of ``RangeIndex(n_samples)`` and ``self.index``
        """

        def gen_unif():
            np_unif = np.random.uniform(size=self.shape)
            if self.ndim > 0:
                return pd.DataFrame(np_unif, index=self.index, columns=self.columns)
            return np_unif

        # if ppf is implemented, we use inverse transform sampling
        if self._has_implementation_of("_ppf") or self._has_implementation_of("ppf"):
            if n_samples is None:
                return self.ppf(gen_unif())
            # else, we generate n_samples i.i.d. samples
            pd_smpl = [self.ppf(gen_unif()) for _ in range(n_samples)]
            if self.ndim > 0:
                df_spl = pd.concat(pd_smpl, keys=range(n_samples))
            else:
                df_spl = pd.DataFrame(pd_smpl)
            return df_spl

        raise NotImplementedError(self._method_error_msg("sample", "error"))

    def plot(self, fun=None, ax=None, **kwargs):
        """Plot the distribution.

        Different distribution defining functions can be selected for plotting
        via the ``fun`` parameter.
        The functions available are the same as the methods of the distribution class,
        e.g., ``"pdf"``, ``"cdf"``, ``"ppf"``.

        For array distribution, the marginal distribution at each entry is plotted,
        as a separate subplot.

        Parameters
        ----------
        fun : str, optional, default="pdf" for continuous distributions, otherwise "cdf"
            the function to plot, one of "pdf", "cdf", "ppf"
        ax : matplotlib Axes object, optional
            matplotlib Axes to plot in
            if not provided, defaults to current axes (``plot.gca``)
        kwargs : keyword arguments
            passed to the plotting function

        Returns
        -------
        fig : matplotlib.Figure, only returned if self is array distribution
            matplotlig Figure object for subplots
        ax : matplotlib.Axes
            the axis or axes on which the plot is drawn
        """
        _check_soft_dependencies("matplotlib", obj="distribution plot")

        from matplotlib.pyplot import subplots

        if fun is None:
            if self.get_tag("distr:measuretype", "mixed") == "continuous":
                fun = "pdf"
            else:
                fun = "cdf"

        if self.ndim > 0:
            if "x_bounds" not in kwargs:
                upper = self.ppf(0.999).values.flatten().max()
                lower = self.ppf(0.001).values.flatten().min()
                x_bounds = (lower, upper)
            else:
                x_bounds = kwargs.pop("x_bounds")
            if "sharex" not in kwargs:
                sharex = True
            else:
                sharex = kwargs.pop("sharex")
            if "sharey" not in kwargs:
                sharey = True
            else:
                sharey = kwargs.pop("sharey")

            x_argname = _get_first_argname(getattr(self, fun))

            def get_ax(ax, i, j, shape):
                """Get axes at iloc i, j - API unifier for 2D and 1D subplot figures.

                Covers inconsistency in matplotlib where creation of (m, 1) matrix
                of subplots creates a 1D object and not a 2D object.
                """
                if shape[1] == 1:
                    return ax[i]
                else:
                    return ax[i, j]

            shape = self.shape
            fig, ax = subplots(shape[0], shape[1], sharex=sharex, sharey=sharey)
            for i, j in np.ndindex(shape):
                d_ij = self.iloc[i, j]
                ax_ij = get_ax(ax, i, j, shape)
                d_ij.plot(
                    fun=fun,
                    ax=ax_ij,
                    x_bounds=x_bounds,
                    print_labels="off",
                    x_argname=x_argname,
                    **kwargs,
                )
            for i in range(shape[0]):
                ax_i0 = get_ax(ax, i, 0, shape)
                ax_i0.set_ylabel(f"{self.index[i]}")
            for j in range(shape[1]):
                ax_0j = get_ax(ax, 0, j, shape)
                ax_0j.set_title(f"{self.columns[j]}")
            fig.supylabel(f"{fun}({x_argname})")
            fig.supxlabel(f"{x_argname}")
            return fig, ax

        # for now, all plots default to this function
        # but this could be changed to a dispatch mechanism
        # e.g., using this line instead
        # plot_fun_name = f"_plot_{fun}"
        plot_fun_name = "_plot_single"

        ax = getattr(self, plot_fun_name)(ax=ax, fun=fun, **kwargs)
        return ax

    def _plot_single(self, ax=None, **kwargs):
        """Plot the pdf of the distribution."""
        import matplotlib.pyplot as plt

        fun = kwargs.pop("fun")
        print_labels = kwargs.pop("print_labels", "on")
        x_argname = kwargs.pop("x_argname", "x")

        # obtain x axis bounds for plotting
        if "x_bounds" in kwargs:
            lower, upper = kwargs.pop("x_bounds")
        elif fun != "ppf":
            lower, upper = self.ppf(0.001), self.ppf(0.999)

        if fun == "ppf":
            lower, upper = 0.001, 0.999

        x_arr = np.linspace(lower, upper, 1000)
        y_arr = [getattr(self, fun)(x) for x in x_arr]
        y_arr = np.array(y_arr)

        if ax is None:
            ax = plt.gca()

        ax.plot(x_arr, y_arr, **kwargs)

        if print_labels == "on":
            ax.set_xlabel(f"{x_argname}")
            ax.set_ylabel(f"{fun}({x_argname})")
        return ax


class _Indexer:
    """Indexer for BaseDistribution, for pandas-like index in loc and iloc property."""

    def __init__(self, ref, method="_loc"):
        self.ref = ref
        self.method = method

    def __call__(self, *args, **kwargs):
        """Error message to tell the user to use [ ] instead of ( )."""
        methodname = self.method[1:]
        clsname = self.ref.__class__.__name__
        raise ValueError(
            f"Error while attempting to index {clsname} probability "
            f"distribution instance via {methodname}: "
            f"Please use square brackets [] for indexing a distribution, i.e., "
            f"mydist.{methodname}[index] or mydist.{methodname}[index1, index2], "
            f"not mydist.{methodname}(index) or mydist.{methodname}(index1, index2)"
        )

    def __getitem__(self, key):
        """Getitem dunder, for use in my_distr.loc[index] and my_distr.iloc[index]."""

        def is_noneslice(obj):
            res = isinstance(obj, slice)
            res = res and obj.start is None and obj.stop is None and obj.step is None
            return res

        ref = self.ref
        indexer = getattr(ref, self.method)

        if isinstance(key, tuple):
            if not len(key) == 2:
                raise ValueError(
                    "there should be one or two keys when calling .loc, "
                    "e.g., mydist[key], or mydist[key1, key2]"
                )
            rows = key[0]
            cols = key[1]
            if is_noneslice(rows) and is_noneslice(cols):
                return ref
            elif is_noneslice(cols):
                return indexer(rowidx=rows, colidx=None)
            elif is_noneslice(rows):
                return indexer(rowidx=None, colidx=cols)
            else:
                return indexer(rowidx=rows, colidx=cols)
        else:
            return indexer(rowidx=key, colidx=None)


class _BaseTFDistribution(BaseDistribution):
    """Adapter for tensorflow-probability distributions."""

    _tags = {
        "python_dependencies": "tensorflow_probability",
        "capabilities:approx": ["energy"],
        "capabilities:exact": ["mean", "var", "pdf", "log_pdf", "cdf", "ppf"],
    }

    def __init__(self, index=None, columns=None, distr=None):
        self.distr = distr

        super().__init__(index=index, columns=columns)

    def __str__(self):
        return self.to_str()

    def pdf(self, x):
        r"""Probability density function.

        Let :math:`X` be a random variables with the distribution of `self`,
        taking values in `(N, n)` `DataFrame`-s
        Let :math:`x\in \mathbb{R}^{N\times n}`.
        By :math:`p_{X_{ij}}`, denote the marginal pdf of :math:`X` at the
        :math:`(i,j)`-th entry.

        The output of this method, for input `x` representing :math:`x`,
        is a `DataFrame` with same columns and indices as `self`,
        and entries :math:`p_{X_{ij}}(x_{ij})`.

        If `self` has a mixed or discrete distribution, this returns
        the weighted continuous part of `self`'s distribution instead of the pdf,
        i.e., the marginal pdf integrate to the weight of the continuous part.

        Parameters
        ----------
        x : `pandas.DataFrame` or 2D np.ndarray
            representing :math:`x`, as above

        Returns
        -------
        `DataFrame` with same columns and index as `self`
            containing :math:`p_{X_{ij}}(x_{ij})`, as above
        """
        if isinstance(x, pd.DataFrame):
            dist_at_x = self.loc[x.index, x.columns]
            tensor = dist_at_x.distr.prob(x.values)
            return pd.DataFrame(tensor, index=x.index, columns=x.columns)
        else:
            dist_at_x = self
            return dist_at_x.distr.prob(x)

    def log_pdf(self, x):
        r"""Logarithmic probability density function.

        Numerically more stable than calling pdf and then taking logartihms.

        Let :math:`X` be a random variables with the distribution of `self`,
        taking values in `(N, n)` `DataFrame`-s
        Let :math:`x\in \mathbb{R}^{N\times n}`.
        By :math:`p_{X_{ij}}`, denote the marginal pdf of :math:`X` at the
        :math:`(i,j)`-th entry.

        The output of this method, for input `x` representing :math:`x`,
        is a `DataFrame` with same columns and indices as `self`,
        and entries :math:`\log p_{X_{ij}}(x_{ij})`.

        If `self` has a mixed or discrete distribution, this returns
        the weighted continuous part of `self`'s distribution instead of the pdf,
        i.e., the marginal pdf integrate to the weight of the continuous part.

        Parameters
        ----------
        x : `pandas.DataFrame` or 2D np.ndarray
            representing :math:`x`, as above

        Returns
        -------
        `DataFrame` with same columns and index as `self`
            containing :math:`\log p_{X_{ij}}(x_{ij})`, as above
        """
        if isinstance(x, pd.DataFrame):
            dist_at_x = self.loc[x.index, x.columns]
            tensor = dist_at_x.distr.log_prob(x.values)
            return pd.DataFrame(tensor, index=x.index, columns=x.columns)
        else:
            dist_at_x = self
            return dist_at_x.distr.log_prob(x)

    def cdf(self, x):
        """Cumulative distribution function."""
        if isinstance(x, pd.DataFrame):
            dist_at_x = self.loc[x.index, x.columns]
            tensor = dist_at_x.distr.cdf(x.values)
            return pd.DataFrame(tensor, index=x.index, columns=x.columns)
        else:
            dist_at_x = self
            return dist_at_x.distr.cdf(x)

    def sample(self, n_samples=None):
        """Sample from the distribution.

        Parameters
        ----------
        n_samples : int, optional, default = None
            number of samples to draw from the distribution

        Returns
        -------
        pd.DataFrame
            samples from the distribution

            * if ``n_samples`` is ``None``:
            returns a sample that contains a single sample from ``self``,
            in ``pd.DataFrame`` mtype format convention, with ``index`` and ``columns``
            as ``self``
            * if n_samples is ``int``:
            returns a ``pd.DataFrame`` that contains ``n_samples`` i.i.d.
            samples from ``self``, in ``pd-multiindex`` mtype format convention,
            with same ``columns`` as ``self``, and row ``MultiIndex`` that is product
            of ``RangeIndex(n_samples)`` and ``self.index``
        """
        if n_samples is None:
            np_spl = self.distr.sample().numpy()
            return pd.DataFrame(np_spl, index=self.index, columns=self.columns)
        else:
            np_spl = self.distr.sample(n_samples).numpy()
            np_spl = np_spl.reshape(-1, np_spl.shape[-1])
            mi = _prod_multiindex(range(n_samples), self.index)
            df_spl = pd.DataFrame(np_spl, index=mi, columns=self.columns)
            return df_spl


def _prod_multiindex(ix1, ix2):
    rows = []

    def add_rows(rows, ix):
        if isinstance(ix, pd.MultiIndex):
            ix = ix.to_frame()
            rows += [ix[col] for col in ix.columns]
        else:
            rows += [ix]
        return rows

    rows = add_rows(rows, ix1)
    rows = add_rows(rows, ix2)
    res = pd.MultiIndex.from_product(rows)
    res.names = [None] * len(res.names)
    return res


def is_scalar_notnone(obj):
    """Check if obj is scalar and not None."""
    return obj is not None and np.isscalar(obj)


def _get_first_argname(fun):
    """Get the name of the first argument of a function as str."""
    from inspect import signature

    return list(signature(fun).parameters.keys())[0]


def _coerce_to_pd_index_or_none(x):
    """Coerce to pd.Index, if not None, else return None."""
    if x is None:
        return None
    if isinstance(x, pd.Index):
        return x
    return pd.Index(x)
