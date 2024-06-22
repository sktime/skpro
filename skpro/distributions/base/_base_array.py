# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Base classes for probability array distribution objects."""

__author__ = ["ShreeshaM07"]

__all__ = ["_BaseArrayDistribution"]

import numpy as np
import pandas as pd

from skpro.base import BaseObject
from skpro.distributions.base import BaseDistribution
from skpro.distributions.base._base import (
    _coerce_to_pd_index_or_none,
    is_scalar_notnone,
)


class _BaseArrayDistribution(BaseDistribution, BaseObject):
    """Base Array probability distribution."""

    def __init__(self, index=None, columns=None):
        self.index = _coerce_to_pd_index_or_none(index)
        self.columns = _coerce_to_pd_index_or_none(columns)

        super().__init__(index=index, columns=columns)

    def _loc(self, rowidx=None, colidx=None):
        if is_scalar_notnone(rowidx) and is_scalar_notnone(colidx):
            return self._at(rowidx, colidx)
        if is_scalar_notnone(rowidx):
            rowidx = pd.Index([rowidx])
        if is_scalar_notnone(colidx):
            colidx = pd.Index([colidx])

        if rowidx is not None:
            row_iloc = pd.Index(self.index.get_indexer_for(rowidx))
        else:
            row_iloc = None
        if colidx is not None:
            col_iloc = pd.Index(self.columns.get_indexer_for(colidx))
        else:
            col_iloc = None
        return self._iloc(rowidx=row_iloc, colidx=col_iloc)

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
            if val is None:
                subset_param_dict[param] = None
                continue
            arr = val
            arr_shape = 2
            # when rowidx and colidx are integer while plotting
            if coerce_scalar:
                arr = arr[rowidx][colidx]
                subset_param_dict[param] = arr
                continue
            # subset the 2D distributions
            if arr_shape == 2 and rowidx is not None:
                _arr_shift = []
                if rowidx.values is not None and colidx is None:
                    rowidx_list = rowidx.values
                    for row in rowidx:
                        _arr_shift.append(arr[row])

                elif rowidx.values is not None and colidx.values is not None:
                    rowidx_list = rowidx.values
                    colidx_list = colidx.values
                    for row in rowidx_list:
                        _arr_shift_row = []
                        for col in colidx_list:
                            _arr_shift_row.append(arr[row][col])
                        _arr_shift.append(_arr_shift_row)
                arr = _arr_shift

            if arr_shape == 2 and rowidx is None:
                _arr_shift = []
                if colidx is not None:
                    colidx_list = colidx.values
                    for row in range(len(arr)):
                        _arr_shift_row = []
                        for col in colidx_list:
                            _arr_shift_row.append(arr[row][col])
                        _arr_shift.append(_arr_shift_row)
                    arr = _arr_shift

            subset_param_dict[param] = arr
        return subset_param_dict

    def _iloc(self, rowidx=None, colidx=None):
        if is_scalar_notnone(rowidx) and is_scalar_notnone(colidx):
            return self._iat(rowidx, colidx)
        if is_scalar_notnone(rowidx):
            rowidx = pd.Index([rowidx])
        if is_scalar_notnone(colidx):
            colidx = pd.Index([colidx])

        if rowidx is not None:
            rowidx = pd.Index(rowidx)
        if colidx is not None:
            colidx = pd.Index(colidx)

        subset_params = self._subset_params(rowidx=rowidx, colidx=colidx)

        def subset_not_none(idx, subs):
            if subs is not None:
                return idx.take(pd.Index(subs))
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

    def _check_single_arr_distr(self, value):
        return (
            isinstance(value[0], int)
            or isinstance(value[0], np.integer)
            or isinstance(value[0], float)
            or isinstance(value[0], np.floating)
        )

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

        # def row_to_col(arr):
        #     """Convert 1D arrays to 2D col arrays, leave 2D arrays unchanged."""
        #     if arr.ndim == 1 and oned_as == "col":
        #         return arr.reshape(-1, 1)
        #     return arr

        # kwargs_as_np = {k: row_to_col(np.array(v)) for k, v in kwargs.items()}
        kwargs_as_np = {k: v for k, v in kwargs.items()}

        if hasattr(self, "index") and self.index is not None:
            kwargs_as_np["index"] = self.index.to_numpy().reshape(-1, 1)
        if hasattr(self, "columns") and self.columns is not None:
            kwargs_as_np["columns"] = self.columns.to_numpy()

        bc_params = self.get_tags()["broadcast_params"]

        if bc_params is None:
            bc_params = kwargs_as_np.keys()

        args_as_np = [kwargs_as_np[k] for k in bc_params]

        if all(self._check_single_arr_distr(value) for value in kwargs_as_np.values()):
            # Convert all values in kwargs_as_np to np.array
            kwargs_as_np = {key: np.array(value) for key, value in kwargs_as_np.items()}
            shape = ()

            if return_shape:
                is_scalar = tuple([True] * (len(args_as_np) - 2))
                # print(kwargs_as_np,shape,is_scalar)
                return kwargs_as_np, shape, is_scalar
            return kwargs_as_np

        shape = (len(args_as_np[0]), len(args_as_np[0][0]))
        # create broadcast_array which will be same shape as the original bins
        # without considering the inner np.array containing the values of the bin edges
        # and bin masses. This will later get replaced by the values after broadcasting
        # index and columns.
        broadcast_array = np.arange(len(args_as_np[0]) * len(args_as_np[0][0])).reshape(
            shape
        )

        index_column_broadcast = [broadcast_array] * (len(args_as_np) - 2)
        index_column_broadcast.append(kwargs_as_np["index"])
        index_column_broadcast.append(kwargs_as_np["columns"])

        bc = np.broadcast_arrays(*index_column_broadcast)
        if dtype is not None:
            bc = [array.astype(dtype) for array in bc]

        for i in range(len(bc) - 2):
            bc[i] = args_as_np[i]

        for i, k in enumerate(bc_params):
            kwargs_as_np[k] = bc[i]

        if return_shape:
            is_scalar = tuple([False] * (len(args_as_np) - 2))
            # print(kwargs_as_np,shape,is_scalar)
            return kwargs_as_np, shape, is_scalar
        return kwargs_as_np
