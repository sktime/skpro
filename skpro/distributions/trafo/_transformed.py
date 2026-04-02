# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Transformed distribution."""

__author__ = ["fkiraly"]

import warnings

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.pipeline import FunctionTransformer

from skpro.compose._transformer import DifferentiableTransformer
from skpro.distributions.base import BaseDistribution


class TransformedDistribution(BaseDistribution):
    r"""Distribution transformed by an entry-wise function.

    Constructed with:

    * ``distribution``: a skpro distribution object, which is transformed
    * ``func`` and ``inverse_func``: functions that are applied to the distribution
    * ``transformer``: an sklearn transformer (forward: dist→output)

    Parameters
    ----------
    distribution : skpro distribution - must be same shape as ``self``

    transformer : sklearn transformer, optional, default = None
        Transformer where ``.transform()`` is the forward direction (dist→output).
        Takes precedence over ``func``/``inverse_func``.

    assume_monotonic : bool, optional, default = True
        whether to assume that the transform is monotonic, i.e., that
        the distribution is transformed in a way that preserves order of sample values.

    transform : callable, default = None
        function that transforms the distribution, must be applicable to array-likes of
        the same shape as ``self``. Only used if ``transformer`` is None.

    inverse_func : callable, optional, default = None
        inverse function of ``func``, must be applicable to array-likes of the
        same shape as ``self``.
        Only used if ``transformer`` is None.


    index : pd.Index, optional, default = RangeIndex
    columns : pd.Index, optional, default = RangeIndex

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from sklearn.preprocessing import FunctionTransformer
    >>> from skpro.distributions.trafo import TransformedDistribution
    >>> from skpro.distributions import Normal

    >>> n = Normal(mu=0, sigma=1)
    >>> # transform the distribution by taking the exponential
    >>> t = TransformedDistribution(distribution=n, func=np.exp)

    If the inverse is known, it can be given to ensure more methods are exact:

    >>> t = TransformedDistribution(
    ...     distribution=n,
    ...     func=np.exp,
    ...     inverse_func=np.log,
    ... )

    It can also be constructed with a ``FunctionTransformer``:
    >>> ft = FunctionTransformer(func=np.log, inverse_func=np.exp)
    >>> t = TransformedDistribution(distribution=n, transformer=ft)

    For internal use with TransformedTargetRegressor, use the factory method:
    >>> from sklearn.preprocessing import StandardScaler
    >>> scaler = StandardScaler().fit([[1.0], [2.0], [3.0]])
    >>> t = TransformedDistribution._from_inverse_transformer(
    ...     distribution=n, transformer=scaler
    ... )
    """

    _tags = {
        "capabilities:approx": [
            "mean",
            "var",
            "energy",
            "cdf",
        ],
        "capabilities:exact": ["ppf"],
        "distr:measuretype": "mixed",
        "distr:paramtype": "composite",
    }

    def __init__(
        self,
        distribution,
        transform=None,
        inverse_func=None,
        transformer=None,
        assume_monotonic=True,
        index=None,
        columns=None,
    ):
        # Store all parameters for get_params
        self.distribution = distribution
        # this is a transformer object
        self.transformer = transformer
        self.assume_monotonic = assume_monotonic
        # transform is a Callable
        self.transform = transform
        self.inverse_func = inverse_func

        self._is_scalar_dist = self.distribution.ndim == 0

        # determine index and columns
        if not self._is_scalar_dist:
            if index is None or columns is None:
                _example = self.distribution.sample()
                n_rows = _example.shape[0]
                n_cols = _example.shape[1]
                if index is None:
                    index = pd.RangeIndex(n_rows)
                if columns is None:
                    columns = pd.RangeIndex(n_cols)

        if transformer is not None:
            self.transformer_ = _coerce_to_diff_transformer(
                transformer, None, index=index, columns=columns
            )

        elif transform is not None:
            self.transformer_ = _coerce_to_diff_transformer(
                transform, inverse_func, index=index, columns=columns
            )

        else:
            raise ValueError(
                "At least one of 'transformer' or 'transform' must be provided."
            )

        super().__init__(index=index, columns=columns)

        # transformed discrete distributions are always discrete
        # (otherwise we only know that they are mixed)
        if distribution.get_tag("distr:measuretype") == "discrete":
            self.set_tags(**{"distr:measuretype": "discrete"})

        # Check inverse function availability
        inner_inverse_diff_status = self.transformer_._get_transform_diff_capabilities()

        # Check if we have external->internal transform (inner inverse)
        # This is the inverse_func that maps from transformed to distribution space
        has_inverse_func = (
            hasattr(self.transformer_, "transformer_")
            and hasattr(self.transformer_.transformer_, "inverse_func")
            and self.transformer_.transformer_.inverse_func is not None
        )

        self._has_inner_inverse = any(
            [
                has_inverse_func,
                inverse_func is not None,
            ]
        )

        # Set capabilities tags given the state of inner inverse and its derivative
        if inner_inverse_diff_status or self._has_inner_inverse:
            exact_methods = []
            approx_methods = ["pdfnorm", "mean", "var", "energy"]

            # ppf can be exact when either monotonic or inner_inverse is available
            exact_methods.append("ppf")

            # cdf can only be exact when separate inner_inverse is provided
            if self._has_inner_inverse:
                exact_methods.append("cdf")
            else:
                # No inner_inverse, cdf will use sampling approximation
                approx_methods.append("cdf")

            # pdf and log_pdf require an inner_inverse derivative
            if inner_inverse_diff_status == "exact":
                exact_methods.extend(["pdf", "log_pdf"])
            elif inner_inverse_diff_status == "approx":
                approx_methods.extend(["pdf", "log_pdf"])

            self.set_tags(
                **{
                    "capabilities:exact": exact_methods,
                    "capabilities:approx": approx_methods,
                }
            )

    @classmethod
    def _from_inverse_transformer(
        cls,
        distribution,
        transformer,
        assume_monotonic=True,
        index=None,
        columns=None,
    ):
        """Create TransformedDistribution from a transformer with inverted semantics.

        This is a private factory method for internal use, primarily by
        TransformedTargetRegressor. When TTR transforms targets, its transformer's
        ``.transform()`` maps original→transformed space, but the distribution lives
        in transformed space. This factory handles the semantic swap.

        Parameters
        ----------
        distribution : skpro distribution
            Distribution in the transformed space.
        transformer : sklearn transformer
            Transformer where ``.inverse_transform()`` should be used as the forward
            direction (transformed→original space for sampling).
        assume_monotonic : bool, optional, default=True
            Whether to assume the transform is monotonic.
        index : pd.Index, optional
            Index for the transformed distribution.
        columns : pd.Index, optional
            Columns for the transformed distribution.

        Returns
        -------
        TransformedDistribution
            Distribution with semantics correctly swapped for the inverse transformer.
        """
        # Create instance by passing bound methods as callables
        # This swaps the transform directions:
        # - transformer.inverse_transform becomes our forward transform (D→T)
        # - transformer.transform becomes our inverse transform (T→D)
        return cls(
            distribution=distribution,
            transform=transformer.inverse_transform,
            inverse_func=transformer.transform,
            assume_monotonic=assume_monotonic,
            index=index,
            columns=columns,
        )

    def _pdf(self, x):
        r"""Probability density function.

        Provides exact "change-of-variables" pdf if either:
          * the user passes a DifferentiableTransformer with transform_func_diff method
            to the transform kwarg,
          * the transformer is a scaler transformer (has `scaler_` attribute)

        Provides approximate "change-of-variables" pdf via numerical differentiation if:
          * the user passes a DifferentiableTransformer without transform_func_diff
            method.
          * the transformer is not a scaler transformer (lacks `scaler_` attribute)
          * the user passes inverse_transform function

        Parameters
        ----------
        x : pd.DataFrame, same shape as ``self``
            points where the pdf is evaluated

        Returns
        -------
        pd.DataFrame, same shape as ``self``
            pdf values at the given points
        """
        dist = self.distribution
        # Use inverse transform (T→D) to map x back to distribution space
        x_ = self.transformer_.inverse_transform(x)
        pdf_out = dist.pdf(x_)

        if isinstance(pdf_out, pd.DataFrame):
            if self.index is not None:
                pdf_out.index = self.index
            if self.columns is not None:
                pdf_out.columns = self.columns
        elif not self._is_scalar_dist:
            pdf_out = pd.DataFrame(pdf_out, index=self.index, columns=self.columns)

        if self.transformer_._get_transform_diff_capabilities():
            # Ensure x is a DataFrame for transform_diff if distribution has columns
            # This avoids sklearn warnings about feature names
            if not isinstance(x, pd.DataFrame) and self.columns is not None:
                x_df = pd.DataFrame(x, columns=self.columns)
                # Use derivative of inverse transform (T→D)
                # x is already in transformed space, so use _inverse_transform_diff
                jacobian = np.abs(
                    self.transformer_._inverse_transform_diff(x_df.values)
                )
            else:
                # Use derivative of inverse transform (T→D)
                # x is already in transformed space, so use _inverse_transform_diff
                if isinstance(x, pd.DataFrame):
                    jacobian = np.abs(
                        self.transformer_._inverse_transform_diff(x.values)
                    )
                else:
                    jacobian = np.abs(self.transformer_._inverse_transform_diff(x))
        else:
            raise ValueError(
                "The TransformedDistribution must have a transform to compute the pdf: "
                "Either pass a DifferentiableTransformer, Transform as transform, "
                "or provide an inverse_transform ufunc."
            )

        return pdf_out * jacobian

    def _log_pdf(self, x):
        r"""Logarithmic probability density function.

        Provides exact "change-of-variables" log_pdf if either:
          * the user passes a DifferentiableTransformer with transform_func_diff method
            to the transform kwarg,
          * the transformer is a scaler transformer (has `scaler_` attribute)

        Provides approximate "change-of-variables" log_pdf via numerical
        differentiation if:
          * the user passes a DifferentiableTransformer without transform_func_diff
            method.
          * the transformer is not a scaler transformer (lacks `scaler_` attribute)
          * the user passes inverse_transform function


        Parameters
        ----------
        x : pd.DataFrame, same shape as ``self``
            points where the log-pdf is evaluated

        Returns
        -------
        pd.DataFrame, same shape as ``self``
            log-pdf values at the given points
        """
        dist = self.distribution
        # Use inverse transform (T→D) to map x back to distribution space
        x_ = self.transformer_.inverse_transform(x)
        log_pdf_out = dist.log_pdf(x_)

        if isinstance(log_pdf_out, pd.DataFrame):
            if self.index is not None:
                log_pdf_out.index = self.index
            if self.columns is not None:
                log_pdf_out.columns = self.columns
        elif not self._is_scalar_dist:
            log_pdf_out = pd.DataFrame(
                log_pdf_out, index=self.index, columns=self.columns
            )

        if self.transformer_._get_transform_diff_capabilities():
            # Ensure x is a DataFrame for transform_diff if distribution has columns
            # This avoids sklearn warnings about feature names
            if not isinstance(x, pd.DataFrame) and self.columns is not None:
                x_df = pd.DataFrame(x, columns=self.columns)
                # Use derivative of inverse transform (T→D)
                # x is already in transformed space, so use _inverse_transform_diff
                jacobian = self.transformer_._inverse_transform_diff(x_df.values)
            else:
                # Use derivative of inverse transform (T→D)
                # x is already in transformed space, so use _inverse_transform_diff
                if isinstance(x, pd.DataFrame):
                    jacobian = self.transformer_._inverse_transform_diff(x.values)
                else:
                    jacobian = self.transformer_._inverse_transform_diff(x)
        else:
            raise ValueError(
                "The TransformedDistribution must have a transform to compute the "
                "log-pdf.",
            )

        return log_pdf_out + np.log(np.abs(jacobian))

    def _iloc(self, rowidx=None, colidx=None):
        if is_scalar_notnone(rowidx) and is_scalar_notnone(colidx):
            return self._iat(rowidx, colidx)
        if is_scalar_notnone(rowidx):
            rowidx = pd.Index([rowidx])
        if is_scalar_notnone(colidx):
            colidx = pd.Index([colidx])

        def subset_not_none(idx, subs):
            if subs is not None:
                return idx.take(subs)
            else:
                return idx

        index_subset = subset_not_none(self.index, rowidx)
        columns_subset = subset_not_none(self.columns, colidx)

        distr = self.distribution.iloc[rowidx, colidx]

        # these parameters are manually subset
        # the other parameters are passed through
        POP_PARAMS = ["distribution", "index", "columns"]

        sk_distr_type = type(self)
        params_dict = self.get_params(deep=False)
        [params_dict.pop(param) for param in POP_PARAMS]

        return sk_distr_type(
            distribution=distr,
            index=index_subset,
            columns=columns_subset,
            **params_dict,
        )

    def _iat(self, rowidx=None, colidx=None):
        if rowidx is None or colidx is None:
            raise ValueError("iat method requires both row and column index")
        self_subset = self.iloc[[rowidx], [colidx]]

        POP_PARAMS = ["distribution", "index", "columns"]

        sk_distr_type = type(self)
        params_dict = self.get_params(deep=False)
        [params_dict.pop(param) for param in POP_PARAMS]

        return sk_distr_type(
            distribution=self_subset.distribution.iat[0, 0],
            **params_dict,
        )

    def _ppf(self, p):
        """Quantile function = percent point function = inverse cdf.

        Parameters
        ----------
        p : 2D np.ndarray, same shape as ``self``
            values to evaluate the ppf at

        Returns
        -------
        2D np.ndarray, same shape as ``self``
            ppf values at the given points
        """
        if not self.assume_monotonic and not self._has_inner_inverse:
            raise ValueError(
                "if inverse_transform is not given, "
                "ppf is implemented only for monotonic transforms, "
                "set `assume_monotonic=True` to use this method"
            )

        # inner_inverse exists
        elif not self.assume_monotonic and self._has_inner_inverse:
            return super().ppf(p)

        if self.ndim != 0:
            p = pd.DataFrame(p, index=self.index, columns=self.columns)

        # Use forward transform (D→T) to map from distribution space to output
        trafo = self.transformer_.transform
        inner_ppf = self.distribution.ppf(p)

        if self.ndim == 0:
            inner_ppf = pd.DataFrame([[float(inner_ppf)]])
        outer_ppf = trafo(inner_ppf)
        if self.ndim == 0:
            outer_ppf = _coerce_to_scalar(outer_ppf)

        if isinstance(outer_ppf, pd.DataFrame):
            # if the transform returns a DataFrame, we ensure the index and columns
            outer_ppf.index = self.index
            outer_ppf.columns = self.columns
        elif not self._is_scalar_dist:
            # if the transform returns a scalar or array, we  convert it to DataFrame
            outer_ppf = pd.DataFrame(outer_ppf, index=self.index, columns=self.columns)

        return outer_ppf

    def _cdf(self, x):
        """Cumulative distribution function.

        Parameters
        ----------
        x : 2D np.ndarray, same shape as ``self``
            values to evaluate the cdf at

        Returns
        -------
        2D np.ndarray, same shape as ``self``
            cdf values at the given points
        """
        # if no inner_inverse use sampling-based approximation
        if not self._has_inner_inverse:
            return super()._cdf(x)

        # Use inverse transform (T→D) to map x back to distribution space
        inv_trafo = self.transformer_.inverse_transform

        if self.ndim != 0:
            x = pd.DataFrame(x, index=self.index, columns=self.columns)
        else:
            x = pd.DataFrame([[float(x)]])

        inv_x = inv_trafo(x)
        if self.ndim == 0:
            inv_x = _coerce_to_scalar(inv_x)

        cdf_res = self.distribution.cdf(inv_x)

        if isinstance(cdf_res, pd.DataFrame):
            # if the transform returns a DataFrame, we ensure the index and columns
            cdf_res.index = self.index
            cdf_res.columns = self.columns
        elif not self._is_scalar_dist:
            # if the transform returns a scalar or array, we  convert it to DataFrame
            cdf_res = pd.DataFrame(cdf_res, index=self.index, columns=self.columns)

        return cdf_res

    def _sample(self, n_samples=None):
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
        # else we sample manually, this will be less efficient due to loops
        if n_samples is None:
            n = 1
        else:
            n = n_samples

        # Use forward transform (D→T)
        trafo = self.transformer_.transform

        samples = []

        for _ in range(n):
            new_sample = trafo(self.distribution.sample())
            if not self._is_scalar_dist:
                if not isinstance(new_sample, pd.DataFrame):
                    new_sample = pd.DataFrame(
                        new_sample, index=self.index, columns=self.columns
                    )
                else:
                    new_sample.index = self.index
                    new_sample.columns = self.columns
            samples.append(new_sample)

        if n_samples is None:
            return samples[0]

        if self._is_scalar_dist:
            return pd.DataFrame(samples)

        # if n_samples is int, we return a DataFrame with MultiIndex
        res = pd.concat(samples, axis=0, keys=range(n))
        return res

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        from sklearn.preprocessing import FunctionTransformer

        from skpro.compose._transformer import DifferentiableTransformer
        from skpro.distributions import Normal

        # use arcsinh as both it and its inverse are defined on all of R
        ft = FunctionTransformer(func=np.arcsinh, inverse_func=np.sinh)

        dft = DifferentiableTransformer(
            ft, inverse_func_diff=lambda x: 1 / ((x**2 + 1) ** 0.5)
        )

        n_scalar = Normal(mu=0, sigma=1)
        # scalar case example
        params1 = {"distribution": n_scalar, "transform": np.exp}

        # array case example
        n_array = Normal(mu=[[1, 2], [3, 4]], sigma=1, columns=pd.Index(["a", "b"]))

        params2 = {
            "distribution": n_array,
            "transform": np.exp,
            "index": pd.Index([1, 2]),
            "columns": pd.Index(["a", "b"]),  # this should override n_row.columns
        }

        # array case example with inverse transform
        n_array = Normal(mu=[[1, 2], [3, 4]], sigma=1, columns=pd.Index(["c", "d"]))
        params3 = {
            "distribution": n_array,
            "transform": lambda x: 2 * x,
            "inverse_func": lambda x: x / 2,
            "index": pd.Index([1, 2]),
            "columns": pd.Index(["a", "b"]),  # this should override n_row.columns
        }

        params4 = {
            "distribution": n_array,
            "transformer": ft,
            "index": pd.Index([1, 2]),
            "columns": pd.Index(["a", "b"]),  # this should override n_row.columns
        }

        # scalar case example with inverse transform
        params5 = {
            "distribution": n_scalar,
            "transform": np.exp,
            "inverse_func": np.log,
        }

        params6 = {
            "distribution": n_array,
            "transformer": dft,
        }

        return [params1, params2, params3, params4, params5, params6]


def is_scalar_notnone(obj):
    """Check if obj is scalar and not None."""
    return obj is not None and np.isscalar(obj)


def _coerce_to_scalar(x):
    """Coerce numpy or pd.DataFrame to numpy float."""
    if isinstance(x, pd.DataFrame):
        if x.shape != (1, 1):
            raise ValueError("input must be of shape (1, 1) to coerce to scalar")
        return x.iat[0, 0]
    elif isinstance(x, np.ndarray):
        if x.shape != (1, 1):
            raise ValueError("input must be of shape (1, 1) to coerce to scalar")
        return x[0, 0]
    else:
        return x


def _coerce_to_diff_transformer(
    transform, inverse_transform=None, index=None, columns=None
):
    """Coerce transform to DifferentiableTransformer.

    Takes a callable, Transformer, or DifferentiableTransformer and returns
    a DifferentiableTransformer.

    Parameters
    ----------
    transform : callable, Transformer, or DifferentiableTransformer
        transformation to be coerced
    inverse_transform : callable, optional
        inverse transformation function, if known
    index : pd.Index, optional
        index to be used if transform is not a fitted transformer
    columns : pd.Index, optional
        columns to be used if transform is not a fitted transformer

    Returns
    -------
    transformer_ : DifferentiableTransformer
        differentiable transformer
    """
    if isinstance(transform, DifferentiableTransformer):
        transformer = transform
    elif isinstance(transform, TransformerMixin):
        transformer = DifferentiableTransformer(transformer=transform)
    elif callable(transform):
        transform_owner = getattr(transform, "__self__", None)
        inverse_owner = getattr(inverse_transform, "__self__", None)

        # Check if methods come from a DifferentiableTransformer
        # If so, extract derivative functions and swap them
        transform_func_diff = None
        inverse_func_diff = None

        if isinstance(transform_owner, DifferentiableTransformer):
            # transform is inv_transform method, inv_transform is transform method
            # So derivative functions need to be swapped
            transform_func_diff = getattr(transform_owner, "inverse_func_diff", None)
            inverse_func_diff = getattr(transform_owner, "transform_func_diff", None)
        elif isinstance(inverse_owner, DifferentiableTransformer):
            # inv_transform is transform method, transform is inv_transform method
            inverse_func_diff = getattr(inverse_owner, "transform_func_diff", None)
            transform_func_diff = getattr(inverse_owner, "inverse_func_diff", None)

        if any(
            [
                isinstance(transform_owner, TransformerMixin),
                isinstance(inverse_owner, TransformerMixin),
            ]
        ):
            warnings.warn(
                "If passing transformer methods `transform` or "
                "`inverse_transform`, consider passing the full transformer"
                "instance instead.",
                UserWarning,
                stacklevel=2,
            )

        ft = FunctionTransformer(
            func=transform, inverse_func=inverse_transform, check_inverse=False
        )

        transformer = DifferentiableTransformer(
            transformer=ft,
            transform_func_diff=transform_func_diff,
            inverse_func_diff=inverse_func_diff,
        )
    else:
        raise ValueError(
            "Transform must be a callable, TransformerMixin, "
            "or DifferentiableTransformer."
        )

    # If transformer is already fitted, use it directly
    # If not fitted, just set columns for later fitting
    return transformer._wrap_if_fitted(columns=columns)
