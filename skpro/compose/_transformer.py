"""Transformers for skpro."""

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, check_is_fitted, clone
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import FunctionTransformer

from skpro.base import BaseEstimator

# Optional import for more advanced numerical differentiation
try:
    import numdifftools as nd

    HAS_NUMDIFFTOOLS = True
except ImportError:
    HAS_NUMDIFFTOOLS = False


class BaseTransformer(BaseEstimator, TransformerMixin):
    """Base class for transformer objects."""

    def __init__(self, transformer):
        """Initialise the transformer objects.

        Parameters
        ----------
        transformer : callable, optional
            Maybe only allow sklearn transformers for now.
        """
        self.transformer = transformer
        super().__init__()

    def _fit_with_fitted_transformer(self, index=None, columns=None):
        """Fit with already fitted transformer if possible."""
        try:
            check_is_fitted(self.transformer)
            self.transformer_ = self.transformer
            self.index = index
            self.columns = columns
        except NotFittedError:
            pass
        return self

    def fit(self, X, y=None):
        """Fit transformer to y.

        Parameters
        ----------
        X : array-like, shape (n_samples,) or (n_samples, n_outputs)
            Target values.
        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : reference to self
        """
        self.transformer_ = clone(self.transformer)
        self.transformer_.fit(X)

        if isinstance(X, pd.DataFrame):
            self.index = X.index
            self.columns = X.columns
        else:
            X = np.asarray(X)
            if X.ndim == 1:
                self.index = pd.RangeIndex(X.shape[0])
                self.columns = pd.Index([0])
            else:
                n_samples, n_cols = X.shape
                if self.index is None:
                    self.index = pd.RangeIndex(n_samples)
                if self.columns is None:
                    self.columns = pd.RangeIndex(n_cols)
        return self

    def transform(self, X, y=None):
        """Transform y using the transformer.

        Parameters
        ----------
        X : array-like, shape (n_samples,) or (n_samples, n_outputs)
            Target values.
        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        Xt : array-like, shape (n_samples,) or (n_samples, n_outputs)
            Transformed target values.
        """
        check_is_fitted(self.transformer_)
        Xt = self.transformer_.transform(X)

        if isinstance(Xt, pd.DataFrame):
            Xt.index = self.index
            Xt.columns = self.columns
        else:
            Xt = np.asarray(Xt)
            if Xt.ndim == 1:
                Xt = pd.Series(Xt, index=self.index, name=self.columns[0])
            else:
                Xt = pd.DataFrame(Xt, index=self.index, columns=self.columns)
        return Xt

    def inverse_transform(self, X, y=None):
        """Inverse transform y using the transformer."""
        check_is_fitted(self.transformer_)
        Xt = self.transformer_.inverse_transform(X)

        if isinstance(Xt, pd.DataFrame):
            Xt.index = self.index
            Xt.columns = self.columns
        else:
            Xt = np.asarray(Xt)
            if Xt.ndim == 1:
                Xt = pd.Series(Xt, index=self.index, name=self.columns[0])
            elif Xt.ndim == 0:
                Xt = Xt.item()
            else:
                Xt = pd.DataFrame(
                    Xt.reshape(len(self.index), len(self.columns)),
                    index=self.index,
                    columns=self.columns,
                )

        return Xt


class BaseDifferentiableTransformer(BaseTransformer):
    """Differentiable transformer."""

    def __init__(self, transformer, transform_func_diff=None, inverse_func_diff=None):
        """Differentiable transformer.

        Parameters
        ----------
        transformer : callable, optional
            Maybe only allow sklearn transformers for now.
        transform_func_diff : callable, optional
            Function to compute the derivative of the transform function.
        inverse_func_diff : callable, optional
            Function to compute the derivative of the inverse transform function.
        """
        self.transform_func_diff = transform_func_diff
        self.inverse_func_diff = inverse_func_diff
        super().__init__(transformer=transformer)

    def transform_diff(self, X):
        """Compute the derivative of the transform function at X.

        Parameters
        ----------
        X : array-like, shape (n_samples,) or (n_samples, 1)
            Input data.
        """
        if self.transform_func_diff is not None:
            return self.transform_func_diff(X)
        elif (
            hasattr(self.transformer_, "scale_")
            and self.transformer_.scale_ is not None
        ):
            return self.transformer_.scale_
        else:
            return self._numerical_diff(
                self.transformer_.transform,
                X,
            )

    def inverse_transform_diff(self, X):
        """Compute the derivative of the inverse transform function at X.

        Parameters
        ----------
        X : array-like, shape (n_samples,) or (n_samples, 1)
            Input data.
        """
        Xt = self.transform(X)

        if self.inverse_func_diff is not None:
            diff = self.inverse_func_diff(Xt)
        elif (
            hasattr(self.transformer_, "scale_")
            and self.transformer_.scale_ is not None
        ):
            diff = np.ones_like(Xt) / self.transformer_.scale_
        else:
            diff = self._numerical_diff(self.transformer_.inverse_transform, Xt)

        return diff

    def _numerical_diff(self, func, X, delta=1e-6):
        """Apply numerical differentiation using central difference.

        Parameters
        ----------
        func : callable
            Function to differentiate.
        X : array-like, shape (n_samples,) or (n_samples, 1)
            Input data.
        delta : float, default=1e-6
            Step size for numerical differentiation.

        Returns
        -------
        diff : array-like, shape (n_samples,) or (n_samples, 1)
            Numerical derivative of the transformation at X.
        """
        X = np.asarray(X)
        original_shape = X.shape
        X_flat = X.flatten()

        if HAS_NUMDIFFTOOLS:
            # Use numdifftools for more accurate differentiation if available
            def func_1d(x_val):
                # Handle scalar input for numdifftools
                if np.isscalar(x_val):
                    x_val = np.array([[x_val]])
                else:
                    x_val = np.array(x_val).reshape(-1, 1)
                return func(x_val).flatten()[0]

            # Calculate derivative for each point
            diff = np.zeros_like(X_flat)
            for i, x_val in enumerate(X_flat):
                derivative_func = nd.Derivative(func_1d, n=1, step=delta)
                diff[i] = derivative_func(x_val)
        else:
            # Use custom central difference implementation
            diff = np.zeros_like(X_flat)

            for i, x_val in enumerate(X_flat):
                # Central difference: f'(x) = (f(x+h) - f(x-h)) / (2*h)
                x_plus = np.array([[x_val + delta]])
                x_minus = np.array([[x_val - delta]])

                f_plus = func(x_plus).flatten()[0]
                f_minus = func(x_minus).flatten()[0]

                diff[i] = (f_plus - f_minus) / (2 * delta)

        return diff.reshape(original_shape)

    def _check_inverse_func(self):
        """Check if inverse function is available."""
        if not hasattr(self.transformer_, "inverse_transform") or (
            hasattr(self.transformer_, "inverse_func")
            and self.transformer_.inverse_func is None
        ):
            return False
        else:
            return True

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Get test parameters for this class.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.

        Returns
        -------
        params : dict or list of dict, default={}
            Parameters to create testing instances of the class.
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `None` if no constructor arguments are required.
        """
        from sklearn.preprocessing import MinMaxScaler

        params1 = {"transformer": MinMaxScaler()}
        params2 = {"transformer": np.log}
        return [params1, params2]


class DifferentiableTransformer(BaseDifferentiableTransformer):
    """Differentiable transformer for TTR with default numerical differentiation."""

    def __init__(self, transformer, transform_func_diff=None, inverse_func_diff=None):
        """Differentiable transformer for TTR with default numerical differentiation.

        Parameters
        ----------
        transformer : callable, optional
            Maybe only allow sklearn transformers for now.
        """
        super().__init__(
            transformer=transformer, transform_func_diff=None, inverse_func_diff=None
        )

    @classmethod
    def _coerce_to_differentiable(cls, transform, index=None, columns=None):
        """Coerce transform to DifferentiableTransformer if possible.

        Takes a BaseDifferentiableTransformer, TransformerMixin, a bound method
        of a TransformerMixin or a function and returns a DifferentiableTransformer.

        Parameters
        ----------
        transform : callable or BaseDifferentiableTransformer or TransformerMixin
            transformation to be coerced
        index : pd.Index, optional
            index to be used if transform is not a fitted transformer
        columns : pd.Index, optional
            columns to be used if transform is not a fitted transformer

        Returns
        -------
        transformer_ : BaseDifferentiableTransformer
            differentiable transformer
        """
        if isinstance(transform, BaseDifferentiableTransformer):
            transformer = transform
        elif isinstance(transform, TransformerMixin):
            transformer = DifferentiableTransformer(transformer=transform)
        elif callable(transform):
            bound_instance = getattr(transform, "__self__", None)

            if isinstance(bound_instance, TransformerMixin):
                transformer = DifferentiableTransformer(transformer=bound_instance)
            else:
                # could add a numerical inverse if desired
                ft = FunctionTransformer(func=transform, check_inverse=False)
                transformer = DifferentiableTransformer(transformer=ft)
        else:
            raise ValueError(
                "Transform must be a callable, a TransformerMixin, "
                "or a BaseDifferentiableTransformer."
            )

        return transformer._fit_with_fitted_transformer(index=index, columns=columns)
