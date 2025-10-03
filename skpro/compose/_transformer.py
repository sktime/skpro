"""Transformers for skpro."""

import numpy as np
from sklearn.base import TransformerMixin, check_is_fitted, clone
from sklearn.exceptions import NotFittedError

from skpro.base import BaseEstimator


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

    def _fit_with_fitted_transformer(self):
        """Fit with already fitted transformer if possible."""
        try:
            check_is_fitted(self.transformer)
            self.transformer_ = self.transformer
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

        # TODO: sklearn <1.2 compat issue
        self.transformer_.set_output(transform="pandas")
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
        Xt = self.transformer_.transform(X)
        return Xt

    def inverse_transform(self, X, y=None):
        """Inverse transform y using the transformer."""
        if self.transformer_ is not None:
            Xt = self.transformer_.inverse_transform(X)
        else:
            Xt = X
        return Xt


class BaseDifferentiableTransformer(BaseTransformer):
    """Differentiable transformer."""

    def __init__(self, transformer, transform_diff_func=None, inverse_diff_func=None):
        """Differentiable transformer.

        Parameters
        ----------
        transformer : callable, optional
            Maybe only allow sklearn transformers for now.
        transform_diff_func : callable, optional
            Function to compute the derivative of the transform function.
        inverse_diff_func : callable, optional
            Function to compute the derivative of the inverse transform function.
        """
        self.transform_diff_func = transform_diff_func
        self.inverse_diff_func = inverse_diff_func
        super().__init__(transformer=transformer)

    def transform_diff(self, X):
        """Compute the derivative of the transform function at X.

        Parameters
        ----------
        X : array-like, shape (n_samples,) or (n_samples, 1)
            Input data.
        """
        if self.transform_diff_func is not None:
            return self.transform_diff_func(X)
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

        if self.inverse_diff_func is not None:
            diff = self.inverse_diff_func(Xt)
        elif (
            hasattr(self.transformer_, "scale_")
            and self.transformer_.scale_ is not None
        ):
            diff = np.ones_like(Xt) / self.transformer_.scale_
        else:
            diff = self._numerical_diff(
                self.transformer_.inverse_transform,
                Xt,
            )

        return diff

    def _numerical_diff(self, func, X):
        """Apply numerical differentiation.

        Parameters
        ----------
        func : callable
            Function to differentiate.
        X : array-like, shape (n_samples,) or (n_samples, 1)
            Input data.

        Returns
        -------
        diff : array-like, shape (n_samples,) or (n_samples, 1)
            Numerical derivative of the transformation at X.
        """
        # TODO: use finite difference
        X = np.asarray(X)
        original_shape = X.shape
        X = X.flatten()
        sort_idx = np.argsort(X)
        x_sorted = X[sort_idx]
        y_sorted = func(x_sorted.reshape(-1, 1)).flatten()
        grad = np.gradient(y_sorted, x_sorted)
        diff = np.zeros_like(X)
        diff[sort_idx] = grad
        return diff.reshape(original_shape)


class DifferentiableTransformer(BaseDifferentiableTransformer):
    """Differentiable transformer for TTR with default numerical differentiation."""

    def __init__(self, transformer):
        """Differentiable transformer for TTR with default numerical differentiation.

        Parameters
        ----------
        transformer : callable, optional
            Maybe only allow sklearn transformers for now.
        """
        super().__init__(transformer=transformer)
