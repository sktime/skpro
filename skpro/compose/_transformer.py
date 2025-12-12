"""Transformers for skpro."""

import warnings

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, check_is_fitted, clone
from sklearn.exceptions import NotFittedError

from skpro.base import BaseEstimator

# Optional import for scipy derivative
HAS_SCIPY_DERIVATIVE = False
scipy_derivative = None
SCIPY_DERIVATIVE_NEW_API = False

try:
    # Try scipy.differentiate first (scipy >= 1.14.0)
    from scipy.differentiate import derivative as scipy_derivative

    HAS_SCIPY_DERIVATIVE = True
    SCIPY_DERIVATIVE_NEW_API = True
except ImportError:
    try:
        # Fall back to scipy.misc (scipy < 1.14.0)
        # Note: scipy.misc.derivative is deprecated and removed in scipy 2.0.0
        from scipy.misc import derivative as scipy_derivative

        HAS_SCIPY_DERIVATIVE = True
        SCIPY_DERIVATIVE_NEW_API = False
    except (ImportError, AttributeError):
        pass


class BaseTransformer(BaseEstimator, TransformerMixin):
    """Base class for transformer objects."""

    def __init__(self):
        """Initialise the transformer objects."""
        super().__init__()

    def fit(self, X, y=None):
        """Fit transformer to X.

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
        if isinstance(X, pd.DataFrame):
            self.columns_ = X.columns
        else:
            self.columns_ = None

        self._fit(X, y)
        self._is_fitted = True
        return self

    def _fit(self, X, y=None):
        """Fit transformer to X.

        Private method to be implemented by subclasses.

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
        raise NotImplementedError("Subclasses must implement _fit method")

    def transform(self, X, y=None):
        """Transform X using the transformer.

        Parameters
        ----------
        X : array-like, shape (n_samples,) or (n_samples, n_outputs)
            Input values to transform.
        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        Xt : pd.DataFrame or pd.Series
            Transformed target values.
        """
        return self._transform(X, y)

    def _transform(self, X, y=None):
        """Transform X using the transformer.

        Private method to be implemented by subclasses.
        Subclasses should return a DataFrame or Series.

        Parameters
        ----------
        X : array-like, shape (n_samples,) or (n_samples, n_outputs)
            Input values to transform.
        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        Xt : pd.DataFrame or pd.Series
            Transformed values.
        """
        raise NotImplementedError("Subclasses must implement _transform method")

    def inverse_transform(self, X, y=None):
        """Inverse transform X using the transformer.

        Parameters
        ----------
        X : array-like, shape (n_samples,) or (n_samples, n_outputs)
            Input values to inverse transform.
        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        Xt : pd.DataFrame or pd.Series
            Inverse transformed values.
        """
        return self._inverse_transform(X, y)

    def _inverse_transform(self, X, y=None):
        """Inverse transform X using the transformer.

        Private method to be implemented by subclasses.
        Subclasses should return a DataFrame or Series.

        Parameters
        ----------
        X : array-like, shape (n_samples,) or (n_samples, n_outputs)
            Input values to inverse transform.
        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        Xt : pd.DataFrame or pd.Series
            Inverse transformed values.
        """
        raise NotImplementedError("Subclasses must implement _inverse_transform method")


class WrapTransformer(BaseTransformer):
    """Transformer that wraps an sklearn-compatible transformer.

    This class handles wrapping an external transformer object and delegates
    fit, transform, and inverse_transform operations to it.
    """

    def __init__(self, transformer):
        """Initialize the wrapper transformer.

        Parameters
        ----------
        transformer : sklearn-compatible transformer
            A transformer object that implements fit and transform methods.
        """
        self.transformer = transformer
        super().__init__()

    def _fit(self, X, y=None):
        """Fit the wrapped transformer to X.

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
        return self

    def _transform(self, X, y=None):
        """Transform X using the wrapped transformer.

        Parameters
        ----------
        X : array-like, shape (n_samples,) or (n_samples, n_outputs)
            Input values to transform.
        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        Xt : pd.DataFrame or pd.Series
            Transformed values.
        """
        check_is_fitted(self.transformer_)
        Xt = self.transformer_.transform(X)

        # Convert to DataFrame/Series if needed
        if isinstance(Xt, (pd.DataFrame, pd.Series)):
            return Xt

        Xt = np.asarray(Xt)
        if Xt.ndim == 0:
            return Xt.item()
        elif Xt.ndim == 1:
            index = (
                X.index
                if isinstance(X, (pd.DataFrame, pd.Series))
                else pd.RangeIndex(len(Xt))
            )
            name = self.columns_[0] if self.columns_ is not None else 0
            return pd.Series(Xt, index=index, name=name)
        else:
            index = (
                X.index
                if isinstance(X, (pd.DataFrame, pd.Series))
                else pd.RangeIndex(Xt.shape[0])
            )
            columns = (
                self.columns_
                if self.columns_ is not None
                else pd.RangeIndex(Xt.shape[1])
            )
            return pd.DataFrame(Xt, index=index, columns=columns)

    def _inverse_transform(self, X, y=None):
        """Inverse transform X using the wrapped transformer.

        Parameters
        ----------
        X : array-like, shape (n_samples,) or (n_samples, n_outputs)
            Input values to inverse transform.
        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        Xt : pd.DataFrame or pd.Series
            Inverse transformed values.
        """
        check_is_fitted(self.transformer_)
        Xt = self.transformer_.inverse_transform(X)

        # Convert to DataFrame/Series if needed
        if isinstance(Xt, (pd.DataFrame, pd.Series)):
            return Xt

        Xt = np.asarray(Xt)
        if Xt.ndim == 0:
            return Xt.item()
        elif Xt.ndim == 1:
            index = (
                X.index
                if isinstance(X, (pd.DataFrame, pd.Series))
                else pd.RangeIndex(len(Xt))
            )
            name = self.columns_[0] if self.columns_ is not None else 0
            return pd.Series(Xt, index=index, name=name)
        else:
            index = (
                X.index
                if isinstance(X, (pd.DataFrame, pd.Series))
                else pd.RangeIndex(Xt.shape[0])
            )
            columns = (
                self.columns_
                if self.columns_ is not None
                else pd.RangeIndex(Xt.shape[1])
            )
            return pd.DataFrame(Xt, index=index, columns=columns)

    def _fit_with_fitted_transformer(self, index=None, columns=None):
        """Fit with already fitted transformer if possible.

        This is useful when wrapping a transformer that has already been fitted.

        Parameters
        ----------
        index : pd.Index, optional
            Not used, kept for API compatibility.
        columns : pd.Index, optional
            Columns to use for output formatting.

        Returns
        -------
        self : reference to self
        """
        self.columns_ = columns

        try:
            check_is_fitted(self.transformer)
            self.transformer_ = self.transformer
            self._is_fitted = True
        except NotFittedError:
            pass

        return self


class BaseDifferentiableTransformer(BaseTransformer):
    """Base class for differentiable transformers.

    This class adds differentiation capabilities to transformers.
    Subclasses must implement _fit, _transform, and _inverse_transform,
    and can optionally provide explicit derivative functions.
    """

    def __init__(self, transform_func_diff=None, inverse_func_diff=None):
        """Initialize differentiable transformer.

        Parameters
        ----------
        transform_func_diff : callable, optional
            Function to compute the derivative of the transform function.
        inverse_func_diff : callable, optional
            Function to compute the derivative of the inverse transform function.
        """
        # Only set if not already set (for cooperative multiple inheritance)
        if not hasattr(self, "transform_func_diff"):
            self.transform_func_diff = transform_func_diff
        if not hasattr(self, "inverse_func_diff"):
            self.inverse_func_diff = inverse_func_diff
        super().__init__()

    def transform_diff(self, X):
        """Compute the derivative of the transform function at X.

        Parameters
        ----------
        X : array-like, shape (n_samples,) or (n_samples, 1)
            Input data.

        Returns
        -------
        diff : array-like
            Derivative of the transform function at X.
        """
        if self.transform_func_diff is not None:
            return self.transform_func_diff(X)
        else:
            return self._transform_diff(X)

    def _transform_diff(self, X):
        """Compute the derivative of the transform function at X.

        Default implementation to be overridden by subclasses.
        Uses numerical differentiation on the _transform method.

        Parameters
        ----------
        X : array-like, shape (n_samples,) or (n_samples, 1)
            Input data.

        Returns
        -------
        diff : array-like
            Derivative of the transform function at X.
        """
        return self._numerical_diff(self._transform, X)

    def inverse_transform_diff(self, X):
        """Compute the derivative of the inverse transform function at X.

        Parameters
        ----------
        X : array-like, shape (n_samples,) or (n_samples, 1)
            Input data.

        Returns
        -------
        diff : array-like
            Derivative of the inverse transform function at X.
        """
        Xt = self.transform(X)

        if self.inverse_func_diff is not None:
            return self.inverse_func_diff(Xt)
        else:
            return self._inverse_transform_diff(Xt)

    def _inverse_transform_diff(self, Xt):
        """Compute the derivative of the inverse transform function.

        Default implementation to be overridden by subclasses.
        Uses numerical differentiation on the _inverse_transform method.

        Parameters
        ----------
        Xt : array-like, shape (n_samples,) or (n_samples, 1)
            Transformed data.

        Returns
        -------
        diff : array-like
            Derivative of the inverse transform function.
        """
        return self._numerical_diff(self._inverse_transform, Xt)

    def _numerical_diff(self, func, X, delta=1e-8):
        """Apply numerical differentiation using central difference.

        Parameters
        ----------
        func : callable
            Function to differentiate.
        X : array-like, shape (n_samples,) or (n_samples, 1)
            Input data.
        delta : float, default=1e-8
            Step size for numerical differentiation.

        Returns
        -------
        diff : array-like, shape (n_samples,) or (n_samples, 1)
            Numerical derivative of the transformation at X.
        """
        X = np.asarray(X)
        original_shape = X.shape
        X_flat = X.flatten()

        if HAS_SCIPY_DERIVATIVE and scipy_derivative is not None:
            # Use scipy.derivative (from differentiate or misc module)
            diff = np.zeros_like(X_flat)

            for i, x_val in enumerate(X_flat):
                # Wrapper function that scipy.derivative can work with
                def func_1d(x):
                    x = np.asarray(x)
                    if x.ndim == 0:
                        x_reshaped = np.array([[x.item()]])
                        result = func(x_reshaped)
                        # Convert to numpy if DataFrame/Series
                        if hasattr(result, "values"):
                            result = result.values
                        result = np.asarray(result).flatten()[0]

                        if SCIPY_DERIVATIVE_NEW_API:
                            return np.array(result)
                        else:
                            return result
                    else:
                        x_reshaped = x.reshape(-1, 1)
                        result = func(x_reshaped)
                        # Convert to numpy if DataFrame/Series
                        if hasattr(result, "values"):
                            result = result.values
                        result = np.asarray(result).flatten()
                        return result.reshape(x.shape)

                if SCIPY_DERIVATIVE_NEW_API:
                    # New API (scipy >= 1.14.0): scipy.differentiate.derivative
                    # Returns a result object with .df attribute
                    result = scipy_derivative(
                        func_1d,
                        x_val,
                        initial_step=delta,
                        maxiter=10,
                        preserve_shape=True,
                    )

                    diff[i] = result.df
                else:
                    # Old API (scipy < 1.14.0): scipy.misc.derivative
                    # Returns the derivative directly, uses 'dx' parameter
                    diff[i] = scipy_derivative(func_1d, x_val, dx=delta)
        else:
            # Warn user that scipy is not available
            warnings.warn(
                "scipy.derivative is not available. Falling back to custom "
                "central difference implementation. For better numerical "
                "differentiation, install scipy >= 0.14.0.",
                UserWarning,
                stacklevel=2,
            )

            # Use custom central difference implementation
            diff = np.zeros_like(X_flat)

            for i, x_val in enumerate(X_flat):
                # Central difference: f'(x) = (f(x+h) - f(x-h)) / (2*h)
                x_plus = np.array([[x_val + delta]])
                x_minus = np.array([[x_val - delta]])

                f_plus = func(x_plus)
                f_minus = func(x_minus)
                # Convert to numpy if DataFrame/Series
                if hasattr(f_plus, "values"):
                    f_plus = f_plus.values
                if hasattr(f_minus, "values"):
                    f_minus = f_minus.values
                f_plus = np.asarray(f_plus).flatten()[0]
                f_minus = np.asarray(f_minus).flatten()[0]

                diff[i] = (f_plus - f_minus) / (2 * delta)

        return diff.reshape(original_shape)


class DifferentiableTransformer(WrapTransformer, BaseDifferentiableTransformer):
    """Differentiable transformer that wraps sklearn transformers.

    Combines WrapTransformer (for wrapping sklearn transformers) with
    BaseDifferentiableTransformer (for differentiation capabilities).
    """

    def __init__(self, transformer, transform_func_diff=None, inverse_func_diff=None):
        """Initialize differentiable transformer.

        Parameters
        ----------
        transformer : sklearn-compatible transformer
            A transformer object that implements fit and transform methods.
        transform_func_diff : callable, optional
            Function to compute the derivative of the transform function.
        inverse_func_diff : callable, optional
            Function to compute the derivative of the inverse transform function.
        """
        self.transform_func_diff = transform_func_diff
        self.inverse_func_diff = inverse_func_diff
        super().__init__(transformer=transformer)

    def _transform_diff(self, X):
        """Compute the derivative of the transform function at X.

        Overrides BaseDifferentiableTransformer._transform_diff to use
        wrapped transformer's scale_ attribute if available.

        Parameters
        ----------
        X : array-like, shape (n_samples,) or (n_samples, 1)
            Input data.

        Returns
        -------
        diff : array-like
            Derivative of the transform function at X.
        """
        if self.transform_func_diff is not None:
            # if explicit derivative function was provided
            return self.transform_func_diff(X)
        elif (
            # if transformer has scale_ attribute (e.g., MinMaxScaler)
            hasattr(self.transformer_, "scale_")
            and self.transformer_.scale_ is not None
        ):
            return np.ones_like(X) * self.transformer_.scale_
        else:
            # Fall back to numerical differentiation
            return self._numerical_diff(self.transformer_.transform, X)

    def _inverse_transform_diff(self, Xt):
        """Compute the derivative of the inverse transform function.

        Overrides BaseDifferentiableTransformer._inverse_transform_diff to use
        wrapped transformer's scale_ attribute if available.

        Parameters
        ----------
        Xt : array-like, shape (n_samples,) or (n_samples, 1)
            Transformed data.

        Returns
        -------
        diff : array-like
            Derivative of the inverse transform function.
        """
        if self.inverse_func_diff is not None:
            # if explicit derivative function was provided
            return self.inverse_func_diff(Xt)

        elif (
            hasattr(self.transformer_, "scale_")
            and self.transformer_.scale_ is not None
        ):
            # if transformer has scale_ attribute
            return np.ones_like(Xt) / self.transformer_.scale_
        else:
            # Fall back to numerical differentiation
            return self._numerical_diff(self.transformer_.inverse_transform, Xt)

    def _get_transform_diff_capabilities(self):
        """Check if transform diff function is available, approx or exact.

        Prima

        Returns
        -------
        str or False
            - False if no inverse function is available
            - "approx" if inverse will use numerical differentiation
            - "exact" if inverse has an explicit function or scale_ attribute
        """
        # Check if transform_func_diff is available at all
        if not self._is_fitted:
            raise NotFittedError(
                "This DifferentiableTransformer instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this method."
            )

        # Check if we have explicit derivative function
        if self.transform_func_diff is not None:
            return "exact"

        # Check if transformer has scale_ attribute (e.g., MinMaxScaler, StandardScaler)
        if (
            hasattr(self.transformer_, "scale_")
            and self.transformer_.scale_ is not None
        ):
            return "exact"

        # Has transform but will use numerical differentiation fallback
        return "approx"

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
