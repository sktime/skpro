"""Transformers for skpro."""

import warnings

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, check_is_fitted, clone
from sklearn.exceptions import NotFittedError

from skpro.base import BaseEstimator


def _get_scipy_derivative():
    """Get scipy derivative function if available.

    Checks for scipy and returns the appropriate derivative function.
    Uses scipy.differentiate for scipy >= 1.14.0, scipy.misc for older versions.

    Returns
    -------
    tuple of (callable or None, bool)
        - derivative function if scipy is available, None otherwise
        - True if using new API (scipy >= 1.14.0), False if old API
    """
    from skbase.utils.dependencies import _check_soft_dependencies

    # Check for new API (scipy >= 1.14.0)
    if _check_soft_dependencies("scipy>=1.14.0", severity="none"):
        from scipy.differentiate import derivative

        return derivative, True

    # Check for old API (scipy < 1.14.0)
    elif _check_soft_dependencies("scipy", severity="none"):
        from scipy.misc import derivative

        return derivative, False

    # scipy is not available
    return None, False


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


class DifferentiableTransformer(BaseTransformer):
    """Differentiable transformer that wraps sklearn transformers.

    Wraps an sklearn-compatible transformer and adds differentiation capabilities
    for computing derivatives of the transform and inverse_transform functions.

    Parameters
    ----------
    transformer : sklearn-compatible transformer
        A transformer object that implements fit and transform methods.
        Can be already fitted or an unfitted transformer.
    transform_func_diff : callable, optional
        Function to compute the derivative of the transform function.
        If not provided, will use numerical differentiation or scale_
        attribute if available.
    inverse_func_diff : callable, optional
        Function to compute the derivative of the inverse transform function.
        If not provided, will use numerical differentiation or scale_
        attribute if available.

    Attributes
    ----------
    transformer_ : fitted transformer
        The fitted transformer. If transformer was already fitted when
        passed to __init__, this references the original. Otherwise,
        it's a clone of transformer that has been fitted.
    columns_ : pd.Index or None
        Column names from the data passed to fit.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from sklearn.preprocessing import StandardScaler
    >>> from skpro.compose import DifferentiableTransformer

    >>> X = pd.DataFrame([[1.0], [2.0], [3.0]], columns=["y"])
    >>> dt = DifferentiableTransformer(StandardScaler())
    >>> df = dt.fit(X)
    >>> dt.transform(X)
              y
    0 -1.224745
    1  0.000000
    2  1.224745

    >>> dt.transform_diff(X)  # Get derivatives
    array([[0.81649658],
           [0.81649658],
           [0.81649658]])
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
        self.transformer = transformer
        self.transform_func_diff = transform_func_diff
        self.inverse_func_diff = inverse_func_diff
        super().__init__()

    def _fit(self, X, y=None):
        """Fit the wrapped transformer to X.

        Always clones and fits the transformer on X, following sklearn conventions.
        This ensures calling .fit(X) actually fits on X regardless of whether
        the transformer was already fitted.

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
        # Always clone to avoid mutating the original, then fit on X
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

        Uses explicit derivative if provided, otherwise tries to use
        scale_ attribute if available, otherwise falls back to numerical
        differentiation.

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
            # Explicit derivative function provided
            return self.transform_func_diff(X)
        elif (
            hasattr(self.transformer_, "scale_")
            and self.transformer_.scale_ is not None
        ):
            # Use scale_ attribute (e.g., MinMaxScaler, StandardScaler)
            return np.ones_like(X) * self.transformer_.scale_
        else:
            # Fall back to numerical differentiation
            return self._numerical_diff(self.transformer_.transform, X)

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

        Uses explicit derivative if provided, otherwise tries to use
        scale_ attribute if available, otherwise falls back to numerical
        differentiation.

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
            # Explicit derivative function provided
            return self.inverse_func_diff(Xt)
        elif (
            hasattr(self.transformer_, "scale_")
            and self.transformer_.scale_ is not None
        ):
            # Use scale_ attribute (inverse derivative is 1/scale_)
            return np.ones_like(Xt) / self.transformer_.scale_
        else:
            # Fall back to numerical differentiation
            return self._numerical_diff(self.transformer_.inverse_transform, Xt)

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

        # Get scipy derivative function if available
        scipy_derivative, use_new_api = _get_scipy_derivative()

        if scipy_derivative is not None:
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

                        if use_new_api:
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

                if use_new_api:
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

    def _wrap_if_fitted(self, columns=None):
        """Mark transformer as fitted if already fitted.

        This is useful for TransformedDistribution which doesn't have
        data to fit with, but may receive an already-fitted transformer.

        Parameters
        ----------
        columns : pd.Index, optional
            Columns to use for output formatting.

        Returns
        -------
        self : reference to self
        """
        self.columns_ = columns

        try:
            check_is_fitted(self.transformer)
            # Already fitted - just use it directly
            self.transformer_ = self.transformer
            self._is_fitted = True
        except NotFittedError:
            # Not fitted - leave as is
            pass

        return self

    def _get_transform_diff_capabilities(self):
        """Check if transform diff function is available, approx or exact.

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
