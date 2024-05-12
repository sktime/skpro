"""Johnson Quantile-Parameterized Distributions."""

# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)

__author__ = [
    "FelixWick",
    "setoguchi-naoki",
]  # interface only. Cyclic boosting authors in cyclic_boosting package

import warnings
from typing import Optional

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.misc import derivative
from scipy.stats import logistic, norm

from skpro.distributions.base import BaseDistribution


class QPD_S(BaseDistribution):
    """Johnson Quantile-Parameterized Distributions with semi-bounded mode.

    see https://repositories.lib.utexas.edu/bitstream/handle/2152
        /63037/HADLOCK-DISSERTATION-2017.pdf
    (Due to the Python keyword, the parameter lambda from
    this reference is named kappa below.)
    A distribution is parameterized by a symmetric-percentile triplet (SPT).

    Parameters
    ----------
    alpha : float
        lower quantile of SPT (upper is ``1 - alpha``)
    qv_low : float or array_like[float]
        quantile function value of ``alpha``
    qv_median : float or array_like[float]
        quantile function value of quantile 0.5
    qv_high : float or array_like[float]
        quantile function value of quantile ``1 - alpha``
    lower : float
        lower bound of semi-bounded range (default is 0)
    version: str, optional, default="normal"
        options are ``normal`` (default) or ``logistic``
    dist_shape: float, optional, default=0.0
        parameter modifying the logistic base distribution via
        sinh/arcsinh-scaling (only active in sinhlogistic version)
        ``dist_shape`` will be removed in version X.X.X.

    Example
    -------
    >>> from skpro.distributions.qpd import QPD_S  # doctest: +SKIP

    >>> qpd = QPD_S(
    ...         alpha=0.2,
    ...         qv_low=[1, 2],
    ...         qv_median=[3, 4],
    ...         qv_high=[5, 6],
    ...         lower=0
    ...       )  # doctest: +SKIP

    >>> qpd.mean()  # doctest: +SKIP
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["setoguchi-naoki", "felix-wick"],
        "maintainers": ["setoguchi-naoki"],
        "python_dependencies": "cyclic_boosting>=1.2.5",
        # estimator tags
        # --------------
        "capabilities:approx": ["pdfnorm", "energy"],
        "capabilities:exact": ["mean", "var", "cdf", "ppf", "pdf", "log_pdf"],
        "distr:measuretype": "continuous",
    }

    def __init__(
        self,
        alpha: float,
        qv_low: float or object,
        qv_median: float or object,
        qv_high: float or object,
        lower: Optional[float] = 0.0,
        version: Optional[str] = "normal",
        dist_shape: Optional[float] = 0.0,
        index=None,
        columns=None,
    ):
        self.qpd = []
        self.alpha = alpha
        self.qv_low = qv_low
        self.qv_median = qv_median
        self.qv_high = qv_high
        self.lower = lower
        self.version = version
        self.dist_shape = dist_shape
        self.index = index
        self.columns = columns

        #  TODO <X.X.X>: change default of 'lower' and to None
        #  update docstring, and remove warning. related to PR-232
        link = "https://github.com/Blue-Yonder-OSS/cyclic-boosting/ \
                blob/a62eccffcd611e87a850b9bf38ea699fbea1fb31/ \
                cyclic_boosting/quantile_matching.py#L12"
        comment = f"Parameter 'dist_shape' of 'QPD_S' will be removed \
                    in version X.X.X. \
                   'J_QPD_S' is used instead of 'J_QPD_extended_S'. \
                    See {link} for details"
        warnings.warn(comment, DeprecationWarning, stacklevel=2)

        comment = "Parameter 'lower' of 'QPD_S' will change \
                   default value from 0.0 to None in version X.X.X. \
                   To retain prior behaviour, set 'lower' to 0.0 explicitly"
        warnings.warn(comment, DeprecationWarning, stacklevel=2)

        super().__init__(index=index, columns=columns)

        from cyclic_boosting.quantile_matching import J_QPD_extended_S

        alpha, qv_low, qv_median, qv_high = _prep_qpd_params(
            self, alpha, qv_low, qv_median, qv_high
        )

        if index is None:
            index = pd.RangeIndex(qv_low.shape[0])
            self.index = index

        if columns is None:
            columns = pd.RangeIndex(1)
            self.columns = columns

        self._shape = (qv_low.shape[0], 1)

        if version == "normal":
            self.phi = norm()
        elif version == "logistic":
            self.phi = logistic()
        else:
            raise Exception("Invalid version.")

        if (np.any(qv_low > qv_median)) or np.any(qv_high < qv_median):
            warnings.warn(
                "The SPT values are not monotonically increasing, "
                "each SPT is sorted by value",
                stacklevel=2,
            )
            idx = np.where((qv_low > qv_median), True, False) + np.where(
                (qv_high < qv_median), True, False
            )
            un_orderd_idx = np.argwhere(idx > 0).tolist()
            warnings.warn(f"sorted index {un_orderd_idx}", stacklevel=2)
            for idx in un_orderd_idx:
                low, mid, high = sorted([qv_low[idx], qv_median[idx], qv_high[idx]])
                qv_low[idx] = low
                qv_median[idx] = mid
                qv_high[idx] = high

        iter = np.nditer(qv_low, flags=["c_index"])
        for _i in iter:
            jqpd = J_QPD_extended_S(
                alpha=alpha,
                qv_low=qv_low[iter.index],
                qv_median=qv_median[iter.index],
                qv_high=qv_high[iter.index],
                l=self.lower,
                version=version,
                shape=dist_shape,
            )
            self.qpd.append(jqpd)
        self.qpd = pd.DataFrame(self.qpd, index=self.index)

    def mean(self, lower=0.0, upper=np.inf):
        """Return expected value of the distribution.

        Returns
        -------
        pd.DataFrame with same rows, columns as `self`
        expected value of distribution (entry-wise)
        """
        #  TODO <X.X.X>: remove warning
        comment = "Parameter 'lower' and 'upper' will be moved to \
                   class parameters in version X.X.X. \
                   To retain prior behaviour, set 'lower' to 0.0 \
                   and 'upper' to large number explicitly"
        warnings.warn(comment, DeprecationWarning, stacklevel=2)

        loc = []
        for idx in self.index:
            qpd = self.qpd.loc[idx, :].values[0]
            l, _ = quad(exp_func, args=(qpd), a=lower, b=upper)
            loc.append(l)
        loc_arr = np.array(loc)
        return pd.DataFrame(loc_arr, index=self.index, columns=self.columns)

    def var(self, lower=0.0, upper=np.inf):
        """Return element/entry-wise variance of the distribution.

        Returns
        -------
        pd.DataFrame with same rows, columns as `self`
        variance of distribution (entry-wise)
        """
        #  TODO <X.X.X>: remove warning
        comment = "Parameter 'lower' and 'upper' will be moved to \
                   class parameters in version X.X.X. \
                   To retain prior behaviour, set 'lower' to 0.0 \
                   and 'upper' to large number explicitly"
        warnings.warn(comment, DeprecationWarning, stacklevel=2)

        mean = self.mean()
        var = []
        for idx in self.index:
            mu = mean.loc[idx, :].to_numpy()
            qpd = self.qpd.loc[idx, :].values[0]
            l, _ = quad(var_func, args=(mu, qpd), a=lower, b=upper)
            var.append(l)
        var_arr = np.array(var)
        return pd.DataFrame(var_arr, index=self.index, columns=self.columns)

    def pdf(self, x: pd.DataFrame):
        """Probability density function.

        this fucntion transform cdf to pdf
        because j-qpd's pdf calculation is bit complex
        """
        pdf = []
        for idx in x.index:
            qpd = self.qpd.loc[idx, :].values[0]
            _x = x.loc[idx, :]
            _pdf = derivative(qpd.cdf, _x, dx=1e-6)
            pdf.append(_pdf)
        pdf_arr = np.array(pdf)
        return pd.DataFrame(pdf_arr, index=x.index, columns=x.columns)

    def ppf(self, p: pd.DataFrame):
        """Quantile function = percent point function = inverse cdf."""
        ppf = []
        for idx in p.index:
            qpd = self.qpd.loc[idx, :].values[0]
            _ppf = qpd.ppf(p.loc[idx, :])
            ppf.append(_ppf)
        ppf_arr = np.array(ppf)
        return pd.DataFrame(ppf_arr, index=p.index, columns=p.columns)

    def cdf(self, x: pd.DataFrame):
        """Cumulative distribution function."""
        cdf = []
        for idx in x.index:
            qpd = self.qpd.loc[idx, :].values[0]
            _cdf = qpd.cdf(x.loc[idx, :])
            cdf.append(_cdf)
        cdf_arr = np.array(cdf)
        return pd.DataFrame(cdf_arr, index=x.index, columns=x.columns)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        params1 = {
            "alpha": 0.2,
            "version": "normal",
            "qv_low": 0.2,
            "qv_median": 0.5,
            "qv_high": 0.8,
            "index": pd.Index([1, 2, 5]),
            "columns": pd.Index(["a"]),
        }
        params2 = {
            "alpha": 0.2,
            "version": "normal",
            "qv_low": [0.2, 0.2, 0.2],
            "qv_median": [0.5, 0.5, 0.5],
            "qv_high": [0.8, 0.8, 0.8],
            "index": pd.Index([1, 2, 5]),
            "columns": pd.Index(["a"]),
        }
        return [params1, params2]


class QPD_B(BaseDistribution):
    """Johnson Quantile-Parameterized Distributions with bounded mode.

    see https://repositories.lib.utexas.edu/bitstream/handle/2152
        /63037/HADLOCK-DISSERTATION-2017.pdf
    (Due to the Python keyword, the parameter lambda from
    this reference is named kappa below).
    A distribution is parameterized by a symmetric-percentile triplet (SPT).

    Parameters
    ----------
    alpha : float
        lower quantile of SPT (upper is ``1 - alpha``)
    qv_low : float or array_like[float]
        quantile function value of ``alpha``
    qv_median : float or array_like[float]
        quantile function value of quantile 0.5
    qv_high : float or array_like[float]
        quantile function value of quantile ``1 - alpha``
    lower : float
        lower bound of semi-bounded range
    upper : float
        upper bound of supported range
    version: str, optional, default="normal"
        options are ``normal`` (default) or ``logistic``
    dist_shape: float, optional, default=0.0
        parameter modifying the logistic base distribution via
        sinh/arcsinh-scaling (only active in sinhlogistic version)
        ``dist_shape`` will be removed in version X.X.X.

    Example
    -------
    >>> from skpro.distributions.qpd import QPD_B  # doctest: +SKIP

    >>> qpd = QPD_B(
    ...         alpha=0.2,
    ...         qv_low=[1, 2],
    ...         qv_median=[3, 4],
    ...         qv_high=[5, 6],
    ...         lower=0,
    ...         upper=10
    ...       )  # doctest: +SKIP

    >>> qpd.mean()  # doctest: +SKIP
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["setoguchi-naoki", "felix-wick"],
        "maintainers": ["setoguchi-naoki"],
        "python_dependencies": "cyclic_boosting>=1.2.5",
        # estimator tags
        # --------------
        "capabilities:approx": ["pdfnorm", "energy"],
        "capabilities:exact": ["mean", "var", "cdf", "ppf", "pdf", "log_pdf"],
        "distr:measuretype": "continuous",
    }

    def __init__(
        self,
        alpha: float,
        qv_low: float or object,
        qv_median: float or object,
        qv_high: float or object,
        lower: float,
        upper: float,
        version: Optional[str] = "normal",
        dist_shape: Optional[float] = 0.0,
        index=None,
        columns=None,
    ):
        self.qpd = []
        self.alpha = alpha
        self.qv_low = qv_low
        self.qv_median = qv_median
        self.qv_high = qv_high
        self.lower = lower
        self.upper = upper
        self.version = version
        self.dist_shape = dist_shape
        self.index = index
        self.columns = columns

        #  TODO <X.X.X>: change default of 'lower' and to None
        #  update docstring, and remove warning
        link = "https://github.com/Blue-Yonder-OSS/cyclic-boosting/ \
                blob/a62eccffcd611e87a850b9bf38ea699fbea1fb31/ \
            cyclic_boosting/quantile_matching.py#L168"
        comment = f"Parameter 'dist_shape' of 'QPD_B' will be removed \
                    in version X.X.X. \
                   'J_QPD_B' is used instead of 'J_QPD_extended_B'. \
                    See {link} for details"
        warnings.warn(comment, DeprecationWarning, stacklevel=2)

        comment = "Parameter 'lower' and 'upper' of 'QPD_B' will change \
                   default value from 0.0 and 1.0 to None and None in version X.X.X. \
                   To retain prior behaviour, set 'lower' and 'upper' to 0.0 and 1.0 \
                   explicitly"
        warnings.warn(comment, DeprecationWarning, stacklevel=2)

        super().__init__(index=index, columns=columns)

        from cyclic_boosting.quantile_matching import J_QPD_extended_B

        alpha, qv_low, qv_median, qv_high = _prep_qpd_params(
            self, alpha, qv_low, qv_median, qv_high
        )

        if index is None:
            index = pd.RangeIndex(qv_low.shape[0])
            self.index = index

        if columns is None:
            columns = pd.RangeIndex(1)
            self.columns = columns

        if version == "normal":
            self.phi = norm()
        elif version == "logistic":
            self.phi = logistic()
        else:
            raise Exception("Invalid version.")

        if (np.any(qv_low > qv_median)) or np.any(qv_high < qv_median):
            warnings.warn(
                "The SPT values are not monotonically increasing, "
                "each SPT is sorted by value",
                stacklevel=2,
            )
            idx = np.where((qv_low > qv_median), True, False) + np.where(
                (qv_high < qv_median), True, False
            )
            un_orderd_idx = np.argwhere(idx > 0).tolist()
            warnings.warn(f"sorted index {un_orderd_idx}", stacklevel=2)
            for idx in un_orderd_idx:
                low, mid, high = sorted([qv_low[idx], qv_median[idx], qv_high[idx]])
                qv_low[idx] = low
                qv_median[idx] = mid
                qv_high[idx] = high

        iter = np.nditer(qv_low, flags=["c_index"])
        for _i in iter:
            jqpd = J_QPD_extended_B(
                alpha=alpha,
                qv_low=qv_low[iter.index],
                qv_median=qv_median[iter.index],
                qv_high=qv_high[iter.index],
                l=lower,
                u=upper,
                version=version,
                shape=dist_shape,
            )
            self.qpd.append(jqpd)
        self.qpd = pd.DataFrame(self.qpd, index=self.index)

    def mean(self, lower=0.0, upper=np.inf):
        """Return expected value of the distribution.

        Returns
        -------
        pd.DataFrame with same rows, columns as `self`
        expected value of distribution (entry-wise)
        """
        #  TODO <X.X.X>: remove warning
        comment = "Parameter 'lower' and 'upper' will be moved to \
                   class parameters in version X.X.X. \
                   To retain prior behaviour, set 'lower' to 0.0 \
                   and 'upper' to large number explicitly"
        warnings.warn(comment, DeprecationWarning, stacklevel=2)

        loc = []
        for idx in self.index:
            qpd = self.qpd.loc[idx, :].values[0]
            l, _ = quad(exp_func, args=(qpd), a=lower, b=upper)
            loc.append(l)
        loc_arr = np.array(loc)
        return pd.DataFrame(loc_arr, index=self.index, columns=self.columns)

    def var(self, lower=0.0, upper=np.inf):
        """Return element/entry-wise variance of the distribution.

        Returns
        -------
        pd.DataFrame with same rows, columns as `self`
        variance of distribution (entry-wise)
        """
        #  TODO <X.X.X>: remove warning
        comment = "Parameter 'lower' and 'upper' will be moved to \
                   class parameters in version X.X.X. \
                   To retain prior behaviour, set 'lower' to 0.0 \
                   and 'upper' to large number explicitly"
        warnings.warn(comment, DeprecationWarning, stacklevel=2)

        mean = self.mean()
        var = []
        for idx in self.index:
            mu = mean.loc[idx, :].to_numpy()
            qpd = self.qpd.loc[idx, :].values[0]
            l, _ = quad(var_func, args=(mu, qpd), a=lower, b=upper)
            var.append(l)
        var_arr = np.array(var)
        return pd.DataFrame(var_arr, index=self.index, columns=self.columns)

    def pdf(self, x: pd.DataFrame):
        """Probability density function.

        this fucntion transform cdf to pdf
        because j-qpd's pdf calculation is bit complex
        """
        pdf = []
        for idx in x.index:
            qpd = self.qpd.loc[idx, :].values[0]
            _x = x.loc[idx, :]
            _pdf = derivative(qpd.cdf, _x, dx=1e-6)
            pdf.append(_pdf)
        pdf_arr = np.array(pdf)
        return pd.DataFrame(pdf_arr, index=x.index, columns=x.columns)

    def ppf(self, p: pd.DataFrame):
        """Quantile function = percent point function = inverse cdf."""
        ppf = []
        for idx in p.index:
            qpd = self.qpd.loc[idx, :].values[0]
            _ppf = qpd.ppf(p.loc[idx, :])
            ppf.append(_ppf)
        ppf_arr = np.array(ppf)
        return pd.DataFrame(ppf_arr, index=p.index, columns=p.columns)

    def cdf(self, x: pd.DataFrame):
        """Cumulative distribution function."""
        cdf = []
        for idx in x.index:
            qpd = self.qpd.loc[idx, :].values[0]
            _cdf = qpd.cdf(x.loc[idx, :])
            cdf.append(_cdf)
        cdf_arr = np.array(cdf)
        return pd.DataFrame(cdf_arr, index=x.index, columns=x.columns)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        params1 = {
            "alpha": 0.2,
            "version": "normal",
            "qv_low": 0.2,
            "qv_median": 0.5,
            "qv_high": 0.8,
            "lower": 0.0,
            "upper": 1.0,
            "index": pd.Index([1, 2, 5]),
            "columns": pd.Index(["a"]),
        }
        params2 = {
            "alpha": 0.2,
            "version": "normal",
            "qv_low": [0.2, 0.2, 0.2],
            "qv_median": [0.5, 0.5, 0.5],
            "qv_high": [0.8, 0.8, 0.8],
            "lower": 0.0,
            "upper": 1.0,
            "index": pd.Index([1, 2, 5]),
            "columns": pd.Index(["a"]),
        }
        return [params1, params2]


class QPD_U(BaseDistribution):
    """Johnson Quantile-Parameterized Distributions with unbounded mode.

    see https://repositories.lib.utexas.edu/bitstream/handle/2152
        /63037/HADLOCK-DISSERTATION-2017.pdf
    (Due to the Python keyword, the parameter lambda from
    this reference is named kappa below).
    A distribution is parameterized by a symmetric-percentile triplet (SPT).

    Parameters
    ----------
    alpha : float
        lower quantile of SPT (upper is ``1 - alpha``)
    qv_low : float or array_like[float]
        quantile function value of ``alpha``
    qv_median : float or array_like[float]
        quantile function value of quantile 0.5
    qv_high : float or array_like[float]
        quantile function value of quantile ``1 - alpha``
    version: str, optional, default="normal"
        options are ``normal`` (default) or ``logistic``
    dist_shape: float, optional, default=0.0
        parameter modifying the logistic base distribution via
        sinh/arcsinh-scaling (only active in sinhlogistic version)

    Example
    -------
    >>> from skpro.distributions.qpd import QPD_U  # doctest: +SKIP

    >>> qpd = QPD_U(
    ...         alpha=0.2,
    ...         qv_low=[1, 2],
    ...         qv_median=[3, 4],
    ...         qv_high=[5, 6],
    ...       )  # doctest: +SKIP

    >>> qpd.mean()  # doctest: +SKIP
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["setoguchi-naoki", "felix-wick"],
        "maintainers": ["setoguchi-naoki"],
        "python_dependencies": "cyclic_boosting>=1.2.5",
        # estimator tags
        # --------------
        "capabilities:approx": ["pdfnorm", "energy"],
        "capabilities:exact": ["mean", "var", "cdf", "ppf", "pdf", "log_pdf"],
        "distr:measuretype": "continuous",
    }

    def __init__(
        self,
        alpha: float,
        qv_low: float or object,
        qv_median: float or object,
        qv_high: float or object,
        version: Optional[str] = "normal",
        dist_shape: Optional[float] = 0.0,
        index=None,
        columns=None,
    ):
        self.qpd = []
        self.alpha = alpha
        self.qv_low = qv_low
        self.qv_median = qv_median
        self.qv_high = qv_high
        self.version = version
        self.dist_shape = dist_shape
        self.index = index
        self.columns = columns

        super().__init__(index=index, columns=columns)

        from cyclic_boosting.quantile_matching import J_QPD_extended_U

        alpha, qv_low, qv_median, qv_high = _prep_qpd_params(
            self, alpha, qv_low, qv_median, qv_high
        )

        if index is None:
            index = pd.RangeIndex(qv_low.shape[0])
            self.index = index

        if columns is None:
            columns = pd.RangeIndex(1)
            self.columns = columns

        if version == "normal":
            self.phi = norm()
        elif version == "logistic":
            self.phi = logistic()
        else:
            raise Exception("Invalid version.")

        if (np.any(qv_low > qv_median)) or np.any(qv_high < qv_median):
            warnings.warn(
                "The SPT values are not monotonically increasing, "
                "each SPT is sorted by value",
                stacklevel=2,
            )
            idx = np.where((qv_low > qv_median), True, False) + np.where(
                (qv_high < qv_median), True, False
            )
            un_orderd_idx = np.argwhere(idx > 0).tolist()
            warnings.warn(f"sorted index {un_orderd_idx}", stacklevel=2)
            for idx in un_orderd_idx:
                low, mid, high = sorted([qv_low[idx], qv_median[idx], qv_high[idx]])
                qv_low[idx] = low
                qv_median[idx] = mid
                qv_high[idx] = high

        iter = np.nditer(qv_low, flags=["c_index"])
        for _i in iter:
            jqpd = J_QPD_extended_U(
                alpha=alpha,
                qv_low=qv_low[iter.index],
                qv_median=qv_median[iter.index],
                qv_high=qv_high[iter.index],
                version=version,
                shape=dist_shape,
            )
            self.qpd.append(jqpd)
        self.qpd = pd.DataFrame(self.qpd, index=self.index)

    def mean(self, lower=0.0, upper=np.inf):
        """Return expected value of the distribution.

        Returns
        -------
        pd.DataFrame with same rows, columns as `self`
        expected value of distribution (entry-wise)
        """
        #  TODO <X.X.X>: remove warning
        comment = "Parameter 'lower' and 'upper' will be moved to \
                   class parameters in version X.X.X. \
                   To retain prior behaviour, set 'lower' to 0.0 \
                   and 'upper' to large number explicitly"
        warnings.warn(comment, DeprecationWarning, stacklevel=2)

        loc = []
        for idx in self.index:
            qpd = self.qpd.loc[idx, :].values[0]
            l, _ = quad(exp_func, args=(qpd), a=lower, b=upper)
            loc.append(l)
        loc_arr = np.array(loc)
        return pd.DataFrame(loc_arr, index=self.index, columns=self.columns)

    def var(self, lower=0.0, upper=np.inf):
        """Return element/entry-wise variance of the distribution.

        Returns
        -------
        pd.DataFrame with same rows, columns as `self`
        variance of distribution (entry-wise)
        """
        #  TODO <X.X.X>: remove warning
        comment = "Parameter 'lower' and 'upper' will be moved to \
                   class parameters in version X.X.X. \
                   To retain prior behaviour, set 'lower' to 0.0 \
                   and 'upper' to large number explicitly"
        warnings.warn(comment, DeprecationWarning, stacklevel=2)

        mean = self.mean()
        var = []
        for idx in self.index:
            mu = mean.loc[idx, :].to_numpy()
            qpd = self.qpd.loc[idx, :].values[0]
            l, _ = quad(var_func, args=(mu, qpd), a=lower, b=upper)
            var.append(l)
        var_arr = np.array(var)
        return pd.DataFrame(var_arr, index=self.index, columns=self.columns)

    def pdf(self, x: pd.DataFrame):
        """Probability density function.

        this fucntion transform cdf to pdf
        because j-qpd's pdf calculation is bit complex
        """
        pdf = []
        for idx in x.index:
            qpd = self.qpd.loc[idx, :].values[0]
            _x = x.loc[idx, :]
            _pdf = derivative(qpd.cdf, _x, dx=1e-6)
            pdf.append(_pdf)
        pdf_arr = np.array(pdf)
        return pd.DataFrame(pdf_arr, index=x.index, columns=x.columns)

    def ppf(self, p: pd.DataFrame):
        """Quantile function = percent point function = inverse cdf."""
        ppf = []
        for idx in p.index:
            qpd = self.qpd.loc[idx, :].values[0]
            _ppf = qpd.ppf(p.loc[idx, :])
            ppf.append(_ppf)
        ppf_arr = np.array(ppf)
        return pd.DataFrame(ppf_arr, index=p.index, columns=p.columns)

    def cdf(self, x: pd.DataFrame):
        """Cumulative distribution function."""
        cdf = []
        for idx in x.index:
            qpd = self.qpd.loc[idx, :].values[0]
            _cdf = qpd.cdf(x.loc[idx, :])
            cdf.append(_cdf)
        cdf_arr = np.array(cdf)
        return pd.DataFrame(cdf_arr, index=x.index, columns=x.columns)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        params1 = {
            "alpha": 0.2,
            "version": "normal",
            "qv_low": 0.2,
            "qv_median": 0.5,
            "qv_high": 0.8,
            "index": pd.Index([1, 2, 5]),
            "columns": pd.Index(["a"]),
        }
        params2 = {
            "alpha": 0.2,
            "version": "normal",
            "qv_low": [0.2, 0.2, 0.2],
            "qv_median": [0.5, 0.5, 0.5],
            "qv_high": [0.8, 0.8, 0.8],
            "index": pd.Index([1, 2, 5]),
            "columns": pd.Index(["a"]),
        }
        return [params1, params2]


def exp_func(x, qpd):
    """Return Expectation."""
    # TODO: scipy.integrate will be removed in scipy 1.12.0
    comment = "Parameter 'qpd' will be removed to in version X.X.X."
    warnings.warn(comment, DeprecationWarning, stacklevel=2)

    pdf = derivative(qpd.cdf, x, dx=1e-6)
    return x * pdf


def var_func(x, mu, qpd):
    """Return Variance."""
    # TODO: scipy.integrate will be removed in scipy 1.12.0
    comment = "Parameter 'qpd' will be removed to in version X.X.X."
    warnings.warn(comment, DeprecationWarning, stacklevel=2)

    pdf = derivative(qpd.cdf, x, dx=1e-6)
    return ((x - mu) ** 2) * pdf


def _prep_qpd_params(self, alpha, qv_low, qv_median, qv_high):
    """Prepare parameters for Johnson Quantile-Parameterized Distributions."""
    if not isinstance(alpha, np.ndarray):
        alpha = np.array([alpha])
    qv_low, qv_median, qv_high = BaseDistribution._get_bc_params(
        self, qv_low, qv_median, qv_high, oned_as="col"
    )
    qv_low = qv_low.flatten()
    qv_median = qv_median.flatten()
    qv_high = qv_high.flatten()
    return alpha, qv_low, qv_median, qv_high
