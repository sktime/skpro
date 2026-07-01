"""This module contains some general/canonical link-mean-function pairs such as

- :class:`~LogLinkMixin`
- :class:`~LogitLinkMixin`
"""


import abc

import numexpr
import numpy as np




class LinkFunction(object, metaclass=abc.ABCMeta):
    r"""Abstract base class for link function computations."""

    @abc.abstractmethod
    def is_in_range(self, values: np.ndarray):
        """Check if values can be transformed by the link function."""
        pass

    @abc.abstractmethod
    def link_func(self, m: np.ndarray):
        """Transform values in m to link space"""
        pass

    @abc.abstractmethod
    def unlink_func(self, l: np.ndarray):
        """Inverse of :meth:`~link_func`"""
        pass


class LogLinkMixin(LinkFunction):
    r"""Link function and mean function for example for Poisson-distributed
    data.

    Supported values are in the range :math:`x > 0`"""

    def unlink_func(self, l: np.ndarray) -> np.ndarray:
        r"""Calculates the inverse of the link function

        .. math::

           \mu = \exp(l)
        """
        return numexpr.evaluate("exp(l)")

    def is_in_range(self, m: np.ndarray) -> bool:
        return np.all(m > 0.0)

    def link_func(self, m: np.ndarray) -> np.ndarray:
        r"""Calculates the log-link

        .. math::

           l = \log(\mu)
        """
        return numexpr.evaluate("log(m)")


class LogitLinkMixin(LinkFunction):
    r"""Link for the logit transformation.

    Supported values are in the range :math:`0 \leq x \leq 1`
    """

    def is_in_range(self, p: np.ndarray) -> bool:
        return np.all(numexpr.evaluate("(p >= 0.0) & (p <= 1.0)"))

    def link_func(self, p: np.ndarray) -> np.ndarray:
        r"""Calculates the logit-link

        .. math::

           l = \log(\frac{p}{1-p})
        """
        return numexpr.evaluate("log(p / (1. - p))")

    def unlink_func(self, l: np.ndarray) -> np.ndarray:
        r"""Inverse of logit-link

        .. math::

           p = \frac{1}{1+ \exp(-l)}
        """
        return numexpr.evaluate("1. / (1. + exp(-l))")


class IdentityLinkMixin(LinkFunction):
    """Identity link"""

    def is_in_range(self, m: np.ndarray):
        return True

    def link_func(self, m: np.ndarray):
        r"""Returns a copy of the input"""
        return m.copy()

    def unlink_func(self, l: np.ndarray):
        r"""Returns a copy of the input"""
        return l.copy()


__all__ = [
    "LinkFunction",
    "LogLinkMixin",
    "LogitLinkMixin",
    "IdentityLinkMixin",
]
