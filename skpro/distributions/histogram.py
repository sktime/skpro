# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Histogram distribution."""

__author__ = ["ShreeshaM07"]

import numpy as np

from skpro.distributions.base import BaseDistribution


class Histogram(BaseDistribution):
    """Histogram Probability Distribution.

    The histogram probability distribution is parameterized
    by the bins and bin densities.

    Parameters
    ----------
    bins : float or array of float 1D
        array has the bin boundaries with 1st element the first bin's
        starting point and rest are the bin ending points of all bins
    bin_mass: array of float 1D
        Mass of the bins or Area of the bins.
        Sum of all the bin_mass must be 1.
    index : pd.Index, optional, default = RangeIndex
    columns : pd.Index, optional, default = RangeIndex
    """

    def __init__(self, bins, bin_mass, index=None, columns=None):
        self.bins = bins
        self.bin_mass = bin_mass

        super().__init__(index=index, columns=columns)

    def _mean(self):
        """Return expected value of the distribution.

        Returns
        -------
        float, sum(bin_mass)/range(bins)
            expected value of distribution (entry-wise)
        """
        bins = self.bins
        # 1 is the cumulative sum of all bin_mass
        return 1 / (max(bins) - min(bins))

    def _var(self):
        r"""Return element/entry-wise variance of the distribution.

        Returns
        -------
        2D np.ndarray, same shape as ``self``
            variance of the distribution (entry-wise)
        """
        bins = self.bins
        bin_mass = self.bin_mass
        bin_width = np.diff(bins)
        mean = self._mean()
        var = np.sum((bin_mass / bin_width - mean) * bin_width) / (
            max(bins) - min(bins)
        )
        return var

    def _pdf(self, x):
        """Probability density function.

        Parameters
        ----------
        x : 1D np.ndarray, same shape as ``self``
            values to evaluate the pdf at

        Returns
        -------
        1D np.ndarray, same shape as ``self``
            pdf values at the given points
        """
        bin_mass = np.array(self.bin_mass.copy())
        bins = self.bins
        pdf = []
        if isinstance(bins, list):
            bin_width = np.diff(bins)
            pdf_arr = bin_mass / bin_width
            for X in x:
                if len(np.where(X < bins)[0]) and len(np.where(X >= bins)[0]):
                    pdf.append(pdf_arr[min(np.where(X < bins)[0]) - 1])
                else:
                    pdf.append(0)
            pdf = np.array(pdf)
            return pdf

    def _cdf(self, x):
        """Cumulative distribution function.

        Parameters
        ----------
        x : 1D np.ndarray, same shape as ``self``
            values to evaluate the cdf at

        Returns
        -------
        1D np.ndarray, same shape as ``self``
            cdf values at the given points
        """
        bins = self.bins
        bin_mass = self.bin_mass
        cdf = []
        pdf = self._pdf(x)
        if isinstance(bins, list):
            for X in x:
                # cum_bin_index is an array of all indices
                # of the bins or bin edges that are less than X.
                cum_bin_index = np.where(X >= bins)[0]
                X_index_in_x = np.where(X == x)
                if len(cum_bin_index) == len(bins):
                    cdf.append(1)
                elif len(cum_bin_index) > 1:
                    cdf.append(
                        np.cumsum(bin_mass)[-2]
                        + pdf[X_index_in_x][0] * (X - bins[cum_bin_index[-1]])
                    )
                elif len(cum_bin_index) == 0:
                    cdf.append(0)
                elif len(cum_bin_index) == 1:
                    cdf.append(pdf[X_index_in_x][0] * (X - bins[cum_bin_index[-1]]))
            cdf = np.array(cdf)
        return cdf


# import pandas as pd

# x = np.array([100, 1, 0.75, 1.8, 2.5, 3, 5, 6, 6.5, 0])
# hist = Histogram(
#     bins=[0.5, 2, 7],
#     bin_mass=[0.3, 0.7],
#     index=pd.Index(np.arange(3)),
#     columns=pd.Index(np.arange(2)),
# )
# pdf = hist._pdf(x)
# print(pdf)
# cdf = hist._cdf(x)
# print(cdf)
# mean = hist._mean()
# print(mean)
# var = hist._var()
# print(var)
