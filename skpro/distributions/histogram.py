# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Histogram distribution."""

__author__ = ["ShreeshaM07"]

import numpy as np

from skpro.distributions.base import BaseDistribution

# import pandas as pd


class Histogram(BaseDistribution):
    """Histogram Probability Distribution.

    The histogram probability distribution is parameterized
    by the bins and bin densities.

    Parameters
    ----------
    bins : float or array of float 1D
        array has the bin boundaries with 1st element the first bin's
        starting point and rest are the bin ending points of all bins
    bin_density: array of float 1D
        The density of bins.
        i.e., it is equal to the empirical probability divided
        by the interval length, or bin width.
    index : pd.Index, optional, default = RangeIndex
    columns : pd.Index, optional, default = RangeIndex
    """

    def __init__(self, bins, bin_density, index=None, columns=None):
        self.bins = bins
        self.bin_density = bin_density

        super().__init__(index=index, columns=columns)

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
        bin_density = np.array(self.bin_density.copy())
        bins = self.bins
        pdf = []
        if isinstance(bins, list):
            bin_width = []
            for i in range(1, len(bins)):
                bin_width.append(bins[i] - bins[i - 1])
            bin_width = np.array(bin_width)
            pdf_arr = bin_density / bin_width
            for i in range(len(x)):
                for j in range(1, len(bins)):
                    if x[i] < bins[j] and x[i] >= bins[j - 1]:
                        pdf.append(pdf_arr[j - 1])
                        break
            pdf = np.array(pdf)
            return pdf


# x=np.array([1,0.75,1.8,2.5,3,5,6,6.5])
# hist = Histogram(bins=[0.5,2,7],bin_density=[0.3,0.7]
# ,index=pd.Index(np.arange(3)),columns=pd.Index(np.arange(2)))
# pdf = hist._pdf(x)
# print(pdf)
