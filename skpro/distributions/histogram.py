# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Histogram distribution."""

__author__ = ["ShreeshaM07"]

from skpro.distributions.base import BaseDistribution


class Histogram(BaseDistribution):
    """Histogram Probability Distribution.

    The histogram probability distribution is parameterized
    by the bins and bin densities.

    Parameters
    ----------
    bin_width : int or array of int 1D
        Equal width bins, bins will be an int.
        variable width bins, bins will be an array of int.
    bin_density: float or array of float 1D
        The density of bins.
        i.e., it is equal to the empirical probability divided
        by the interval length, or bin width.
    index : pd.Index, optional, default = RangeIndex
    columns : pd.Index, optional, default = RangeIndex
    """

    def __init__(self, bin_width, bin_density, index=None, columns=None):
        self.bin_width = bin_width
        self.bin_density = bin_density

        super.__init__(index=index, columns=columns)

    def _cut(self, x):
        # extends the min and max values by 0.1% of range
        # to include it in the histogram
        bins = []
        range_x = max(x) - min(x)
        bin_width = self._bc_params["bin_width"]
        if isinstance(bin_width, int):
            bins.append(min(x) - 0.001 * (range_x))
            nbins = range_x / bin_width
            for i in range(1, nbins):
                bins.append(min(x) + i * bin_width)
            bins.append(max(x) + 0.001 * (range_x))
        elif isinstance(bin_width, list):
            bins.append(min(x) - 0.001 * (range_x))
            nbins = len(bin_width)
            for bw in bin_width:
                bins.append(min(x) + bw)
            bins.append(max(x) + 0.001 * (range_x))

        self.bins = bins
        return bins

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
        pdf_arr = self._bc_params["bin_density"].copy()
        return pdf_arr
