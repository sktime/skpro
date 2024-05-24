# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Histogram distribution."""

__author__ = ["ShreeshaM07"]

import numpy as np
import pandas as pd

from skpro.distributions.base import BaseDistribution


class Histogram(BaseDistribution):
    """Histogram Probability Distribution.

    The histogram probability distribution is parameterized
    by the bins and bin densities.

    Parameters
    ----------
    bins : tuple(float,float,int) or array of float 1D
        1. tuple(first bin's start point, last bin's end point, number of bins)
        Used when bin widths are equal.
        2. array has the bin boundaries with 1st element the first bin's
        starting point and rest are the bin ending points of all bins
    bin_mass: array of float 1D
        Mass of the bins or Area of the bins.
        Note: Sum of all the bin_mass must be 1.
    index : pd.Index, optional, default = RangeIndex
    columns : pd.Index, optional, default = RangeIndex
    """

    _tags = {
        "authors": ["ShreeshaM07"],
        "capabilities:approx": ["pdfnorm", "energy"],
        "capabilities:exact": ["mean", "var", "pdf", "log_pdf", "cdf", "ppf"],
        "distr:measuretype": "continuous",
        "distr:paramtype": "parametric",
        # "broadcast_init": "on",
    }

    def __init__(self, bins, bin_mass, index=None, columns=None):
        self.bins = bins
        self.bin_mass = bin_mass

        super().__init__(index=index, columns=columns)

    def _convert_tuple_to_array(self, bins):
        bins_to_list = (bins[0], bins[1], bins[2])
        bins = []
        bin_width = (bins_to_list[1] - bins_to_list[0]) / bins_to_list[2]
        for b in range(bins_to_list[2]):
            bins.append(bins_to_list[0] + b * bin_width)
        bins.append(bins_to_list[1])
        return bins

    def _energy_self(self):
        r"""Energy of self, w.r.t. self.

        :math:`\mathbb{E}[|X-Y|]`, where :math:`X, Y` are i.i.d. copies of self.

        Private method, to be implemented by subclasses.

        Returns
        -------
        2D np.ndarray, same shape as ``self``
            energy values w.r.t. the given points
        """
        bins = self.bins
        # bin_mass = self.bin_mass
        # convert the bins into a list
        if isinstance(bins, tuple):
            bins = self._convert_tuple_to_array(bins)

    def _energy_x(self, x):
        r"""Energy of self, w.r.t. a constant frame x.

        :math:`\mathbb{E}[|X-x|]`, where :math:`X` is a copy of self,
        and :math:`x` is a constant.

        Private method, to be implemented by subclasses.

        Parameters
        ----------
        x : 1D np.ndarray, same shape as ``self``
            values to compute energy w.r.t. to

        Returns
        -------
        1D np.ndarray, same shape as ``self``
            energy values w.r.t. the given points
        """
        bins = self.bins
        energy_arr = []
        # convert the bins into a list
        if isinstance(bins, tuple):
            bins = self._convert_tuple_to_array(bins)

        if isinstance(bins, list):
            mean = self._mean()

            is_outside = np.logical_or(x < bins[0], x > bins[-1])
            # is_inside = 1 - is_outside

            if is_outside:
                energy_arr = abs(mean - x)
            # else:
            #     bin_idx_pre_x = np.where(x >= bins)[0][-1]
            #     still in progress ...

            return energy_arr

    def _mean(self):
        """Return expected value of the distribution.

        Returns
        -------
        float, sum(bin_mass)/range(bins)
            expected value of distribution (entry-wise)
        """
        bins = self.bins
        bin_mass = np.array(self.bin_mass)
        # convert the bins into a list
        if isinstance(bins, tuple):
            bins = self._convert_tuple_to_array(bins)

        if isinstance(bins, list):
            from numpy.lib.stride_tricks import sliding_window_view

            win_sum_bins = np.sum(sliding_window_view(bins, window_shape=2), axis=1)
            mean = 0.5 * np.dot(win_sum_bins, bin_mass)
            return mean

    def _var(self):
        r"""Return element/entry-wise variance of the distribution.

        Returns
        -------
        2D np.ndarray, same shape as ``self``
            variance of the distribution (entry-wise)
        """
        bins = self.bins
        bin_mass = np.array(self.bin_mass)

        # convert the bins into a list
        if isinstance(bins, tuple):
            bins = self._convert_tuple_to_array(bins)

        if isinstance(bins, list):
            from numpy.lib.stride_tricks import sliding_window_view

            win_sum_bins = np.sum(sliding_window_view(bins, window_shape=2), axis=1)
            mean = self._mean()
            win_prod_bins = np.prod(sliding_window_view(bins, window_shape=2), axis=1)
            var = np.dot(bin_mass / 3, (win_sum_bins**2 - win_prod_bins)) - mean**2
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

        # convert the bins into a list
        if isinstance(bins, tuple):
            bins = self._convert_tuple_to_array(bins)

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

    def _log_pdf(self, x):
        """Logarithmic probability density function.

        Parameters
        ----------
        x : 2D np.ndarray, same shape as ``self``
            values to evaluate the pdf at

        Returns
        -------
        2D np.ndarray, same shape as ``self``
            log pdf values at the given points
        """
        bin_mass = np.array(self.bin_mass.copy())
        bins = self.bins
        lpdf = []

        # convert the bins into a list
        if isinstance(bins, tuple):
            bins = self._convert_tuple_to_array(bins)

        if isinstance(bins, list):
            bin_width = np.diff(bins)
            lpdf_arr = np.log(bin_mass / bin_width)
            for X in x:
                if len(np.where(X < bins)[0]) and len(np.where(X >= bins)[0]):
                    lpdf.append(lpdf_arr[min(np.where(X < bins)[0]) - 1])
                else:
                    lpdf.append(0)
            lpdf = np.array(lpdf)
            return lpdf

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

        # convert the bins into a list
        if isinstance(bins, tuple):
            bins = self._convert_tuple_to_array(bins)

        if isinstance(bins, list):
            cum_sum_mass = np.cumsum(bin_mass)
            for X in x:
                # cum_bin_index is an array of all indices
                # of the bins or bin edges that are less than X.
                cum_bin_index = np.where(X >= bins)[0]
                X_index_in_x = np.where(X == x)
                if len(cum_bin_index) == len(bins):
                    cdf.append(1)
                elif len(cum_bin_index) > 1:
                    cdf.append(
                        cum_sum_mass[cum_bin_index[-2]]
                        + pdf[X_index_in_x][0] * (X - bins[cum_bin_index[-1]])
                    )
                elif len(cum_bin_index) == 0:
                    cdf.append(0)
                elif len(cum_bin_index) == 1:
                    cdf.append(pdf[X_index_in_x][0] * (X - bins[cum_bin_index[-1]]))
            cdf = np.array(cdf)
        return cdf

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
        bins = self.bins
        bin_mass = self.bin_mass
        ppf = []

        # convert the bins into a list
        if isinstance(bins, tuple):
            bins = self._convert_tuple_to_array(bins)

        if isinstance(bins, list):
            cum_sum_mass = np.cumsum(bin_mass)
            # print(cum_sum_mass)
            pdf_bins = self._pdf(np.array(bins))
            # print('pdf: ',pdf_bins)
            for P in p:
                cum_bin_index_P = np.where(P >= cum_sum_mass)[0]
                # print(cum_bin_index_P[0])
                if P < 0 or P > 1:
                    ppf.append(np.NaN)
                elif len(cum_bin_index_P) == 0:
                    X = P / pdf_bins[len(cum_bin_index_P)]
                    ppf.append(X)
                elif len(cum_bin_index_P) > 0:
                    if P - cum_sum_mass[cum_bin_index_P[-1]] > 0:
                        X = (
                            bins[cum_bin_index_P[-1] + 1]
                            + (P - cum_sum_mass[cum_bin_index_P[-1]])
                            / pdf_bins[len(cum_bin_index_P)]
                        )
                    else:
                        X = bins[cum_bin_index_P[-1] + 1]
                    ppf.append(X)

        ppf = np.array(ppf)
        return ppf

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        # array case examples
        params1 = {
            "bins": [0, 1, 2, 3, 4],
            "bin_mass": [0.1, 0.2, 0.3, 0.4],
            "index": pd.Index([1, 2, 3, 4]),
            "columns": pd.Index(["a"]),
        }

        params2 = {
            "bins": (0, 4, 4),
            "bin_mass": [0.1, 0.2, 0, 0.7],
        }

        return [params1, params2]
