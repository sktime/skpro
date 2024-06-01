# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Histogram distribution."""

__author__ = ["ShreeshaM07"]

import numpy as np
import pandas as pd

from skpro.distributions.base import BaseArrayDistribution


class Histogram(BaseArrayDistribution):
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
        "broadcast_init": "on",
    }

    def _convert_tuple_to_array(self, bins):
        bins_to_list = (bins[0], bins[1], bins[2])
        bins = []
        bin_width = (bins_to_list[1] - bins_to_list[0]) / bins_to_list[2]
        for b in range(bins_to_list[2]):
            bins.append(bins_to_list[0] + b * bin_width)
        bins.append(bins_to_list[1])
        return bins

    def __init__(self, bins, bin_mass, index=None, columns=None):
        # convert the bins into a list
        for i in range(len(bins)):
            for j in range(len(bins[i])):
                if isinstance(bins[i][j], tuple):
                    bins[i][j] = self._convert_tuple_to_array(bins[i][j])
                bins[i][j] = np.array(bins[i][j])
                bin_mass[i][j] = np.array(bin_mass[i][j])

        # bins = [[self._convert_tuple_to_array(item) if
        # isinstance(item, tuple) else item
        #      for item in inner_list] for inner_list in bins]
        self.bins = bins
        self.bin_mass = bin_mass

        super().__init__(index=index, columns=columns)

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
        x : 2D np.ndarray, same shape as ``self``
            values to compute energy w.r.t. to

        Returns
        -------
        2D np.ndarray, same shape as ``self``
            energy values w.r.t. the given points
        """
        # still needs work
        bins = self.bins
        bin_mass = self.bin_mass
        energy_arr = []
        mean = self._mean()
        cdf = self._cdf(x)
        pdf = self._pdf(x)
        from numpy.lib.stride_tricks import sliding_window_view

        for i in range(len(bins)):
            energy_arr_row = []
            for j in range(len(bins[0])):
                bins_hist = bins[i][j]
                bin_mass_hist = bin_mass[i][j]
                X = x[i][j]
                is_outside = X < bins_hist[0] or X > bins_hist[-1]
                if is_outside:
                    energy_arr_row.append(abs(mean[i][j] - X))
                else:
                    # consider X lies in kth bin
                    # so kth bin's start index is
                    k_1_bins = np.where(X >= bins_hist)[0][-1]
                    win_sum_bins = np.sum(
                        sliding_window_view(bins_hist, window_shape=2), axis=1
                    )
                    # upto kth bin excluding kth
                    X_upto_k = X * cdf[i][j] - 0.5 * np.dot(
                        win_sum_bins[:k_1_bins], bin_mass_hist[:k_1_bins]
                    )
                    # after kth bin excluding kth
                    X_after_k = 0.5 * np.dot(
                        win_sum_bins[k_1_bins + 1 :], bin_mass_hist[k_1_bins + 1 :]
                    ) - X * (1 - cdf[i][j])
                    # in the kth bin
                    X_in_k = (
                        0.5
                        * pdf[i][j]
                        * (
                            bins_hist[k_1_bins] ** 2
                            + bins_hist[k_1_bins + 1] ** 2
                            - 2 * X**2
                        )
                    )
                    energy_arr_row.append(X_upto_k + X_in_k + X_after_k)
            energy_arr.append(energy_arr_row)
        energy_arr = np.array(energy_arr)
        return energy_arr

    def _mean(self):
        """Return expected value of the distribution.

        Returns
        -------
        float, sum(bin_mass)/range(bins)
            expected value of distribution (entry-wise)
        """
        bins = self.bins
        bin_mass = self.bin_mass
        mean = []
        from numpy.lib.stride_tricks import sliding_window_view

        for i in range(len(bins)):
            mean_row = []
            for j in range(len(bins[0])):
                bins_hist = bins[i][j]
                bin_mass_hist = bin_mass[i][j]
                win_sum_bins = np.sum(
                    sliding_window_view(bins_hist, window_shape=2), axis=1
                )
                mean_row.append(0.5 * np.dot(win_sum_bins, bin_mass_hist))
            mean.append(mean_row)
        return np.array(mean)

    def _var(self):
        r"""Return element/entry-wise variance of the distribution.

        Returns
        -------
        2D np.ndarray, same shape as ``self``
            variance of the distribution (entry-wise)
        """
        bins = self.bins
        bin_mass = self.bin_mass
        var = []
        mean = self._mean()
        from numpy.lib.stride_tricks import sliding_window_view

        for i in range(len(bins)):
            var_row = []
            for j in range(len(bins[0])):
                bins_hist = bins[i][j]
                bin_mass_hist = bin_mass[i][j]
                win_sum_bins = np.sum(
                    sliding_window_view(bins_hist, window_shape=2), axis=1
                )
                win_prod_bins = np.prod(
                    sliding_window_view(bins_hist, window_shape=2), axis=1
                )
                var_row.append(
                    np.dot(bin_mass_hist / 3, (win_sum_bins**2 - win_prod_bins))
                    - mean[i][j] ** 2
                )
            var.append(var_row)
        var = np.array(var)
        return var

    def _pdf(self, x):
        """Probability density function.

        Parameters
        ----------
        x : 2D np.ndarray, same shape as ``self``
            values to evaluate the pdf at

        Returns
        -------
        2D np.ndarray, same shape as ``self``
            pdf values at the given points
        """
        bin_mass = self.bin_mass
        bins = self.bins
        pdf = []
        # bins_hist contains the bins edges of each histogram
        for i in range(len(bins)):
            pdf_row = []
            for j in range(len(bins[i])):
                bins_hist = bins[i][j]
                bin_mass_hist = bin_mass[i][j]
                bin_width = np.diff(bins_hist)
                pdf_arr = bin_mass_hist / bin_width
                X = x[i][j]
                if len(np.where(X < bins_hist)[0]) and len(np.where(X >= bins_hist)[0]):
                    pdf_row.append(pdf_arr[min(np.where(X < bins_hist)[0]) - 1])
                else:
                    pdf_row.append(0)
            pdf.append(pdf_row)
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
        bin_mass = self.bin_mass
        bins = self.bins
        lpdf = []

        for i in range(len(bins)):
            lpdf_row = []
            for j in range(len(bins[0])):
                X = x[i][j]
                bins_hist = bins[i][j]
                bin_mass_hist = bin_mass[i][j]
                bin_width = np.diff(bins_hist)
                lpdf_arr = np.log(bin_mass_hist / bin_width)
                if len(np.where(X < bins_hist)[0]) and len(np.where(X >= bins_hist)[0]):
                    lpdf_row.append(lpdf_arr[min(np.where(X < bins_hist)[0]) - 1])
                else:
                    lpdf_row.append(0)
            lpdf.append(lpdf_row)
        lpdf = np.array(lpdf)
        return lpdf

    def _cdf(self, x):
        """Cumulative distribution function.

        Parameters
        ----------
        x : 2D np.ndarray, same shape as ``self``
            values to evaluate the cdf at

        Returns
        -------
        2D np.ndarray, same shape as ``self``
            cdf values at the given points
        """
        bins = self.bins
        bin_mass = self.bin_mass
        cdf = []
        pdf = self._pdf(x)

        for i in range(len(bins)):
            cdf_row = []
            for j in range(len(bins[0])):
                X = x[i][j]
                bins_hist = bins[i][j]
                bin_mass_hist = bin_mass[i][j]
                cum_sum_mass = np.cumsum(bin_mass_hist)
                # cum_bin_index is an array of all indices
                # of the bins or bin edges that are less than X.
                cum_bin_index = np.where(X >= bins_hist)[0]
                if len(cum_bin_index) == len(bins_hist):
                    cdf_row.append(1)
                elif len(cum_bin_index) > 1:
                    cdf_row.append(
                        cum_sum_mass[cum_bin_index[-2]]
                        + pdf[i][j] * (X - bins_hist[cum_bin_index[-1]])
                    )
                elif len(cum_bin_index) == 0:
                    cdf_row.append(0)
                elif len(cum_bin_index) == 1:
                    cdf_row.append(pdf[i][j] * (X - bins_hist[cum_bin_index[-1]]))
            cdf.append(cdf_row)
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

        for i in range(len(bins)):
            ppf_row = []
            for j in range(len(bins[0])):
                P = p[i][j]
                bins_hist = bins[i][j]
                bin_mass_hist = bin_mass[i][j]
                cum_sum_mass = np.cumsum(bin_mass_hist)
                # manually finding pdf of 1D array at all bin edges
                pdf_bins = []
                bin_width = np.diff(bins_hist)
                pdf_arr = bin_mass_hist / bin_width
                for bh in bins_hist:
                    if len(np.where(bh < bins_hist)[0]) and len(
                        np.where(bh >= bins_hist)[0]
                    ):
                        pdf_bins.append(pdf_arr[min(np.where(bh < bins_hist)[0]) - 1])
                    else:
                        pdf_bins.append(0)
                pdf_bins = np.array(pdf_bins)
                # find a way to calculate pdf for 1D array ...
                cum_bin_index_P = np.where(P >= cum_sum_mass)[0]
                if P < 0 or P > 1:
                    ppf_row.append(np.NaN)
                elif len(cum_bin_index_P) == 0:
                    X = P / pdf_bins[len(cum_bin_index_P)]
                    ppf_row.append(round(X, 4))
                elif len(cum_bin_index_P) > 0:
                    if P - cum_sum_mass[cum_bin_index_P[-1]] > 0:
                        X = (
                            bins_hist[cum_bin_index_P[-1] + 1]
                            + (P - cum_sum_mass[cum_bin_index_P[-1]])
                            / pdf_bins[len(cum_bin_index_P)]
                        )
                    else:
                        X = bins_hist[cum_bin_index_P[-1] + 1]
                    ppf_row.append(round(X, 4))
            ppf.append(ppf_row)
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
