# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Histogram distribution."""

__author__ = ["ShreeshaM07"]

import numpy as np
import pandas as pd

from skpro.distributions.base import _BaseArrayDistribution


class Histogram(_BaseArrayDistribution):
    """Histogram Probability Distribution.

    The histogram probability distribution is parameterized
    by the ``bins`` and ``bin_mass``.

    Parameters
    ----------
    bins : tuple(float, float, int) or numpy.array of float 1D or 2D list of size m x n
        1. tuple(first bin's start point, last bin's end point, number of bins)
           Used when ``bin widths`` are ``equal``.
            example:
                    bins:(0,4,4)
        2. array has the bin boundaries with 1st element the first bin's
           starting point and rest are the bin ending points of all bins
            example:
                    bins:[0, 1, 2, 3, 4]
        3. 2D list of size m x n containing m*n float numpy.arrays or tuple
           like ``case 1``.
            example:
                bins:
                [
                        [[0, 1, 2, 3, 4], [5, 5.5, 5.8, 6.5, 7, 7.5]],
                        [(2, 12, 5), [0, 1, 2, 3, 4]],
                        [[1.5, 2.5, 3.1, 4, 5.4], [-4, -2, -1.5, 5, 10]]
                ]
    bin_mass : array of float 1D or 2D list of size m x n
        1. Array has the mass of the bins or area of the bins.
           example:
                bin_mass:[0.1, 0.2, 0.3, 0.4]
           ``Note``: ``len(bin_mass)`` will be ``(len(bins)-1)``.
           ``Note``: ``sum(bin_mass)`` must be ``1``.
        2. 2D list of size m x n containing m*n float numpy.arrays
           each satisfying ``case 1``.
            example:
                bin_mass:
                [
                        [[0.1, 0.2, 0.3, 0.4], [0.25, 0.1, 0, 0.4, 0.25]],
                        [[0.1, 0.2, 0.4, 0.2, 0.1], [0.4, 0.3, 0.2, 0.1]],
                        [[0.06, 0.15, 0.09, 0.7], [0.4, 0.15, 0.325, 0.125]]
                ]
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

    def _check_single_array_distr(self, bins, bin_mass):
        all1Ds = (
            isinstance(bins[0], float)
            or isinstance(bins[0], np.floating)
            or isinstance(bins[0], int)
            or isinstance(bins[0], np.integer)
        )
        all1Ds = (
            all1Ds
            and isinstance(bin_mass[0], int)
            or isinstance(bin_mass[0], np.integer)
            or isinstance(bin_mass[0], float)
            or isinstance(bin_mass[0], np.floating)
            and np.array(bin_mass).ndim == 1
        )
        return all1Ds

    def __init__(self, bins, bin_mass, index=None, columns=None):
        if isinstance(bins, tuple):
            bins = self._convert_tuple_to_array(bins)
            self.bins = np.array(bins)
            self.bin_mass = np.array(bin_mass)
        elif self._check_single_array_distr(bins, bin_mass):
            self.bins = np.array(bins)
            self.bin_mass = np.array(bin_mass)
        else:
            # convert the bins into a list
            for i in range(len(bins)):
                for j in range(len(bins[i])):
                    if isinstance(bins[i][j], tuple):
                        bins[i][j] = self._convert_tuple_to_array(bins[i][j])
                    bins[i][j] = np.array(bins[i][j])
                    bin_mass[i][j] = np.array(bin_mass[i][j])
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
        bin_mass = self.bin_mass
        energy_arr = []
        from numpy.lib.stride_tricks import sliding_window_view

        if self._check_single_array_distr(bins, bin_mass):
            bins_hist = bins
            bin_mass_hist = bin_mass
            win_centre_bins = 0.5 * np.sum(
                sliding_window_view(bins_hist, window_shape=2), axis=1
            )
            expected_value = 0
            for i in range(len(bin_mass_hist)):
                for j in range(len(bin_mass_hist)):
                    expected_value += (
                        bin_mass_hist[i]
                        * bin_mass_hist[j]
                        * abs(win_centre_bins[i] - win_centre_bins[j])
                    )
            energy_arr = expected_value
            return expected_value

        for row in range(len(bins)):
            energy_arr_row = []
            for col in range(len(bins[0])):
                bins_hist = bins[row][col]
                bin_mass_hist = bin_mass[row][col]
                win_centre_bins = 0.5 * np.sum(
                    sliding_window_view(bins_hist, window_shape=2), axis=1
                )
                expected_value = 0
                for i in range(len(bin_mass_hist)):
                    for j in range(len(bin_mass_hist)):
                        expected_value += (
                            bin_mass_hist[i]
                            * bin_mass_hist[j]
                            * abs(win_centre_bins[i] - win_centre_bins[j])
                        )
                energy_arr_row.append(expected_value)
            energy_arr.append(energy_arr_row)
        energy_arr = np.array(energy_arr)
        if energy_arr.ndim > 0:
            energy_arr = np.sum(energy_arr, axis=1)
        return energy_arr

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
        bins = self.bins
        bin_mass = self.bin_mass
        energy_arr = []
        mean = self._mean()
        cdf = self._cdf(x)
        pdf = self._pdf(x)
        x = np.array(x)
        from numpy.lib.stride_tricks import sliding_window_view

        if self._check_single_array_distr(bins, bin_mass):
            bins_hist = np.array(bins)
            bin_mass_hist = np.array(bin_mass)
            X = x
            is_outside = X < bins_hist[0] or X > bins_hist[-1]
            if is_outside:
                energy_arr = abs(mean - X)
                return energy_arr
            else:
                # consider X lies in kth bin
                # so kth bin's start index is
                k_1_bins = np.where(X >= bins_hist)[0][-1]
                win_sum_bins = np.sum(
                    sliding_window_view(bins_hist, window_shape=2), axis=1
                )
                # upto kth bin excluding kth
                X_upto_k = X * cdf - 0.5 * np.dot(
                    win_sum_bins[:k_1_bins], bin_mass_hist[:k_1_bins]
                )
                # if X is in last bin
                if k_1_bins >= len(bin_mass_hist) - 1:
                    energy_arr = X_upto_k
                    return energy_arr
                # in the kth bin
                X_in_k = (
                    0.5
                    * pdf
                    * (
                        bins_hist[k_1_bins] ** 2
                        + bins_hist[k_1_bins + 1] ** 2
                        - 2 * X**2
                    )
                )
                # after kth bin excluding kth
                X_after_k = 0.5 * np.dot(
                    win_sum_bins[k_1_bins + 1 :], bin_mass_hist[k_1_bins + 1 :]
                ) - X * (1 - cdf)
                energy_arr = X_upto_k + X_in_k + X_after_k
                return energy_arr

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
                    # if X is in last bin
                    if k_1_bins >= len(bin_mass_hist) - 1:
                        energy_arr_row.append(X_upto_k)
                        continue
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
                    # after kth bin excluding kth
                    X_after_k = 0.5 * np.dot(
                        win_sum_bins[k_1_bins + 1 :], bin_mass_hist[k_1_bins + 1 :]
                    ) - X * (1 - cdf[i][j])
                    energy_arr_row.append(X_upto_k + X_in_k + X_after_k)
            energy_arr.append(energy_arr_row)
        energy_arr = np.array(energy_arr)
        if energy_arr.ndim > 0:
            energy_arr = np.sum(energy_arr, axis=1)
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

        if self._check_single_array_distr(bins, bin_mass):
            bins = np.array(bins)
            bin_mass = np.array(bin_mass)
            win_sum_bins = np.sum(sliding_window_view(bins, window_shape=2), axis=1)
            mean = 0.5 * np.dot(win_sum_bins, bin_mass)
            return mean

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

        if self._check_single_array_distr(bins, bin_mass):
            bins = np.array(bins)
            bin_mass = np.array(bin_mass)
            win_sum_bins = np.sum(sliding_window_view(bins, window_shape=2), axis=1)
            win_prod_bins = np.prod(sliding_window_view(bins, window_shape=2), axis=1)
            var = np.dot(bin_mass / 3, (win_sum_bins**2 - win_prod_bins)) - mean**2
            return var

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
        if self._check_single_array_distr(bins, bin_mass):
            bins = np.array(bins)
            bin_mass = np.array(bin_mass)
            bin_width = np.diff(bins)
            pdf_arr = bin_mass / bin_width
            X = x
            if len(np.where(X < bins)[0]) and len(np.where(X >= bins)[0]):
                pdf = pdf_arr[min(np.where(X < bins)[0]) - 1]
            else:
                pdf = 0
            return pdf
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

        from warnings import warn

        if self._check_single_array_distr(bins, bin_mass):
            bins = np.array(bins)
            bin_mass = np.array(bin_mass)
            bin_width = np.diff(bins)
            if 0 in bin_mass:
                warn("0 value detected in bin_mass is replaced by a positive const")
                small_value = 1e-100
                bin_mass = np.where(bin_mass == 0, small_value, bin_mass)
            lpdf_arr = np.log(bin_mass / bin_width)
            X = x
            if len(np.where(X < bins)[0]) and len(np.where(X >= bins)[0]):
                lpdf = lpdf_arr[min(np.where(X < bins)[0]) - 1]
            else:
                lpdf = 0
            return lpdf

        x = np.array(x)
        for i in range(len(bins)):
            lpdf_row = []
            for j in range(len(bins[0])):
                X = x[i][j]
                bins_hist = bins[i][j]
                bin_mass_hist = bin_mass[i][j]
                bin_width = np.diff(bins_hist)
                if 0 in bin_mass_hist:
                    warn("0 value detected in bin_mass is replaced by a positive const")
                    small_value = 1e-100
                    bin_mass_hist = np.where(
                        bin_mass_hist == 0, small_value, bin_mass_hist
                    )
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

        if self._check_single_array_distr(bins, bin_mass):
            X = x
            bins_hist = bins
            bin_mass_hist = bin_mass
            cum_sum_mass = np.cumsum(bin_mass_hist)
            # cum_bin_index is an array of all indices
            # of the bins or bin edges that are less than X.
            cum_bin_index = np.where(X >= bins_hist)[0]
            if len(cum_bin_index) == len(bins_hist):
                cdf = 1
            elif len(cum_bin_index) > 1:
                cdf = cum_sum_mass[cum_bin_index[-2]] + pdf * (
                    X - bins_hist[cum_bin_index[-1]]
                )
            elif len(cum_bin_index) == 0:
                cdf = 0
            elif len(cum_bin_index) == 1:
                cdf = pdf * (X - bins_hist[cum_bin_index[-1]])

            return cdf

        x = np.array(x)

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
        p = np.array(p)

        if self._check_single_array_distr(bins, bin_mass):
            bins = np.array(bins)
            bin_mass = np.array(bin_mass)
            cum_sum_mass = np.cumsum(bin_mass)
            # print(cum_sum_mass)
            pdf_bins = []
            for bin in bins:
                pdf_bins.append(self._pdf(bin))
            P = p
            cum_bin_index_P = np.where(P >= cum_sum_mass)[0]
            if P < 0 or P > 1:
                X = np.NaN
            elif len(cum_bin_index_P) == 0:
                X = bins[0] + P / pdf_bins[len(cum_bin_index_P)]
            elif len(cum_bin_index_P) > 0:
                if P - cum_sum_mass[cum_bin_index_P[-1]] > 0:
                    X = (
                        bins[cum_bin_index_P[-1] + 1]
                        + (P - cum_sum_mass[cum_bin_index_P[-1]])
                        / pdf_bins[len(cum_bin_index_P)]
                    )
                else:
                    X = bins[cum_bin_index_P[-1] + 1]

            return X

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
                    X = bins_hist[0] + P / pdf_bins[len(cum_bin_index_P)]
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
            "bins": [
                [[0, 1, 2, 3, 4], [5, 5.5, 5.8, 6.5, 7, 7.5]],
                [(2, 12, 5), [0, 1, 2, 3, 4]],
                [[1.5, 2.5, 3.1, 4, 5.4], [-4, -2, -1.5, 5, 10]],
            ],
            "bin_mass": [
                [[0.1, 0.2, 0.3, 0.4], [0.25, 0.1, 0, 0.4, 0.25]],
                [[0.1, 0.2, 0.4, 0.2, 0.1], [0.4, 0.3, 0.2, 0.1]],
                [[0.06, 0.15, 0.09, 0.7], [0.4, 0.15, 0.325, 0.125]],
            ],
            "index": pd.Index(np.arange(3)),
            "columns": pd.Index(np.arange(2)),
        }

        return [params1]
