# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Mixture distribution."""

import numpy as np
import pandas as pd
from skbase.base import BaseMetaObject

from skpro.distributions.base import BaseDistribution


class Mixture(BaseMetaObject, BaseDistribution):
    """Mixture of distributions.

    Parameters
    ----------
    distributions : list of tuples (str, BaseDistribution) or BaseDistribution
        list of mixture components
    weights : 1D or 2D array-like, optional, default = None
        Mixture weights. Can be:

        - 1D array of shape ``(n_components,)`` — global weights shared across
          all rows. Will be normalized to sum to 1.
        - 2D array of shape ``(n_rows, n_components)`` — per-row weights.
          Each row will be normalized to sum to 1.

        If not provided, uniform mixture is assumed.
    indep_rows : bool, optional, default = True
        if True, rows are sampled independently from the mixture components.
        If False, the same component is used for all rows.
        Relevant only in ``sample`` method and non-marginal outputs.
    indep_cols : bool, optional, default = True
        if True, columns are sampled independently from the mixture components.
        If False, the same component is used for all columns.
        Relevant only in ``sample`` method and non-marginal outputs.
    index : pd.Index, optional, default = inferred from component distributions
    columns : pd.Index, optional, default = inferred from component distributions

    Examples
    --------
    >>> from skpro.distributions.mixture import Mixture
    >>> from skpro.distributions.normal import Normal

    >>> n1 = Normal(mu=[[0, 1], [2, 3], [4, 5]], sigma=1)
    >>> n2 = Normal(mu=3, sigma=2, index=n1.index, columns=n1.columns)
    >>> m = Mixture(distributions=[("n1", n1), ("n2", n2)], weights=[0.3, 0.7])
    >>> mixture_sample = m.sample(n_samples=10)
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["fkiraly", "MurtuzaShaikh26"],
        # estimator tags
        # --------------
        "capabilities:approx": ["pdfnorm", "energy", "ppf"],
        "capabilities:exact": ["mean", "var", "pdf", "log_pdf", "cdf"],
        "distr:measuretype": "mixed",
        "distr:paramtype": "composite",
        "named_object_parameters": "_distributions",
    }

    def __init__(
        self,
        distributions,
        weights=None,
        indep_rows=True,
        indep_cols=True,
        index=None,
        columns=None,
    ):
        self.distributions = distributions
        self.weights = weights
        self.indep_rows = indep_rows
        self.indep_cols = indep_cols

        self._distributions = self._coerce_to_named_object_tuples(distributions)
        n_dists = len(self._distributions)

        if weights is None:
            self._weights = np.ones(n_dists) / n_dists
            self._row_weights = False
        else:
            w = np.array(weights, dtype=float)
            if w.ndim == 2:
                # row-wise weights: normalize each row to sum to 1
                row_sums = w.sum(axis=1, keepdims=True)
                self._weights = w / row_sums
                self._row_weights = True
            else:
                self._weights = w / w.sum()
                self._row_weights = False

        if index is None:
            index = self._distributions[0][1].index

        if columns is None:
            columns = self._distributions[0][1].columns

        super().__init__(index=index, columns=columns)

    def _iloc(self, rowidx=None, colidx=None):
        dists = self._distributions

        dists_subset = [(x[0], x[1].iloc[rowidx, colidx]) for x in dists]

        index_subset = dists_subset[0][1].index
        columns_subset = dists_subset[0][1].columns

        if self._row_weights and rowidx is not None:
            weights = self._weights[rowidx]
        else:
            weights = self.weights

        return Mixture(
            distributions=dists_subset,
            weights=weights,
            index=index_subset,
            columns=columns_subset,
        )

    def _iat(self, rowidx=None, colidx=None):
        dists = self._distributions

        dists_subset = [(x[0], x[1].iat[rowidx, colidx]) for x in dists]

        if self._row_weights and rowidx is not None:
            weights = self._weights[rowidx]
        else:
            weights = self.weights

        return Mixture(distributions=dists_subset, weights=weights)

    def _mean(self):
        r"""Return expected value of the distribution.

        Let :math:`X` be a random variable with the distribution of `self`.
        Returns the expectation :math:`\mathbb{E}[X]`

        Returns
        -------
        pd.DataFrame with same rows, columns as `self`
        expected value of distribution (entry-wise)
        """
        return self._average("mean")

    def _var(self):
        r"""Return element/entry-wise variance of the distribution.

        Let :math:`X` be a random variable with the distribution of `self`.
        Returns :math:`\mathbb{V}[X] = \mathbb{E}\left(X - \mathbb{E}[X]\right)^2`

        Returns
        -------
        pd.DataFrame with same rows, columns as `self`
        variance of distribution (entry-wise)
        """
        var_mean = self._average("var")
        mixture_mean = self._average("mean")

        means = [d.mean() for _, d in self._distributions]
        mean_var = [(m - mixture_mean) ** 2 for m in means]

        if self.ndim > 0:
            var_mean_var = self._average_df(mean_var)
        elif self._row_weights:
            var_mean_var = np.sum(
                [w * v for w, v in zip(self._weights, mean_var)]
            )
        else:
            var_mean_var = np.average(mean_var, weights=self._weights)

        return var_mean + var_mean_var

    def _average(self, method, x=None, weights=None):
        """Average a method over the mixture components."""
        if x is None:
            args = ()
        else:
            args = (x,)

        vals = [getattr(d, method)(*args) for _, d in self._distributions]

        if self.ndim > 0:
            return self._average_df(vals, weights=weights)
        elif self._row_weights:
            if weights is None:
                weights = self._weights
            return np.sum([w * v for w, v in zip(weights, vals)])
        else:
            return np.average(vals, weights=weights)

    def _average_df(self, df_list, weights=None):
        """Average a list of ``pd.DataFrame`` objects, with weights.

        Supports both global weights (1D) and row-wise weights (2D).
        """
        if weights is None and hasattr(self, "_weights"):
            weights = self._weights
        elif weights is None:
            weights = np.ones(len(df_list)) / len(df_list)

        weights = np.asarray(weights)
        n_df = len(df_list)

        if weights.ndim == 2:
            # row-wise weights: weights[i, k] is the weight for row i, comp k
            df_weighted = [
                df.multiply(weights[:, k], axis=0)
                for k, df in enumerate(df_list)
            ]
        else:
            df_weighted = [df * w for df, w in zip(df_list, weights)]

        df_concat = pd.concat(df_weighted, axis=1, keys=range(n_df))
        df_res = df_concat.T.groupby(level=-1).sum().T
        return df_res

    def _pdf(self, x):
        """Probability density function."""
        return self._average("pdf", x)

    def _cdf(self, x):
        """Cumulative distribution function."""
        return self._average("cdf", x)

    def _sample(self, n_samples=None):
        """Sample from the distribution.

        Parameters
        ----------
        n_samples : int, optional, default = None
            number of samples to draw from the distribution

        Returns
        -------
        pd.DataFrame
            samples from the distribution

            * if ``n_samples`` is ``None``:
            returns a sample that contains a single sample from ``self``,
            in ``pd.DataFrame`` mtype format convention, with ``index`` and ``columns``
            as ``self``
            * if n_samples is ``int``:
            returns a ``pd.DataFrame`` that contains ``n_samples`` i.i.d.
            samples from ``self``, in ``pd-multiindex`` mtype format convention,
            with same ``columns`` as ``self``, and row ``MultiIndex`` that is product
            of ``RangeIndex(n_samples)`` and ``self.index``
        """
        indep_rows = self.indep_rows
        indep_cols = self.indep_cols

        if n_samples is None:
            N = 1
        else:
            N = n_samples

        # deal with fully dependent case
        if not indep_rows and not indep_cols or self.ndim == 0:
            return self._sample_blocked(n_samples=n_samples)

        # we know that indep_rows and indep_cols are True, and self.ndim > 0
        rd_size = list(self.shape)
        full_size = list(self.shape)
        rd_size[0] *= N
        full_size[0] *= N

        if indep_rows:
            rd_size[0] = 1
        if indep_cols:
            rd_size[1] = 1

        n_dist = len(self._distributions)
        weights = self._weights

        if self._row_weights:
            # per-row component selection
            n_rows = full_size[0]
            n_cols = rd_size[1] if indep_cols else 1
            # tile row weights for n_samples if needed
            if N > 1:
                tiled_weights = np.tile(weights, (N, 1))
            else:
                tiled_weights = weights
            selector = np.array([
                np.random.choice(n_dist, size=n_cols, p=tiled_weights[i])
                for i in range(n_rows)
            ])
        else:
            selector = np.random.choice(n_dist, size=rd_size, p=weights)
        indicators = [selector == i for i in range(n_dist)]
        indicators = [np.broadcast_to(ind, full_size) for ind in indicators]

        dists = [d[1] for d in self._distributions]
        raw_samples = [d.sample(N).values for d in dists]
        masked_samples = [ind * raw for ind, raw in zip(indicators, raw_samples)]
        masked_samples = np.array(masked_samples)
        sample = masked_samples.sum(axis=0)

        if n_samples is None:
            spl_index = self.index
        else:
            spl_index = pd.MultiIndex.from_product([range(N), self.index])

        spl = pd.DataFrame(sample, index=spl_index, columns=self.columns)
        return spl

    def _sample_blocked(self, n_samples):
        """Sample from the distribution with blocked rows and columns."""
        if n_samples is None:
            N = 1
        else:
            N = n_samples

        n_dist = len(self._distributions)

        if self._row_weights:
            # for blocked case with row weights, use first row's weights
            p = self._weights[0]
        else:
            p = self._weights
        selector = np.random.choice(n_dist, size=N, p=p)

        samples = [self._distributions[i][1].sample() for i in selector]

        if n_samples is None:
            return samples[0]
        elif self.ndim > 0:
            return pd.concat(samples, axis=0, keys=range(N))
        else:  # if self.ndim == 0 and n_samples is not None
            return pd.DataFrame(samples, columns=self.columns)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        from skpro.distributions.normal import Normal

        # 2D array case
        index = pd.RangeIndex(3)
        columns = pd.Index(["a", "b"])
        normal1 = Normal(mu=0, sigma=1, index=index, columns=columns)
        normal2 = Normal(mu=[[0, 1], [2, 3], [4, 5]], sigma=1, columns=columns)

        dists = [("normal1", normal1), ("normal2", normal2)]
        dists2 = [normal1, normal2]  # to check case without names

        params1 = {"distributions": dists}
        params2 = {"distributions": dists2, "weights": [0.3, 0.7]}

        # scalar case
        normal3 = Normal(mu=0, sigma=1)
        normal4 = Normal(mu=3, sigma=2)
        dists = [("normal3", normal3), ("normal4", normal4)]
        dists2 = [normal3, normal4]

        params3 = {"distributions": dists2}
        params4 = {"distributions": dists, "weights": [0.3, 0.7]}

        # more than 2 distributions
        normal5 = Normal(mu=[[0, 1], [2, 3], [4, 5]], sigma=2, columns=columns)
        normal6 = Normal(mu=[[0, 1], [2, 3], [4, 5]], sigma=0.5, columns=columns)
        dists3 = [normal1, normal2, normal5, normal6]
        params5 = {"distributions": dists3}

        # row-wise weights: 3 rows, 2 components
        row_weights = [[0.9, 0.1], [0.5, 0.5], [0.1, 0.9]]
        normal7 = Normal(mu=0, sigma=1, index=index, columns=columns)
        normal8 = Normal(mu=[[0, 1], [2, 3], [4, 5]], sigma=1, columns=columns)
        dists_rw = [("normal7", normal7), ("normal8", normal8)]
        params6 = {"distributions": dists_rw, "weights": row_weights}

        return [params1, params2, params3, params4, params5, params6]
