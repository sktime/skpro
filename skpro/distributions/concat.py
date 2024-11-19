# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Concat operation."""

__author__ = ["SaiRevanth25"]

import pandas as pd


class concat:
    """Concatenate the given distributions along specified axis.

    Parameters
    ----------
    distributions : list
        list of distributions
    axis : {0/'index', 1/'columns'}, default 0
        The axis to concatenate along

    Examples
    --------
    >>> import skpro.distributions.concat as skpro
    >>> d1 = Normal(mu=[[1, 2], [3, 4]], sigma=1)
    >>> d2 = Normal(mu=0, sigma = [[2, 42]])
    >>> skpro.concat([d1,d2]).mean()

            0	1
        0	1	2
        1	3	4
        2	0	0

    >>> skpro.concat([d1,d2]).var()

            0	1
        0	1	1
        1	1	1
        2	4	1764


    >>> d3 = Gamma(alpha=[[5, 2]], beta=4)
    >>> d4 = Laplace(mu= [5,7], scale=[2,8])

    >>> skpro.concat([d2,d3,d4]).pdf(x=1)

                    0	 1
        Normal	4.0000	1764.000
        Gamma	0.3125	0.125
        Laplace	8.0000	128.000

    """

    def __init__(self, distributions, axis=0):
        """Initialize concat with list of distributions and axis for concatenation."""
        self.distributions = distributions
        self.axis = axis
        self.distribution_names = [dist.name for dist in distributions]

    def mean(self):
        """Calculate and concatenate means for each distribution."""
        means = [dist.mean() for dist in self.distributions]
        concatenated = pd.concat(means, axis=self.axis, ignore_index=True)
        concatenated.index = self._generate_index(len(concatenated))
        return concatenated

    def var(self):
        """Calculate and concatenate variances for each distribution."""
        variances = [dist.var() for dist in self.distributions]
        concatenated = pd.concat(variances, axis=self.axis, ignore_index=True)
        concatenated.index = self._generate_index(len(concatenated))
        return concatenated

    def pdf(self, x):
        """Concatenate PDFs of the distributions for a given value of `x`."""
        pdfs = []
        for dist in self.distributions:
            try:
                pdf_values = dist.pdf(x)
                pdfs.append(pdf_values)
            except ValueError as e:
                raise ValueError(
                    f"Error in pdf computation for distribution {dist.name}: {str(e)}"
                )

        concatenated = pd.concat(pdfs, axis=self.axis, ignore_index=True)
        concatenated.index = self._generate_index(len(concatenated))
        return concatenated

    def _generate_index(self, length):
        """Generate index for concatenated result."""
        if length != len(self.distribution_names):
            return pd.RangeIndex(start=0, stop=length)
        return self.distribution_names

    # todo: Constructing a new distribution when the two distributions are same.
    def _concat_distr(self):
        """Construct a new distrbution when the distributions are same."""
        pass
