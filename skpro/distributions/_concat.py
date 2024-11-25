# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Base class for concat operation."""

__author__ = ["SaiRevanth25"]

import pandas as pd
from skbase.base._meta import _MetaObjectMixin


class ConcatDistr(_MetaObjectMixin):
    """Concatenate the given distributions along specified axis.

    Parameters
    ----------
    distributions : list
        list of distributions
    axis : {0/'index', 1/'columns'}, default 0
        The axis to concatenate along
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

    def cdf(self, x):
        """Concatenate CDFs for each distribution at a given value of `x`."""
        cdfs = []
        for dist in self.distributions:
            try:
                cdf_values = dist.cdf(x)
                cdfs.append(cdf_values)
            except ValueError as e:
                raise ValueError(
                    f"Error in cdf computation for distribution {dist.name}: {str(e)}"
                )

        concatenated = pd.concat(cdfs, axis=self.axis, ignore_index=True)
        concatenated.index = self._generate_index(len(concatenated))
        return concatenated

    def _generate_index(self, length):
        """Generate index for concatenated result."""
        if length != len(self.distribution_names):
            return pd.RangeIndex(start=0, stop=length)
        return self.distribution_names

    # todo: Constructing a new distribution when the two distributions are same.
    def _constr_distribution(self):
        """Construct a new distrbution when the distributions are same."""
        pass
