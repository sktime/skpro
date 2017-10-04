import numpy as np

from sklearn.neighbors import KernelDensity


def ecdf(a, return_func=False):
    """ Returns the empirical distribution function of a sample

    Parameters
    ----------
    a: array
        Input array representing a sample
    return_func: bool
        If true, a function that represents the ecdf is returned
        instead of x and y values

    Returns
    -------
    mixed   Empirical cdf of the input sample
    """
    xs = np.sort(np.array(a))
    ys = np.arange(1, len(xs)+1)/float(len(xs))

    if not return_func:
        return xs, ys

    def func(x):
        index = np.searchsorted(xs, x)
        index = len(ys) - 1 if index >= len(ys) else index
        return ys[index]

    return func


def _kde(self):
    if self.kde_ is None:
        self.kde_ = [
            KernelDensity().fit(self.sample(index)[:, np.newaxis])
            for index in range(len(self.X))
        ]

    return self.kde_


def pdf(self, x):
    return [
        np.exp(self._kde()[index].score_samples(x[:, np.newaxis]))[0]
        for index in range(len(self.X))
    ]


def sample(self, index):
    return self.estimator.samples_["y_pred"][index, :]


def _ecdf(self):
    if self.ecdf_ is None:
        self.ecdf_ = [
            ecdf(self.sample(index))
            for index in range(len(self.X))
        ]

    return self.ecdf_


def cdf(self, x):
    return [self._ecdf()[index](x) for index in range(len(self.X))]