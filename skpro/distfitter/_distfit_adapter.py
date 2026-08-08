"""Generic scalar distribution wrapping a named scipy distribution.

Used to turn the output of ``distfit`` (a scipy distribution name plus
shape/loc/scale parameters) into a fitted ``skpro`` ``BaseDistribution``,
by reusing the existing ``_ScipyAdapter`` machinery.
"""
# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)

from skpro.distributions.adapters.scipy import _ScipyAdapter

__all__ = ["_DistfitDistribution"]


class _DistfitDistribution(_ScipyAdapter):
    """Scalar distribution wrapping a scipy distribution selected by ``distfit``.

    Parameters
    ----------
    dist_name : str
        Name of a distribution in ``scipy.stats``, e.g. ``"norm"``, ``"gamma"``.
    shape_args : tuple, optional (default=())
        Positional shape parameters for the ``scipy.stats`` distribution.
    dist_loc : float, optional (default=0.0)
        Location parameter, passed as ``loc`` to the ``scipy.stats`` distribution.
    dist_scale : float, optional (default=1.0)
        Scale parameter, passed as ``scale`` to the ``scipy.stats`` distribution.
    """

    _tags = {
        "authors": ["areychana"],
        "capabilities:exact": ["mean", "var", "pdf", "log_pdf", "cdf", "ppf"],
        "distr:measuretype": "continuous",
        "distr:paramtype": "parametric",
    }

    def __init__(self, dist_name, shape_args=(), dist_loc=0.0, dist_scale=1.0):
        self.dist_name = dist_name
        self.shape_args = shape_args
        self.dist_loc = dist_loc
        self.dist_scale = dist_scale

        super().__init__(index=None, columns=None)

    def _get_scipy_object(self):
        import scipy.stats

        return getattr(scipy.stats, self.dist_name)

    def _get_scipy_param(self):
        args = list(self.shape_args)
        kwds = {"loc": self.dist_loc, "scale": self.dist_scale}
        return args, kwds
