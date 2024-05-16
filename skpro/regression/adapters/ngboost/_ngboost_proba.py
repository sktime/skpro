# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Adapter to ngboost probabilistic and survival regressors."""

__author__ = ["ShreeshaM07"]


class NGBoostAdapter:
    """Adapter to interconvert NGBoost and skpro BaseDistributions.

    NGBoostAdapter is a Mixin class adapter for the NGBoostRegressor
    and NGBoostSurvival classes to convert the distributions
    from ngboost to skpro and vice versa.

    It uses a boolean survival parameter to distinguish
    whether it is being called from the NGBoostRegressor
    or from NGBoostSurvival.
    """

    def _dist_to_ngboost_instance(self, dist, survival=False):
        """Convert string to NGBoost object.

        Parameters
        ----------
        dist : string
            the input string for the type of Distribution.
            It then creates an object of that particular NGBoost Distribution.
        survival : boolean, default = False
            It denotes whether it is called from the NGBoostSurvival
            or from NGBoostRegressor.

        Returns
        -------
        NGBoost Distribution object.
        """
        from ngboost.distns import Exponential, Laplace, LogNormal, Normal, Poisson, T

        ngboost_dists = {
            "Normal": Normal,
            "Laplace": Laplace,
            "TDistribution": T,
            "Poisson": Poisson,
            "LogNormal": LogNormal,
            "Exponential": Exponential,
        }
        # default Normal distribution
        dist_ngboost = Normal
        # default LogNormal distribution if Survival prediction
        if survival is True:
            dist_ngboost = LogNormal
        # replace default with other distribution if present
        if dist in ngboost_dists:
            dist_ngboost = ngboost_dists[dist]

        return dist_ngboost

    def _ngb_dist_to_skpro(self, **kwargs):
        """Convert NGBoost distribution object to skpro BaseDistribution object.

        Parameters
        ----------
        pred_mean, pred_std and index and columns.

        Returns
        -------
        skpro_dist (skpro.distributions.BaseDistribution):
        Converted skpro distribution object.
        """
        from skpro.distributions.exponential import Exponential
        from skpro.distributions.laplace import Laplace
        from skpro.distributions.lognormal import LogNormal
        from skpro.distributions.normal import Normal
        from skpro.distributions.poisson import Poisson
        from skpro.distributions.t import TDistribution

        ngboost_dists = {
            "Normal": Normal,
            "Laplace": Laplace,
            "TDistribution": TDistribution,
            "Poisson": Poisson,
            "LogNormal": LogNormal,
            "Exponential": Exponential,
        }

        skpro_dist = None

        if self.dist in ngboost_dists:
            skpro_dist = ngboost_dists[self.dist](**kwargs)

        return skpro_dist
