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

    def _ngb_skpro_dist_params(
        self,
        pred_dist,
        index,
        columns,
        **kwargs,
    ):
        import numpy as np

        # The returned values of the Distributions from NGBoost
        # are different. So based on that they are split into these
        # categories of loc,scale,mu and s.
        # Distribution type | Parameters
        # ------------------|-----------
        # Normal            | loc = mean, scale = standard deviation
        # TDistribution     | loc = mean, scale = standard deviation
        # Poisson           | mu = mean
        # LogNormal         | s = standard deviation, scale = exp(mean)
        #                   |     (see scipy.stats.lognorm)
        # Laplace           | loc = mean, scale = scale parameter
        # Exponential       | scale = 1/rate
        # Normal, Laplace, TDistribution and Poisson have not yet
        # been implemented for Survival analysis.

        dist_params = {
            "Normal": ["loc", "scale"],
            "Laplace": ["loc", "scale"],
            "TDistribution": ["loc", "scale"],
            "Poisson": ["mu"],
            "LogNormal": ["scale", "s"],
            "Exponential": ["scale"],
        }

        skpro_params = {
            "Normal": ["mu", "sigma"],
            "Laplace": ["mu", "scale"],
            "TDistribution": ["mu", "sigma"],
            "Poisson": ["mu"],
            "LogNormal": ["mu", "sigma"],
            "Exponential": ["rate"],
        }

        if self.dist in dist_params and self.dist in skpro_params:
            ngboost_params = dist_params[self.dist]
            skp_params = skpro_params[self.dist]
            for ngboost_param, skp_param in zip(ngboost_params, skp_params):
                kwargs[skp_param] = pred_dist.params[ngboost_param]
                if self.dist == "LogNormal" and ngboost_param == "scale":
                    kwargs[skp_param] = np.log(pred_dist.params[ngboost_param])
                if self.dist == "Exponential" and ngboost_param == "scale":
                    kwargs[skp_param] = 1 / pred_dist.params[ngboost_param]

                kwargs[skp_param] = self._check_y(y=kwargs[skp_param])
                # returns a tuple so taking only first index of the tuple
                kwargs[skp_param] = kwargs[skp_param][0]
            kwargs["index"] = index
            kwargs["columns"] = columns

        return kwargs

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
