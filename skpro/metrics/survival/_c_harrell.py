"""Concordance index, Harrell's."""
# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)

import numpy as np
import pandas as pd

from skpro.metrics.base import BaseDistrMetric


class ConcordanceHarrell(BaseDistrMetric):
    r"""Concordance index (Harrell).

    Fraction of concordant test index pairs among all comparale pairs,
    as proposed in [1]_, commonly known as Harrell's C-index, Harrell's C,
    or simply concordance index,
    if not in delination of other C-indices (e.g., Uno's C-index).

    For ground truth samples :math:`y_i, c_i, i = 1 \dots N`,
    and predicted inverse risk scores :math:`s_i, i = 1 \dots N`,
    a pair of test non-equal test indices :math:`i \lneq j` is concordant if
    :math:`(y_i > y_j) \land (s_i > s_j)` or :math:`(y_i < y_j) \land (s_i < s_j)`.
    If :math:`(s_i = s_j)`, the pair is counted as concordant if :math:`y_i = y_j`,
    and :math:`c_i = c_j = 0`, otherwise it is considered a tie,
    counted as half concordant, half discordant by default.

    A pair of test indices :math:`i \lneq j` is said to be comparable
    if one of the following conditions holds:

    * :math:`y_i > y_j` and :math:`c_j = 0`
    * :math:`y_i < y_j` and :math:`c_i = 0`
    * :math:`y_i = y_j` and :math:`c_i c_j = 0`

    This metric supports multiple options for inverse risk scores,
    including any method evaluates of predictive distributions.

    The default is the predictive mean survival time.

    `evaluate` computes the concordance index.
    `evaluate_by_index` produces, for one test sample,
    the fraction of concordant pairs among all pairs with this sample as first index.
    `multivariate` controls averaging over variables.

    Parameters
    ----------
    score : str, optional, default='mean'
        The type of inverse risk score to use.
        Calls predict_proba, then the method of the same name as `score`.
        Examples include 'mean', 'median', 'quantile', 'cdf'.

    score_args : dict, optional, default=None
        Additional arguments to pass to the score method, e.g., quantiles.

    higher_score_is_lower_risk : bool, optional, default=True
        If True, higher score is considered lower risk, and vice versa,
        that is, the score is assumed to be an inverse risk score.
        If False, the score is assumed to be a risk score, and a
        negative sign is applied to the score.

    tie_score : float, optional, default=0.5
        The value to use for ties in the risk scores,
        as a relative value to counting as concordant.
        1 is counting as concordant, 0 is counting as discordant.
        0.5 is counting as half concordant, half discordant.

    normalization : str, {'overall', 'index'}, optional, default='overall'
        Determines the normalization of the concordance index, whether
        fractions of concordant pairs are averaged primarily overall,
        or primarily per index. In both cases, ``evaluate`` returns the
        arithmetic mean of ``evaluate_by_index``.

        * If ``'overall'``, ``evaluate``
        returns the fraction of concordant among all comparable pairs.
        This is as in [1]_.
        In ``evaluate_by_index``, fraction denominators are the
        number of comparable pairs overall, divided by the number of samples,
        instead of the number of comparable pairs in which the index is the first index.

        * If ``'index'``, ``evaluate`` returns the average, over indices,
        of the fraction of concordant pairs among
        all comparable pairs in which the index is the first index.
        In ``evaluate_by_index``, entries are the fraction of concordant pairs
        among all comparable pairs in which the index is the first index,
        without further normalization.

    multioutput : {'raw_values', 'uniform_average'} or array-like of shape
        (n_outputs,), default='uniform_average'
        Defines whether and how to aggregate metric for across variables.
        If 'uniform_average' (default), errors are mean-averaged across variables.
        If array-like, errors are weighted averaged across variables, values as weights.
        If 'raw_values', does not average errors across variables, columns are retained.
    multivariate : bool, optional, default=False
        if True, behaves as multivariate log-loss
        C-index is computed for entire row, results one score per row
        if False, is univariate log-loss
        C-index is computed per variable marginal, results in many scores per row

    References
    ----------
    .. [1] Harrell FE, Califf R.M., Pryor DB, Lee KL, Rosati RA.
      Multivariable prognostic models: issues in developing models,
      evaluating assumptions and adequacy, and measuring and reducing errors.
      Statistics in Medicine, 15(4), 361-87, 1996.
    """

    _tags = {
        "authors": "fkiraly",
        "capability:survival": True,
        "scitype:y_pred": "pred_proba",
        "lower_is_better": False,
    }

    def __init__(
        self,
        score="mean",
        score_args=None,
        higher_score_is_lower_risk=True,
        tie_score=0.5,
        normalization="overall",
        multioutput="uniform_average",
        multivariate=False,
    ):
        self.score = score
        self.score_args = score_args
        self.higher_score_is_lower_risk = higher_score_is_lower_risk
        self.tie_score = tie_score
        self.normalization = normalization
        self.multivariate = multivariate
        super().__init__(multioutput=multioutput)

    def _evaluate_by_index(self, y_true, y_pred, **kwargs):
        C_true = kwargs.get("C_true", None)

        # retrieve parameters
        tie_score = self.tie_score
        normalization = self.normalization
        score_args = self.score_args
        if score_args is None:
            score_args = {}

        # convert to numpy and remember index and columns
        ix = y_true.index
        cols = y_true.columns

        y_true = y_true.to_numpy()
        if C_true is not None:
            C_true = C_true.to_numpy()
        else:
            C_true = np.zeros_like(y_true)

        risk_scores = getattr(y_pred, self.score)(**score_args)
        if not self.higher_score_is_lower_risk:
            risk_scores = -risk_scores
        risk_scores = risk_scores.to_numpy()

        ncomp_mat = np.zeros_like(y_true)
        nconc_mat = np.zeros_like(y_true)
        result = np.zeros_like(y_true)

        # compute concordance index for each index
        for j in range(y_true.shape[1]):
            yj = y_true[:, j]
            Cj = C_true[:, j] == 1
            nCj = ~Cj
            rj = risk_scores[:, j]
            for i in range(y_true.shape[0]):
                yij = yj[i]
                rij = rj[i]
                Cij = Cj[i]
                nCij = ~Cij
                one_unc = ~(Cj & Cij)

                # mark concordant pairs (no ties)
                comp1 = nCij & (yj > yij)  # comparable, > type
                conc1 = comp1 & (rj > rij)
                comp2 = nCj & (yj < yij)  # comparable, < type
                conc2 = comp2 & (rj < rij)

                # mark concordant pairs, handling ties
                conc3 = nCij & nCj & ((yj == yij) & (rj == rij))
                conc = conc1 | conc2 | conc3

                # count concordant pairs
                nconc = conc.sum() - nCij  # sum, subtract i=j if counted above
                # i=j was counted iff it was not censored

                # handle ties in total of concordant pairs
                if tie_score != 0:
                    nconc = nconc.astype(float)
                    nconc += np.sum((yj != yij) & (rj == rij)) * tie_score
                    nconc += np.sum(one_unc & (yj == yij) & (rj == rij)) * tie_score

                # count comparable pairs
                comp3 = one_unc & (yj == yij)
                comp = comp1 | comp2 | comp3
                ncomp = comp.sum() - nCij  # subtract i=j, but only if counted above

                nconc_mat[i, j] = nconc
                ncomp_mat[i, j] = ncomp

        # normalization
        if normalization == "overall":
            # weighting is such that average over rows
            # results in the overall C-index
            ncomp_total = ncomp_mat.sum(axis=0)
            nspl = len(ncomp_mat)
            result = (nspl / ncomp_total) * nconc_mat
        else:  # normalization == "index"
            # weighting is such that rows contain simple fractions
            # but the average over rows is not the overall C-index,
            # as number of comparable pairs is in general not the same for each index
            result = nconc_mat / ncomp_mat

        res_df = pd.DataFrame(result, index=ix, columns=cols)

        if self.multivariate:
            return pd.DataFrame(res_df.mean(axis=1), columns=["C_Harrell"])
        else:
            return res_df
