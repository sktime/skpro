import numpy as np
from sklearn.model_selection import cross_validate
# from sklearn.metrics.scorer import check_scoring


class RetrievesScores:

    def __init__(self, scorer, score=True, std=False):
        self.scorer = scorer
        self.score = score
        self.std = std

    def __call__(self, estimator, X, y):
        score, std = self.scorer(estimator, X, y, return_std=True)

        if self.score and self.std:
            return score, std
        elif self.std:
            return std
        else:
            return score


def cross_val_score(estimator, X, y=None, groups=None, scoring=None, cv=None,
                    n_jobs=1, verbose=0, fit_params=None, pre_dispatch='2*n_jobs'):
    """Evaluate a score using cross-validation

    Parameters
    ----------
    estimator : estimator object implementing 'fit'
        The object to use to fit the data.
    X : array-like
        The data to fit. Can be for example a list, or an array.
    y : array-like, optional, default: None
        The target variable to try to predict in the case of
        supervised learning.
    groups : array-like, with shape (n_samples,), optional
        Group labels for the samples used while splitting the dataset into
        train/test set.
    scoring : string, callable or None, optional, default: None
        A string (see model evaluation documentation) or
        a scorer callable object / function with signature
        ``scorer(estimator, X, y)``.
    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
        - None, to use the default 3-fold cross validation,
        - integer, to specify the number of folds in a `(Stratified)KFold`,
        - An object to be used as a cross-validation generator.
        - An iterable yielding train, test splits.
        For integer/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used.
        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.
    n_jobs : integer, optional
        The number of CPUs to use to do the computation. -1 means
        'all CPUs'.
    verbose : integer, optional
        The verbosity level.
    fit_params : dict, optional
        Parameters to pass to the fit method of the estimator.
    pre_dispatch : int, or string, optional
        Controls the number of jobs that get dispatched during parallel
        execution. Reducing this number can be useful to avoid an
        explosion of memory consumption when more jobs get dispatched
        than CPUs can process. This parameter can be:

            - None, in which case all the jobs are immediately
              created and spawned. Use this for lightweight and
              fast-running jobs, to avoid delays due to on-demand
              spawning of the jobs

            - An int, giving the exact number of total jobs that are
              spawned

            - A string, giving an expression as a function of n_jobs,
              as in '2*n_jobs'

    Returns
    -------
    scores : numpy.array, shape=(len(list(cv)), 2)
        Array of scores of the estimator for each run of the cross validation
        with their corresponding uncertainty.

    See Also
    ---------
    :func:`skpro.metrics.make_scorer`:
        Make a scorer from a performance metric or loss function.
    """

    # To ensure multimetric format is not supported
    scorer = check_scoring(estimator, scoring=scoring)

    if n_jobs == 1:
        # If we are not multiprocessing it's possible to
        # use a wrapper function to retrieve the std values
        test_scores = []

        def scoring_task(estimator, X, y):
            score, std = scorer(estimator, X, y, return_std=True)
            test_scores.append([score, std])

            return score
    else:
        # We allow multiprocessing by passing in two scoring functions.
        # That is far from ideal since we call the scorer twice,
        # so any improvement is welcome
        score_scorer = RetrievesScores(scorer, score=True, std=False)
        std_scorer = RetrievesScores(scorer, score=False, std=True)
        scoring_task = {'score': score_scorer, 'std': std_scorer}

    cv_results = cross_validate(estimator=estimator, X=X, y=y, groups=groups,
                                scoring=scoring_task, cv=cv,
                                return_train_score=False,
                                n_jobs=n_jobs, verbose=verbose,
                                fit_params=fit_params,
                                pre_dispatch=pre_dispatch)

    if n_jobs == 1:
        return np.array(test_scores)
    else:
        return np.column_stack((cv_results['test_score'], cv_results['test_std']))
