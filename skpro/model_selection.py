from sklearn.model_selection import cross_val_score as sklearn_cross_val_score
import numpy as np


def cross_val_score(estimator, X, y=None, groups=None, scoring=None, cv=None,
                    verbose=0, fit_params=None):
    """Evaluate a score by cross-validation

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
    verbose : integer, optional
        The verbosity level.
    fit_params : dict, optional
        Parameters to pass to the fit method of the estimator.
    Returns
    -------
    scores : array of float, shape=(len(list(cv)),)
        Array of scores of the estimator for each run of the cross validation.
    Examples
    --------
    >>> from sklearn import datasets, linear_model
    >>> from sklearn.model_selection import cross_val_score
    >>> diabetes = datasets.load_diabetes()
    >>> X = diabetes.data[:150]
    >>> y = diabetes.target[:150]
    >>> lasso = linear_model.Lasso()
    >>> print(cross_val_score(lasso, X, y))  # doctest: +ELLIPSIS
    [ 0.33150734  0.08022311  0.03531764]
    See Also
    ---------
    :func:`sklearn.model_selection.cross_validate`:
        To run cross-validation on multiple metrics and also to return
        train scores, fit times and score times.
    :func:`sklearn.metrics.make_scorer`:
        Make a scorer from a performance metric or loss function.
    """

    scores = []

    def cv_scoring(estimator, X, y):
        score, std = scoring(estimator, X, y, return_std=True)
        scores.append([score, std])

        return score

    sklearn_cross_val_score(estimator, X, y, groups=groups, scoring=cv_scoring,
                            cv=cv, verbose=verbose, fit_params=fit_params)

    return np.array(scores)