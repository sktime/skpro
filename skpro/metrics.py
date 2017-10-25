import numpy as np


def sample_loss(loss, return_std=False):
    """ Averages the loss of a sample

    Parameters
    ----------
    loss: np.array
        Loss sample
    return_std: boolean, default=False
        If true, the standard deviation of the
        loss sample will be returned

    Returns
    -------
    np.array
        Sample loss (with standard deviation if ``return_std`` is True)
    """
    loss = loss[~np.isnan(loss)]
    if return_std:
        return np.mean(loss), np.std(loss) / np.sqrt(len(loss))
    else:
        return np.mean(loss)


def make_scorer(score_func, greater_is_better=True, return_std=False, **kwargs):
    """Make a scorer from a performance metric or loss function.

    This factory function wraps scoring functions for use in GridSearchCV
    and cross_val_score. It takes a score function, such as ``log_loss``,
    and returns a callable that scores an estimator's output.

    Parameters
    ----------
    score_func : callable,
        Score function (or loss function) with signature
        ``score_func(y, y_pred, **kwargs)``.

    greater_is_better : boolean, default=True
        Whether score_func is a score function (default), meaning high is good,
        or a loss function, meaning low is good. In the latter case, the
        scorer object will sign-flip the outcome of the score_func.

    return_std: boolean, default=False
        If true, the scorer returns a standard deviation

    **kwargs : additional arguments
        Additional parameters to be passed to score_func.

    Returns
    -------
    scorer : callable
        Callable object that returns a scalar score; greater is better.
    """

    def scorer(estimator, X_test, y_test, return_std=return_std):
        sign = 1 if greater_is_better else -1
        y_pred = estimator.predict(X_test)
        return sign * score_func(y_pred, y_test, return_std=return_std, **kwargs)

    return scorer


def gneiting_loss(y_true, dist_pred, sample=True, return_std=False):
    """ Gneiting loss

    Parameters
    ----------
    y_true: np.array
        The true labels
    dist_pred: ProbabilisticEstimator.Distribution
        The predicted distribution
    sample: boolean, default=True
        If true, loss will be averaged across the sample
    return_std: boolean, default=False
        If true, the standard deviation of the
        loss sample will be returned

    Returns
    -------
    np.array
        Loss (with standard deviation if ``return_std`` is True)
    """
    lp2 = getattr(dist_pred, 'lp2', False)
    if not lp2:
        raise Exception('The estimator does not provide an lp2 integration')

    loss = -2 * dist_pred.pdf(y_true) + lp2()

    if sample is True:
        return sample_loss(loss, return_std)

    return loss


def linearized_log_loss(y_true, dist_pred, range=1e-10, sample=True, return_std=False):
    """ Linearized log loss

    Parameters
    ----------
    y_true: np.array
        The true labels
    dist_pred: ProbabilisticEstimator.Distribution
        The predicted distribution
    range: float
        Threshold value of linearization
    sample: boolean, default=True
        If true, loss will be averaged across the sample
    return_std: boolean, default=False
        If true, the standard deviation of the
        loss sample will be returned

    Returns
    -------
    np.array
        Loss (with standard deviation if ``return_std`` is True)
    """
    pdf = dist_pred.pdf(y_true)

    def f(x):
        if x <= range:
            return (-1 / range) * x - np.log(range) + 1
        else:
            return -np.log(x)

    f = np.vectorize(f)
    loss = f(pdf)

    if sample:
        return sample_loss(loss, return_std)

    return loss


def log_loss(y_true, dist_pred, sample=True, return_std=False):
    """ Log loss

    Parameters
    ----------
    y_true: np.array
        The true labels
    dist_pred: ProbabilisticEstimator.Distribution
        The predicted distribution
    sample: boolean, default=True
        If true, loss will be averaged across the sample
    return_std: boolean, default=False
        If true, the standard deviation of the
        loss sample will be returned

    Returns
    -------
    np.array
        Loss (with standard deviation if ``return_std`` is True)
    """
    pdf = dist_pred.pdf(y_true)
    loss = -np.log(pdf)

    if sample:
        return sample_loss(loss, return_std)

    return loss


def rank_probability_loss(y_true, dist_pred, sample=True, return_std=False):
    """ Rank probability loss

    .. math::
        L(F,y) = -int_-\infty^y F(x)^2 dx - int_y^\infty (1-F(x))^2 dx

    Parameters
    ----------
    y_true: np.array
        The true labels
    dist_pred: ProbabilisticEstimator.Distribution
        The predicted distribution
    sample: boolean, default=True
        If true, loss will be averaged across the sample
    return_std: boolean, default=False
        If true, the standard deviation of the
        loss sample will be returned

    Returns
    -------
    np.array
        Loss (with standard deviation if ``return_std`` is True)
    """
    def term(index, one_minus=False):
        def integrand(x):
            if one_minus:
                return (1 - dist_pred[index].cdf(x)) ** 2
            else:
                return dist_pred[index].cdf(x) ** 2

        return integrand

    from scipy.integrate import quad as integrate

    loss = -1 * np.array([
        # -int_ -\infty ^ y F(x)² dx
        - integrate(term(index), -np.inf, y_true[index])[0]
        # – int_y ^\infty(1 - F(x))² dx
        - integrate(term(index, one_minus=True), y_true[index], np.inf)[0]
        for index in range(len(dist_pred))
    ])

    if sample:
        return sample_loss(loss, return_std)

    return loss