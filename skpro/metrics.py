import numpy as np


def sample_loss(loss, return_std=False):
    loss = loss[~np.isnan(loss)]
    if return_std:
        return np.mean(loss), np.std(loss) / np.sqrt(len(loss))
    else:
        return np.mean(loss)


def make_scorer(loss_function, greater_is_better=True, return_std=False):
    def scorer(estimator, X_test, y_test, return_std=return_std):
        sign = 1 if greater_is_better else -1
        y_pred = estimator.predict(X_test)
        return sign * loss_function(y_pred, y_test, return_std=return_std)

    return scorer


def gneiting_loss(dist_pred, y, sample=True, return_std=False):
    lp2 = getattr(dist_pred, 'lp2', False)
    if not lp2:
        raise Exception('The estimator does not provide an lp2 integration')

    loss = -2 * dist_pred.pdf(y) + lp2()

    if sample is True:
        return sample_loss(loss, return_std)

    return loss


def linearized_log_loss(dist_pred, y, range=1e-10, sample=True, return_std=False):
    pdf = dist_pred.pdf(y)

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


def log_loss(dist_pred, y, sample=True, return_std=False):
    pdf = dist_pred.pdf(y)
    loss = -np.log(pdf)

    if sample:
        return sample_loss(loss, return_std)

    return loss


def brier_loss(dist_pred, y, sample=True, return_std=False):
    pdf = dist_pred.pdf(y)

    loss = (1 - pdf) ** 2

    if sample:
        return sample_loss(loss, return_std)

    return loss
