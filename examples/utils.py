# -*- coding: utf-8 -*-
import numpy as np
from matplotlib import pyplot

from skpro.metrics import log_loss


def plot_performance(y_test, y_pred, filename=None):
    """
    Visualises the prediction performance

    Parameters
    ----------
    y_test  Ground truth
    y_pred  Predictions
    filename    If string, figure will be saved to file

    Returns
    -------
    Matplotlib plot
    """

    fig, ax1 = pyplot.subplots()

    ax1.plot(y_test, y_test, "g.", label="Optimum")
    sigma = np.std(y_pred)
    ax1.errorbar(
        y_test,
        y_pred.point(),
        yerr=sigma,
        label="Predictions",
        fmt="b.",
        ecolor="r",
        linewidth=0.5,
    )
    ax1.set_ylabel("Predicted $y_{pred}$")
    ax1.set_xlabel("Correct label $y_{true}$")
    ax1.legend(loc="best")

    losses = log_loss(y_test, y_pred, sample=False)
    ax2 = ax1.twinx()
    overall = "{0:.2f}".format(np.mean(losses)) + " +/- {0:.2f}".format(np.std(losses))
    ax2.set_ylabel("Loss")
    ax2.plot(y_test, losses, "y_", label="Loss: " + overall)
    ax2.tick_params(colors="y")
    ax2.legend(loc=1)

    if not isinstance(filename, str):
        pyplot.show()
    else:
        pyplot.savefig(filename, transparent=True, bbox_inches="tight")
