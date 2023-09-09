"""Legacy module: test utils."""
# LEGACY MODULE - TODO: remove or refactor

import numpy as np


def assert_close_prediction(y_hat, y_true, fraction=0.75, within=0.25):
    """Check that defined fraction of predictions lies in a certain tolerance.

    Parameters
    ----------
    y_hat   Predictions
    y_true  True values
    fraction Fraction of close values
    within  Relative tolerance to assume when comparing the values

    Raises
    ------
    AssertionError
    """
    predictions_within_tolerance = np.count_nonzero(
        np.isclose(y_hat, y_true, rtol=within)
    )
    target = len(y_true) * fraction

    assert predictions_within_tolerance > target
