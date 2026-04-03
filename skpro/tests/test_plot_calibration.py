import pandas as pd
import pytest

from skpro.utils.plotting import plot_calibration


def test_plot_calibration_runs():
    """Test that plot_calibration runs without errors."""
    pytest.importorskip("matplotlib")
    y_true = pd.Series([1, 2, 3, 4])
    y_pred = pd.DataFrame(
        {
            (0, 0.1): [1, 2, 2, 3],
            (0, 0.5): [2, 3, 3, 4],
            (0, 0.9): [3, 4, 4, 5],
        }
    )
    fig, ax = plot_calibration(y_true, y_pred)
    assert fig is not None
    assert ax is not None


def test_plot_calibration_with_ax():
    """Test that plot_calibration accepts an existing ax."""
    pytest.importorskip("matplotlib")
    from matplotlib import pyplot as plt

    y_true = pd.Series([1, 2, 3, 4])
    y_pred = pd.DataFrame(
        {
            (0, 0.1): [1, 2, 2, 3],
            (0, 0.9): [3, 4, 4, 5],
        }
    )
    _, ax = plt.subplots()
    result_ax = plot_calibration(y_true, y_pred, ax=ax)
    assert result_ax is ax
