import numpy as np
import pandas as pd
import pandas.testing as pdt
from scipy.stats import norm

from skpro.regression.shrinking_interval import ShrinkingNormalIntervalRegressor


def _make_xy(values):
    X = pd.DataFrame({"x": np.arange(len(values))})
    y = pd.DataFrame({"y": values})
    return X, y


def test_mean_sd_interval_width_shrinks_with_n():
    X_test = pd.DataFrame({"x": [0, 1]})

    X_small, y_small = _make_xy([0.0, 2.0, 0.0, 2.0])
    X_large, y_large = _make_xy([0.0, 2.0] * 8)

    reg_small = ShrinkingNormalIntervalRegressor(method="mean_sd").fit(X_small, y_small)
    reg_large = ShrinkingNormalIntervalRegressor(method="mean_sd").fit(X_large, y_large)

    int_small = reg_small.predict_interval(X_test, coverage=[0.9])
    int_large = reg_large.predict_interval(X_test, coverage=[0.9])

    width_small = int_small.iloc[:, 1] - int_small.iloc[:, 0]
    width_large = int_large.iloc[:, 1] - int_large.iloc[:, 0]

    assert (width_large < width_small).all()


def test_predict_interval_and_quantiles_shapes_and_values():
    X_train, y_train = _make_xy([0.0, 2.0, 0.0, 2.0])
    X_test = pd.DataFrame({"x": [10, 11, 12]})

    coverage = [0.8]
    alpha = [0.1, 0.5, 0.9]

    for method in ["mean_sd", "quantile"]:
        reg = ShrinkingNormalIntervalRegressor(method=method).fit(X_train, y_train)

        pred_int = reg.predict_interval(X_test, coverage=coverage)
        pred_q = reg.predict_quantiles(X_test, alpha=alpha)

        assert pred_int.shape == (len(X_test), 2)
        assert pred_q.shape == (len(X_test), 3)

        expected_int_cols = pd.MultiIndex.from_product(
            [["y"], coverage, ["lower", "upper"]], names=["var", "coverage", "bound"]
        )
        expected_q_cols = pd.MultiIndex.from_product(
            [["y"], alpha], names=["var", "alpha"]
        )

        pdt.assert_index_equal(pred_int.columns, expected_int_cols)
        pdt.assert_index_equal(pred_q.columns, expected_q_cols)

        if method == "mean_sd":
            mean = float(np.mean(y_train.to_numpy()))
            std = float(np.std(y_train.to_numpy(), ddof=1))
            n = len(y_train)

            z_interval = abs(norm.ppf((1 + coverage[0]) / 2))
            half_width = z_interval * std / np.sqrt(n)
            expected_int = np.column_stack(
                [
                    np.full(len(X_test), mean - half_width),
                    np.full(len(X_test), mean + half_width),
                ]
            )

            expected_q = np.column_stack(
                [
                    (
                        np.full(
                            len(X_test),
                            mean
                            + np.sign(a - 0.5) * abs(norm.ppf(a)) * std / np.sqrt(n),
                        )
                        if a != 0.5
                        else np.full(len(X_test), mean)
                    )
                    for a in alpha
                ]
            )
        else:
            y_values = y_train.to_numpy()
            expected_int = np.column_stack(
                [
                    np.full(
                        len(X_test),
                        np.percentile(
                            y_values,
                            100 * (1 - coverage[0]) / 2,
                            axis=0,
                        )[0],
                    ),
                    np.full(
                        len(X_test),
                        np.percentile(
                            y_values,
                            100 * (1 + coverage[0]) / 2,
                            axis=0,
                        )[0],
                    ),
                ]
            )
            expected_q = np.column_stack(
                [
                    np.full(
                        len(X_test),
                        np.percentile(y_values, 100 * a, axis=0)[0],
                    )
                    for a in alpha
                ]
            )

        pdt.assert_frame_equal(
            pred_int.reset_index(drop=True),
            pd.DataFrame(expected_int, columns=pred_int.columns),
            check_dtype=False,
        )
        pdt.assert_frame_equal(
            pred_q.reset_index(drop=True),
            pd.DataFrame(expected_q, columns=pred_q.columns),
            check_dtype=False,
        )


def test_small_n_mean_sd_edge_case():
    X_train, y_train = _make_xy([5.0])
    X_test = pd.DataFrame({"x": [0, 1]})

    reg = ShrinkingNormalIntervalRegressor(method="mean_sd").fit(X_train, y_train)

    pred_int = reg.predict_interval(X_test, coverage=[0.9])
    pred_q = reg.predict_quantiles(X_test, alpha=[0.1, 0.5, 0.9])

    assert np.isfinite(pred_int.to_numpy()).all()
    assert np.isfinite(pred_q.to_numpy()).all()
    assert np.allclose(pred_int.to_numpy(), 5.0)
    assert np.allclose(pred_q.to_numpy(), 5.0)
