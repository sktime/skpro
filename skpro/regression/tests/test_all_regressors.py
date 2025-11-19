"""Automated tests based on the skbase test suite template."""
import pandas as pd
import pytest
from skbase.testing import QuickTester

from skpro.datatypes import check_is_mtype, check_raise
from skpro.distributions.base import BaseDistribution
from skpro.tests.test_all_estimators import BaseFixtureGenerator, PackageConfig

TEST_ALPHAS = [0.05, [0.1], [0.25, 0.75], [0.3, 0.1, 0.9]]


class TestAllRegressors(PackageConfig, BaseFixtureGenerator, QuickTester):
    """Generic tests for all regressors in the mini package."""

    # class variables which can be overridden by descendants
    # ------------------------------------------------------

    # which object types are generated; None=all, or scitype string
    # passed to skpro.registry.all_objects as object_type
    object_type_filter = "regressor_proba"

    def test_input_output_contract(self, object_instance):
        """Tests that output of predict methods is as specified."""
        import pandas as pd
        from sklearn.datasets import load_diabetes
        from sklearn.model_selection import train_test_split

        X, y = load_diabetes(return_X_y=True, as_frame=True)
        X = X.iloc[:50]
        y = y.iloc[:50]
        y = pd.DataFrame(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y)

        # fit - just once for all predict output methods
        regressor = object_instance
        regressor.fit(X_train, y_train)

        # test predict output contract
        y_pred = regressor.predict(X_test)

        assert isinstance(y_pred, pd.DataFrame)
        assert (y_pred.index == X_test.index).all()
        assert (y_pred.columns == y_train.columns).all()
        # check that return is float dtype
        for col in y_pred.columns:
            assert pd.api.types.is_float_dtype(y_pred[col])

        # test predict_proba output contract
        y_pred_proba = regressor.predict_proba(X_test)

        assert isinstance(y_pred_proba, BaseDistribution)
        assert (y_pred_proba.index == X_test.index).all()
        assert (y_pred_proba.columns == y_train.columns).all()
        assert y_pred_proba.shape == y_test.shape

        assert isinstance(y_pred_proba.sample(), pd.DataFrame)
        assert y_pred_proba.sample().shape == y_test.shape

        # test predict_interval output contract with default coverage
        y_pred_interval = regressor.predict_interval(X_test)

        assert isinstance(y_pred_interval, pd.DataFrame)
        assert (y_pred_interval.index == X_test.index).all()
        check_raise(
            y_pred_interval, "pred_interval", "Proba", "predict_interval return"
        )

        # test predict_quantiles output contract with default alpha
        y_pred_quantiles = regressor.predict_quantiles(X_test)

        assert isinstance(y_pred_quantiles, pd.DataFrame)
        assert (y_pred_quantiles.index == X_test.index).all()
        check_raise(
            y_pred_quantiles, "pred_quantiles", "Proba", "predict_quantiles return"
        )

        # test predict_var output contract
        y_pred_var = regressor.predict_var(X_test)

        assert isinstance(y_pred_var, pd.DataFrame)
        assert (y_pred_var.index == X_test.index).all()
        assert (y_pred_var.columns == y_train.columns).all()

    def _check_predict_quantiles(self, pred_quantiles, X_test, y_train, alpha):
        """Check expected quantile prediction output."""
        # check expected type
        valid, msg, _ = check_is_mtype(
            pred_quantiles,
            mtype="pred_quantiles",
            scitype="Proba",
            return_metadata=True,
            var_name="predict_quantiles return",
            msg_return_dict="list",
        )  # type: ignore
        assert valid, msg

        # check index
        assert (pred_quantiles.index == X_test.index).all()

        # check columns
        expected_columns = y_train.columns
        expected_quantiles = [alpha] if isinstance(alpha, float) else alpha
        expected = pd.MultiIndex.from_product([expected_columns, expected_quantiles])

        found = pred_quantiles.columns
        msg = (
            "columns of returned quantile prediction DataFrame do not"
            f"match up with expected columns. Expected: {expected},"
            f"found: {found}"
        )
        assert all(expected == found), msg

    def _check_predict_intervals(self, pred_ints, X_test, y_train, coverage):
        """Check expected interval prediction output."""
        # check expected type
        valid, msg, _ = check_is_mtype(
            pred_ints,
            mtype="pred_interval",
            scitype="Proba",
            return_metadata=True,
            var_name="predict_interval return",
            msg_return_dict="list",
        )  # type: ignore
        assert valid, msg

        # check index
        assert (pred_ints.index == X_test.index).all()

        # check columns
        expected_columns = y_train.columns
        expected_coverages = [coverage] if isinstance(coverage, float) else coverage
        expected = pd.MultiIndex.from_product(
            [expected_columns, expected_coverages, ["lower", "upper"]]
        )

        found = pred_ints.columns
        msg = (
            "columns of returned prediction interval DataFrame do not"
            f"match up with expected columns. Expected: {expected},"
            f"found: {found}"
        )
        assert all(expected == found), msg

    @pytest.mark.parametrize(
        "alpha", TEST_ALPHAS, ids=[f"alpha={a}" for a in TEST_ALPHAS]
    )
    def test_pred_quantiles_interval(self, object_instance, alpha):
        """Test predict_interval and predict_quantiles output with different alpha."""
        import pandas as pd
        from sklearn.datasets import load_diabetes
        from sklearn.model_selection import train_test_split

        X, y = load_diabetes(return_X_y=True, as_frame=True)
        X = X.iloc[:50]
        y = y.iloc[:50]
        y = pd.DataFrame(y)

        X_train, X_test, y_train, _ = train_test_split(X, y)

        regressor = object_instance
        regressor.fit(X_train, y_train)

        # check predict_interval output contract
        pred_ints = regressor.predict_interval(X_test, alpha)
        self._check_predict_intervals(pred_ints, X_test, y_train, alpha)

        # check predict_quantiles output contract
        pred_q = regressor.predict_quantiles(X_test, alpha)
        self._check_predict_quantiles(pred_q, X_test, y_train, alpha)

    def test_online_update(self, object_instance):
        """Test online update of regressor."""
        import pandas as pd
        from sklearn.datasets import load_diabetes
        from sklearn.model_selection import train_test_split

        X, y = load_diabetes(return_X_y=True, as_frame=True)
        X = X.iloc[:70]
        y = y.iloc[:70]
        y = pd.DataFrame(y)

        X_train, X_test, y_train, _ = train_test_split(X, y)
        X_fit, X_update, y_fit, y_update = train_test_split(X_train, y_train)
        X_upd1, X_upd2, y_upd1, y_upd2 = train_test_split(X_update, y_update)

        regressor = object_instance
        regressor.fit(X_fit, y_fit)

        regressor.update(X_upd1, y_upd1)
        y_pred1 = regressor.predict(X_upd1)
        y_pred2 = regressor.predict(X_upd2)

        # check predict output contract
        assert isinstance(y_pred2, pd.DataFrame)
        assert (y_pred1.index == X_upd1.index).all()
        assert (y_pred1.columns == y_fit.columns).all()
        assert (y_pred2.index == X_upd2.index).all()
        assert (y_pred2.columns == y_fit.columns).all()

        regressor.update(X_upd2, y_upd2)
        y_pred_test = regressor.predict(X_test)

        # check predict output contract
        assert isinstance(y_pred_test, pd.DataFrame)
        assert (y_pred_test.index == X_test.index).all()
        assert (y_pred_test.columns == y_fit.columns).all()

    # pyGAM adapter specific tests
    def test_pygam_adapter_distributions(self, object_instance):
        """Test pyGAM adapter with different distributions."""
        from skpro.tests.test_switch import run_test_for_class
        from skpro.regression.adapters.pygam import PyGAMAdapter

        # only run for PyGAMAdapter instances
        if not isinstance(object_instance, PyGAMAdapter):
            pytest.skip("Not a PyGAMAdapter instance")

        # skip if pygam isn't available
        if not run_test_for_class(PyGAMAdapter):
            pytest.skip("pygam dependencies not available")

        import pandas as pd
        from sklearn.datasets import load_diabetes
        from sklearn.model_selection import train_test_split

        X, y = load_diabetes(return_X_y=True, as_frame=True)
        y = pd.DataFrame(y)
        X = X.iloc[:100]
        y = y.iloc[:100]
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

        # check that distribution gets detected correctly
        adapter = object_instance
        adapter.fit(X_train, y_train)

        # make sure it detected the distribution
        assert hasattr(adapter, "_actual_distribution")
        assert adapter._actual_distribution in ["normal", "poisson", "gamma", "inverse_gaussian"]

        # make sure predictions work
        y_pred = adapter.predict(X_test)
        y_pred_proba = adapter.predict_proba(X_test)

        assert y_pred.shape == y_test.shape
        assert y_pred_proba.shape == y_test.shape

    def test_pygam_adapter_get_test_params(self):
        """Test that PyGAMAdapter.get_test_params works correctly."""
        from skpro.regression.adapters.pygam import PyGAMAdapter

        params = PyGAMAdapter.get_test_params()
        assert params is not None

        # if pygam isn't installed, it returns a special marker
        if isinstance(params, dict) and params.get("estimator") == "runtests-no-pygam":
            return

        # otherwise should get a list of parameter sets
        assert isinstance(params, list)
        assert len(params) > 0

        # each param set should create a valid instance
        for param_set in params:
            adapter = PyGAMAdapter(**param_set)
            assert isinstance(adapter, PyGAMAdapter)
