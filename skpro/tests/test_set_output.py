import pytest

from skpro.utils.set_output import check_output_config  # SUPPORTED_OUTPUTS,
from skpro.utils.validation._dependencies import _check_soft_dependencies

# TODO - write functions that ensures that the column values from the single level
# columsn frame matches the multi-index columsn frame


@pytest.fixture
def estimator():
    from sklearn.linear_model import LinearRegression

    from skpro.regression.residual import ResidualDouble

    # refactor to use ResidualDouble with Linear Regression
    _estimator = ResidualDouble(LinearRegression())
    return _estimator


def test_check_transform_config_happy(estimator):
    # check to make sure that regression estimators have the transform config
    # with default value "default"
    assert estimator.get_config()["transform"] == "default"

    estimator.set_output(transform="pandas")
    assert estimator.get_config()["transform"] == "pandas"
    valid, dense_config = check_output_config(estimator)
    assert valid
    assert dense_config["dense"] == ("pd_DataFrame_Table", "Table")

    if _check_soft_dependencies(["polars", "pyarrow"], severity="none"):
        estimator.set_output(transform="polars")
        assert estimator.get_config()["transform"] == "polars"
        valid, dense_config = check_output_config(estimator)
        assert valid
        assert dense_config["dense"] == ("polars_eager_table", "Table")


def test_check_transform_config_negative(estimator):
    estimator.set_output(transform="foo")
    with pytest.raises(
        ValueError,
        # match=f"set_output container must be in {SUPPORTED_OUTPUTS}, found foo.",
    ):
        check_output_config(estimator)


def test_check_transform_config_none(estimator):
    valid, dense = check_output_config(estimator)
    assert not valid
    assert dense == {}
