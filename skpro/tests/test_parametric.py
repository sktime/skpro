# -*- coding: utf-8 -*-
import numpy as np
import pytest
from scipy.stats import norm
from sklearn.linear_model import LinearRegression

import skpro.tests.utils as utils
from skpro.metrics.metrics import linearized_log_loss
from skpro.regression.parametric.parametric import ParametricEstimator
from skpro.regression.parametric.residuals import ResidualEstimator
from skpro.workflow.manager import DataManager


def test_baseline():
    data = DataManager("boston")

    model = ParametricEstimator()
    y_pred = model.fit(data.X_train, data.y_train).predict(data.X_test)

    mu = np.mean(data.y_train)
    sigma = np.std(data.y_train)

    # is the dummy prediction working?
    assert (y_pred.point() == np.ones((len(data.X_test))) * mu).all()
    assert (y_pred.std() == np.ones((len(data.X_test))) * sigma).all()

    # does subsetting work?
    assert len(y_pred[1:3].point()) == 2
    assert len(y_pred[1:3].lp2()) == 2

    # pdf, cdf?
    x = np.random.randint(0, 10)
    i = np.random.randint(0, len(data.X_test) - 1)

    assert y_pred[i].pdf(x) == norm.pdf(x, mu, sigma)
    assert y_pred[i].cdf(x) == norm.cdf(x, mu, sigma)


def test_simple_model():
    data = DataManager("boston")

    model = ParametricEstimator(LinearRegression(), LinearRegression())
    y_pred = model.fit(data.X_train, data.y_train).predict(data.X_test)

    utils.assert_close_prediction(y_pred.point(), data.y_test, within=0.5)


@pytest.mark.skip(reason="loss assert fails sporadically")
def test_residual_prediction():
    data = DataManager("boston")

    baseline_model = ParametricEstimator(LinearRegression())
    model = ParametricEstimator(
        point=LinearRegression(), std=ResidualEstimator(LinearRegression())
    )

    baseline = baseline_model.fit(data.X_train, data.y_train).predict(data.X_test)
    y_pred = model.fit(data.X_train, data.y_train).predict(data.X_test)

    baseline_loss = linearized_log_loss(data.y_test, baseline)
    y_pred_loss = linearized_log_loss(data.y_test, y_pred)

    assert baseline_loss > y_pred_loss
