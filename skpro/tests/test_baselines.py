import numpy as np
import utils

if False:
    from skpro.workflow.manager import DataManager
    from skpro.baselines import DensityBaseline
    from skpro.metrics import linearized_log_loss


def test_density_baseline():
    data = DataManager('boston')

    model = DensityBaseline()
    y_pred = model.fit(data.X_train, data.y_train).predict(data.X_test)

    # median prediction working?
    mu = np.mean(data.y_train)
    sigma = np.std(data.y_train)
    assert (y_pred.point() == np.ones((len(data.X_test))) * mu).all()
    assert (y_pred.std() == np.ones((len(data.X_test))) * sigma).all()

    # pdf, cdf working?
    x = np.random.randint(0, 10)
    i = np.random.randint(0, len(data.X_test) - 1)
    assert isinstance(y_pred[i].pdf(x), float)
    assert isinstance(y_pred[i].cdf(x), float)

    # mean prediction is useful?
    utils.assert_close_prediction(y_pred.point(), data.y_test, within=0.75)

    # loss calculation working?
    assert isinstance(linearized_log_loss(data.y_test, y_pred), float)