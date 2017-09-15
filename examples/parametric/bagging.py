from sklearn.datasets import load_boston
from sklearn.tree import DecisionTreeRegressor

from skpro.workflow.table import Table, IdModifier, SortModifier
from skpro.workflow.cross_validation import CrossValidationController, CrossValidationView
from skpro.metrics import log_loss, gneiting_loss
from skpro.workflow import Model
from skpro.ensemble import BaggingRegressor
from skpro.workflow.utils import InfoView, InfoController
from skpro.workflow.manager import DataManager
from skpro.parametric import ParamtericEstimator
from skpro.parametric.estimators import Constant

# Load the dataset
X, y = load_boston(return_X_y=True)
data = DataManager(X, y, name='Boston')

if True:  # DEBUG
    import numpy as np
    from sklearn.model_selection import cross_val_score

    def evaluate(model):
        model.fit(data.X_train, data.y_train)
        y_pred = model.predict(data.X_test)

        mse = np.sum((np.abs(y_pred - data.y_test))) / len(y_pred)
        # print('MSE: ', mse)

        scores = cross_val_score(model, data.X, data.y, cv=3)

        print('CV: %f+-%f' % (np.mean(scores), np.std(scores) / np.sqrt(len(scores))))


    print('Without bagging: ')
    model = DecisionTreeRegressor()
    evaluate(model)

    print('With bagging:')
    from sklearn.ensemble import BaggingRegressor as SklearnBaggingRegressor
    from sklearn.base import clone
    bagged_model = SklearnBaggingRegressor(clone(model), n_estimators=50, bootstrap=True, oob_score=True) #, max_samples=0.5, max_features=0.5, bootstrap=False)

    evaluate(model)

    exit()

tbl = Table()

# Adding controllers displayed as columns
tbl.add(InfoController(), InfoView())

for loss_func in [gneiting_loss, log_loss]:
    tbl.add(
        controller=CrossValidationController(data, loss_func=loss_func),
        view=CrossValidationView()
    )

# Sort by score in the last column, i.e. log_loss
tbl.modify(SortModifier(key=lambda x: x[-1]['data']['score']))
# Use ID modifier to display model numbers
tbl.modify(IdModifier())

# Compose the models displayed as rows
models = []

for point_estimator in [RandomForestRegressor()]:#, LinearRegression()]:
    for std_estimator in [RandomForestRegressor(), Constant('mean(y)')]:
        model = ParamtericEstimator(point=point_estimator, std=std_estimator)
        models.append(Model(model))
        models.append(Model(
            #BaggingRegressor(model, bootstrap=False, n_estimators=1)
            BaggingRegressor(model, max_samples=0.1, max_features=0.1, bootstrap=False, n_estimators=100)
        ))


tbl.print(models)