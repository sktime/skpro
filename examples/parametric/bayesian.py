from sklearn.ensemble import RandomForestRegressor

from skpro.bayesian.pymc3.estimators import LinearRegression
from skpro.metrics import log_loss
from skpro.parametric import ParamtericEstimator
from skpro.parametric.estimators import Constant
from skpro.workflow import Model
from skpro.workflow.cross_validation import CrossValidationController, CrossValidationView
from skpro.workflow.manager import DataManager
from skpro.workflow.table import Table, IdModifier, SortModifier
from skpro.workflow.utils import InfoView, InfoController

data = DataManager('boston')

tbl = Table()

# Adding controllers displayed as columns
tbl.add(InfoController(), InfoView())

tbl.add(
    controller=CrossValidationController(data, loss_func=log_loss),
    view=CrossValidationView()
)

# Sort by score in the last column, i.e. log_loss
tbl.modify(SortModifier(key=lambda x: x[-1]['data']['score']))
# Use ID modifier to display model numbers
tbl.modify(IdModifier())

# Compose the models displayed as rows
models = [
    Model(ParamtericEstimator(point=RandomForestRegressor(), std=Constant('mean(y)'))),
    Model(ParamtericEstimator(point_std=LinearRegression()))
]

tbl.print(models)