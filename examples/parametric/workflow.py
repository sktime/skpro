from skpro.workflow.table import Table, IdModifier, SortModifier
from skpro.workflow.cross_validation import CrossValidationController, CrossValidationView
from skpro.metrics import log_loss, gneiting_loss
from skpro.workflow import Model
from skpro.workflow.utils import InfoView, InfoController
from skpro.workflow.manager import DataManager
from sklearn.datasets import load_boston
from skpro.parametric import ParamtericEstimator
from skpro.parametric.estimators import Constant
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

# Load the dataset
X, y = load_boston(return_X_y=True)
data = DataManager(X, y, name='Boston')

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
for point_estimator in [RandomForestRegressor(), LinearRegression()]:
    for std_estimator in [Constant('mean(y)'), Constant(42)]:
        model = ParamtericEstimator(point=point_estimator, std=std_estimator)
        models.append(Model(model))

tbl.print(models)