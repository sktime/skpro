# LEGACY MODULE - TODO: remove or refactor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from skpro.metrics import linearized_log_loss, log_loss
from skpro.parametric import ParametricEstimator
from skpro.parametric.estimators import Constant
from skpro.workflow import Model
from skpro.workflow.cross_validation import (
    CrossValidationController,
    CrossValidationView,
)
from skpro.workflow.manager import DataManager
from skpro.workflow.table import IdModifier, RankModifier, SortModifier, Table
from skpro.workflow.utils import InfoController, InfoView

# Load the dataset
data = DataManager("boston")

tbl = Table()

# Adding controllers displayed as columns
tbl.add(InfoController(), InfoView())

for loss_func in [linearized_log_loss, log_loss]:
    tbl.add(
        controller=CrossValidationController(data, loss_func=loss_func),
        view=CrossValidationView(),
    )

# Rank results
tbl.modify(RankModifier())
# Sort by score in the last column, i.e. log_loss
tbl.modify(SortModifier(key=lambda x: x[-1]["data"]["score"]))
# Use ID modifier to display model numbers
tbl.modify(IdModifier())

# Compose the models displayed as rows
models = []
for point_estimator in [
    RandomForestRegressor(),
    LinearRegression(),
    Constant("mean(y)"),
]:
    for std_estimator in [Constant("std(y)"), Constant(42)]:
        model = ParametricEstimator(point=point_estimator, std=std_estimator)
        models.append(Model(model))

tbl.print(models)
