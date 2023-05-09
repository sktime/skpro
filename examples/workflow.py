# -*- coding: utf-8 -*-
from sklearn.linear_model import LinearRegression

from skpro.baselines import DensityBaseline
from skpro.metrics import log_loss
from skpro.parametric import ParametricEstimator
from skpro.parametric.estimators import Constant
from skpro.workflow import Model
from skpro.workflow.cross_validation import grid_optimizer
from skpro.workflow.manager import DataManager
from skpro.workflow.table import Table

tbl = Table()

# Loads and represents the data
data = DataManager("boston")

# Adds a model information column
tbl.info()
# Defines the cross validation using the log_loss metric and grid hyperparameter search
tbl.cv(data, log_loss, tune=True, optimizer=grid_optimizer(n_jobs=-1, verbose=0))

# Run the models against the workflow and print the results
tbl.print(
    [
        # Baseline ...
        Model(DensityBaseline()),
        # ... and parametric composite model
        Model(
            ParametricEstimator(LinearRegression(), Constant("std(y)")),
            # ... which hyperparameter shall be optimized
            tuning={"point__normalize": [True, False]},
        ),
    ]
)
