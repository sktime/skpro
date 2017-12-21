Meta-modelling API
******************

Meta-modelling API concepts are required for a modular implementation of higher-order modelling features such as hyper-parameter optimization or ensemble methods. Following the principle of composition the meta-algorithms are implemented as modular compositions of the simpler algorithms. Technically, the meta-strategies are realised by meta-estimators that are estimator-like objects which perform certain methods with a given estimators. They hence take estimator type objects as some of their initializing inputs, and when initialized exhibit the fit-predict logic that implements the meta-algorithm when instantiated on the wrapped estimators.

Hyperparamter optimization
--------------------------

The optimization of model hyperparameter, for instance, can be implemented using scikit's grid or random search meta-estimators, for example:

.. literalinclude:: ../examples/parametric/hyperparameters.py
    :language: python

Read the `scikit documentation <http://scikit-learn.org/stable/modules/grid_search.html>`_ for more information.

Pipelines
---------

Probabilistic estimators work well with scikit-learn's ``Pipeline`` meta-estimator that allows to combine multiple estimators into one. Read the `pipeline documentation <http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html>`_ to learn more.


Ensemble methods
----------------

The framework provides experimental support for ensemble methods. Currently, this includes bagging in a regression setting which is implemented by the ``BaggingRegressor`` estimator in the ensemble module. The meta-estimator fits base regressors (i.e. probabilistic estimators) on random subsets of the original dataset and then aggregates their individual predictions in a distribution interface to form a final prediction. The implementation is based on scikit's meta-estimator of the same name but introduces support for the probabilistic setting.

The following example demonstrates the use of the bagging procedure:

.. literalinclude:: ../examples/parametric/bagging.py
    :language: python

To learn more, you may also read `scikit's documentation of the BaggingRegressor <http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingRegressor.html>`_.