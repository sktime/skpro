User Guide
**********

This guide will give a short overview of the basic functions in ``skpro`` package.
For further details you may explore the `API documentation <api/modules.html>`_.

.. note:: skpro uses many of scikit-learn's building principles and conventions. If you aren't familiar with the scikit-learn package you might want to explore its `project documentation <http://scikit-learn.org/stable/tutorial/basic/tutorial.html>`_.

Overview
--------

The figure below gives an overview about central elements and concepts of skpro and how it extends the scikit-learn toolbox. To understand skpro, it is firstly helpful to quickly review scikit-learn’s classical prediction workflow, particularly its seminal ``Estimator`` object. In scikit-learn, an ``Estimator`` object represents a certain prediction strategy (e.g. Linear regression), that can be fitted using the ``fit(X, y)`` function. The fitted estimator can then be used to obtain the predictions on new data using the ``predict(X)`` method. Finally, the predicted values can be compared with the actual targets using one of the available classical loss functions.

.. figure:: _static/overview.png
   :width: 90%

   Overview of the skpro prediction framework and how it extends the *scikit-learn*
   package.

skpro seeks to replicate this general pattern and introduces the ``ProbabilisticEstimator`` class that encapsulates the
probabilistic prediction models. Like the ``Estimator`` class it offers a fit and predict method but returns a probability distribution as prediction (``Distribution`` class). The returned distribution objects provide methods to obtain relevant distribution properties, for example the distribution's probability density function (``y_pred.pdf(x)``).

The predictions obtained from skpro's estimators are hence of a genuine probabilistic kind that represent predicted probability distributions for each data point. For example, if predictions for a vector ``X`` of length k are obtained, the returned ``y_pred`` object represents k predicted distributions. ``y_pred[i]`` therefore provides access to the point prediction (e.g. mean) of the i-th distribution, ``y_pred.std()`` will return a vector of length k that contains the standard deviations of the predicted distribution, and so forth. In many cases, such as plotting and error calculation, the distributions objects can thus be handled like scikit's commonly returned prediction vectors.

Metrics
-------

To evaluate the accuracy of the predicted distributions, skpro provides probabilistic loss metrics. To calculate the loss between prediction and the true target values, you can choose from a variety of available functions in the ``skpro.metrics`` module. In the default setting, all loss functions return the averaged loss of the sample. If you'd like to obtain the point-wise loss instead, set ``sample=False``. You can also obtain the confidence interval of the loss by setting ``return_std`` to ``True``. For a detailed documentation of the metrics package read the :doc:`API documentation <api/modules>`.

Meta-estimators
---------------

Meta-estimators are estimator-like objects that can be used to perform methods on given estimators. skpro's probabilistic estimators are widely compatible with the meta-estimators of scikit-learn or derived from the scikit library.

Hyperparamter optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~

The optimization of model hyperparameter, for instance, can be implemented using scikit's grid or random search meta-estimators, for example:

.. literalinclude:: ../examples/parametric/hyperparameters.py
    :language: python

Read the `scikit documentation <http://scikit-learn.org/stable/modules/grid_search.html>`_ for more information.

Pipelines
~~~~~~~~~

Probabilistic estimators work well with scikit-learn's ``Pipeline`` meta-estimator that allows to combine multiple estimators into one. Read the `Pipeline documentation <http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html>`_ to learn more.


Bagging
~~~~~~~

A bagging meta-estimator is currently in development. Check out the :doc:`skpro\.ensemble <api/modules>` module for a current preview.

Probabilistic Estimators
------------------------

How can probabilistic prediction models be learned, specifically
strategies that predict probability distributions? skpro offers a variety of  strategies that are documented in the :doc:`module documention <api/modules>`. To understand the principles of the probabilistic estimators, let's look at the perhaps simplest strategy of *parametric estimation*.

Parametric estimation
~~~~~~~~~~~~~~~~~~~~~

It uses classical estimators to predict the defining parameters of continuous distributions. The idea is that the prediction of a normal distribution can be brought down to a prediction of its defining parameters mean :math:`\mu` and standard deviation :math:`\sigma`. Likewise we can predict a Laplacian distribution by predicting its defining parameter location :math:`\mu` and scale :math:`b`. More general, we seek to obtain *point estimates* and *variance predictions* that are plugged into the definition of the respective predicted distribution. The point estimates can be understood as equivalent to the classical predictions in non-probabilistic settings, for example an estimated housing price. While these estimates are definite in the classical setting, the probabilistic point estimates can be interpreted as the expected value of the predicted distribution (e.g. as :math:`\mu` in the case of a Normal distribution). The variance predictions, on the other hand, estimate the uncertainty of the point prediction and account for the expected fluctuation or deviation of the probabilistic prediction (e.g. :math:`\sigma` of the Normal distribution). The variance estimates can, for instance, account for the reliability of the price forecast and have no equivalent in the classical setting. Given the estimated point and variance parameters, various distribution types (e.g. Normal, Laplace etc.) can take them to form the predicted distribution output. Which type is selected can be decided based on the data which is being modelled, for instance, by choosing the distribution type that minimizes the probabilistic loss for provided point and variance estimate. In this way, suitable probabilistic predictions, that is predicted distributions, can be obtained.

The prediction-via-parameter strategy has the obvious advantage that existing classic learning algorithms can be reused in the probabilistic setting. In fact, in this paradigm the same algorithm that is used to predict a housing price can be employed to obtain the point prediction which represents the mean of the predicted price distribution for this house. It is, however, an open question how the variance predictions that are understood to estimate the probabilistic uncertainty of these point predictions can be obtained. An intuitive idea is to use the residuals of the point estimations, since they represent the magnitude of error committed during point prediction and hence suggest how correct or certain these predictions actually were. In the supervised setting, where the correct training labels :math:`y_i` are provided, we can easily obtain the absolute training residuals :math:`\varepsilon_{\text{train}, i} = |\hat{y}_i - y_i`\ \| of the point predictions :math:`\hat{y}_i`. Since training and test data are assumed to be i.i.d. sampled from the same generative distribution, we can estimate the test residuals based on the training residuals. More precisely, we fit a residual model using the training features and calculated training residuals (:math:`x_i`, :math:`\varepsilon_{\text{train}, i}`). Using the trained residual model, we are then able to estimate the test residuals :math:`\hat{\varepsilon}_{\text{test}, j}` for given test features :math:`x_j^*`. Note that the obtained residuals are the residuals of the distributional parameter estimation and not of the overall distribution estimate. It is, however, reasonable to assume that higher residuals in the prediction of the distribution’s parameter imply higher residuals of the overall distributional prediction. We thus regard :math:`\hat{\varepsilon}_{\text{test}, j}` as a prediction of the distribution’s deviation parameter (e.g. :math:`\sigma` in :math:`\mathcal{N}(\mu, \sigma)`), that is the variance prediction of the overall strategy. Note that we calculated the absolute residuals to account for the non-negativity of the variance. Alternatively, the strategy can be modified by fitting the squared or logarithmic training residuals to the residual model and back transforming the estimated test residuals using the square root and exponential function respectively.

The ``skpro.parametric.ParametricEstimator`` object implements such a "parametric" strategy. Currently, two-parametric continuous distributions are supported (e.g. Normal and Laplace distribution) where point and variance prediction are used as mean and variance parameter or location and scale parameter respectively. Specifically, the parametric estimator takes the arguments *point* for the point estimator, *std* for the variance estimator and *shape* to define the assumed distribution form (e.g. Normal or Laplace). During fitting the estimator automatically fits the provided point and variance estimators; accordingly, on predicting, it retrieves their estimations to compose the overall predicted distribution interface of the specified shape. The parametric estimator object also supports combined estimation in which the same estimator instance is used to obtain both point and variance prediction. The combined estimator has to be passed to the optional ``point_std`` parameter while the *point* and *std* can be used to specify how point and variance estimation should be retrieved from it. Hence, the parametric estimator can be considered a function that maps the distribution interface onto the actual learning algorithms of the provided estimators.

The example below shows the definition of a parametric model that uses a RandomForestRegressor as point estimator and the feature mean as variance predictor.

.. literalinclude:: ../examples/parametric/simple.py
    :language: python
    :emphasize-lines: 9-14
    :lines: 1-23

Estimators
^^^^^^^^^^

Since the parametric distribution estimation only relies on estimators that implement the actual prediction mechanisms, it is generally possible to employ any of scikit-learn’s classical estimators. In addition to the estimators in the scikit-learn library, skpro implements own estimator objects, for instance, a constant estimator which was used in the previous example to define a constant variance prediction.

**Constant estimator** The constant estimator simply predicts a constant value which is pre-defined or calculated from the training data. The estimator is particularly useful for control strategies, e.g. a baseline that omits the training data features and makes an uninformed guess by calculating the constant mean of the dependent variable.

**Residual estimator** The residual estimator implements the mentioned residual prediction strategy in which training residuals are used to fit another residual estimator. To this end, the estimator takes three arguments. First, a reference to the estimator which residuals should be estimated (that is normally the point predictor). Second, the model that should be used for the residual prediction (e.g. another estimator). Third, the method of residual calculation (e.g. squared or absolute error).

Example
^^^^^^^

The following code example illustrates the resulting overall syntax that defines a probabilistic baseline model *baseline* using the parametric estimator.

.. code:: python

    # Initiate model
    baseline = ParametricEstimator(
        shape='norm',       # Distribution type
        point=Constant(42),        # Point estimator
            std=ResidualEstimator( # Variance estimator
                'point',             # Base estimator
                Constant('mean(y)'), # Residual estimator
                'abs_error'          # Calculation method
            )
    )
    # Train the model on training data
    baseline.fit(X_train, y_train)
    # Obtain the probabilistic predictions for test data
    y_pred = baseline.predict(X_test)

The resulting prediction *y\_pred* is a normal distribution with mean equals :math:`42` and the standard deviation is mean of the absolute training residuals. While the syntax of this definition is identical with the definition syntax of scikit-learn, the model crucially returns probabilistic predictions, that is probability distributions.

Developing own models
---------------------

skpro can be extended to implement own models. All probabilistic models have to implement the API of the abstract base class ``skpro.base.ProbabilisticEstimator``. The example below illustrates a possible implementation of a random guess model that predicts normal distributions with random mean and variance.

.. literalinclude:: ../examples/custom_model.py
    :language: python