User Guide
**********

This guide will give a short overview of the basic functions of the ``skpro`` package.
For further details you may explore the `API documentation <api/modules.html>`_.

.. note:: skpro uses many of scikit-learn's building principles and conventions. If you aren't familiar with the scikit-learn package you may read its  `basic tutorial <http://scikit-learn.org/stable/tutorial/basic/tutorial.html>`_.

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

Probabilistic Estimators
------------------------

How can probabilistic prediction models be learned, specifically  strategies that predict probability distributions? skpro offers a variety of strategies, specifically:

* :doc:`Baseline strategies <baselines>`, for instance a kernel density estimation on the labels
* :doc:`Parametric estimation <parametric>`, that estimates parameters of the predicted distributions
* :doc:`integrations with other vendor packages <vendors>` such as ``PyMC3``

For a full documentation you may read the respective :doc:`module documention <api/modules>` but to understand the principles of the probabilistic estimators we recommend starting with the :doc:`parametric estimation <parametric>`.

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

Like in sklearn, probabilistic estimators can be bagged using the meta-estimator ``BaggingRegressor``, for instance:

.. literalinclude:: ../examples/parametric/bagging.py
    :language: python

Check out the :doc:`skpro\.ensemble <api/modules>` module to learn more.
