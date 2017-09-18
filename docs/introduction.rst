Introduction
************

Probabilistic forecasting makes predictions in settings where even perfect prior knowledge does not allow for impeccable forecast and provides appropriate measures of the uncertainty that is associated with them. In a supervised context, probabilistic prediction problems have been tackled through various strategies in both the frequentist and the Bayesian domain.

``skpro`` presents a supervised domain-agnostic framework that allows for probabilistic modelling, namely the prediction of probability distributions for individual data points. The package is based on the popular machine learning library `scikit-learn`_ and is being developed open source under the direction of `Dr Franz Király`_ and is released under an :doc:`MIT license <license>`.

Features
^^^^^^^^

The package offers a variety of features and specifically allows for

- implementation of probabilistic prediction strategies in the supervised contexts
- comparison of frequentist and Baysian prediction methods
- strategy optimization through hyperparamter tuning and ensemble methods (e.g. bagging)
- workflow automation

.. note::
    We are currently in public beta.

A motivating example
^^^^^^^^^^^^^^^^^^^^

Let's have a look at a well-known, simple example to see what this is about: Boston Housing price prediction. The skpro specific lines are highlighted below.

.. literalinclude:: ../examples/parametric/simple.py
    :language: python
    :emphasize-lines: 9-14, 21-23
    :lines: 1-23

>>> Loss: 4.054001+-0.001248

If you are familiar with scikit-learn you will recognise that we define and train a model on the boston housing dataset and obtain the test prediction *y_pred*. Furthermore, we use a loss function to calculate the loss between the predicted points and the true values -- nothing unexpected there.

Crucially, however, the skpro model does not just return a list of numbers or point predictions here. Instead, *y_pred* is a probablistic prediction, i.e. it represents probability distributions for each individual data point.
We can, for instance, obtain the distribution's standard deviation or even its probability density function: ::

    deviation = y_pred.std()
    density = y_pred.pdf(x)

The skpro predictions are hence forecasts of a genuine probabilistic kind that are a primary interest for many real-world applications.

This is in a nutshell what skpro is about. Head to the :doc:`installation instructions <installation>` or continue to the comprehensive :doc:`user guide <user_guide>` to learn more.

.. _scikit-learn: http://scikit-learn.org/
.. _Dr Franz Király: https://www.ucl.ac.uk/statistics/people/franz-kiraly