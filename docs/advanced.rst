Extending and advanced usage
****************************

skpro can be extended using custom models.

Developing custom models
------------------------

skpro can be extended to implement own models. All probabilistic models have to implement the API of the abstract base class ``skpro.base.ProbabilisticEstimator``. The example below illustrates a possible implementation of a random guess model that predicts normal distributions with random mean and variance.

.. literalinclude:: ../examples/custom_model.py
    :language: python


Integrating vendor models
-------------------------

Please refer to the ``skpro.base`` module documentation to explore how vendor models, e.g. of the Bayesian type, can be integrated into skpro.