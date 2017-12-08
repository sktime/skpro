Integrating models
******************

skpro can be extended using custom models.

Developing custom models
------------------------

skpro can be extended to implement own models. All probabilistic models have to implement the API of the abstract base class ``skpro.base.ProbabilisticEstimator``. The example below illustrates a possible implementation of a random guess model that predicts normal distributions with random mean and variance.

.. literalinclude:: ../examples/custom_model.py
    :language: python


Integrating vendor models
-------------------------

To integrate existing models into the framework, the user can implement own subclasses of the probabilistic estimator. The API, however, also offers simplified model integration using the derived ``VendorEstimator`` object that takes a ``VendorInterface`` and a ``DensityAdapter``. The vendor interface must only define ``on_fit`` and ``on_predict`` events that are invoked automatically. The results of the fit-predict procedure are exposed as public variables of the interface. The adapter, on the other hand, then describes how distributional properties are generated from the interfaced vendor model, i.e. the VendorInterface’s public properties. Given a vendor interface and appropriate adapter, a vendor estimator can be used like any other probabilistic estimator of the framework.

Bayesian integration
~~~~~~~~~~~~~~~~~~~~

A notable example of the model integration API is the Bayesian case. To integrate a Bayesian model one can implement the ``BayesianVendorInterface`` and its ``samples`` method that is ought to return a predictive posterior sample. Combined with a ``skpro.density.DensityAdapter`` like the ``KernelDensityAdapter`` that transforms the sample into estimated densities, the Bayesian model can then be used as a probabilistic estimator.