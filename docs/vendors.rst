Vendors integrations
********************

A major motivation of the skpro project is a unified, domain-agnostic and approachable model assessment workflow. While many different packages solve the tasks of probabilistic modelling within the frequentist and Bayesian domain, it is hard to compare the models across the different packages in a consistent, meaningful and convenient way.

Therefore, skpro provides integrations of existing prediction algorithms from other frameworks to make them accessible to a fair and convenient model comparison workflow.

Currently, Bayesian methods are integrated via the predictive posterior samples they produce. Various adapter allow to transform these posterior samples into skpro's unified distribution interface that offers easy access to essential properties of the predicted distributions.


PyMC3
-----

The following example of a Bayesian Linear Regression demonstrates the PyMC3 integration. Crucially, the model definition method defines the shared ``y_pred`` variable that represent the employed model.

.. literalinclude:: ../examples/vendors/pymc.py
    :language: python

Please refer to PyMC3's own `project documentation <http://docs.pymc.io/index.html>`_ to learn more about available PyMCs model definitions.

Integrate other models
----------------------

skpro's base classes provide scaffold to quickly integrate arbitrary models of the Bayesian or frequentist type. Please read the documentation on :doc:`extension and model integration <extending>` to learn more.
