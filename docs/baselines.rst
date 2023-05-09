Baseline strategies
*******************

skpro offers simple baseline strategy strategies for model validation.

DensityBaseline
---------------

The ``DensityBaseline`` strategy wraps scikit-learn's KernelDensity estimation to predict a density using the training labels.

The following example illustrates the baseline usage on Bosting housing data:

.. literalinclude:: ../examples/simple.py
    :language: python

>>> Loss: 3.444260+-0.062277

Please refer to the :doc:`module documentation <api/modules>` to learn more.
