.. _tags_ref:

Object and estimator tags
=========================

Every first-class object in ``skpro``
is tagged with a set of tags that describe its properties and capabilities,
or control its behavior.

Tags are key-value pairs, where the key is a string with the name of the tag.
The value of the tag can have arbitrary type, and describes a property, capability,
or controls behaviour of the object, depending on the tag.

This API reference lists all tags available in ``skpro``, and key utilities
for their usage.


Inspecting tags, retrieving by tags
------------------------------------

Tags can be inspected at runtime using the following utilities:

* to get the tags of an object, use the ``get_tags`` method.
  An object's tags can depend on its hyper-parameters.
* to get the tags of a class, use the ``get_class_tags`` method of the class.
  A class's tags are static and do not depend on its hyper-parameters.
  By default, class tags that may vary for instances take the most "capable" value,
  in the case of capabilities.
* to programmatically retrieve all tags available in ``skpro``
  or for a particular type of object, at runtime, use the ``registry.all_tags``
  utility.

.. currentmodule:: skpro.registry

.. autosummary::
    :toctree: auto_generated/
    :template: function.rst

    all_tags


.. _packaging_tags:

General tags, packaging
-----------------------

This section lists tags that are general and apply to all objects in ``skpro``.
These tags are typically used for typing, packaging and documentation purposes.

.. currentmodule:: skpro.registry._tags

.. autosummary::
    :toctree: auto_generated/
    :template: function.rst
    :nosignatures:

    object_type
    estimator_type
    reserved_params
    maintainers
    authors
    python_version
    python_dependencies
    python_dependencies_alias
    license_type


.. _regressor_tags:

Probabilistic regressor tags
-----------------------------

This section lists tags applying to probabilistic regressors
(``"regressor_proba"`` type).
These tags are used to describe capabilities, properties, and behavior of
probabilistic regressors.

.. currentmodule:: skpro.registry._tags

.. autosummary::
    :toctree: auto_generated/
    :template: function.rst
    :nosignatures:

    capability__survival
    capability__multioutput
    capability__missing
    capability__update
    X_inner_mtype
    y_inner_mtype
    C_inner_mtype


.. _distribution_tags:

Distribution tags
-----------------

This section lists tags applying to probability distributions
(``"distribution"`` type).
These tags are used to describe capabilities, properties, and behavior of
distributions.

.. currentmodule:: skpro.registry._tags

.. autosummary::
    :toctree: auto_generated/
    :template: function.rst
    :nosignatures:

    capabilities_approx
    capabilities_exact
    capabilities_undefined
    distr_measuretype
    distr_paramtype
    approx_mean_spl
    approx_var_spl
    approx_energy_spl
    approx_spl
    bisect_iter
    broadcast_params
    broadcast_init
    broadcast_inner


.. _metric_tags:

Metric tags
-----------

This section lists tags applying to probabilistic metrics (``"metric"`` type).

.. currentmodule:: skpro.registry._tags

.. autosummary::
    :toctree: auto_generated/
    :template: function.rst
    :nosignatures:

    scitype_y_pred
    lower_is_better


.. _meta_object_tags:

Meta-object tags
----------------

Tags relating to meta-object composition (pipelines, ensembles, etc.).

.. currentmodule:: skpro.registry._tags

.. autosummary::
    :toctree: auto_generated/
    :template: function.rst
    :nosignatures:

    named_object_parameters
    fitted_named_object_parameters


.. _testing_tags:

Testing and CI tags
-------------------

These tags control behaviour of objects in ``skpro`` continuous integration
tests.

They are primarily useful for developers managing CI behaviour of individual
objects.

.. currentmodule:: skpro.registry._tags

.. autosummary::
    :toctree: auto_generated/
    :template: function.rst
    :nosignatures:

    tests_libs
    tests_vm
    tests_skip_by_name
    tests_python_dependencies
