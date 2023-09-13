.. _utils_ref:

Utility functions
=================

``skpro`` has a number of modules dedicated to utilities:

* :mod:`skpro.datatypes`, which contains utilities for data format checks and conversion.
* :mod:`skpro.registry`, which contains utilities for estimator and tag search
* :mod:`skpro.utils`, which contains generic utility functions.


Data Format Checking and Conversion
-----------------------------------

:mod:`skpro.datatypes`

.. automodule:: skpro.datatypes
    :no-members:
    :no-inherited-members:

.. currentmodule:: skpro.datatypes

.. autosummary::
    :toctree: auto_generated/
    :template: function.rst

    convert_to
    convert
    check_raise
    check_is_mtype
    check_is_scitype
    mtype
    scitype
    mtype_to_scitype
    scitype_to_mtype


Estimator Search and Retrieval, Estimator Tags
----------------------------------------------

:mod:`skpro.registry`

.. automodule:: skpro.registry
    :no-members:
    :no-inherited-members:

.. currentmodule:: skpro.registry

.. autosummary::
    :toctree: auto_generated/
    :template: function.rst

    all_objects
    all_tags
    check_tag_is_valid


Plotting
--------

:mod:`skpro.utils.plotting`

.. automodule:: skpro.utils.plotting
    :no-members:
    :no-inherited-members:

.. currentmodule:: skpro.utils.plotting

.. autosummary::
    :toctree: auto_generated/
    :template: function.rst

    plot_crossplot_interval
    plot_crossplot_std
    plot_crossplot_loss


Estimator Validity Checking
---------------------------

:mod:`skpro.utils.estimator_checks`

.. automodule:: skpro.utils.estimator_checks
    :no-members:
    :no-inherited-members:

.. currentmodule:: skpro.utils.estimator_checks

.. autosummary::
    :toctree: auto_generated/
    :template: function.rst

    check_estimator
