.. _tags_reference:

===============
Tags Reference
===============

Tags are metadata attributes that describe properties and capabilities of estimators and other objects in ``skpro``.

They are used for:

* Filtering and searching estimators
* Specifying test requirements and conditions
* Documenting estimator capabilities
* Enabling automated estimator selection and composition

Common Tags
===========

**object_type**
    Type of object, e.g., "regressor_proba", "distribution", "metric"

**estimator_type**
    Type of estimator, e.g., "regressor", "transformer", "distribution"

**capability:survival**
    Whether the estimator supports survival/time-to-event prediction with censoring.
    Value: ``True`` or ``False``

**handles_missing_data**
    Whether the estimator can handle missing values in input features.
    Value: ``True`` or ``False``

**requires_y**
    Whether the estimator requires a target variable for fitting.
    Value: ``True`` or ``False``

**handles_multioutput**
    Whether the estimator can handle multiple target variables (multioutput regression).
    Value: ``True`` or ``False``

Using Tags in Code
===================

Tags can be accessed from estimator classes using the tag registry:

.. code-block:: python

    from skpro.registry import all_objects, all_tags

    # Get all available tags
    tags = all_tags()

    # Get all regressors with survival capability
    survival_regressors = all_objects(
        object_types="regressor_proba",
        filter_tags={"capability:survival": True},
        as_dataframe=True
    )

    # Get all estimators that handle missing data
    missing_data_estimators = all_objects(
        filter_tags={"handles_missing_data": True},
        as_dataframe=True
    )

    # Get tag value for a specific estimator
    from skpro.regression import LinearRegressor
    survival_capable = LinearRegressor.get_tag("capability:survival")

For a complete list of available tags and their descriptions, see the
:mod:`skpro.registry` module documentation.
