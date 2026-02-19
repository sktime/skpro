.. _getting_started:

===========
Get Started
===========

The following information is designed to get users up and running with
``skpro`` quickly. For more detailed information, see the links in each
of the subsections.

Installation
============

``skpro`` currently supports:

* environments with python version 3.8, 3.9, 3.10, 3.11, or 3.12.
* operating systems Mac OS X, Unix-like OS, Windows 8.1 and higher
* installation via ``PyPi`` or ``conda``

Please see the :ref:`installation <full_install>` guide for step-by-step instructions on the package installation.

Quick Start
===========

Here's a quick example to get you started with ``skpro`` for probabilistic regression:

.. code-block:: python

    # Import necessary libraries
    from sklearn.datasets import load_diabetes
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    
    from skpro.regression.residual import ResidualDouble
    from skpro.metrics import CRPS

    # Load and prepare data
    X, y = load_diabetes(return_X_y=True, as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a probabilistic regressor
    # Using RandomForest for mean prediction and residual modeling for uncertainty
    reg_mean = RandomForestRegressor(n_estimators=100, random_state=42)
    reg_proba = ResidualDouble(reg_mean)

    # Fit the model
    reg_proba.fit(X_train, y_train)

    # Make probabilistic predictions
    y_pred_proba = reg_proba.predict_proba(X_test)  # Full distribution
    y_pred_interval = reg_proba.predict_interval(X_test, coverage=0.9)  # 90% interval
    y_pred_quantiles = reg_proba.predict_quantiles(X_test, alpha=[0.05, 0.5, 0.95])  # Quantiles

    # Evaluate predictions using CRPS (Continuous Ranked Probability Score)
    crps = CRPS()
    score = crps(y_test, y_pred_proba)
    print(f"CRPS score: {score:.3f}")

    # Display example predictions
    print("\\nExample probabilistic predictions:")
    print(f"Prediction intervals (90%):\\n{y_pred_interval.head()}")
    print(f"\\nQuantiles (5%, 50%, 95%):\\n{y_pred_quantiles.head()}")

This example demonstrates:

* **Data preparation** using sklearn datasets
* **Probabilistic modeling** with ``ResidualDouble`` to add uncertainty estimates
* **Multiple prediction types**: distributions, intervals, and quantiles
* **Model evaluation** using probabilistic metrics like CRPS

Key Concepts
============

``skpro`` extends the familiar scikit-learn interface with probabilistic capabilities:

* **``fit(X, y)``** - Train the model on features ``X`` and target ``y``
* **``predict_proba(X)``** - Return full predictive distributions
* **``predict_interval(X, coverage)``** - Return prediction intervals
* **``predict_quantiles(X, alpha)``** - Return quantile predictions
* **``predict_var(X)``** - Return predictive variance

The output formats are pandas-compatible, making it easy to integrate with existing data science workflows.

Next Steps
==========

* :ref:`Tutorials <tutorials>` - Comprehensive notebooks for learning
* :ref:`User Guide <user_guide>` - Detailed usage patterns
* :ref:`API Reference <api_reference>` - Complete documentation
* :ref:`Examples <examples>` - More advanced use cases

.. _scikit-learn: https://scikit-learn.org/stable/index.html
