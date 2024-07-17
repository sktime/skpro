.. _set_output:

==========
skpro ``set_output`` API for Regression Models
==========

The following example will demonstrate how to use the ``set_output`` API
to configure the output container of your probabilistic predictions. Currently
for regression estimators, the following predict functions are supported:
``predict_quantile``, ``predict_interval``, ``predict_var``, and ``predict``.

Available ``mtypes`` include ``pd.DataFrame`` from the ``pandas`` library
and ``pl.DataFrame`` from the ``polars`` library.

We first load an sklearn dataset and import an skpro regression estimator.

.. code-block :: python

    #import our dataset
    from sklearn.datasets import load_diabetes
    from sklearn.model_selection import train_test_split

    #import our estimator
    from sklearn.linear_model import LinearRegression
    from skpro.regression.residual import ResidualDouble

    X, y = load_diabetes(return_X_y=True, as_frame=True)
    X = X.iloc[:75]
    y = y.iloc[:75]
    y = pd.DataFrame(y)
    X_train, X_test, y_train, _ = train_test_split(
    X, y, test_size=0.33, random_state=42
    )

    estimator = ResidualDouble(LinearRegression())

Next, we will call the ``set_output`` method built into the estimator.

.. code-block :: python

    estimator.set_output(transform = "polars")

After we fit the model, we can then call the ``predict`` function and the
output will automatically be converted into a polars DataFrame

.. code-block :: python

    estimator.fit(X_train, y_train)

    estimator.predict(X_test)

*output polars dataframe here*
