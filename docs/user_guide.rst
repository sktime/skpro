User Guide
**********

This guide will give a short overview of the basic functions in skpro package.
For further details you may check the `reference guide <api/modules.html>`_ as well.

Overview
--------

Figure [fig:overview] gives an overview about central elements and
concepts of the implemented API and how it extends the scikit-learn
toolbox. To understand the probabilistic extension, it is firstly
helpful to quickly review scikit-learn’s classical prediction workflow,
particularly its seminal *Estimator* object. In scikit-learn, an
Estimator object represents a certain prediction strategy (e.g. Linear
regression), that can be fitted using the *fit(X, y)* function. The
fitted estimator can then be used to obtain the predictions on new data
using the *predict(X)* method. Finally, the predicted values can be
compared with the actual targets using one of the available classical
loss functions.

Our probabilistic extension of scikit-learn seeks to replicate this
general pattern, specifically the idea of encapsulating the
probabilistic prediction models in an estimator-like structure that
offers a fit and predict method. As apparent in the overview figure
[fig:overview], we introduce “ProbabilisticEstimator” objects to extend into the
probabilistic domain.

.. figure:: figures/overview.pdf
   :alt: Overview of the implemented *scikit-learn*-based probabilistic
   prediction framework.
   :width: 90.0%

   Overview of the implemented *scikit-learn*-based probabilistic
   prediction framework.


API structure
-------------

In the following, we give an overview of the implemented classes that
inherit from the previously discussed central base classes
*Distribution* and *Estimator*.

Distributions
~~~~~~~~~~~~~

Since in this study distributions are predicted through an estimation of
their defining parameters, we implemented a “parametric” distribution
object (cp. sec. [sec:distribution-via-parameter-prediction]).
Currently, two-parametric continuous distributions are supported (e.g.
Normal and Laplace distribution) where point and variance prediction are
used as mean and variance parameter or location and scale parameter
respectively. Specifically, the implemented *Parametric* object that
inherits from *Distribution* takes the arguments *point* for the point
estimator, *std* for the variance estimator and *shape* to define the
assumed distribution form (e.g. Normal or Laplace). During fitting
(*fit(X, y)*) the Parametric object automatically fits the provided
point and variance estimators; accordingly, on predicting
(*predict(X)*), it retrieves their estimations to compose the overall
predicted distribution interface of the specified shape. The Parametric
object also supports combined estimation in which the same estimator
instance is used to obtain both point and variance prediction. The
combined estimator has to be passed to the optional *point\_std*
parameter while the *point* and *std* can be used to specify how point
and variance estimation should be retrieved from it. Hence, the
Parametric object can be considered a function that maps the
distribution interface onto the actual learning algorithms of the
provided estimators (cp. fig. [fig:overview]).

Estimators
~~~~~~~~~~

Consequently, the distribution estimation relies on estimators that
implement the actual prediction mechanisms. Since we follow the
estimator API of scikit-learn, it is generally possible to employ any of
scikit-learn’s classical estimators. In addition to the estimators in
the scikit-learn library, we implemented the following estimator
objects.

Constant estimator (C)
^^^^^^^^^^^^^^^^^^^^^^

The most basic estimator predicts a constant value which is pre-defined
or calculated from the training data. The estimator is particularly
useful for control strategies, e.g. a baseline that omits the training
data features and makes an uninformed guess by calculating the constant
mean of the dependent variable.

Residual estimator (RE)
^^^^^^^^^^^^^^^^^^^^^^^

The estimator implements the residual prediction strategy in which
training residuals are used to fit another residual estimator
(cp. sec. [sec:residual-estimation]). To this end, the RE takes three
arguments. First, a reference to the estimator which residuals should be
estimated (that is normally the point predictor). Second, the model that
should be used for the residual prediction (e.g. another estimator).
Third, the method of residual calculation (e.g. squared or absolute
error).

Code example
~~~~~~~~~~~~

The following code example illustrates the resulting overall syntax that
defines a baseline model *baseline* using the parametric distribution
class *Parametric*:

.. code:: python

    # Initiate model
    baseline = Parametric(
        shape='norm',       # Distribution type
        point=C(42),        # Point estimator
            std=RE(             # Variance estimator
                'point',        # Base estimator
                C('mean(y)'),   # Residual estimator
                'abs_error'     # Calculation method
            )
    )
    # Train the model on training data
    baseline.fit(X_train, y_train)
    # Obtain the predictions for test data
    y_pred = baseline.predict(X_test)

The resulting prediction *y\_pred* is a normal distribution with mean
equals :math:`42` and the standard deviation is mean of the absolute
training residuals. Crucially, the syntax in this probabilistic model
definition is identical with the model definition syntax of
scikit-learn. We will later denote such a model as:

| :math:`\mathcal{N}`\ (p=C(42), s=RE(p, C(mean(y)), abs\_error))

and write more general:

| DistributionType(p=PointEstimator, s=VarianceEstimator))

Meta-estimators
---------------

Meta-estimators are estimator-like objects that perform certain methods
with a given Estimators
:raw-latex:`\autocite[for an extended discussion see][sec. 3.1]{buitinck_api_2013}`.
Perhaps most notably, is scikit-learn’s **meta-estimators for
hyper-parameter optimization** that optimizes the hyperparameters of a
given estimator (either through exhaustive grid search or randomized
parameter optimization, cp. sec. [sec:model-tuning]). Much effort in the
development of our API extension has been invested in achieving
compatibility with scikit-learn where possible. One benefit of these
efforts is the compatibility of *Distribution* objects with the existing
meta-estimators in scikit-learn. It is thus possible to tune
hyperparameters of a probabilistic prediction model (e.g. a parametric
distribution predictor) using the usual meta estimator of scikit-learn.
Accordingly, it is possible to use scikit-learn’s **pipelines
meta-estimator** to combine multiple estimation steps into a single
model. This allows one, for instance, to conveniently prepend
data-pre-processing for the actual prediction algorithm.

Metrics and visualisations
--------------------------

To evaluate the accuracy of the predicted distributions, the API
provides probabilistic loss metrics (cp. overview figure
[fig:overview]). Specifically, the log-loss and the Gneiting loss, as
described in section [sec:probabilistic-losses], were implemented. For
consistency, the signatures of the provided loss functions are unified
and correspond with the classical loss functions that are provided by
scikit-learn. Like the scikit-learn package, the metrics package
provides a helper function to transform a given loss function into a
score function, which is used, for instance, in cross-validation and
hyperparameter optimization.

To support the analysis, different plot methods are available that take
the predicted distribution and the corresponding ground truth to
visualise the performance, residuals, or Q-Q-comparison.

Workflow automation
-------------------

Unlike scikit-learn, which only provides a loose library of validation
components, we propose an object-oriented structure that standardizes
the prediction workflows. The objective is to support efficient model
management and fair model assessment in unified framework. After all,
the user should only be concerned with the definition and development of
models while leaving the tedious tasks of result aggregation to the
framework.

Model-view-controller structure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Our workflow framework is build up of three fundamental components:
model, controller, view. The model object contains the actual prediction
algorithm that was defined by the user (e.g. a distribution object). It
thus unifies and simplifies the management of learning algorithms. It
allows to store information and configuration for the algorithm it
contains, e.g. a name or a range of hyperparameters that should be
optimized. In future, it might support saving of trained models for
later use. Secondly, a controller represents an action or task that can
be done with a model to obtain certain information. A scoring
controller, for instance, might take a dataset and loss function and
return the loss score of the model on this dataset. The controller can
save the obtained data for later use. Finally, a view object takes what
a controller returns to present it to the user. A scoring view, for
example, could take a raw score value and format it in power mode. The
separation of controller and view level is advantageous since controller
tasks like the training of a model to obtain a score can be
computationally expensive. Thus, a reformation of an output should not
require the revaluation of the task. Moreover, if a view only displays a
part of the information it yet useful to store the full information the
controller returned.

Our framework currently implements one major controller, the **Cross
validation controller (CV)**, and multiple views to display scores and
model information. The CV controller encapsulates the cross-validation
procedure described in section [sec:cross-validation]. It takes a
dataset and loss function and returns the fold-losses as well as the
overall loss with confidence interval for a given model (cp. eq.
[eq:cv-model-performance]). If the model specifies a range of
hyperparameters for tuning, the controller automatically optimizes the
hyperparamters in a nested cross-validation procedure and additionally
returns the found best hyperparameters (cp. sec. [sec:model-tuning]).

The model-view-controller structure (MVC) encapsulates a fundamental
procedure in machine learning: perform a certain task with a certain
model and display the results. Thanks to its unified API, the MVC
building blocks can then be easily used for result aggregation and
comparison.

Result aggregation and comparison
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

At its current stage, the workflow framework support a simple way of
results aggregation and comparison, namely a results table. A table can
be easily defined by providing controller-view-pairs as columns and
models as rows. The framework will then evaluate the table cells by
running the controller task for the respective models and render the
results table using the specified views. Note that the evaluation of the
controller tasks and the process of rendering the table is decoupled. It
is therefore possible to access the “raw” table with all the information
each controller returned and then render the table with the reduced
information that is actually needed. Furthermore, the decoupling allows
for manipulation or enhancement of the raw data before rendering. The
raw table data can, for example, be sorted by the model performances.
Notably, the table supports so-called rank-sorting. Rank sorting is, for
instance, useful if models are compared on different datasets and ought
to be sorted by their overall performance. In this case, it is
unsuitable to simply average the dataset’s performance scores since the
value ranges might differ considerably between the different datasets.
Instead, it is useful to rank the performances on each dataset and then
average the model’s rank on each dataset to obtain the overall rank.
Table [tbl:results-table-example] shows an example of such a rank sorted
result table that is typically generated by the workflow framework and
that will be used to present the results of the numerical experiments in
the following section.

center

+-----+-------------------+--------------------------------+--------------------------------+----+----+
| #   | Model             | CV(Dataset A, loss function)   | CV(Dataset B, loss function)   |    |    |
+=====+===================+================================+================================+====+====+
| 0   | Example model 1   | (2) 12\ :math:`\pm`\ 1\*       | (1) 3\ :math:`\pm`\ 2\*        |    |    |
+-----+-------------------+--------------------------------+--------------------------------+----+----+
| 1   | Example model 2   | (1) 5\ :math:`\pm`\ 0.5\*      | (2) 9\ :math:`\pm`\ 1\*        |    |    |
+-----+-------------------+--------------------------------+--------------------------------+----+----+
| 2   | Example model 3   | (3) 28\ :math:`\pm`\ 3\*       | (3) 29\ :math:`\pm`\ 4\*       |    |    |
+-----+-------------------+--------------------------------+--------------------------------+----+----+

Table: Example and explanation of a rank-sorted results table that can
be easily created in the workflow framework: Models are listed in the
rows of the table while the columns present the cross-validated
performance of a certain dataset and loss function. The numbers in
parentheses denote the model’s performance rank in the respective
column. The models are sorted by the average model rank, displaying
models with the best performances (that is the lowest losses) on top of
the table.

