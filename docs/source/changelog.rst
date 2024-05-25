=========
Changelog
=========

All notable changes to this project beginning with version 0.1.0 will be
documented in this file. The format is based on
`Keep a Changelog <https://keepachangelog.com/en/1.0.0/>`_ and we adhere
to `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_. The source
code for all `releases <https://github.com/sktime/skpro/releases>`_
is available on GitHub.

You can also subscribe to ``skpro``'s
`PyPi release <https://libraries.io/pypi/skpro>`_.

For planned changes and upcoming releases, see roadmap in the
`issue tracker <https://github.com/sktime/skpro/issues>`_.

[2.3.1] - 2024-05-26
====================

Maintenance release with ``scikit-learn 1.5.X`` and ``scikit-base 0.8.X``
compatibility and minor enhancements.

Dependency changes
~~~~~~~~~~~~~~~~~~

* ``scikit-base`` bounds have been updated to ``>=0.6.1,<0.9.0``.
* ``scikit-learn`` bounds have been updated to ``>=0.24.0,<1.6.0``.

Deprecations and removals
~~~~~~~~~~~~~~~~~~~~~~~~~

* in probabilistic regressor tuners ``GridSearchCV``, ``RandomizedSearchCV``,
  use of ``joblib`` backend specific parameters ``n_jobs``,
  ``pre_dispatch`` has been deprecated, and will be removed in ``skpro`` 2.5.0.
  Users should pass backend parameters via the ``backend_params`` parameter instead.

Enhancements
~~~~~~~~~~~~

* [ENH] make ``get_packages_with_changed_specs`` safe to mutation of return
  (:pr:`348`) :user:`fkiraly`
* [ENH] EnbPI regressor for conformal prediction
  intervals (:pr:`343`) :user:`fkiraly`
* [ENH] improved default function to plot via ``BaseDistribution.plot``,
  depending on distribution type (:pr:`353`) :user:`fkiraly`
* [ENH] iid array distribution (:pr:`347`) :user:`fkiraly`
* [ENH] Correct algorithm in ``EnbpiRegressor`` (:pr:`351`) :user:`fkiraly`
* [ENH] Gamma Distribution (:pr:`355`) :user:`ShreeshaM07`
* [ENH] Alpha distribution (:pr:`356`) :user:`SaiRevanth25`

Fixes
~~~~~

* [BUG] fix ``test_run_test_for_class`` test logic (:pr:`345`) :user:`fkiraly`
* [BUG] fix ``random_state`` handling in ``BootstrapRegressor``
  (:pr:`344`) :user:`fkiraly`
* [BUG] fix ``spl`` index when subsetting ``Empirical`` distribution
  via ``iat`` (:pr:`352`) :user:`fkiraly`

Maintenance
~~~~~~~~~~~

* [MNT] isolate imports in ``changelog.py`` build util (:pr:`339`) :user:`fkiraly`
* [MNT] remove legacy base modules (:pr:`80`) :user:`fkiraly`
* [MNT] [Dependabot](deps): Update sphinx-design requirement from ``<0.6.0`` to
  ``<0.7.0`` (:pr:`357`) :user:`dependabot[bot]`
* [MNT] [Dependabot](deps): Update scikit-learn requirement from ``<1.5.0,>=0.24.0``
  to ``>=0.24.0,<1.6.0`` (:pr:`354`) :user:`dependabot[bot]`
* [MNT] Update ``scikit-base`` requirement from
  ``<0.8.0,>=0.6.1`` to ``>=0.6.1,<0.9.0`` (:pr:`366`) :user:`fkiraly`

Documentation
~~~~~~~~~~~~~

* [DOC] minor docs improvements (:pr:`359`) :user:`fkiraly`
* [DOC] fix download shields in readme (:pr:`360`) :user:`fkiraly`
* [DOC] fixing download shields in README (:pr:`361`) :user:`fkiraly`
* [DOC] fixing download shields in README (:pr:`362`) :user:`fkiraly`

Contributors
~~~~~~~~~~~~

:user:`fkiraly`,
:user:`SaiRevanth25`,
:user:`ShreeshaM07`


[2.3.0] - 2024-05-16
====================

* new tutorial notebooks for survival prediction and probability distributions (:pr:`303`, :pr:`305`) :user:`fkiraly`
* interface to ``ngboost`` probabilistic regressor and survival predictor (:pr:`215`, :pr:`301`, :pr:`309`, :pr:`332`) :user:`ShreeshaM07`
* interface to Poisson regressor from ``sklearn`` (:pr:`213`) :user:`nilesh05apr`
* probability distributions rearchitecture, including scalar valued distributions, e.g., ``Normal(mu=0, sigma=1)`` - see "core interface changes"
* probability distributions: illustrative and didactic plotting functionality, e.g., ``my_normal.plot("pdf")`` (:pr:`275`) :user:`fkiraly`
* more distributions: beta, chi-squared, delta, exponential, uniform - :user:`an20805`,
  :user:`malikrafsan`, :user:`ShreeshaM07`, :user:`sukjingitsit`

Core interface changes
~~~~~~~~~~~~~~~~~~~~~~

Probability distributions have been rearchitected with API improvements:

* all changes are fully downwards compatible with the previous API.
* distributions can now be scalar valued, e.g., ``Normal(mu=0, sigma=1)``.
  More generally, all distributions behave as scalar distributions if
  ``index`` and ``columns`` are not passed and all parameters passed are scalar.
  or scalar-like. In this case, methods such as ``pdf``,
  ``cdf`` or ``sample`` will return scalar (float) values instead of ``pd.DataFrame``.
* ``ndim`` and ``shape`` - distributions now possess an ``ndim`` property, which evaluates to 0 for
  scalar distributions, and 2 otherwise. The ``shape`` property evaluates to
  the empty tuple for scalar distributions, and to a 2-tuple with the shape for
  array-like distributions. This is in line with ``numpy`` conventions.
* ``plot`` - distributions now have a ``plot`` method, which can be used to plot any
  method of the distribution. The method is called as ``my_distr.plot("pdf")``
  or ``my_distribution.plot("cdf")``, or similar.
  If the distribution is scalar, this will create a single ``matplotlib`` plot in
  an ``ax`` object. DataFrame-like distributions will create a plot for each
  marginal component, returning ``fig`` with an array of ``ax`` objects, of same
  shape as the distribution object.
* ``head``, ``tail`` - distributions now possess ``head`` and ``tail`` methods,
  which return the first
  and last ``n`` rows of the distribution, respectively. This is useful for
  inspecting the distribution object in a Jupyter notebook, in particular when
  combined with ``plot``.
* ``at``, ``iat`` - distributions now possess ``at`` and ``iat`` subsetters,
  which can be used to
  subset a DataFrame-like distribution to a scalar distribution at a given
  integer index or location index, respectively.
* ``pdf``, ``pmf`` - all distributions
  now possess a ``pdf`` and ``pmf`` method, for probability density
  function and probability mass function. These are available for all distributions,
  continuous, discrete, and mixed. ``pdf`` returns the density of the continuous part
  of the distribution, ``pmf`` the mass of the discrete part. Continuous distributions
  will return 0 for ``pmf``, discrete distributions will return 0 for ``pdf``.
  Logarithmic versions of these methods are available as ``log_pdf`` and ``log_pmf``,
  these may be more numerically stable.
* ``surv``, ``haz`` - distributions now possess
  shorthand methods to return survival function evaluates,
  ``surv``, and hazard function evaluates, ``haz``. These are available for
  all distributions. In case of mixed distributions, hazard is computed with the
  continuous part of the distribution.
* ``distr:paramtype`` tag - distributions are now annotated with a new public tag:
  ``distr:paramtype`` indicates whether
  the distribution is ``"parametric"``, ``"non-parametric"``, or ``"composite"``.
  Parametric distributions have only numpy array-like or categorical parameters.
  Non-parametric distributions may have further types of parameters such as data-like,
  but no distributions. Composite distributions have other distributions as parameters.
* ``to_df``, ``get_params_df`` - parametric distributions
  now provide methods ``to_df``, ``get_params_df``,
  which allow to return distribution parameters coerced to ``DataFrame``, or ``dict``
  of ``DataFrame``, keyed by parameter names, respectively.
* the extension contract for distributions has been changed to a boilerplate layered
  design. Extenders will now implement private methods such as ``_pdf``, ``_cdf``,
  instead of overriding the public interface. This allows for more flexibility in
  boilerplate design, and ensures more consistent behavior across distributions.
  The new extension contract is documented in the new ``skpro`` extension template,
  ``extension_templates/distributions.py``.

Deprecations and removals
~~~~~~~~~~~~~~~~~~~~~~~~~

* At version 2.4.0, the ``bound`` parameter will be removed
  from the ``CyclicBoosting`` probabilistic
  supervised regression estimator, and will be replaced by use of ``lower`` or
  ``upper``. To retain previous behaviour, users should replace ``bound="U"``
  with ``upper=None`` and ``lower=None``; ``bound="L"`` with ``upper=None`` and
  ``lower`` set to the value of the lower bound; and ``bound="B"`` with both
  ``upper`` and ``lower`` set to the respective values.
  To silence the warnings and prevent exceptions occurring from 2.4.0,
  users should not explicitly set ``bounds``, and ensure values for any subsequent
  parameters are set as keyword arguments, not positional arguments.

Enhancements
~~~~~~~~~~~~

Probability distributions
~~~~~~~~~~~~~~~~~~~~~~~~~

* [ENH] probability distributions - boilerplate refactor (:pr:`265`) :user:`fkiraly`
* [ENH] probability distributions: convenience feature to coerce ``index`` and ``columns`` to ``pd.Index`` (:pr:`276`) :user:`fkiraly`
* [ENH] distribution ``quantile`` method for scalar distributions (:pr:`277`) :user:`fkiraly`
* [ENH] systematic suite tests for scalar probability distributions (:pr:`278`) :user:`fkiraly`
* [ENH] scalar test cases for probability distributions (:pr:`279`) :user:`fkiraly`
* [ENH] activate tests for distribution base class defaults (:pr:`266`) :user:`fkiraly`
* [ENH] probability distributions: illustrative and didactic plotting functionality (:pr:`275`) :user:`fkiraly`
* [ENH] Uniform Continuous distribution (:pr:`223`) :user:`an20805`
* [ENH] Chi-Squared Distribution (:pr:`217`) :user:`sukjingitsit`
* [ENH] Adapter for Scipy Distributions (:pr:`287`) :user:`malikrafsan`
* [ENH] simplify coercion in ``BaseDistribution._log_pdf`` and ``_pdf`` default (:pr:`293`) :user:`fkiraly`
* [ENH] Beta Distribution (:pr:`298`) :user:`malikrafsan`
* [ENH] distributions: ``pmf`` and ``log_pmf`` method (:pr:`295`) :user:`fkiraly`
* [ENH] Delta distribution (:pr:`299`) :user:`fkiraly`
* [ENH] distributions: survival and hazard function and defaults (:pr:`294`) :user:`fkiraly`
* [ENH] improved ``Empirical`` distribution - scalar mode, new API compatibility (:pr:`307`) :user:`fkiraly`
* [ENH] increase distribution default ``plot`` resolution (:pr:`308`) :user:`fkiraly`
* [ENH] distribution ``get_params`` in data frame format (:pr:`285`) :user:`fkiraly`
* [ENH] ``head`` and ``tail`` for distribution objects (:pr:`310`) :user:`fkiraly`
* [ENH] full support of hierarchical ``MultiIndex`` ``index`` in ``Empirical`` distribution, tests (:pr:`314`) :user:`fkiraly`
* [ENH] ``at`` and ``iat`` subsetters for distributions (:pr:`274`) :user:`fkiraly`
* [ENH] ``Exponential`` distribution (:pr:`325`) :user:`ShreeshaM07`
* [ENH] ``Mixture`` distribution upgrade - refactor to new extension interface, support scalar case (:pr:`315`) :user:`fkiraly`
* [ENH] native implementation of Johnson QPD family, explicit pdf (:pr:`327`) :user:`fkiraly`
* [ENH] improved defaults for ``BaseDistribution`` ``_mean``, ``_var``, and ``_energy_x`` (:pr:`330`) :user:`fkiraly`

Probabilistic regression
~~~~~~~~~~~~~~~~~~~~~~~~

* [ENH] interface to ``ngboost`` (:pr:`215`) :user:`ShreeshaM07`
* [ENH] interfacing Poisson regressor from sklearn (:pr:`213`) :user:`nilesh05apr`
* [ENH] refactor ``NGBoostRegressor`` to inherit ``NGBoostAdapter`` (:pr:`309`) :user:`ShreeshaM07`
* [ENH] ``Exponential`` dist in ``NGBoostRegressor``, ``NGBoostSurvival`` (:pr:`332`) :user:`ShreeshaM07`

Survival and time-to-event prediction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* [ENH] Delta point prediction baseline regressor (:pr:`300`) :user:`fkiraly`
* [ENH] Interface ``NGBSurvival`` from ``ngboost`` (:pr:`301`) :user:`ShreeshaM07`
* [ENH] in ``ConditionUncensored`` reducer, ensure coercion to float of ``C`` (:pr:`318`) :user:`fkiraly`

Test framework
~~~~~~~~~~~~~~

* [MNT] faster collection of differential tests through caching, test if pyproject change (:pr:`296`) :user:`fkiraly`

Fixes
~~~~~

Probability distributions
~~~~~~~~~~~~~~~~~~~~~~~~~

* [BUG] bugfixes for distribution base class default methods (:pr:`281`) :user:`fkiraly`
* [BUG] fix ``Empirical`` index to be ``pd.MultiIndex`` for hierarchical data index (:pr:`286`) :user:`fkiraly`
* [BUG] update Johnson QPDistributions with bugfixes and vectorization (cyclic-boosting ver.1.4.0) (:pr:`232`) :user:`setoguchi-naoki`
* [BUG] ``BaseDistribution._var``: fix missing factor 2 in Monte Carlo variance default method (:pr:`331`) :user:`fkiraly`

Survival and time-to-event prediction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* [BUG] fix ``CoxPH`` handling of ``statsmodels`` ``status`` variable (:pr:`306`) :user:`fkiraly`
* [BUG] fix survival metrics if ``C_true=None`` is passed (:pr:`316`) :user:`fkiraly`

Maintenance
~~~~~~~~~~~

* [MNT] [Dependabot](deps): Update ``sphinx-gallery`` requirement from ``<0.16.0`` to ``<0.17.0`` (:pr:`288`) :user:`dependabot[bot]`
* [MNT] move GHA runners consistently to ``ubuntu-latest``, ``windows-latest``, ``macos-13`` (:pr:`272`) :user:`fkiraly`
* [MNT] set macos runner for release workflow to ``macos-13`` (:pr:`273`) :user:`fkiraly`
* [MNT] fix binder environment (:pr:`297`) :user:`fkiraly`
* [MNT] moving ensemble regressors to ``regression.ensemble`` (:pr:`302`) :user:`fkiraly`
* [MNT] remove ``findiff`` soft dependency (:pr:`328`) :user:`fkiraly`
* [MNT] deprecation handling for ``CyclicBoosting`` (:pr:`329`) :user:`fkiraly`, :user:`setoguchi-naoki`
* [MNT] fix repository variables in changelog generator (:pr:`333`) :user:`fkiraly`

Documentation
~~~~~~~~~~~~~

* [DOC] add ``zenodo`` citation badge in README (:pr:`262`) :user:`fkiraly`
* [DOC] fix typo in changelog link (:pr:`263`) :user:`fkiraly`
* [DOC] typo fixes in Fisk AFT docstring (:pr:`264`) :user:`fkiraly`
* [DOC] fix minor typos in the changelog (:pr:`268`) :user:`fkiraly`
* [DOC] fixes to extension templates (:pr:`270`) :user:`fkiraly`
* [DOC] remove legacy examples (:pr:`271`) :user:`fkiraly`
* [DOC] correcting 2024 changelog dates (:pr:`280`) :user:`fkiraly`
* [DOC] add missing contributors to ``all-contributorsrc`` - :user:`an20805`, :user:`duydl`, :user:`sukjingitsit` (:pr:`284`) :user:`fkiraly`
* [DOC] tutorial notebook for probability distributions (:pr:`303`) :user:`fkiraly`
* [DOC] tutorial notebook for survival prediction (:pr:`305`) :user:`fkiraly`
* [DOC] visualizations for first intro vignette in intro notebook and minor updates (:pr:`311`) :user:`fkiraly`
* [DOC] improve docstrings of metrics (:pr:`317`) :user:`fkiraly`
* [DOC] Fix typos throughout the codebase (:pr:`338`) :user:`szepeviktor`

Contributors
~~~~~~~~~~~~

:user:`an20805`,
:user:`fkiraly`,
:user:`malikrafsan`,
:user:`nilesh05apr`,
:user:`setoguchi-naoki`,
:user:`ShreeshaM07`,
:user:`sukjingitsit`,
:user:`szepeviktor`


[2.2.2] - 2024-04-20
====================

Highlights
----------

* ``lifelines`` predictive survival regressors are available as ``skpro`` estimators:
  accelerated failure time (Fisk, Log-normal, Weibull), CoxPH variants,
  Aalen additive model (:pr:`247`, :pr:`258`, :pr:`260`) :user:`fkiraly`
* ``scikit-survival`` predictive survival regressors are available as ``skpro`` estimators:
  CoxPH variants, CoxNet, survival tree and forest, survival gradient boosting (:pr:`237`) :user:`fkiraly`
* GLM regressor using ``statsmodels`` ``GLM``, with Gaussian link (:pr:`222`) :user:`julian-fong`
* various survival type distributions added: log-normal, logistic, Fisk (=log-logistic), Weibull
  (:pr:`218`, :pr:`241`, :pr:`242`, :pr:`259`) :user:`bhavikar`, :user:`malikrafsan`, :user:`fkiraly`
* Poisson distribution added (:pr:`226`) :user:`fkiraly`


Core interface changes
~~~~~~~~~~~~~~~~~~~~~~

Probability distributions
~~~~~~~~~~~~~~~~~~~~~~~~~

* Probability distributions (``BaseDistribution``) now have a ``len`` method,
  which returns the number of number of rows of the distribution, this is the same
  as the ``len`` of a ``pd.DataFrame`` returned by ``sample``.
* the interface now supports discrete distributions and those with integer support.
  Such distributions implement ``pmf`` and ``log_pmf`` methods.

Enhancements
------------

Probability distributions
~~~~~~~~~~~~~~~~~~~~~~~~~

* [ENH] Log-normal probability distribution (:pr:`218`) :user:`bhavikar`
* [ENH] Poisson distribution (:pr:`226`) :user:`fkiraly`
* [ENH] make ``Empirical`` distribution compatible with multi-index rows (:pr:`233`) :user:`fkiraly`
* [ENH] empirical quantile parameterized distribution (:pr:`236`) :user:`fkiraly`
* [ENH] add ``len`` of ``BaseDistribution``, test ``shape``, ``len``, indices (:pr:`239`) :user:`fkiraly`
* [ENH] Logistic distribution (:pr:`241`) :user:`malikrafsan`
* [ENH] Weibull distribution (:pr:`242`) :user:`malikrafsan`
* [ENH] delegator class for distributions (:pr:`252`) :user:`fkiraly`
* [ENH] Johnson QP-distributions - add some missing capability tags (:pr:`253`) :user:`fkiraly`
* [ENH] remove stray ``_get_bc_params`` from ``LogNormal`` (:pr:`256`) :user:`fkiraly`
* [ENH] Fisk distribution aka log-logistic distribution (:pr:`259`) :user:`fkiraly`

Probabilistic regression
~~~~~~~~~~~~~~~~~~~~~~~~

* [ENH] ``GLMRegressor`` using statsmodels ``GLM`` with Gaussian link (:pr:`222`) :user:`julian-fong`
* [ENH] added test parameters for probabilistic metrics (:pr:`234`) :user:`fkiraly`

Survival and time-to-event prediction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* [ENH] adapter to ``scikit-survival``, all distributional survival regressors interfaced (:pr:`237`) :user:`fkiraly`
* [ENH] adapter to ``lifelines``, most distributional survival regressors interfaced (:pr:`247`) :user:`fkiraly`
* [ENH] log-normal AFT model from ``lifelines`` (:pr:`258`) :user:`fkiraly`
* [ENH] log-logistic/Fisk AFT model from ``lifelines`` (:pr:`260`) :user:`fkiraly`

Test framework
~~~~~~~~~~~~~~

* [ENH] refactor test scenario creation to be lazy rather than on module load (:pr:`245`) :user:`fkiraly`

Fixes
-----

Probability distributions
~~~~~~~~~~~~~~~~~~~~~~~~~

* [BUG] bugfixes to QPD distributions - ``QPD_U``, ``QPD_S`` (:pr:`194`) :user:`fkiraly`
* [BUG] fixes to lognormal distribution  (:pr:`261`) :user:`fkiraly`

Documentation
-------------

* [DOC] documentation improvement for probabilistic metrics (:pr:`234`) :user:`fkiraly`
* [DOC] add :user:`julian-fong` to ``all-contributorsrc`` (:pr:`238`) :user:`fkiraly`
* [DOC] docstring with mathematical description for ``QPD_Empirical`` (:pr:`253`) :user:`fkiraly`

Maintenance
-----------

* [MNT] fix version pointer in readthedocs ``json`` (:pr:`225`) :user:`fkiraly`
* [MNT] fix broken api source links in latest docs version (:pr:`243`) :user:`duydl`

Contributors
------------

:user:`bhavikar`,
:user:`duydl`,
:user:`fkiraly`,
:user:`julian-fong`,
:user:`malikrafsan`


[2.2.1] - 2024-03-03
====================

Minor bugfix and maintenance release.

Contents
--------

* [ENH] migrate tests of distribution prediction metrics to ``skbase`` class
  (:pr:`208`) :user:`fkiraly`
* [BUG] fix dispatching of censoring information in probabilistic metrics
  (:pr:`208`) :user:`fkiraly`
* [BUG] fix missing location/scale in ``TDistribution`` (:pr:`210`) :user:`ivarzap`


[2.2.0] - 2024-02-08
====================

Highlights
----------

* interface to ``cyclic_boosting`` package (:pr:`144`) :user:`setoguchi-naoki`, :user:`FelixWick`
* framework support for probabilistic survival/time-to-event prediction with right censored data (:pr:`157`) :user:`fkiraly`
* basic set of time-to-event prediction estimators and survival prediction metrics (:pr:`161`, :pr:`198`) :user:`fkiraly`
* Johnson Quantile-Parameterized Distributions (QPD) with bounded and unbounded mode (:pr:`144`) :user:`setoguchi-naoki`, :user:`FelixWick`
* abstract parallelization backend, for benchmarking and tuning (:pr:`160`) :user:`fkiraly`, :user:`hazrulakmal`

Dependency changes
~~~~~~~~~~~~~~~~~~

* ``pandas`` bounds have been updated to ``>=1.1.0,<2.3.0``.

Core interface changes
~~~~~~~~~~~~~~~~~~~~~~

BaseObject and base framework
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* estimators and objects now record author and maintainer information in the new
  tags ``"authors"`` and ``"maintainers"``. This is required only for estimators
  in ``skpro`` proper and compatible third party packages. It is also used to generate
  mini-package headers used in lookup functionality of the ``skpro`` webpage.
* the ``model_selection`` and ``benchmarking`` utilities now support abstract
  parallelization backends via the ``backend`` and ``backend_params`` arguments.
  This has been standardized to use the same backend options and syntax as the
  abstract parallelization backend in ``sktime``.

Probabilistic regression
~~~~~~~~~~~~~~~~~~~~~~~~

* all probabilistic regressors now accept an argument ``C`` in ``fit``,
  to pass censoring information. This is for API compatibility with survival
  and is ignored when passed to non-survival regressors, corresponding to the
  naive reduction strategy of "ignoring censoring information".
* existing pipelines, tuners and ensemble methods have been extended to support
  survival prediction - if ``C`` if passed, it is passed to the underlying
  components.

Survival and time-to-event prediction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* support for probabilistic survival or time-to-event prediction estimators
  with right censored data has been introduced. The interface and base class
  is identical to the tabular probabilistic regression interface, with the
  addition of a ``C`` argument to the ``fit`` methods.
  Regressors that genuinely support survival prediction have the
  ``capability: survival`` tag set to ``True`` in their metadata.
* an extension template for survival prediction has been added to the
  ``skpro`` extension templates, in ``extension_templates``
* the interface for probabilistic performance metrics has been extended to
  also accept censoring information, which can be passed via the optional ``C_true``
  argument, to all performance metrics. Metrics genuinely supporting survival
  prediction have the ``capability: survival`` tag set to ``True``. Other metrics
  still take the ``C_true`` argument, but ignore it. This corresponds to the
  naive reduction strategy of "ignoring censoring information".
* for pipelining and tuning, the existing compositors in ``model_selection``
  and ``regression.compose`` can be used, see above.
* for benchmarking, the existing benchmarking framework in ``benchmarking``
  can be used, it has been extended to support survival prediction and censoring
  information.

Enhancements
------------

BaseObject and base framework
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* [ENH] author and maintainer tags, tags documented in regressor extension template
  (:pr:`187`) :user:`fkiraly`

Probability distributions
~~~~~~~~~~~~~~~~~~~~~~~~~

* [ENH] Johnson Quantile-Parameterized Distributions (QPD) with bounded and
  unbounded mode (:pr:`144`) :user:`setoguchi-naoki`, :user:`FelixWick`

Probabilistic regression
~~~~~~~~~~~~~~~~~~~~~~~~

* [ENH] Cyclic boosting interface (:pr:`144`) :user:`setoguchi-naoki`, :user:`FelixWick`
* [ENH] abstract parallelization backend, refactor of ``evaluate`` and tuners,
  extend evaluate and tuners to survival predictors (:pr:`160`) :user:`fkiraly`, :user:`hazrulakmal`

Survival and time-to-event prediction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* [ENH] support for survival/time-to-event prediction, statsmodels Cox PH model
  (:pr:`157`) :user:`fkiraly`
* [ENH] survival prediction compositor - reducers to tabular probabilistic regression
  (:pr:`161`) :user:`fkiraly`
* [ENH] survival prediction metrics - framework support and tests, SPLL, Harrell C
  (:pr:`198`) :user:`fkiraly`

Fixes
-----

Probabilistic regression
~~~~~~~~~~~~~~~~~~~~~~~~

* [BUG] fix API non-compliance in ``sklearn`` variance prediction adapter (:pr:`192`) :user:`fkiraly`
* [BUG] fix defaulting logic for ``_predict_interval`` and ``_predict_quantiles`` when only ``_predict_var`` is implemented (:pr:`191`) :user:`fkiraly`
* [BUG] fix ``CyclicBoosting._predict_quantiles`` (:pr:`195`) :user:`fkiraly`
* [BUG] fix fallback for ``pdfnorm`` method, add metrics to tests (:pr:`204`) :user:`fkiraly`

Test framework
~~~~~~~~~~~~~~

* [BUG] fix lookup for specialized test classes (:pr:`189`) :user:`fkiraly`

Documentation
-------------

* [DOC] API reference for performance metrics (:pr:`206`) :user:`fkiraly`
* [DOC] README update for 2.2.0 (:pr:`207`) :user:`fkiraly`

Maintenance
-----------

* [MNT] [Dependabot](deps): Bump styfle/cancel-workflow-action from ``0.12.0`` to ``0.12.1`` (:pr:`183`) :user:`dependabot`
* [MNT] skip ``CyclicBoosting`` and QPD tests until #189 failures are resolved (:pr:`193`) :user:`fkiraly`
* [MNT] [Dependabot](deps-dev): Update pandas requirement from ``<2.2.0,>=1.1.0`` to ``>=1.1.0,<2.3.0`` (:pr:`182`) :user:`dependabot`
* [MNT] [Dependabot](deps): Bump codecov/codecov-action from 3 to 4 by (:pr:`201`) :user:`dependabot`
* [MNT] [Dependabot](deps): Bump pre-commit/action from ``3.0.0`` to ``3.0.1`` (:pr:`202`) :user:`dependabot`

Contributors
------------

:user:`FelixWick`,
:user:`fkiraly`,
:user:`hazrulakmal`,
:user:`setoguchi-naoki`


[2.1.3] - 2023-01-22
====================

``sklearn`` compatibility update:

* compatibility with ``sklearn 1.4.X``
* addition of ``feature_names_in_`` and ``n_features_in_`` default attributes
  to ``BaseProbaRegressor``, written to ``self`` in ``fit``

Dependency changes
~~~~~~~~~~~~~~~~~~

* ``sklearn`` bounds have been updated to ``<1.4.0,>=0.24.0``.

Core interface changes
~~~~~~~~~~~~~~~~~~~~~~

Probabilistic regression
^^^^^^^^^^^^^^^^^^^^^^^^

* probabilistic regressors will now always save attributes ``feature_names_in_``
  and ``n_features_in_`` to ``self`` in ``fit``.
  ``feature_names_in_`` is an 1D ``np.ndarray`` of feature names seen in ``fit``,
  ``n_features_in_`` is an ``int``, and equal to ``len(feature_names_in_)``.
* this ensures compatibility with ``sklearn``, where these attributes are expected.
* the new attributes can also be queried via the existing ``get_fitted_params``
  interface.

Enhancements
------------

* [ENH] in ``BaseRegressorProba.fit``, use ``"feature_names"`` metadata field
  to store feature names and write to ``self`` in ``fit`` (:pr:`180`) :user:`dependabot`

Maintenance
-----------

* [MNT] [Dependabot](deps): Bump ``actions/dependency-review-action``
  from 3 to 4 (:pr:`178`) :user:`dependabot`
* [MNT] [Dependabot](deps-dev): Update polars requirement from ``<0.20.0``
  to ``<0.21.0`` (:pr:`176`) :user:`dependabot`
* [MNT] [Dependabot](deps-dev): Update ``sphinx-issues`` requirement
  from ``<4.0.0`` to ``<5.0.0`` (:pr:`179`) :user:`dependabot`
* [MNT] [Dependabot](deps-dev): Update ``scikit-learn`` requirement
  from ``<1.4.0,>=0.24.0`` to ``>=0.24.0,<1.5.0`` (:pr:`177`) :user:`dependabot`


[2.1.2] - 2024-01-07
====================

Highlights
----------

* ``sklearn`` based probabilistic regressors - Gaussian processes, Bayesian linear regression (:pr:`166`) :user:`fkiraly`
* ``SklearnProbaReg`` - general interface adapter to ``sklearn`` regressors with variance prediction model (:pr:`163`) :user:`fkiraly`

Dependency changes
~~~~~~~~~~~~~~~~~~

* ``scikit-base`` bounds have been updated to ``<0.8.0,>=0.6.1``.
* ``polars`` (data container soft dependency) bounds have been updated to allow python 3.12.

Enhancements
------------

Data types, checks, conversions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* [ENH] ``n_features`` and ``feature_names`` metadata field for table mtypes (:pr:`150`) :user:`fkiraly`
* [ENH] ``check_is_mtype`` dict type return, improved input check error messages in ``BaseRegressorProba`` (:pr:`151`) :user:`fkiraly`

Probability distributions
~~~~~~~~~~~~~~~~~~~~~~~~~

* [ENH] adapter from ``scipy`` ``rv_discrete`` to ``skpro`` ``Empirical`` (:pr:`155`) :user:`fkiraly`

Probabilistic regression
~~~~~~~~~~~~~~~~~~~~~~~~

* [ENH] ``sklearn`` wrappers to str-coerce columns of ``pd.DataFrame`` before passing (:pr:`148`) :user:`fkiraly`
* [ENH] clean up copy-paste leftovers in ``BaseProbaRegressor`` (:pr:`156`) :user:`fkiraly`
* [ENH] adapter for ``sklearn`` probabilistic regressors (:pr:`163`) :user:`fkiraly`
* [ENH] add tags to ``SklearnProbaReg`` (:pr:`168`) :user:`fkiraly`
* [ENH] interfacing all concrete ``sklearn`` probabilistic regressors (:pr:`166`) :user:`fkiraly`

Test framework
~~~~~~~~~~~~~~

* [ENH] scenario tests for mixed ``pandas`` column index types (:pr:`145`) :user:`fkiraly`
* [ENH] scitype inference utility, test class register, test class test condition (:pr:`159`) :user:`fkiraly`

Fixes
-----

Probabilistic regression
~~~~~~~~~~~~~~~~~~~~~~~~

* [BUG] in probabilistic regressors, ensure correct index treatment if ``X: pd.DataFrame`` and ``y: np.ndarray`` are passed (:pr:`146`) :user:`fkiraly`

Documentation
-------------

* [DOC] update ``AUTHORS.rst`` file (:pr:`147`) :user:`fkiraly`

Maintenance
-----------

* [MNT] [Dependabot](deps): Bump ``actions/upload-artifact`` from 3 to 4 (:pr:`154`) :user:`dependabot`
* [MNT] [Dependabot](deps): Bump ``actions/download-artifact`` from 3 to 4 (:pr:`153`) :user:`dependabot`
* [MNT] [Dependabot](deps): Bump ``actions/setup-python`` from 4 to 5 (:pr:`152`) :user:`dependabot`
* [MNT] [Dependabot](deps-dev): Update ``sphinx-gallery`` requirement from ``<0.15.0`` to ``<0.16.0`` (:pr:`149`) :user:`dependabot`
* [MNT] [Dependabot](deps-dev): Update ``scikit-base`` requirement from ``<0.7.0,>=0.6.1`` to ``>=0.6.1,<0.8.0`` (:pr:`169`) :user:`dependabot`
* [MNT] adding ``codecov.yml`` and turning coverage reports informational (:pr:`165`) :user:`fkiraly`
* [MNT] handle deprecation of ``pandas.DataFrame.applymap`` (:pr:`170`) :user:`fkiraly`
* [MNT] handle ``polars`` deprecations (:pr:`171`) :user:`fkiraly`


[2.1.1] - 2023-11-02
====================

Highlights
----------

* probabilistic regressor: multiple quantile regression (:pr:`108`) :user:`Ram0nB`
* probabilistic regressor: interface to ``MapieRegressor`` from ``mapie`` package
  (:pr:`136`) :user:`fkiraly`
* framework support for ``polars`` via mtypes (:pr:`130`) :user:`fkiraly`

Enhancements
------------

Data types, checks, conversions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* [ENH] ``polars`` mtypes for data tables (:pr:`130`) :user:`fkiraly`

Probabilistic regression
~~~~~~~~~~~~~~~~~~~~~~~~

* [ENH] probabilistic regressors - input checks and support for more input types
  (:pr:`129`) :user:`fkiraly`
* [ENH] multiple quantile regression (:pr:`108`) :user:`Ram0nB`
* [ENH] interface ``MapieRegressor`` from ``mapie`` (:pr:`136`) :user:`fkiraly`

Test framework
~~~~~~~~~~~~~~

* [ENH] integrate ``check_estimator`` with ``TestAllEstimators`` and
  ``TestAllRegressors`` for python command line estimator testing
  (:pr:`138`) :user:`fkiraly`
* [ENH] improved conditional testing (:pr:`140`) :user:`fkiraly`

Documentation
-------------

* [DOC] fix math in ``plotting`` docstrings (:pr:`121`) :user:`fkiraly`
* [DOC] improved probabilistic tabular regressor extension template
  (:pr:`137`) :user:`fkiraly`
* [DOC] typo fixes in regression extension template (:pr:`139`) :user:`fkiraly`

Maintenance
-----------

* [MNT] point readthedocs ``json`` switcher variable to GitHub
  (:pr:`125`) :user:`fkiraly`
* [MNT] change test OS versions to latest (:pr:`126`) :user:`fkiraly`

Fixes
-----

* [BUG] fix test fixture generation logic (:pr:`142`) :user:`fkiraly`
* [BUG] fix retrieval in ``all_objects`` if ``filter_tags`` is provided
  (:pr:`141`) :user:`fkiraly`

Contributors
------------
:user:`fkiraly`,
:user:`Ram0nB`


[2.1.0] - 2023-10-09
====================

Python 3.12 compatibility release.

Contents
--------

* [MNT] [Dependabot](deps-dev): Update ``numpy`` requirement from
  ``<1.25,>=1.21.0`` to ``>=1.21.0,<1.27`` (:pr:`118`) :user:`dependabot`
* [MNT] Python 3.12 support - for ``skpro`` release 2.1.0 (:pr:`109`) :user:`fkiraly`


[2.0.1] - 2023-10-08
====================

Release with minor maintenance actions and enhancements.

Enhancements
------------

* [ENH] basic "test all estimators" suite (:pr:`89`) :user:`fkiraly`

Documentation
-------------

* [DOC] improvements to notebook 1 (:pr:`106`) :user:`fkiraly`

Maintenance
-----------

* [MNT] address deprecation of ``skbase.testing.utils.deep_equals``
  (:pr:`111`) :user:`fkiraly`
* [MNT] activate ``dependabot`` for version updates and maintenance
  (:pr:`110`) :user:`fkiraly`
* [MNT] [Dependabot](deps): Bump ``styfle/cancel-workflow-action`` from 0.9.1 to 0.12.0
  (:pr:`113`) :user:`dependabot`
* [MNT] [Dependabot](deps): Bump ``actions/dependency-review-action`` from 1 to 3
  (:pr:`114`) :user:`dependabot`
* [MNT] [Dependabot](deps): Bump ``actions/checkout`` from 3 to 4
  (:pr:`115`) :user:`dependabot`
* [MNT] [Dependabot](deps): Bump ``actions/download-artifact`` from 2 to 3
  (:pr:`116`) :user:`dependabot`
* [MNT] [Dependabot](deps): Bump ``actions/upload-artifact`` from 2 to 3
  (:pr:`117`) :user:`dependabot`


[2.0.0] - 2023-09-13
====================

Re-release of ``skpro``, newly rearchitected using ``skbase``!

Try out ``skpro v2`` on `Binder <https://mybinder.org/v2/gh/sktime/skpro/main?filepath=examples>`_!

Contributions, bug reports, and feature requests are welcome on the `issue tracker <https://github.com/sktime/skpro/issues>`_

or on the `community Discord <https://discord.com/invite/54ACzaFsn7>`_.

Contributors
------------
:user:`Alex-JG3`,
:user:`fkiraly`,
:user:`frthjf`

[1.0.1] - 2019-02-18
====================

First stable release of ``skpro``, last release before hiatus.

[1.0.0b] - 2017-12-08
=====================

First public release (beta) of ``skpro``.
