=========
Changelog
=========

All notable changes to this project beggining with version 0.1.0 will be
documented in this file. The format is based on
`Keep a Changelog <https://keepachangelog.com/en/1.0.0/>`_ and we adhere
to `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_. The source
code for all `releases <https://github.com/sktime/skbase/releases>`_
is available on GitHub.

You can also subscribe to ``skpro``'s
`PyPi release <https://libraries.io/pypi/skpro>`_.

For planned changes and upcoming releases, see our :ref:`roadmap`.


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


[2.1.2] - 2023-01-07
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
