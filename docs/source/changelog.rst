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
