.. _history:

=======
History
=======

``skpro`` was started in 2017 by Franz Király and his then-student Frithjof Gressmann
as a `scikit-learn`_ like python package for probabilistic supervised regression.

``skpro`` was then abandoned, from 2019, at version 1.0.1, as development in
Franz Király research group continued to be focused on ``sktime``.

In 2022-23, ``sktime``'s base module was turned into a separate package,
`skbase`_, intended as a workbench to allow easy templating and creation of
`scikit-learn`-likes.

Using the templating scaffold of ``skbase``, ``skpro`` was finally revived
in 2023 by Franz Király, Frithjof Gressmann, and Alex Gregory,
built upon a fully rearchitectured, ``skbase`` reliant API,
as version 2.0.0.

The joint base interface enables mutual compabitibilty between ``skpro``, ``sklearn``,
and ``sktime``, with ``skpro`` probabilistic regressors being potential components used
for probabilistic forecasting in ``sktime``.

Development is supported by members of the ``sktime`` project,
new core developers and the broader community (see
`contributors <contributors.md>`_).

If you are interested in contributing, check out our
:ref:`Contributing <contrib_guide>` guide.

.. _scikit-learn: https://scikit-learn.org/stable/index.html
.. _skbase: https://skbase.readthedocs.io/en/latest/
.. _sktime: https://www.sktime.net/en/stable/index.html
