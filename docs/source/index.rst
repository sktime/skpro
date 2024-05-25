.. _home:

================
Welcome to skpro
================

``skpro`` is a library for supervised probabilistic prediction and
tabular probability distributions in python.

Features
========

``skpro`` provides unified, ``sklearn`` and ``skbase`` compatible interfaces to:

* tabular **supervised regressors for probabilistic prediction** - interval, quantile and distribution predictions
* tabular **probabilistic time-to-event and survival prediction** - instance-individual survival distributions
* **metrics to evaluate probabilistic predictions**, e.g., pinball loss, empirical coverage, CRPS
* **reductions** to turn ``sklearn`` regressors into probabilistic ``skpro`` regressors, such as bootstrap or conformal
* building **pipelines and composite models**, including tuning via probabilistic performance metrics
* symbolic **probability distributions** with value domain of ``pandas.DataFrame``-s and ``pandas``-like interface

Technical specification
=======================

* In-memory computation of a single machine, no distributed computing
* Medium-sized data in pandas and NumPy based containers
* Modular, principled and object-oriented API
* Using interactive Python interpreter, no command-line interface or graphical user interface

Contents
========

.. toctree::
   :maxdepth: 1
   :hidden:

   get_started
   users
   installation
   api_reference
   get_involved
   developer_guide
   about
   examples

From here, you can navigate to:

.. grid:: 1 2 2 2
    :gutter: 3

    .. grid-item-card:: Get Started
        :text-align: center

        Get started using ``skpro`` quickly.

        +++

        .. button-ref:: get_started
            :color: primary
            :click-parent:
            :expand:

            Get Started

    .. grid-item-card:: User Documentation
        :text-align: center

        Find user documentation.

        +++

        .. button-ref:: users
            :color: primary
            :click-parent:
            :expand:

            Users

    .. grid-item-card:: API Reference
        :text-align: center

        Understand ``skpro``'s API.

        +++

        .. button-ref:: api_reference
            :color: primary
            :click-parent:
            :expand:

            API Reference

    .. grid-item-card:: Get Involved
        :text-align: center

        Find out how you can contribute.

        +++

        .. button-ref:: contribute
            :color: primary
            :click-parent:
            :expand:

            Get Involved

    .. grid-item-card:: Changelog
        :text-align: center

        See how the package has changed.

        +++

        .. button-ref:: changelog
            :color: primary
            :click-parent:
            :expand:

            Changelog

    .. grid-item-card:: About
        :text-align: center

        Learn more about ``skpro``.

        +++

        .. button-ref:: about
            :color: primary
            :click-parent:
            :expand:

            Learn More
