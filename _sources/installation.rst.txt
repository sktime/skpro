Installation
************

The installation of the latest stable version is easy using the python package manager `pip`_. ::

    pip install skpro

``skpro`` depends on the ``scikit-learn`` package and its respective dependencies ``numpy`` and ``scipy`` which will be pulled in automatically during installation.
Furthermore, you may install optional package dependencies that enhance the workflow components (i.e. `uncertainties`_ and `tabulate`_). ::

    pip install skpro[workflow]

That's it. You are now ready to go. We recommend reading the :doc:`user guide <user_guide>` to get started.

Bleeding edge
^^^^^^^^^^^^^

To test or develop new features you may want to install the latest package version from the development branch (bleeding edge installation).

Clone the source from our `public code repository`_ on GitHub and change into the skpro directory. Make sure that all dependencies are installed: ::

    pip install -r requirements.txt

Then run ::

    python setup.py develop

to install the package into the activated Python environment. To build the documentation run ::

    python setup.py docs

Note that bleeding edge installations are likely contain bugs are not recommended for productive environments.

If you like to contribute to documentation please refer to our :doc:`contribution guide <contributing>`.


.. _pip: http://www.pip-installer.org/
.. _uncertainties: http://pythonhosted.org/uncertainties/
.. _tabulate: https://pypi.python.org/pypi/tabulate
.. _public code repository: https://github.com/alan-turing-institute/skpro