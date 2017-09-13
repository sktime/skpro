Installation
************

Installation is easy using the python package manager `pip`_. ::

    pip install skpro

skpro depends on the scikit-learn package and its respective dependencies numpy and scipy which will be pulled in automatically during installation.
Furthermore, you may install optional package dependencies that enhance the workflow components (i.e. `uncertainties`_ and `tabulate`_). ::

    pip install skpro[workflow]

That's it. You are now ready to go. We recommend reading the :doc:`/introduction` for a quick start.

.. _pip: http://www.pip-installer.org/
.. _uncertainties: http://pythonhosted.org/uncertainties/
.. _tabulate: https://pypi.python.org/pypi/tabulate