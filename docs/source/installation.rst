.. _full_install:
.. _installation:

Installation
============

``skpro`` currently supports:

* Python versions 3.10, 3.11, 3.12, 3.13, and 3.14.
* Operating systems Mac OS X, Unix-like OS, Windows 8.1 and higher.

See the full list of `precompiled wheels available on PyPI`_.

.. contents::
   :local:

For frequent issues with installation, consult the `Troubleshooting`_ section.

There are three different installation types, depending on your use case:

* Installing stable ``skpro`` releases - for most users and production environments.
* Installing the latest unstable ``skpro`` development version - for pre-release tests.
* Full developer setup - for contributors and extension developers.

Each setup is explained below.

Installing release versions
---------------------------

For:

* Most users
* Use in production environments

Installing skpro from PyPI
~~~~~~~~~~~~~~~~~~~~~~~~~~

``skpro`` releases are available via `PyPI`_. To install ``skpro`` with core
dependencies, excluding soft dependencies, via ``pip`` type:

.. code-block:: bash

    pip install skpro

To install ``skpro`` with maximum dependencies, including soft dependencies,
install with the ``all_extras`` modifier:

.. code-block:: bash

    pip install "skpro[all_extras]"

.. warning::

    The soft dependencies included in ``all_extras`` are only necessary to have
    all optional estimators and integrations available, or to run all tests.
    For most user or developer scenarios, installing ``all_extras`` is not
    necessary. If you are unsure, install ``skpro`` with core dependencies and
    install soft dependencies as needed.

Installing skpro from conda
~~~~~~~~~~~~~~~~~~~~~~~~~~~

``skpro`` releases are available via ``conda`` from `conda-forge`_. To install
``skpro`` with core dependencies via ``conda`` type:

.. code-block:: bash

    conda install -c conda-forge skpro

.. note::

    The ``conda-forge`` package can lag behind the latest PyPI release and may
    support a different set of Python versions. Check the `conda-forge package`_
    metadata if you need a specific ``skpro`` or Python version.

Installing latest unstable development version
----------------------------------------------

For:

* pre-release tests, for example early testing of new features
* not for reliable production use
* not for contributors or extenders

This type of ``skpro`` installation obtains a latest static snapshot of the
repository. It is intended for users who want to build or test code using a
version of the library that contains the latest updates.

.. note::

    For a full editable developer setup, read the section
    `Full developer setup for contributors and extension developers`_ below.

To install the latest version of ``skpro`` directly from the repository, use
``pip``:

.. code-block:: bash

    pip install git+https://github.com/sktime/skpro.git

To install from a specific branch, use:

.. code-block:: bash

    pip install git+https://github.com/sktime/skpro.git@<branch_name>

Alternatively, install the latest version from a local clone of the repository.
For steps on how to obtain a local clone, follow the :ref:`git workflow
<git_workflow>`.

.. code-block:: bash

    pip install .

The ``.`` may be replaced with a full or relative path to the root directory of
the local clone.

.. _dev_install:

Full developer setup for contributors and extension developers
--------------------------------------------------------------

For:

* contributors to the ``skpro`` project
* developers of extensions in closed code bases
* developers of 3rd party extensions released as open source

To develop ``skpro`` locally, or to contribute to the project, set up:

* a local clone of the ``skpro`` repository
* a virtual environment with an editable install of ``skpro`` and its developer
  dependencies

The following steps guide you through the process.

1. Follow the :ref:`git workflow <git_workflow>` to fork and clone the
   repository.

2. Set up a new virtual environment. The following commands use ``conda``,
   which tends to be beginner friendly. The process is similar for ``venv`` or
   other virtual environment managers.

   .. warning::

       Using ``conda`` via one of the commercial distributions such as Anaconda
       is in general not free for commercial use and may incur costs or
       liabilities. Consider using free distributions and channels for package
       management, and be aware of applicable terms and conditions.

In the ``conda`` terminal:

3. Navigate to your local ``skpro`` folder:

   .. code-block:: bash

       cd skpro

4. Create a new environment with a supported Python version:

   .. code-block:: bash

       conda create -n skpro-dev python=3.11

   .. warning::

       If you already have an environment called ``skpro-dev`` from a previous
       attempt, remove it first or choose a different environment name.

5. Activate the environment:

   .. code-block:: bash

       conda activate skpro-dev

6. Build an editable version of ``skpro`` with developer dependencies:

   .. code-block:: bash

       pip install -e ".[dev]"

   If you also want to install all optional soft dependencies, install them
   individually after the developer install, or install all of them with:

   .. code-block:: bash

       pip install -e ".[all_extras,dev]"

   If you are working on documentation, install the documentation dependencies:

   .. code-block:: bash

       pip install -e ".[dev,docs]"

7. If everything has worked, you should see a message that ``skpro`` was
   successfully installed.

Troubleshooting
---------------

Module not found
~~~~~~~~~~~~~~~~

The most frequent reason for *module not found* errors is installing ``skpro``
with minimum dependencies and using functionality that requires a soft
dependency. To resolve this, install the missing package, or install ``skpro``
with maximum dependencies as described above.

ImportError
~~~~~~~~~~~

Import errors are often caused by an improperly linked virtual environment.
Make sure that your environment is activated and linked to the IDE or notebook
kernel you are using. If you are using Jupyter notebooks, follow the `Jupyter
virtual environment instructions`_ for adding your virtual environment as a new
kernel.

Other Startup Resources
-----------------------

Virtual environments
~~~~~~~~~~~~~~~~~~~~

Two good options for virtual environment managers are:

* `conda`_ - beginner friendly, but may incur license fees for commercial use
  if using a commercial distribution.
* `venv`_ - included with Python and suitable for many local workflows.

Be sure to link your new virtual environment as the Python kernel in whatever
IDE you are using. For VS Code, see the `VS Code Python environments`_
documentation.

References
----------

The installation instructions are adapted from ``sktime``'s
`installation instructions <https://www.sktime.net/en/stable/installation.html>`_.

.. _precompiled wheels available on PyPI: https://pypi.org/simple/skpro/
.. _PyPI: https://pypi.org/project/skpro/
.. _conda-forge: https://conda-forge.org/
.. _conda-forge package: https://anaconda.org/conda-forge/skpro
.. _Jupyter virtual environment instructions: https://janakiev.com/blog/jupyter-virtual-envs/
.. _conda: https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html
.. _venv: https://docs.python.org/3/library/venv.html
.. _VS Code Python environments: https://code.visualstudio.com/docs/python/environments
