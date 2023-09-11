.. _deps:

============
Dependencies
============

Types of dependencies
=====================

``skpro`` has three types of dependencies:

* "core" dependencies that are required for ``skpro`` to install and run
* "soft" dependencies that are required to import or use specific,
  non-core functionality
* "developer" dependencies are required for developing ``skpro`` but not
  required of end-users (e.g., ``pre-commit``)
* "test" dependencies are required for running ``skpro``'s unit tests

Making it easy to install and use ``skpro`` in a variety of projects is
on of ``skpro``'s goals. Therefore, we seeks to minimizing the number of
dependencies needed to provide the proejct's functionality.

Soft Dependencies
=================

A soft dependency is a dependency that is only required to import
certain modules, but not necessary to use most of the package's functionality.
Accordingly, soft dependencies are not automatically installed alongside
``skpro``. If you want to install soft dependencies, you'll either need
to do so manually or use follow the installation instructions and install
the optional "[all_extras]" version of the package.

Adding a soft dependency
------------------------

The project tries to keep its dependencies, including soft dependencies,
minimized. Core developers will consider the pros and cons of any additional
soft dependency when reviewing Pull Requests. In the event you need to add a new
soft dependency or changing the version of an existing one,
the following files need to be updated:

*  `pyproject.toml <https://github.com/sktime/skpro/blob/main/pyproject.toml>`_,
   adding the dependency or version bounds in the ``all_extras`` dependency set.
   Following the `PEP 621 <https://www.python.org/dev/peps/pep-0621/>`_ convention,
   all dependencies including build time dependencies and optional dependencies
   are specified in this file.

Informative warnings or error messages for missing soft dependencies should be raised,
in a situation where a user would need them. This is handled through our
``_check_soft_dependencies`` `utility
<https://github.com/sktime/skpro/blob/main/skpro/testing/utils/validation/_dependencies.py>`_.

There are specific conventions to add such warnings in ``BaseObject``-s.
To add ``BaseObject`` with a soft dependency, ensure the following:

*  imports of the soft dependency only happen inside the object
   (e.g., a particular methods or in ``__init__`` after calls to
   ``super(cls).__init__``).
*  the ``python_dependencies`` tag of the ``BaseObject`` is populated with a ``str``,
   or a ``list`` of ``str``, of import dependencies. Exceptions will automatically
   raised when constructing the ``BaseObject`` in an environment without the
   required packages.
*  in the python module containing the ``BaseObject``, the
   ``_check_soft_dependencies`` utility is called at the top of the module,
   with ``severity="warning"``. This will raise an informative warning message
   at module import.
*  In a case where the package import differs from the package name (i.e.,
   ``import package_string`` is different from
   ``pip install different-package-string``), the ``_check_soft_dependencies``
   utility should be used in ``__init__``. Both the warning and constructor call
   should use the ``package_import_alias`` argument for this.
*  If the soft dependencies require specific python versions, the ``python_version``
   tag should also be populated, with a PEP 440 compliant version specification
   ``str`` such as ``"<3.10"`` or ``">3.6,~=3.8"``.

Core or developer dependences
=============================

Core and developer dependencies can only be added by core developers after
discussion in a core developer meeting or in the project's communication channels.

When adding a new core dependency or changing the version of an existing one,
the following files need to be updated:

*  `pyproject.toml <https://github.com/sktime/skpro/blob/main/pyproject.toml>`_,
   adding the dependency or version bounds in the ``dependencies`` dependency set.

When adding a new developer dependency or changing the version of an existing one,
the following files need to be updated:

*  `pyproject.toml <https://github.com/sktime/skpro/blob/main/pyproject.toml>`_,
   adding the dependency or version bounds in the ``dev`` dependency set.
