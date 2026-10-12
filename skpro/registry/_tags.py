"""Register of estimator and object tags.

Note for extenders: new tags should be added as classes inheriting from ``_BaseTag``,
in this module. The tag register ``OBJECT_TAG_REGISTER`` is constructed
automatically from these classes. No other place in the code is necessary
to add new tags, but new tags should be added to the API reference,
in ``docs/source/api_reference/tags.rst``.

To add a new tag, define a class inheriting from ``_BaseTag``, as follows:

* the class name should be the tag name, with ``:`` replaced by ``__``,
  e.g., class ``capability__survival`` for the tag ``"capability:survival"``.
* the class docstring is the documentation of the tag, and should follow the
  pattern of the existing tags: a one-line summary, followed by a list with
  string name, tag category, valid values, examples, and default value,
  followed by a longer description of the tag.
* the ``_tags`` dictionary of the class should be filled in as follows:

  - ``"tag_name"``: string, name of the tag as used in the ``_tags`` dictionary
  - ``"parent_type"``: string or list of string, scitype(s) the tag applies to,
    must be in ``get_obj_scitype_list()``
  - ``"tag_type"``: expected type of the tag value, see element 2 of
    ``OBJECT_TAG_REGISTER`` below
  - ``"short_descr"``: string, plain English description of the tag,
    at most 80 characters
  - ``"user_facing"``: bool, whether the tag is user facing (``True``),
    or developer and framework facing only (``False``)

This module exports the following:

---
OBJECT_TAG_REGISTER - list of tuples

each tuple corresponds to a tag, elements as follows:
    0 : string - name of the tag as used in the _tags dictionary
    1 : string - name of the scitype this tag applies to
                 must be in get_obj_scitype_list()
    2 : string - expected type of the tag value
        should be one of:
            "bool" - valid values are True/False
            "int" - valid values are all integers
            "str" - valid values are all strings
            "list" - valid values are all lists of arbitrary elements
            "dict" - valid values are all dictionaries
            ("str", list_of_string) - any string in list_of_string is valid
            ("list", list_of_string) - any individual string and sub-list is valid
            ("list", "str") - any individual string or list of strings is valid
        validity can be checked by check_tag_is_valid (see below)
    3 : string - plain English description of the tag

---

OBJECT_TAG_TABLE - pd.DataFrame
    OBJECT_TAG_REGISTER in table form, as pd.DataFrame
        rows of OBJECT_TABLE correspond to elements in OBJECT_TAG_REGISTER

OBJECT_TAG_LIST - list of string
    elements are 0-th entries of OBJECT_TAG_REGISTER, in same order

---

check_tag_is_valid(tag_name, tag_value) - checks whether tag_value is valid for tag_name
"""


import inspect
import sys

import pandas as pd
from skbase.base import BaseObject

from skpro.registry._base_classes import get_obj_scitype_list

# ---------------------------------------------------------
# Tag Class Definitions
# ---------------------------------------------------------


class _BaseTag(BaseObject):
    """Base class for all tags in ``skpro``.

    Tags are defined as classes inheriting from this class.
    The tag's metadata is stored in the ``_tags`` dictionary of the class,
    and the class docstring is the tag's documentation.

    The register of all tags, ``OBJECT_TAG_REGISTER``,
    is constructed from all classes inheriting from this class.
    """

    _tags = {
        "object_type": "tag",
        "tag_name": "fill_this_in",  # name of the tag used in the _tags dictionary
        "parent_type": "object",  # scitype of the parent object, str or list of str
        "tag_type": "str",  # type of the tag value
        "short_descr": "describe the tag here",  # short tag description, max 80 chars
        "user_facing": True,  # whether the tag is user-facing
    }


# --------------------------
# All objects and estimators
# --------------------------


class reserved_params(_BaseTag):
    """Parameters reserved by the base class and present in all child classes.

    - String name: ``"reserved_params"``
    - Private tag, developer and framework facing
    - Values: list of str, names of parameters
    - Example: ``["index", "columns"]``
    - Default: no reserved parameters (``None``)

    Some base classes in ``skpro`` define parameters that are present
    in all inheriting classes, for instance, ``index`` and ``columns``
    for distributions, or ``multioutput`` and ``score_average`` for metrics.

    The ``reserved_params`` tag of an object is a list of strings,
    the names of the parameters reserved by the base class.

    Reserved parameters may be handled by the base class in a way that deviates
    from ``scikit-learn`` conventions for parameters, for instance,
    distributions coerce ``index`` and ``columns`` to ``pandas`` index objects
    in ``__init__``.

    The tag is used in the ``skpro`` test framework, to exempt reserved
    parameters from checks of constructor and ``set_params`` behaviour,
    and from test parameter coverage checks.
    """

    _tags = {
        "tag_name": "reserved_params",
        "parent_type": "object",
        "tag_type": "list",
        "short_descr": "list of reserved parameter names",
        "user_facing": False,
    }


class object_type(_BaseTag):
    """Scientific type of the object.

    Typing tag for all objects in ``skpro``.

    - String name: ``"object_type"``
    - Public metadata tag
    - Values: string or list of strings
    - Example: ``"regressor_proba"``
    - Example 2: ``["metric", "metric_distr"]`` (polymorphic object)
    - Default: not set, interpreted as ``"object"``

    In ``skpro``, every object has a scientific type (scitype),
    determining the type of object and unified interface,
    e.g., probabilistic regressor, probability distribution, performance metric.

    The ``object_type`` tag of an object is a string, or list of strings,
    specifying the scitype of the object.
    For instance, a probabilistic regressor has scitype ``"regressor_proba"``.

    In case of a list, the object is polymorphic, and can assume (class),
    or simultaneously satisfy different interfaces (object).

    Valid scitypes are listed by ``skpro.registry.get_obj_scitype_list``,
    and the corresponding base classes by ``skpro.registry.get_base_class_register``.

    The full list of scitypes in the current version is:
    """

    _tags = {
        "tag_name": "object_type",
        "parent_type": "object",
        "tag_type": ("list", "str"),
        "short_descr": "type of object, e.g., 'regressor_proba', 'distribution'",
        "user_facing": True,
    }


# dynamically add a pretty printed list of scitypes to the docstring
# guard against docstrings being stripped, e.g., when running python -OO
if object_type.__doc__ is not None:
    for _name, _desc in get_obj_scitype_list(return_descriptions=True):
        object_type.__doc__ += f'\n    - ``"{_name}"``: {_desc}'


class estimator_type(_BaseTag):
    """Type of estimator, legacy tag - use ``object_type`` instead.

    - String name: ``"estimator_type"``
    - Private tag, developer and framework facing
    - Values: string
    - Example: ``"regressor_proba"``
    - Default: ``"estimator"``

    The ``estimator_type`` tag of an estimator is a string,
    specifying the type of the estimator,
    e.g., ``"regressor_proba"`` for probabilistic regressors,
    or ``"distfitter"`` for distribution fitters.

    This is a legacy tag, which is retained for backwards compatibility.
    The type of objects in ``skpro`` is determined by the ``object_type`` tag,
    for instance, via the ``skpro.registry.scitype`` utility.
    """

    _tags = {
        "tag_name": "estimator_type",
        "parent_type": "estimator",
        "tag_type": "str",
        "short_descr": "type of estimator, e.g., 'regressor_proba', legacy tag",
        "user_facing": False,
    }


# Packaging information
# ---------------------


class maintainers(_BaseTag):
    """Current maintainers of the object, GitHub IDs.

    Part of packaging metadata for the object.

    - String name: ``"maintainers"``
    - Public metadata tag
    - Values: string or list of strings
    - Example: ``["fkiraly", "maintainer2"]``
    - Example 2: ``"fkiraly"``
    - Default: ``"skpro developers"``

    The ``maintainers`` tag of an object is a string or list of strings,
    each string being a GitHub handle of a maintainer of the object.

    Maintenance extends to the specific class in ``skpro`` only,
    and not interfaced packages or dependencies.

    Maintainers should be tagged on issues and PRs related to the object.
    For roles and responsibilities in the ``skpro`` project,
    see the governance document, :ref:`governance`.

    To find an object's maintainers, use ``get_tag("maintainers")`` on the object.

    In case of classes not owned by specific algorithm maintainers,
    the tag defaults to the ``skpro`` core team, ``"skpro developers"``.
    """

    _tags = {
        "tag_name": "maintainers",
        "parent_type": "object",
        "tag_type": ("list", "str"),
        "short_descr": "list of current maintainers of the object,"
        " each maintainer a GitHub handle",
        "user_facing": True,
    }


class authors(_BaseTag):
    """Authors of the object, GitHub IDs.

    Part of packaging metadata for the object.

    - String name: ``"authors"``
    - Public metadata tag
    - Values: string or list of strings
    - Example: ``["fkiraly", "author2"]``
    - Example 2: ``"fkiraly"``
    - Default: ``"skpro developers"``

    The ``authors`` tag of an object is a string or list of strings,
    each string being a GitHub handle of an author of the object.

    Authors are credited for the original implementation of the object,
    and contributions to the object.

    In case of light wrappers around third or second party packages,
    author credits should include authors of the wrapped object.

    Authors are not necessarily maintainers of the object,
    and do not need to be tagged on issues and PRs related to the object.

    To find an object's authors, use ``get_tag("authors")`` on the object.
    """

    _tags = {
        "tag_name": "authors",
        "parent_type": "object",
        "tag_type": ("list", "str"),
        "short_descr": "list of authors of the object, each author a GitHub handle",
        "user_facing": True,
    }


class python_version(_BaseTag):
    """Python version requirement specifier for the object (PEP 440).

    Part of packaging metadata for the object.

    - String name: ``"python_version"``
    - Private tag, developer and framework facing
    - Values: str, PEP 440 compliant version specifier
    - Example: ``">=3.10"``
    - Default: no restriction (``None``)

    ``skpro`` manages objects and estimators like mini-packages,
    with their own dependencies and compatibility requirements.
    Dependencies are specified in the tags:

    - ``"python_version"``: Python version specifier (PEP 440) for the object
    - ``"python_dependencies"``: list of required Python packages (PEP 440)
    - ``"env_marker"``: environment marker for the object (PEP 508)

    The ``python_version`` tag of an object is a PEP 440 compliant version specifier
    string, specifying python version compatibility of the object.

    The tag is used in packaging metadata for the object,
    and is used internally to check compatibility of the object with
    the build environment, to raise informative error messages.

    Developers can use ``_check_python_version`` from ``skbase.utils.dependencies``
    to check compatibility of the python constraint of the object
    with the current build environment, or
    ``_check_estimator_deps`` to check compatibility of the object
    (including further checks) with the current build environment.

    See also the developer guide on dependencies, :ref:`deps`.
    """

    _tags = {
        "tag_name": "python_version",
        "parent_type": "object",
        "tag_type": "str",
        "short_descr": "python version specifier (PEP 440) for"
        " estimator, or None = all versions ok",
        "user_facing": False,
    }


class python_dependencies(_BaseTag):
    """Python package dependency requirement specifiers for the object (PEP 440).

    Part of packaging metadata for the object.

    - String name: ``"python_dependencies"``
    - Private tag, developer and framework facing
    - Values: str or list of str, each str a PEP 440 compliant dependency specifier
    - Example: ``"lifelines"``
    - Example 2: ``["scikit-survival>=0.22", "scikit-learn<1.8"]``
    - Default: no requirements beyond ``skpro`` core dependencies (``None``)

    ``skpro`` manages objects and estimators like mini-packages,
    with their own dependencies and compatibility requirements.
    Dependencies are specified in the tags:

    - ``"python_version"``: Python version specifier (PEP 440) for the object
    - ``"python_dependencies"``: list of required Python packages (PEP 440)
    - ``"env_marker"``: environment marker for the object (PEP 508)

    The ``python_dependencies`` tag of an object is string or list of strings,
    each string a PEP 440 compliant version specifier,
    specifying python dependency requirements of the object.

    If passed as a list, conditions are combined with logical AND.
    Optionally, lists within a list can be used to combine conditions with logical OR.

    The tag is used in packaging metadata for the object,
    and is used internally to check compatibility of the object with
    the build environment, to raise informative error messages.

    Valid dependency specifications with plain English descriptions:

    * ``"lifelines"``: ``lifelines`` must be present
    * ``"scikit-survival>=0.22"``: ``scikit-survival`` must be version 0.22 or higher
    * ``["scikit-survival>=0.22", "scikit-learn<1.8"]``: ``scikit-survival``
      must be version 0.22 or higher, and ``scikit-learn`` must be lower than 1.8
    * ``[["xgboost", "lightgbm"], "pandas>=2.0"]``:
      ``pandas`` must be version 2.0 or higher, and at least one of
      ``xgboost`` or ``lightgbm`` must be present

    Developers should note that package names in the PEP 440 specifier strings
    are identical with the package names used in ``pip install`` commands or on PyPI,
    which in general is not the same as the import name of the package,
    e.g., ``"scikit-survival"`` as in ``pip install scikit-survival``,
    and not ``"sksurv"``, as in ``import sksurv``.

    Developers can use ``_check_soft_dependencies`` from ``skbase.utils.dependencies``
    to check compatibility of the python constraint of the object
    with the current build environment, or
    ``_check_estimator_deps`` to check compatibility of the object
    (including further checks) with the current build environment.

    See also the developer guide on dependencies, :ref:`deps`.
    """

    _tags = {
        "tag_name": "python_dependencies",
        "parent_type": "object",
        "tag_type": ("list", "str"),
        "short_descr": "python dependencies of estimator as str or list of str",
        "user_facing": False,
    }


class env_marker(_BaseTag):
    """Environment marker requirement for the object (PEP 508).

    Part of packaging metadata for the object.

    - String name: ``"env_marker"``
    - Private tag, developer and framework facing
    - Values: str, PEP 508 compliant environment marker
    - Example: ``"platform_system == 'Linux'"``
    - Default: no environment marker (``None``)

    ``skpro`` manages objects and estimators like mini-packages,
    with their own dependencies and compatibility requirements.
    Dependencies are specified in the tags:

    - ``"python_version"``: Python version specifier (PEP 440) for the object
    - ``"python_dependencies"``: list of required Python packages (PEP 440)
    - ``"env_marker"``: environment marker for the object (PEP 508)

    The ``env_marker`` tag of an object is a string,
    specifying a PEP 508 compliant environment marker for the object,
    for instance, to restrict the object to specific operating systems.

    The tag is used in packaging metadata for the object,
    and is used internally to check compatibility of the object with
    the build environment, to raise informative error messages.

    Developers can use ``_check_env_marker`` from ``skbase.utils.dependencies``
    to check compatibility of the environment marker of the object
    with the current build environment, or
    ``_check_estimator_deps`` to check compatibility of the object
    (including further checks) with the current build environment.
    Checks of the ``env_marker`` tag require ``scikit-base`` 0.8.2 or later.
    """

    _tags = {
        "tag_name": "env_marker",
        "parent_type": "object",
        "tag_type": "str",
        "short_descr": "environment marker (PEP 508) requirement for"
        " estimator, or None = no marker",
        "user_facing": False,
    }


class python_dependencies_alias(_BaseTag):
    """Import names of python dependencies, legacy tag.

    Part of packaging metadata for the object.

    - String name: ``"python_dependencies_alias"``
    - Private tag, developer and framework facing
    - Values: dict, with str keys and str values
    - Example: ``{"scikit-survival": "sksurv"}``
    - Default: no aliases (``None``)

    The ``python_dependencies_alias`` tag of an object is a dictionary,
    mapping package names as used in the ``python_dependencies`` tag,
    i.e., names as in ``pip install``, to import names of the packages,
    for packages where the two differ.
    For instance, ``scikit-survival`` is imported as ``sksurv``.

    This is a legacy tag, retained for backwards compatibility.
    It is not used by the ``skpro`` framework, or by current versions
    of the dependency checkers in ``skbase.utils.dependencies``,
    which identify packages by their distribution name, as in ``pip install``.
    New objects do not need to set this tag.
    """

    _tags = {
        "tag_name": "python_dependencies_alias",
        "parent_type": "object",
        "tag_type": "dict",
        "short_descr": "should be provided if import name differs from package name",
        "user_facing": False,
    }


class license_type(_BaseTag):
    """License type of packages interfaced by the object.

    Part of packaging metadata for the object.

    - String name: ``"license_type"``
    - Public metadata tag
    - Values: str, one of ``"permissive"``, ``"copyleft"``, ``"copyright"``
    - Example: ``"copyleft"``
    - Default: not set (``None``)

    Some objects in ``skpro`` interface third party packages, whose license
    may differ from the license of ``skpro``.

    The ``license_type`` tag of an object is a string,
    specifying the type of license of the interfaced package:

    * ``"permissive"``: permissive open source license, e.g., BSD, MIT, Apache
    * ``"copyleft"``: copyleft open source license, e.g., GPL
    * ``"copyright"``: other licenses, e.g., proprietary licenses

    The tag is informative, for users to identify the license type of
    packages that are used when using the object.
    """

    _tags = {
        "tag_name": "license_type",
        "parent_type": "object",
        "tag_type": "str",
        "short_descr": "license type for interfaced"
        " packages: 'copyleft', 'permissive', 'copyright'",
        "user_facing": True,
    }


# CI and test flags
# -----------------


class tests__libs(_BaseTag):
    """Important library dependencies of the object, for test triggers.

    Part of packaging metadata for the object, used only in ``skpro`` CI.

    - String name: ``"tests:libs"``
    - Private tag, developer and framework facing
    - Values: list of str, or None
    - Example: ``["skpro.libs.cyclic_boosting"]``
    - Default: ``None``

    ``skpro``'s CI framework regularly tests estimators in pull requests,
    usually only estimators that have changed, via ``run_test_for_class``.

    The ``tests:libs`` tag of an object is a list of strings,
    it specifies important library dependencies of the object within ``skpro``,
    as module names.

    Setting this tag triggers testing the estimator whenever any of the modules
    in the ``tests:libs`` tag have changed, in addition to the other
    test trigger conditions such as a direct change to the object class.

    Developers should not specify framework imports here, e.g., ``skpro.base``,
    but any modules that contain estimator specific logic, which are not
    identical with the location of the class.

    The ``tests:libs`` tag is not used in user facing checks, error messages,
    or recommended build processes otherwise.
    """

    _tags = {
        "tag_name": "tests:libs",
        "parent_type": "object",
        "tag_type": ("list", "str"),
        "short_descr": "list of library dependencies required for tests",
        "user_facing": False,
    }


class tests__vm(_BaseTag):
    """Whether to spin up a separate VM to test the estimator.

    Part of packaging metadata for the object, used only in ``skpro`` CI.

    - String name: ``"tests:vm"``
    - Private tag, developer and framework facing
    - Values: boolean, ``True`` / ``False``
    - Example: ``True``
    - Default: ``False``

    ``skpro``'s CI framework regularly tests estimators in pull requests,
    usually only estimators that have changed, via ``run_test_for_class``.

    The ``tests:vm`` tag of an object is a boolean,
    it specifies whether the estimator should be tested in a separate VM,
    with a fresh environment set up using the ``python_dependencies`` tag,
    and the ``tests:python_dependencies`` tag.
    Estimators with the tag set to ``True`` are excluded from the main test suite,
    and tested only in their own VM.

    This tag should be set to ``True`` for estimators that have a complex
    dependency setup, or that are known to have issues with the default
    ``skpro`` CI environment.
    It can also be used for estimators with soft dependencies that occur
    only in one or few specific estimators.
    Otherwise, it should be used sparingly.

    The ``tests:vm`` tag is not used in user facing checks, error messages,
    or recommended build processes otherwise.
    """

    _tags = {
        "tag_name": "tests:vm",
        "parent_type": "object",
        "tag_type": "bool",
        "short_descr": "whether tests require their own VM to run",
        "user_facing": False,
    }


class tests__skip_by_name(_BaseTag):
    """A list of test names that should be skipped for this object.

    Part of packaging metadata for the object, used only in ``skpro`` CI.

    - String name: ``"tests:skip_by_name"``
    - Private tag, developer and framework facing
    - Values: list of str, or None
    - Example: ``["test_class_has_doctest_example"]``
    - Default: ``None``

    ``skpro``'s CI framework regularly tests estimators in pull requests,
    usually only estimators that have changed, via ``run_test_for_class``.

    The ``tests:skip_by_name`` tag of an object is list of strings,
    with strings being names of tests that should be skipped for the object.
    The names should be the same as names of test functions in the "test all"
    suite, and will be the same as test names in ``check_estimator`` returns.
    If set to ``None`` (default), no tests are skipped.

    WARNING: this tag should be used with caution.
    When it is set, developers should leave a comment
    next to the tag, explaining why the tests are skipped,
    and optimally link from the comment to an open issue with the purpose
    of resolving the skipped test(s).

    The ``tests:skip_by_name`` tag is not used in user facing checks, error messages,
    or recommended build processes otherwise.
    """

    _tags = {
        "tag_name": "tests:skip_by_name",
        "parent_type": "object",
        "tag_type": ("list", "str"),
        "short_descr": "list of test names to skip when running estimator checks on CI",
        "user_facing": False,
    }


class tests__python_dependencies(_BaseTag):
    """Python package dependency requirement specifiers for tests (PEP 440).

    Part of packaging metadata for the object, used only in ``skpro`` CI.

    - String name: ``"tests:python_dependencies"``
    - Private tag, developer and framework facing
    - Values: str or list of str, each str a PEP 440 compliant dependency specifier
    - Example: ``"lifelines"``
    - Example 2: ``["lifelines", "scikit-survival>=0.22"]``
    - Default: no requirements beyond ``python_dependencies`` (``None``)

    ``skpro``'s CI framework regularly tests estimators in pull requests.

    The ``tests:python_dependencies`` tag specifies additional environment
    dependencies required for testing the object, in a VM setup,
    via the ``tests:vm`` tag.

    These dependencies will not be highlighted to the user when using the
    estimator, and are used only in the CI testing setup.

    This tag should be used, for example, if the test instances in
    ``get_test_params`` require additional packages not required for the main
    functionality of the object.

    The ``tests:python_dependencies`` tag of an object is a string,
    a list of strings, or a nested list of strings, with same format, convention,
    and meaning as ``python_dependencies``.

    It is developer facing only, and is not used in user facing checks,
    error messages, or recommended build processes otherwise.
    """

    _tags = {
        "tag_name": "tests:python_dependencies",
        "parent_type": "object",
        "tag_type": ("list", "str"),
        "short_descr": "additional python dependencies needed in tests (PEP 440)",
        "user_facing": False,
    }


# ------------------
# BaseProbaRegressor
# ------------------


class capability__survival(_BaseTag):
    """Capability: the object can use censoring information, for survival analysis.

    - String name: ``"capability:survival"``
    - Public capability tag
    - Values: boolean, ``True`` / ``False``
    - Example: ``True``
    - Default: ``False``

    This tag applies to probabilistic regressors, distribution fitters,
    and performance metrics.

    If the tag is ``True``, the object can make use of censoring information,
    i.e., the object is suitable for survival analysis, or time-to-event
    prediction with right censored data.
    Censoring information is passed as argument ``C`` to ``fit`` and ``update``
    of regressors and distribution fitters, and as argument ``C_true``
    to metrics.

    If the tag is ``False``, the object assumes all observations to be uncensored,
    and ignores censoring information if passed.

    Probabilistic regressors with this tag set to ``True`` are also called
    survival regressors. They can be listed via
    ``all_objects("regressor_proba", filter_tags={"capability:survival": True})``.
    """

    _tags = {
        "tag_name": "capability:survival",
        "parent_type": ["regressor_proba", "metric", "distfitter"],
        "tag_type": "bool",
        "short_descr": "whether estimator can use censoring information,"
        " for survival analysis",
        "user_facing": True,
    }


class capability__multioutput(_BaseTag):
    """Capability: the regressor can handle multi-output targets.

    - String name: ``"capability:multioutput"``
    - Public capability tag
    - Values: boolean, ``True`` / ``False``
    - Example: ``True``
    - Default: ``False``

    This tag applies to probabilistic regressors.

    If the tag is ``True``, the regressor can handle multivariate targets natively,
    i.e., the target ``y`` passed to ``fit`` may have more than one column,
    and predictions are made for all columns of ``y``.

    If the tag is ``False``, the regressor is designed for univariate targets only,
    i.e., ``y`` with a single column.

    This condition is specific to target data ``y``,
    the features ``X`` may have multiple columns in either case.
    """

    _tags = {
        "tag_name": "capability:multioutput",
        "parent_type": "regressor_proba",
        "tag_type": "bool",
        "short_descr": "whether estimator supports multioutput regression",
        "user_facing": True,
    }


class capability__missing(_BaseTag):
    """Capability: the regressor can handle missing values in features ``X``.

    - String name: ``"capability:missing"``
    - Public capability tag
    - Values: boolean, ``True`` / ``False``
    - Example: ``False``
    - Default: ``True``

    This tag applies to probabilistic regressors.

    If the tag is ``True``, the regressor can handle missing values,
    e.g., ``np.nan``, in the features ``X`` passed to ``fit``, ``predict``,
    and other prediction methods.

    If the tag is ``False``, the regressor cannot handle missing values in ``X``,
    and may raise an error if missing values are present.
    """

    _tags = {
        "tag_name": "capability:missing",
        "parent_type": "regressor_proba",
        "tag_type": "bool",
        "short_descr": "whether estimator supports missing values",
        "user_facing": True,
    }


class capability__update(_BaseTag):
    """Capability: the regressor can be updated with new data, on-line learning.

    - String name: ``"capability:update"``
    - Public capability tag
    - Values: boolean, ``True`` / ``False``
    - Example: ``True``
    - Default: ``False``

    This tag applies to probabilistic regressors.

    The tag specifies whether the regressor can be run in stream or on-line mode,
    with an ``update`` method. Depending on the context, literature
    may refer to this as on-line learning, incremental learning, or stream learning.

    If the tag is ``True``, the ``update`` method is implemented and can be used
    to update the fitted regressor with a new batch of data.

    If the tag is ``False``, calling ``update`` has no effect,
    the call is ignored and the data passed is discarded.

    Compositors in ``skpro.regression.online`` can be used to add
    on-line learning capabilities to regressors without the capability,
    for instance, by re-fitting the regressor on all data seen so far.
    """

    _tags = {
        "tag_name": "capability:update",
        "parent_type": "regressor_proba",
        "tag_type": "bool",
        "short_descr": "whether estimator supports online updates via update",
        "user_facing": True,
    }


class X_inner_mtype(_BaseTag):
    """The machine type(s) the estimator can deal with internally for ``X``.

    - String name: ``"X_inner_mtype"``
    - Extension developer tag
    - Values: str or list of str, from the list of ``"Table"`` mtype strings
    - Example: ``"pd_DataFrame_Table"``
    - Example 2: ``["pd_DataFrame_Table", "numpy2D"]``
    - Default: ``"pd_DataFrame_Table"``

    This tag applies to probabilistic regressors and distribution fitters.

    Estimators in ``skpro`` support a variety of input data types, following
    one of many possible machine types, short: mtype specifications,
    e.g., ``pandas.DataFrame``, ``numpy.ndarray``, or ``polars.DataFrame``.

    Internally, the estimator may support only a subset of these types,
    for instance due to the implementation of the estimator, or due to
    interfacing with external libraries that use a specific data format.

    The ``skpro`` extension contracts allow the extender to specify the
    internal mtype support, in this case the boilerplate layer of the base class
    guarantees that arguments passed to the methods the extender implements,
    such as ``_fit`` and ``_predict``, are of the correct type,
    by carrying out the necessary conversions.

    For instance, an extender implementing ``_fit`` with an ``X`` argument
    and ``X_inner_mtype`` set to ``"pd_DataFrame_Table"`` can assume that the ``X``
    argument is a ``pandas.DataFrame``, while users can pass any supported mtype
    to the public ``fit`` method.

    If a list of mtypes is specified, inputs with an mtype on the list
    are passed through without conversion, and inputs with any other mtype
    are converted to the first mtype in the list.

    Tags named ``X_inner_mtype``, ``y_inner_mtype``, and ``C_inner_mtype``
    apply this specification to the respective arguments in the method signatures.

    Valid mtype strings are listed in ``skpro.datatypes.MTYPE_REGISTER``.
    """

    _tags = {
        "tag_name": "X_inner_mtype",
        "parent_type": ["regressor_proba", "distfitter"],
        "tag_type": ("list", "str"),
        "short_descr": "which machine type(s) is the"
        " internal _fit/_predict able to deal with?",
        "user_facing": False,
    }


class y_inner_mtype(_BaseTag):
    """The machine type(s) the estimator can deal with internally for ``y``.

    - String name: ``"y_inner_mtype"``
    - Extension developer tag
    - Values: str or list of str, from the list of ``"Table"`` mtype strings
    - Example: ``"pd_DataFrame_Table"``
    - Example 2: ``["pd_DataFrame_Table", "numpy2D"]``
    - Default: ``"pd_DataFrame_Table"``

    This tag applies to probabilistic regressors.

    The ``y_inner_mtype`` tag specifies the machine type(s) of the target ``y``
    passed to the methods the extender implements, such as ``_fit`` and ``_update``.
    Point predictions returned by ``_predict`` should also be of this mtype,
    they are converted back to the mtype of ``y`` seen in ``fit`` by the base class.

    For details on inner mtype tags, see the ``X_inner_mtype`` tag.
    """

    _tags = {
        "tag_name": "y_inner_mtype",
        "parent_type": "regressor_proba",
        "tag_type": ("list", "str"),
        "short_descr": "which machine type(s) is the"
        " internal _fit/_predict able to deal with?",
        "user_facing": False,
    }


class C_inner_mtype(_BaseTag):
    """The machine type(s) the estimator can deal with internally for ``C``.

    - String name: ``"C_inner_mtype"``
    - Extension developer tag
    - Values: str or list of str, from the list of ``"Table"`` mtype strings
    - Example: ``"pd_DataFrame_Table"``
    - Example 2: ``["pd_DataFrame_Table", "numpy2D"]``
    - Default: ``"pd_DataFrame_Table"``

    This tag applies to probabilistic regressors and distribution fitters.

    The ``C_inner_mtype`` tag specifies the machine type(s) of the censoring
    information ``C`` passed to the methods the extender implements,
    such as ``_fit`` and ``_update``.
    It is used only if the ``capability:survival`` tag is ``True``,
    otherwise ``C`` is ignored.

    For details on inner mtype tags, see the ``X_inner_mtype`` tag.
    """

    _tags = {
        "tag_name": "C_inner_mtype",
        "parent_type": ["regressor_proba", "distfitter"],
        "tag_type": ("list", "str"),
        "short_descr": "which machine type(s) is the "
        "internal _fit/_predict able to deal with?",
        "user_facing": False,
    }


# ----------------
# BaseDistribution
# ----------------


class capabilities__approx(_BaseTag):
    """Methods of the distribution that return approximate results.

    - String name: ``"capabilities:approx"``
    - Public capability tag
    - Values: list of str, names of methods of the distribution
    - Example: ``["energy", "pdfnorm"]``
    - Default: ``["energy", "mean", "var", "pdfnorm"]``

    This tag applies to probability distributions.

    The ``capabilities:approx`` tag of a distribution is a list of strings,
    the names of public methods of the distribution, such as ``"mean"``,
    ``"var"``, ``"energy"``, ``"pdf"``, ``"cdf"``, or ``"ppf"``,
    which are available, but return approximate results,
    e.g., computed by Monte Carlo sampling or numerical integration.

    Methods that return numerically exact results are listed in the
    ``capabilities:exact`` tag. Methods listed in neither tag are not considered
    to be supported by the distribution.

    The precision of default approximations can be controlled by the tags
    ``approx_mean_spl``, ``approx_var_spl``, ``approx_energy_spl``,
    ``approx_spl``, and ``bisect_iter``.
    """

    _tags = {
        "tag_name": "capabilities:approx",
        "parent_type": "distribution",
        "tag_type": ("list", "str"),
        "short_descr": "methods of distr that are approximate",
        "user_facing": True,
    }


class capabilities__exact(_BaseTag):
    """Methods of the distribution that return numerically exact results.

    - String name: ``"capabilities:exact"``
    - Public capability tag
    - Values: list of str, names of methods of the distribution
    - Example: ``["mean", "var", "pdf", "log_pdf", "cdf", "ppf"]``
    - Default: not set, no exact methods

    This tag applies to probability distributions.

    The ``capabilities:exact`` tag of a distribution is a list of strings,
    the names of public methods of the distribution, such as ``"mean"``,
    ``"var"``, ``"energy"``, ``"pdf"``, ``"cdf"``, or ``"ppf"``,
    which return numerically exact results,
    e.g., computed from closed form expressions.

    Methods that are available, but return approximate results, are listed in the
    ``capabilities:approx`` tag. Methods listed in neither tag are not considered
    to be supported by the distribution.
    """

    _tags = {
        "tag_name": "capabilities:exact",
        "parent_type": "distribution",
        "tag_type": ("list", "str"),
        "short_descr": "methods of distr that are numerically exact",
        "user_facing": True,
    }


class capabilities__undefined(_BaseTag):
    """Methods of the distribution that are mathematically undefined.

    - String name: ``"capabilities:undefined"``
    - Public capability tag
    - Values: list of str, names of methods of the distribution
    - Example: ``["mean", "var"]``
    - Default: not set, no undefined methods

    This tag applies to probability distributions.

    For some distributions, certain quantities are mathematically undefined,
    for instance, the Cauchy distribution has neither mean nor variance.

    The ``capabilities:undefined`` tag of a distribution is a list of strings,
    the names of public methods whose return is mathematically undefined
    for the distribution, e.g., ``["mean", "var"]`` for the Cauchy distribution.

    The tag is used in the ``skpro`` test framework, where only methods listed
    in this tag are allowed to return values that are not finite numbers.
    """

    _tags = {
        "tag_name": "capabilities:undefined",
        "parent_type": "distribution",
        "tag_type": ("list", "str"),
        "short_descr": "methods of distr that are mathematically undefined",
        "user_facing": True,
    }


class distr__measuretype(_BaseTag):
    """Measure type of the distribution - continuous, discrete, or mixed.

    - String name: ``"distr:measuretype"``
    - Public property tag
    - Values: str, one of ``"continuous"``, ``"discrete"``, ``"mixed"``
    - Example: ``"continuous"``
    - Default: ``"mixed"``

    This tag applies to probability distributions.

    The ``distr:measuretype`` tag of a distribution specifies the type
    of the distribution's measure:

    * ``"continuous"``: absolutely continuous distribution, with a density,
      available via ``pdf`` and ``log_pdf``. The probability mass function
      ``pmf`` is zero everywhere.
    * ``"discrete"``: discrete distribution, with a probability mass function,
      available via ``pmf`` and ``log_pmf``. The density ``pdf`` is zero
      everywhere.
    * ``"mixed"``: mixed or other measure type, e.g., with both continuous
      and discrete components.

    The tag is used by the base class to determine the default behaviour
    of methods such as ``pdf``, ``pmf``, and plotting.
    """

    _tags = {
        "tag_name": "distr:measuretype",
        "parent_type": "distribution",
        "tag_type": ("str", ["continuous", "discrete", "mixed"]),
        "short_descr": "measure type of distr",
        "user_facing": True,
    }


class distr__paramtype(_BaseTag):
    """Parametrization type of the distribution.

    - String name: ``"distr:paramtype"``
    - Public property tag
    - Values: str, one of ``"general"``, ``"parametric"``, ``"nonparametric"``,
      ``"composite"``
    - Example: ``"parametric"``
    - Default: ``"general"``

    This tag applies to probability distributions.

    The ``distr:paramtype`` tag of a distribution specifies the type of
    parametrization of the distribution:

    * ``"parametric"``: parametric distribution, with a fixed number of
      numerical parameters, e.g., ``Normal`` with parameters ``mu`` and ``sigma``
    * ``"nonparametric"``: nonparametric distribution, parametrized by data
      such as a sample, e.g., ``Empirical``
    * ``"composite"``: distribution composed from other distribution objects,
      e.g., ``Mixture`` or ``IID``
    * ``"general"``: other or unspecified parametrization type

    Only distributions with value ``"parametric"`` support access to
    parameters in data frame format, via ``get_params_df`` and ``to_df``.
    """

    _tags = {
        "tag_name": "distr:paramtype",
        "parent_type": "distribution",
        "tag_type": ("str", ["general", "parametric", "nonparametric", "composite"]),
        "short_descr": "parametrization type of distribution",
        "user_facing": True,
    }


class approx_mean_spl(_BaseTag):
    """Sample size used in approximations of the mean.

    - String name: ``"approx_mean_spl"``
    - Public configuration tag
    - Values: int, positive
    - Example: ``10000``
    - Default: ``1000``

    This tag applies to probability distributions.

    If the mean of a distribution is not available in closed form,
    the default implementation of ``mean`` approximates it,
    by integrating ``ppf`` with ``approx_mean_spl`` equidistant nodes,
    if ``ppf`` is available, otherwise by the arithmetic mean of
    ``approx_mean_spl`` samples.

    Larger values increase the precision of the approximation,
    at the cost of computation time.
    The value can be changed via ``set_tags``.
    """

    _tags = {
        "tag_name": "approx_mean_spl",
        "parent_type": "distribution",
        "tag_type": "int",
        "short_descr": "sample size used in MC estimates of mean",
        "user_facing": True,
    }


class approx_var_spl(_BaseTag):
    """Sample size used in approximations of the variance.

    - String name: ``"approx_var_spl"``
    - Public configuration tag
    - Values: int, positive
    - Example: ``10000``
    - Default: ``1000``

    This tag applies to probability distributions.

    If the variance of a distribution is not available in closed form,
    the default implementation of ``var`` approximates it,
    by integration with ``approx_var_spl`` equidistant nodes,
    if ``ppf`` is available, otherwise by the arithmetic mean of
    ``approx_var_spl`` squared differences of samples.

    Larger values increase the precision of the approximation,
    at the cost of computation time.
    The value can be changed via ``set_tags``.
    """

    _tags = {
        "tag_name": "approx_var_spl",
        "parent_type": "distribution",
        "tag_type": "int",
        "short_descr": "sample size used in MC estimates of var",
        "user_facing": True,
    }


class approx_energy_spl(_BaseTag):
    """Sample size used in approximations of the energy.

    - String name: ``"approx_energy_spl"``
    - Public configuration tag
    - Values: int, positive
    - Example: ``10000``
    - Default: ``1000``

    This tag applies to probability distributions.

    If the energy of a distribution is not available in closed form,
    the default implementation of ``energy`` approximates it,
    by integration with ``approx_energy_spl`` equidistant nodes,
    if ``ppf`` is available, otherwise by the arithmetic mean of
    ``approx_energy_spl`` samples.

    Larger values increase the precision of the approximation,
    at the cost of computation time.
    The value can be changed via ``set_tags``.
    """

    _tags = {
        "tag_name": "approx_energy_spl",
        "parent_type": "distribution",
        "tag_type": "int",
        "short_descr": "sample size used in MC estimates of energy",
        "user_facing": True,
    }


class approx_spl(_BaseTag):
    """Sample size used in other approximations, e.g., of ``pdfnorm``.

    - String name: ``"approx_spl"``
    - Public configuration tag
    - Values: int, positive
    - Example: ``10000``
    - Default: ``1000``

    This tag applies to probability distributions.

    The ``approx_spl`` tag specifies the sample size used in default
    Monte Carlo approximations of quantities not covered by the more specific
    tags ``approx_mean_spl``, ``approx_var_spl``, and ``approx_energy_spl``,
    for instance, in the default implementation of ``pdfnorm``.

    Larger values increase the precision of the approximation,
    at the cost of computation time.
    The value can be changed via ``set_tags``.
    """

    _tags = {
        "tag_name": "approx_spl",
        "parent_type": "distribution",
        "tag_type": "int",
        "short_descr": "sample size used in other MC estimates",
        "user_facing": True,
    }


class bisect_iter(_BaseTag):
    """Maximum number of iterations of the bisection method used in ``ppf``.

    - String name: ``"bisect_iter"``
    - Public configuration tag
    - Values: int, positive
    - Example: ``10000``
    - Default: ``1000``

    This tag applies to probability distributions.

    If the quantile function ``ppf`` of a distribution is not available
    in closed form, but ``cdf`` is, the default implementation of ``ppf``
    inverts ``cdf`` numerically, via the bisection method,
    with at most ``bisect_iter`` iterations.

    Larger values increase the precision of the approximation,
    at the cost of computation time.
    The value can be changed via ``set_tags``.
    """

    _tags = {
        "tag_name": "bisect_iter",
        "parent_type": "distribution",
        "tag_type": "int",
        "short_descr": "max iters for bisection method in ppf",
        "user_facing": True,
    }


class broadcast_params(_BaseTag):
    """Distribution parameters to broadcast to the shape of the distribution.

    - String name: ``"broadcast_params"``
    - Extension developer tag
    - Values: list of str, names of parameters, or ``None``
    - Example: ``["mu", "sigma"]``
    - Default: ``None``, all parameters are broadcast

    This tag applies to probability distributions.

    Array-valued distributions in ``skpro`` have a 2D shape,
    and parameters are broadcast to this shape,
    following ``numpy`` broadcasting rules.

    The ``broadcast_params`` tag of a distribution is a list of strings,
    the names of parameters that are broadcast together with ``index``
    and ``columns``, to determine the shape of the distribution.
    If ``None``, all parameters of the distribution are broadcast.

    Parameters not listed, for instance non-numerical parameters,
    are not broadcast.
    """

    _tags = {
        "tag_name": "broadcast_params",
        "parent_type": "distribution",
        "tag_type": ("list", "str"),
        "short_descr": "distribution parameters to broadcast",
        "user_facing": False,
    }


class broadcast_init(_BaseTag):
    """Whether to broadcast parameters and infer shape in ``__init__``.

    - String name: ``"broadcast_init"``
    - Extension developer tag
    - Values: str, one of ``"on"``, ``"off"``
    - Example: ``"on"``
    - Default: ``"off"``

    This tag applies to probability distributions.

    If ``"on"``, the base class ``__init__`` broadcasts the parameters in
    ``broadcast_params``, infers the shape of the distribution from them,
    and sets ``index`` and ``columns`` to ``RangeIndex`` defaults if not passed.

    If ``"off"``, no broadcasting is carried out in ``__init__``,
    and the shape of the distribution is determined by ``index`` and ``columns``.
    This is used by distributions deviating from assumptions on input
    parameters, e.g., ``Empirical``.

    Extenders should set the tag to ``"on"``, as in the extension template,
    unless parameters of the distribution are not array-like.
    """

    _tags = {
        "tag_name": "broadcast_init",
        "parent_type": "distribution",
        "tag_type": ("str", ["on", "off"]),
        "short_descr": "whether to initialize broadcast parameters in __init__",
        "user_facing": False,
    }


class broadcast_inner(_BaseTag):
    """Whether inner methods of the distribution are vectorized or scalar.

    - String name: ``"broadcast_inner"``
    - Extension developer tag
    - Values: str, one of ``"array"``, ``"scalar"``
    - Example: ``"array"``
    - Default: ``"array"``

    This tag applies to probability distributions.

    The ``broadcast_inner`` tag specifies whether the private methods
    implementing the distribution logic, e.g., ``_pdf`` or ``_mean``,
    are vectorized over array-valued parameters (``"array"``),
    or assume scalar parameters (``"scalar"``),
    with broadcasting carried out in the boilerplate layer.

    Currently, all distributions in ``skpro`` use the value ``"array"``,
    and the base class does not change behaviour based on this tag.
    """

    _tags = {
        "tag_name": "broadcast_inner",
        "parent_type": "distribution",
        "tag_type": ("str", ["array", "scalar"]),
        "short_descr": "if inner logic is vectorized ('array') or scalar ('scalar')",
        "user_facing": False,
    }


# ---------------
# BaseProbaMetric
# ---------------


class scitype__y_pred(_BaseTag):
    """The type of probabilistic prediction expected by the metric, for ``y_pred``.

    - String name: ``"scitype:y_pred"``
    - Public property tag
    - Values: str, one of ``"pred_proba"``, ``"pred_interval"``,
      ``"pred_quantiles"``
    - Example: ``"pred_interval"``
    - Default: ``"pred_proba"``

    This tag applies to performance metrics.

    The tag specifies the type of the predicted target data ``y_pred``
    expected by the metric:

    * ``"pred_proba"``: distribution predictions, as returned by ``predict_proba``
    * ``"pred_interval"``: interval predictions, as returned by ``predict_interval``
    * ``"pred_quantiles"``: quantile predictions, as returned by
      ``predict_quantiles``

    The tag is used, for instance, in benchmarking, to determine which prediction
    method of a regressor is called to produce ``y_pred`` for the metric.
    """

    _tags = {
        "tag_name": "scitype:y_pred",
        "parent_type": "metric",
        "tag_type": "str",
        "short_descr": "expected input type for y_pred in performance metric",
        "user_facing": True,
    }


class lower_is_better(_BaseTag):
    """Property: whether lower metric values are better.

    - String name: ``"lower_is_better"``
    - Public property tag
    - Values: boolean, ``True`` / ``False``
    - Example: ``True``
    - Default: ``True``

    This tag applies to performance metrics.

    If the tag is ``True``, lower values of the metric are considered better,
    i.e., the metric is a loss.
    If the tag is ``False``, higher values of the metric are considered better,
    i.e., the metric is a score.

    The tag is used, for instance, in tuning, to determine the best
    parameter setting.
    """

    _tags = {
        "tag_name": "lower_is_better",
        "parent_type": "metric",
        "tag_type": "bool",
        "short_descr": "whether lower (True) or higher (False) is better",
        "user_facing": True,
    }


# ----------------------------
# BaseMetaObject reserved tags
# ----------------------------


class named_object_parameters(_BaseTag):
    """Name of the attribute containing the components of a meta-object.

    - String name: ``"named_object_parameters"``
    - Extension developer tag
    - Values: str, name of an attribute of the object
    - Example: ``"_steps"``
    - Default: ``"steps"``

    This tag applies to meta-objects inheriting from ``BaseMetaObject``
    or ``BaseMetaEstimator`` from ``skbase``,
    e.g., pipelines or ensembles with a list of named components.

    The ``named_object_parameters`` tag specifies the name of the attribute
    containing the components, as a list of ``(name, object)`` tuples.
    ``skbase`` uses this attribute in ``get_params`` and ``set_params``
    of meta-objects, to expose components and their parameters.
    """

    _tags = {
        "tag_name": "named_object_parameters",
        "parent_type": "object",
        "tag_type": "str",
        "short_descr": "name of component list attribute for meta-objects",
        "user_facing": False,
    }


class fitted_named_object_parameters(_BaseTag):
    """Name of the attribute containing the fitted components of a meta-estimator.

    - String name: ``"fitted_named_object_parameters"``
    - Extension developer tag
    - Values: str, name of an attribute of the object
    - Example: ``"steps_"``
    - Default: not set

    This tag applies to meta-estimators inheriting from ``BaseMetaEstimator``
    from ``skbase``, e.g., pipelines or ensembles with a list of named components.

    The ``fitted_named_object_parameters`` tag specifies the name of the attribute
    containing the fitted components, as a list of ``(name, estimator)`` tuples.
    ``skbase`` uses this attribute in ``get_fitted_params`` of meta-estimators,
    to expose fitted components and their fitted parameters.
    """

    _tags = {
        "tag_name": "fitted_named_object_parameters",
        "parent_type": "estimator",
        "tag_type": "str",
        "short_descr": "name of fitted component list attribute for meta-objects",
        "user_facing": False,
    }


# ---------------------------------------------------------
# Registry Generation Logic
# ---------------------------------------------------------


def _construct_tag_register():
    """Construct the tag register from all tag classes in this module.

    Done in a function, to avoid loop variables leaking into the module namespace,
    which would make the last tag class visible under a second name.
    """
    register = []
    tag_classes = inspect.getmembers(sys.modules[__name__], inspect.isclass)

    for _, cl in tag_classes:
        if cl.__name__ == "_BaseTag" or not issubclass(cl, _BaseTag):
            continue

        cl_tags = cl.get_class_tags()
        tag_name = cl_tags.get("tag_name", "unknown_tag")
        parent_type = cl_tags.get("parent_type", "object")
        tag_type = cl_tags.get("tag_type", "str")
        short_descr = cl_tags.get("short_descr", "")

        if isinstance(parent_type, list):
            for p_type in parent_type:
                register.append((tag_name, p_type, tag_type, short_descr))
        else:
            register.append((tag_name, parent_type, tag_type, short_descr))

    return register


OBJECT_TAG_REGISTER = _construct_tag_register()
OBJECT_TAG_TABLE = pd.DataFrame(OBJECT_TAG_REGISTER)
OBJECT_TAG_LIST = OBJECT_TAG_TABLE[0].unique().tolist()


def check_tag_is_valid(tag_name, tag_value):
    """Check validity of a tag value.

    Parameters
    ----------
    tag_name : string, name of the tag
    tag_value : object, value of the tag

    Raises
    ------
    KeyError - if tag_name is not a valid tag in OBJECT_TAG_LIST
    ValueError - if the tag_valid is not a valid for the tag with name tag_name
    """
    if tag_name not in OBJECT_TAG_LIST:
        raise KeyError(f"{tag_name} is not a valid tag")

    tag_row = OBJECT_TAG_TABLE[OBJECT_TAG_TABLE[0] == tag_name]
    tag_type = tag_row.iloc[0, 2]

    # Validation logic for strings/types
    if isinstance(tag_type, str):
        if tag_type == "bool" and not isinstance(tag_value, bool):
            raise ValueError(f"{tag_name} must be bool, found {type(tag_value)}")
        if tag_type == "int" and not isinstance(tag_value, int):
            raise ValueError(f"{tag_name} must be int, found {type(tag_value)}")
        if tag_type == "str" and not isinstance(tag_value, str):
            raise ValueError(f"{tag_name} must be str, found {type(tag_value)}")
        if tag_type == "list" and not isinstance(tag_value, list):
            raise ValueError(f"{tag_name} must be list, found {type(tag_value)}")

    # Validation logic for complex types (tuples)
    elif isinstance(tag_type, tuple):
        if tag_type[0] == "str":
            if tag_value not in tag_type[1]:
                raise ValueError(
                    f"{tag_name} must be one of {tag_type[1]}, found {tag_value}"
                )

        elif tag_type[0] == "list" and tag_type[1] == "str":
            if not isinstance(tag_value, (str, list)):
                raise ValueError(
                    f"{tag_name} must be str or list of str, found {type(tag_value)}"
                )
            if isinstance(tag_value, list) and not all(
                isinstance(x, str) for x in tag_value
            ):
                raise ValueError(f"{tag_name} must be a list of strings.")

        elif tag_type[0] == "list" and isinstance(tag_type[1], list):
            if not isinstance(tag_value, list) or not set(tag_value).issubset(
                tag_type[1]
            ):
                raise ValueError(f"{tag_name} must be a subset of {tag_type[1]}")
