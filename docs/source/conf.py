# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Configure skpro Sphinx documentation."""

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import datetime
import inspect
import os
import sys

import skpro

# -- Path setup --------------------------------------------------------------

# When we build the docs on readthedocs, we build the package and want to
# use the built files in order for sphinx to be able to properly read the
# Cython files. Hence, we do not add the source code path to the system path.
env_rtd = os.environ.get("READTHEDOCS")
# Check if on Read the docs
if not env_rtd == "True":
    sys.path.insert(0, os.path.abspath("../.."))

# -- Project information -----------------------------------------------------

current_year = datetime.datetime.now().year
project = "skpro"
project_copyright = f"2017 - {current_year} (BSD-3-Clause License)"
author = "skpro Developers"


# The full version, including alpha/beta/rc tags
release = skpro.__version__

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "numpydoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.linkcode",  # link to GitHub source code via linkcode_resolve()
    "nbsphinx",  # integrates example notebooks
    "sphinx_gallery.load_style",
    "myst_parser",
    "sphinx_panels",
    "sphinx_issues",
    "sphinx_design",
]

myst_enable_extensions = ["colon_fence"]

# -- Internationalization ------------------------------------------------
# specifying the natural language populates some key tags
language = "en"

# Use bootstrap CSS from theme.
panels_add_bootstrap_css = False

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# The main toctree document.
master_doc = "index"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

# add_module_names = False

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# Members and inherited-members default to showing methods and attributes from
# a class or those inherited.
# Member-order orders the documentation in the order of how the members are
# defined in the source code.
autodoc_default_options = {
    "members": True,
    "inherited-members": True,
    "member-order": "bysource",
}

# If true, '()' will be appended to :func: etc. cross-reference text.
add_function_parentheses = False

# Link to GitHub repo for github_issues extension
issues_github_path = "sktime/skpro"


def linkcode_resolve(domain, info):
    """Return URL to source code corresponding.

    Parameters
    ----------
    domain : str
    info : dict

    Returns
    -------
    url : str
    """

    def find_source():
        # try to find the file and line number, based on code from numpy:
        # https://github.com/numpy/numpy/blob/main/doc/source/conf.py#L286
        obj = sys.modules[info["module"]]
        for part in info["fullname"].split("."):
            obj = getattr(obj, part)

        fn = inspect.getsourcefile(obj)
        fn = os.path.relpath(fn, start=os.path.dirname(skpro.__file__))
        source, lineno = inspect.getsourcelines(obj)
        return fn, lineno, lineno + len(source) - 1

    if domain != "py" or not info["module"]:
        return None
    try:
        filename = "skpro/%s#L%d-L%d" % find_source()
    except Exception:
        filename = info["module"].replace(".", "/") + ".py"
    return f"https://github.com/sktime/skpro/blob/{version_match}/{filename}"


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.

html_theme = "pydata_sphinx_theme"

# Define the json_url for our version switcher.
json_url = "https://github.com/sktime/skpro/blob/main/docs/source/_static/switcher.json"

# This uses code from the py-data-sphinx theme's own conf.py
# Define the version we use for matching in the version switcher.
version_match = os.environ.get("READTHEDOCS_VERSION")

# If READTHEDOCS_VERSION doesn't exist, we're not on RTD
# If it is an integer, we're in a PR build and the version isn't correct.
if not version_match or version_match.isdigit():
    # For local development, infer the version to match from the package.
    if "dev" in release or "rc" in release:
        version_match = "latest"
        # We want to keep the relative reference if we are in dev mode
        # but we want the whole url if we are effectively in a released version
        json_url = "_static/switcher.json"
    else:
        version_match = "v" + release

html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/sktime/skpro",
            "icon": "fab fa-github",
        },
        {
            "name": "Discord",
            "url": "https://discord.com/invite/54ACzaFsn7",
            "icon": "fab fa-discord",
        },
        {
            "name": "LinkedIn",
            "url": "https://www.linkedin.com/company/scikit-time/",
            "icon": "fab fa-linkedin",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/skpro",
            "icon": "fa-solid fa-box",
        },
    ],
    "icon_links_label": "Quick Links",
    "show_nav_level": 1,
    "show_prev_next": False,
    "use_edit_page_button": False,
    "navbar_start": ["navbar-logo", "version-switcher"],
    "navbar_center": ["navbar-nav"],
    "switcher": {
        "json_url": json_url,
        "version_match": version_match,
    },
    "header_links_before_dropdown": 6,
}
html_logo = "images/skpro-banner.png"
html_context = {
    "github_user": "sktime",
    "github_repo": "skpro",
    "github_version": "main",
    "doc_path": "docs/source/",
    "default_mode": "light",
}
html_sidebars = {"**": ["sidebar-nav-bs.html", "sidebar-ethical-ads.html"]}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# -- Options for HTMLHelp output ---------------------------------------------
# Output file base name for HTML help builder.
htmlhelp_basename = "skprodoc"


# -- Options for LaTeX output ------------------------------------------------
# latex_elements = {
# The paper size ('letterpaper' or 'a4paper').
# 'papersize': 'letterpaper',
# The font size ('10pt', '11pt' or '12pt').
# 'pointsize': '10pt',
# Additional stuff for the LaTeX preamble.
# 'preamble': '',
# Latex figure (float) alignment
# 'figure_align': 'htbp',
# }

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (
        master_doc,
        "skpro.tex",
        "skpro Documentation",
        "skpro developers",
        "manual",
    ),
]

# -- Extension configuration -------------------------------------------------

# -- Options for numpydoc extension ------------------------------------------
# see http://stackoverflow.com/q/12206334/562769
numpydoc_show_class_members = True
# this is needed for some reason...
# see https://github.com/numpy/numpydoc/issues/69
numpydoc_class_members_toctree = False

numpydoc_validation_checks = {"all", "GL01", "SA01", "EX01"}

# -- Options for sphinx-copybutton extension----------------------------------
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True

# -- Options for nbsphinx extension ------------------------------------------
nbsphinx_execute = "never"  # always  # whether to run notebooks
nbsphinx_allow_errors = False  # False
nbsphinx_timeout = 600  # seconds, set to -1 to disable timeout

# add Binder launch buttom at the top
current_file = "{{ env.doc2path( env.docname, base=None) }}"

# make sure Binder points to latest stable release, not main
binder_base = "https://mybinder.org/v2/gh//skpro/"
binder_url = binder_base + f"{version_match}?filepath={current_file}"
nbsphinx_prolog = f"""
.. |binder| image:: https://mybinder.org/badge_logo.svg
.. _Binder: {binder_url}

|Binder|_
"""

# add link to original notebook at the bottom
notebook_url = f"https://github.com/sktime/skpro/tree/{version_match}/{current_file}"
nbsphinx_epilog = f"""
----

Generated using nbsphinx_. The Jupyter notebook can be found here_.

.. _here: {notebook_url}
.. _nbsphinx: https://nbsphinx.readthedocs.io/
"""

# -- Options for intersphinx extension ---------------------------------------

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    "python": (f"https://docs.python.org/{sys.version_info.major}", None),
    "scikit-learn": ("https://scikit-learn.org/stable/", None),
    "sktime": ("https://www.sktime.net/en/stable/", None),
}
