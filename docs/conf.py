# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

'''
. - current path of the conf.py
.. - parent path of the conf.py
../.. - parent of the parent path, etc.
'''
sys.path.insert(0, os.path.abspath('.'))
sys.path.append(os.path.abspath('../..'))
sys.path.append(os.path.abspath('..'))

# -- Project information -----------------------------------------------------

project = 'torch-tda'
copyright = '2021, Brad Nelson, Yuan Luo'
author = 'Brad Nelson, Yuan Luo'

# The full version, including alpha/beta/rc tags
release = '0.0.0'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
    'sphinx.ext.autosummary',
    'myst_parser',
    'nbsphinx',
    'sphinx_copybutton',
    'sphinx.ext.mathjax',
    'IPython.sphinxext.ipython_console_highlighting',
]

autodoc_default_options = {
    'members': None,
    'imported-members': True,
    'undoc-members': True,
    'show-inheritance': True,
}

autodoc_mock_imports = ["torch"]

autosummary_generate = True

# Tell sphinx what the primary language being documented is.
primary_domain = 'py'

# Tell sphinx what the pygments highlight language should be.
highlight_language = 'py'

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# on_rtd is whether we are on readthedocs.org, this line of code grabbed from docs.readthedocs.org
on_rtd = os.environ.get('READTHEDOCS', None) == 'True'

if not on_rtd:  # only import and set the theme if we're building docs locally
    import sphinx_rtd_theme
    html_theme = 'sphinx_rtd_theme'
    html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
