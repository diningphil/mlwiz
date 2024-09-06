# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/main/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("../"))


# -- Project information -----------------------------------------------------

project = "MLWiz"
copyright = "2024, Federico Errica"
author = "Federico Errica"

# The full version, including alpha/beta/rc tags
release = "1.0.0"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.napoleon",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------

autodoc_mock_imports = [
    "PyYAML",
    "requests",
    "tensorboard",
    "tqdm",
    "ogb",
    "aioredis",
    "ray",
    "gpustat",
    "torch",
    "torch-scatter",
    "torch-sparse",
    "torch-cluster",
    "torch-spline-conv",
    "torch-scatter",
    "torch-cluster",
    "torch-sparse",
    "torch_geometric",
    "tqdm",
    "ray",
    "pandas",
    "numpy",
    "sklearn",
]

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# Logo of the doc
html_logo = html_static_path[0] + "/" + "mlwiz-logo.png"
html_theme_options = {
    "logo_only": False,
    "display_version": True,
}