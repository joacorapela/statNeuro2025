# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'statsNeuro2025'
copyright = '2025, Joaquin Rapela'
author = 'Joaquin Rapela'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx_gallery.gen_gallery',
    'sphinx.ext.mathjax',
]

sphinx_gallery_conf = {
    'examples_dirs': '../../../examples/sphinx_gallery',   # path to your example scripts
    'gallery_dirs': 'auto_examples',  # path to where to save gallery generated output
}

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
