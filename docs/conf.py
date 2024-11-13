# Configuration file for the Sphinx documentation builder.
import logging
import sys
import os
from datetime import datetime
import sphinx_rtd_theme # type: ignore


# -- Project information -----------------------------------------------------
project = 'capibara-gpt'
copyright = f'{datetime.now().year}, Marco Durán'
author = 'Marco Durán'
release = '1.2.2'
version = '1.1'

# Configuración del tema
html_theme = 'sphinx_rtd_theme'
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

# Extensiones
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx_rtd_theme',
    # ... otras extensiones que puedas tener
]

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_type_aliases = None

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

# Template settings
templates_path = ['_templates']
exclude_patterns = [
    '_build',
    'Thumbs.db',
    '.DS_Store',
    '**.ipynb_checkpoints',
    'build',
    'dist'
]

# Source suffix
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# The master toctree document
master_doc = 'index'

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_theme_options = {
    'navigation_depth': 4,
    'collapse_navigation': False,
    'sticky_navigation': True,
    'includehidden': True,
}

# Custom sidebar templates
html_sidebars = {
    '**': [
        'relations.html',
        'searchbox.html',
        'navigation.html',
        'custom_sidebar.html',
        'versions.html',
        'toggle_sidebar.html',
    ]
}

# -- Options for LaTeX output ---------------------------------------------
latex_elements = {
    'papersize': 'letterpaper',
    'pointsize': '10pt',
    'preamble': '',
    'figure_align': 'htbp',
    'extraclassoptions': 'openany,oneside',
    
}

# Intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'jax': ('https://jax.readthedocs.io/en/latest/', None),
    'flax': ('https://flax.readthedocs.io/en/latest/', None),
    'optax': ('https://optax.readthedocs.io/en/latest/', None),
    'xla': ('https://www.tensorflow.org/xla', None),
    'tpu': ('https://cloud.google.com/tpu', None),
    'tensorflow': ('https://www.tensorflow.org/api_docs', None),
}

# -- Extension configuration -------------------------------------------------
todo_include_todos = True
viewcode_follow_imported_members = True
add_module_names = False

# -- Options for MathJax -------------------------------------------------
mathjax_path = 'https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js'

# -- Options for logging -------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -- Setup for notebooks -------------------------------------------------
nbsphinx_execute = 'never'
nbsphinx_allow_errors = True

# -- Additional configuration -------------------------------------------
nitpicky = True
nitpick_ignore = [
    ('py:class', 'optional'),
    ('py:class', 'Any'),
]

# -- GitHub repository -------------------------------------------------
html_context = {
    "display_github": True,
    "github_user": "anachroni",
    "github_repo": "capibara-gpt",
    "github_version": "main",
    "conf_py_path": "/docs/",
}
