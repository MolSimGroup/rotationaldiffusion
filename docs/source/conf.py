# -*- coding: utf-8 -*-
# Configuration file for the Sphinx documentation builder.

# -- Path setup --------------------------------------------------------------

# In case the project was not installed
import os
import sys
sys.path.insert(0, os.path.abspath("../.."))
import rotationaldiffusion 


# -- Project information -----------------------------------------------------

project = "rotationaldiffusion"
copyright = (
    "2024, Simon Holtbruegge."
)
author = "Simon Holtbruegge"

# The short X.Y version
version = "0.8"
# The full version, including alpha/beta/rc tags
release = "0.8"


# -- General configuration ---------------------------------------------------

#needs_sphinx = "6.2.1"
extensions = [
    "sphinx.ext.autodoc",
#    "sphinx.ext.mathjax",
#    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.extlinks",
    "sphinxcontrib.bibtex",
#    "mdanalysis_sphinx_theme",
]

autosummary_generate = True
# This skips generating an autodoc of the test module
# when using the autosummary directive that is included
# by default in api.rst

templates_path = ["_templates"]
source_suffix = ".rst"
master_doc = "index"
language = "en"
exclude_patterns = ["build", "Thumbs.db", ".DS_Store"]
pygments_style = "default"
toc_object_entries = False

# -- Options for HTML output -------------------------------------------------

#html_theme = "mdanalysis_sphinx_theme"
#html_theme_options = {"mda_official": False}
#html_theme = "sphinx_rtd_theme"
#html_theme = "alabaster"
html_theme = "furo"


html_logo = "_static/logo/placeholder_logo.png"
html_favicon = "_static/logo/placeholder_favicon.svg"

html_static_path = ["_static"]


# -- Options for HTMLHelp output ---------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = "rotationaldiffusiondoc"


# -- Options for LaTeX output ------------------------------------------------

latex_elements = {
    # The paper size ("letterpaper" or "a4paper").
    #
    # "papersize": "letterpaper",

    # The font size ("10pt", "11pt" or "12pt").
    #
    # "pointsize": "10pt",

    # Additional stuff for the LaTeX preamble.
    #
    # "preamble": "",

    # Latex figure (float) alignment
    #
    # "figure_align":results.orientations_as_mat (np.ndarray (n_frames, 3, 3)) – Orientations represented as matrices. "htbp",
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, "rotationaldiffusion.tex", "rotationaldiffusion Documentation",
     "rotationaldiffusion", "manual"),
]


# -- Options for manual page output ------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (master_doc, "rotationaldiffusion", "rotationaldiffusion Documentation",
     [author], 1)
]


# -- Options for Texinfo output ----------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (master_doc, "rotationaldiffusion", "rotationaldiffusion Documentation",
     author, "rotationaldiffusion", "A (hopefully soon-to-be) MDAKit for studying rotational diffusion from Molecular Dynamics simulations.",
     "Miscellaneous"),
]


# -- Extension configuration -------------------------------------------------

# autodoc extension
autoclass_content = "both"
autodoc_class_signature = "mixed"
autodoc_default_options = {
    'members': True,
    'inherited-members': True,
    'show-inheritance': True
}
autodoc_mock_imports = ['rotationaldiffusion.tests']

# napoleon extension
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = False
napoleon_use_ivar = False
napoleon_use_rtype = False
napoleon_preprocess_types = True
napoleon_type_aliases = {
    "AtomGroup": ":class:`AtomGroup <MDAnalysis.core.groups.AtomGroup>`",
    "Universe": ":class:`Universe <MDAnalysis.core.universe.Universe>`",
    "ndarray": ":class:`ndarray <numpy.ndarray>`"
}

# intersphinx extension
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "mdanalysis": ("https://docs.mdanalysis.org/stable/", None),
    "numpy": ('https://numpy.org/doc/stable/', None),
}

# extlinks extension
extlinks = {
    "MDAnalysis": ('https://www.mdanalysis.org/%s', 'MDAnalysis %s')
}

# sphinxcontrib-bibtex extension
bibtex_bibfiles = ['references.bib']
bibtex_reference_style = "author_year"