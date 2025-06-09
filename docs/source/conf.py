# -- Path setup --------------------------------------------------------------
from doc.sphinx.source.conf import toc_object_entries

import RotationalDiffusion as rd


# -- Project information -----------------------------------------------------
project = 'RotationalDiffusion'
author = 'Simon L. Holtbrügge'
release = '0.8.0'


# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.coverage',
    'sphinx.ext.doctest',
    #'sphinx.ext.mathjax',
    'numpydoc',
    'sphinxcontrib.bibtex',
    'nbsphinx',
    'nbsphinx_link'
]

templates_path = ["_templates"]
exclude_patterns = ["build", "Thumbs.db", ".DS_Store"]
toc_object_entries = True


# -- Autodoc configuration ---------------------------------------------------
autodoc_default_options = {
    'members': True,
#    'undoc-members': True,
#    'inherited-members': True,
#    'show-inheritance': True,
#    'member-order': 'groupwise',
#    'special-members': '__init__',
#    'exclude-members': '__weakref__',
}

#autodoc_inherit_docstrings = True
#autodoc_member_order = 'bysource'
#autodoc_mock_imports = []


# -- Options for HTML output -------------------------------------------------
html_theme = 'pydata_sphinx_theme'
html_theme_options = {
    'github_url': 'https://github.com/MolSimGroup/rotationaldiffusion',
    'logo': {'text': 'RotationalDiffusion'},
    'show_toc_level': 3,
}

# -- Extension configuration -------------------------------------------------
# autosummary configuration
autosummary_generate = True
autosummary_generate_overwrite = True

# intersphinx extension
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "mdanalysis": ("https://docs.mdanalysis.org/stable/", None),
    "numpy": ('https://numpy.org/doc/stable/', None),
}

# doctest extension
doctest_global_setup = """
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='MDAnalysis')
"""

# numpydoc extension
numpydoc_xref_param_type = True
numpydoc_xref_aliases = {
    'AtomGroup': 'MDAnalysis.core.groups.AtomGroup',
    'Universe': 'MDAnalysis.core.universe.Universe',
    'BackendBase': 'MDAnalysis.analysis.backends.BackendBase',
}
numpydoc_xref_ignore = {'optional', 'default', 'shape', 'of', 'thereof',
                        'N', 'S', 'T', 'n_frames', 'subclass'}

# sphinxcontrib-bibtex extension
bibtex_bibfiles = ['references.bib']
bibtex_reference_style = "author_year"

# nbsphinx extension
nbsphinx_execute = 'never'
