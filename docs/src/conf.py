from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, ROOT)


# -- Project information -----------------------------------------------------

project = "vesin"
copyright = f"{datetime.now().date().year}, vesin developers"


def setup(app):
    subprocess.run(["doxygen", "Doxyfile"], cwd=os.path.join(ROOT, "docs"))


# -- General configuration ---------------------------------------------------

needs_sphinx = "4.4.0"

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx_design",
    "breathe",
]

# Add any paths that contain templates here, relative to this directory.
# templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["Thumbs.db", ".DS_Store"]

autoclass_content = "both"
autodoc_member_order = "bysource"
autodoc_typehints_format = "short"
autodoc_type_aliases = {
    "npt.ArrayLike": "numpy.typing.ArrayLike",
}


breathe_projects = {
    "vesin": os.path.join(ROOT, "docs", "build", "xml"),
}
breathe_default_project = "vesin"
breathe_domain_by_extension = {
    "h": "c",
}

breathe_default_members = ("members", "undoc-members")
cpp_private_member_specifier = ""

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "metatensor" : ("https://docs.metatensor.org/latest", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "ase": ("https://wiki.fysik.dtu.dk/ase/", None),
}

html_theme = "furo"
html_title = "Vesin"
# html_static_path = ["_static"]
