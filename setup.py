# This is not the actual setup.py for this project, see `python/vesin/setup.py` for it.
# Instead, this file is here to enable `pip install .` from a git checkout or `pip
# install git+https://...` without having to specify a subdirectory

import os

from setuptools import setup

ROOT = os.path.realpath(os.path.dirname(__file__))

setup(
    name="vesin-git",
    version="0.0.0",
    install_requires=[
        f"vesin @ file://{ROOT}/python/vesin",
    ],
    extras_require={
        "torch": [
            f"vesin-torch @ file://{ROOT}/python/vesin-torch",
        ]
    },
    packages=[],
)
