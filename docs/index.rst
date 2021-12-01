.. torch-tda documentation master file, created by
   sphinx-quickstart on Fri Aug 20 12:44:03 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to torch-tda's documentation!
=====================================

This package provides utilities for using constructions in topological data analysis
with automatic differentiation.  It wraps functionality from
* [BATS](https://comptop.github.io/BATS.py) for persistent homology
* [persim](https://persim.scikit-tda.org/en/latest/) for computations comparing persistence diagrams
* [topologylayer](https://github.com/bruel-gabrielsson/TopologyLayer) for polynomial features of barcodes

The design is inspired by and draws from [topologylayer](https://github.com/bruel-gabrielsson/TopologyLayer).  Key differences are that `torch-tda` uses `bats` for faster topological computations, and the two packages have different feature sets.


.. toctree::
   :maxdepth: 2
   :caption: Contents:
   
   installation
   examples/index
   api



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
