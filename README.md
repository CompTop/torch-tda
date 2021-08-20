# Torch-TDA

Automatic differentiation for topological data analysis.

This package provides utilities for using constructions in topological data analysis
with automatic differentiation.  It wraps functionality from
* [BATS](https://comptop.github.io/BATS.py) for persistent homology
* [persim](https://persim.scikit-tda.org/en/latest/) for computations comparing persistence diagrams
* [topologylayer](https://github.com/bruel-gabrielsson/TopologyLayer) for polynomial features of barcodes

The design is inspired by and draws from [topologylayer](https://github.com/bruel-gabrielsson/TopologyLayer).  Key differences are that `torch-tda` uses `bats` for faster topological computations, and the two packages have different feature sets.
