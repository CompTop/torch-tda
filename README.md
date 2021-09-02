# Torch-TDA

Automatic differentiation for topological data analysis.

This package provides utilities for using constructions in topological data analysis
with automatic differentiation.  It wraps functionality from
* [BATS](https://comptop.github.io/BATS.py) for persistent homology
* [persim](https://persim.scikit-tda.org/en/latest/) for computations comparing persistence diagrams
* [topologylayer](https://github.com/bruel-gabrielsson/TopologyLayer) for polynomial features of barcodes

The design is inspired by and draws from [topologylayer](https://github.com/bruel-gabrielsson/TopologyLayer).  Key differences are that `torch-tda` uses `bats` for faster topological computations, and the two packages have different feature sets.

## Installation

First, it is recommended to set up a conda environment
```
conda create -n bats
conda activate bats
```

If you want the development version of BATS.py, install it from source.   Otherwise
```
pip install bats-tda
```

Now, you can setup with `setup.py`
```
python setup.py install
```

## Documentation

You can generate documentation using Sphinx
```
cd docs
pip install -r requirements.txt
make html
xdg-open _build/html/index.html # or just open this file
```
