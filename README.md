# Torch-TDA

Automatic differentiation for topological data analysis. 

This package provides utilities for using constructions in topological data analysis
with automatic differentiation.  It wraps functionality from
* [BATS](https://comptop.github.io/BATS.py) for persistent homology
* [persim](https://persim.scikit-tda.org/en/latest/) for computations comparing persistence diagrams
* [topologylayer](https://github.com/bruel-gabrielsson/TopologyLayer) for polynomial features of barcodes

The design is inspired by and draws from [topologylayer](https://github.com/bruel-gabrielsson/TopologyLayer).  Key differences are that `torch-tda` uses `bats` for faster topological computations, and the two packages have different feature sets.

Here is the [documentation and examples](https://torch-tda.readthedocs.io/en/latest/). 

## Use

Package installation provides a package under the `torch_tda` namespace.  Functionality is primarily under `torch_tda.nn`, which provides several PyTorch layers.

```python
import torch_tda
```

## Installation

First, it is recommended to set up a conda environment
```
conda create -n bats
conda activate bats
```

If you are installing `torch_tda` from the development version, please install the development version of BATS.py, install it from source.   

Otherwise, you can use the latest release of BATS.py
```
pip install bats-tda
```
**Attension**: please use Linux OS to install `bats-tda` for now and the support for Mac OS will come soon. 

Now, you can setup with `setup.py`
```
python setup.py install
```

## Documentation

If you want to contribute to the documenation, you can add some jupyter notebooks to the `docs/examples` folder and then generate documentation using Sphinx
```
cd docs
pip install -r requirements.txt
make html
xdg-open _build/html/index.html # or just open this file
```
