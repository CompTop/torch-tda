# Torch-TDA

A package used to do automatic differentiation in Pytorch for topological data analysis. 

This package provides utilities for using constructions in topological data analysis
with automatic differentiation.  It wraps functionality from
* [BATS](https://comptop.github.io/BATS.py) for Basic Applied Topology computation.
* [persim](https://persim.scikit-tda.org/en/latest/) for computations comparing persistence diagrams.
* [hera](https://bitbucket.org/grey_narn/hera/src/master/) a library for fast calculation of bottleneck distance and Wasserstein distance on persistence diagrams.
* [topologylayer](https://github.com/bruel-gabrielsson/TopologyLayer) for polynomial features of barcodes.

The design is heavily inspired by [topologylayer](https://github.com/bruel-gabrielsson/TopologyLayer).  Key differences are that we use `bats` for faster topological computations.

Here is the [documentation and examples](https://torch-tda.readthedocs.io/en/latest/). 
## Reference
If you want to reference this project, please consider referencing one of these two papers or both because it is a side-product of the two papers:

- [Accelerating iterated persistent homology computations with warm starts](https://www.sciencedirect.com/science/article/pii/S0925772124000117)
- [Topology-Preserving Dimensionality Reduction via Interleaving Optimization](https://arxiv.org/abs/2201.13012)



## Use

Package installation provides a package under the `torch_tda` namespace.  Functionality is primarily under `torch_tda.nn`, which provides several PyTorch layers.

```python
import torch_tda
```

## Installation

**Attension**: please use Linux OS to install for now and the support for Mac OS will come soon. 

First, it is recommended to set up a conda environment
```
conda create -n bats
conda activate bats
```

If you are installing `torch_tda` from the development version, please install the development version of BATS.py, install it from source.   

Otherwise, you can use the latest release of [BATS.py](https://comptop.github.io/BATS.py)
```
pip install bats-tda
```
Now, to use `torch_tda`, you will need Pytorch pre-installed and then 
```
pip install torch-tda
```

### Hera
If you want to compute bottleneck distance, you will need to install [hera](https://github.com/anigmetov/hera) by 1. following their installation description to build from source and install the Python binding  2. (Optional if you can put the Python package in the correct position) write a simple `setup.py` file to install the package into environment path. 

### Build from source
If you want to install from source, after cloning the repository, you can setup with `setup.py`
```
python setup.py install
```
**Attention:** 
- Note that installing by `pip install torch-tda` in the cloned repository will probably install the older version from remote, see the [explanation](https://stackoverflow.com/questions/14617136/why-is-pip-installing-an-old-version-of-my-package). So, DO NOT use `pip` if you want to build from the latest source. 

- If you have installed an older version of `torch-tda` and want to update to the newest version, please uninstall it first by `pip uninstall torch-tda`. 

## Documentation

If you want to contribute to the documenation, you can add some jupyter notebooks to the `docs/examples` folder and then generate documentation using Sphinx
```
cd docs
pip install -r requirements.txt
make html
xdg-open _build/html/index.html # or just open this file
```
