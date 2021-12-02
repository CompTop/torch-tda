# Installation

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
Now, to use 'torch_tda', you will need Pytorch pre-installed and then 
```
pip install torch-tda
```

## Build from source
If you want to install from source, after cloning the repository, you can setup with `setup.py`
```
python setup.py install
```
**Attention!!!!:** Please do not install by `pip install torch-tda` in the cloned repository, becuase `pip` will probably install the older version from remote, see the [explanation](https://stackoverflow.com/questions/14617136/why-is-pip-installing-an-old-version-of-my-package).

