#!/usr/bin/env python
# Notes helpful for write setup.py
# How to write a setup.py file
# https://godatadriven.com/blog/a-practical-guide-to-using-setup-py/

# Use `pip install -e .` to intall in editable mode
# So you would use this when trying to install a package locally, 
# most often in the case when you are developing it on your system. 
# It will just link the package to the original location, 
# basically meaning any changes to the original package 
# would reflect directly in your environment.

from setuptools import setup
import setuptools

with open('README.md') as f:
    long_description = f.read()

setup(name='torch-tda',
      version='0.0.1',
      description='Automatic differentiation for topological data analysis',
      long_description=long_description,
      long_description_content_type="text/markdown",
      author='Brad Nelson, Yuan Luo',
      author_email='bradnelson@uchicago.edu, yuanluo@uchicago.edu',
      url='https://github.com/CompTop/torch-tda',
      project_urls={
        "Documentation": "https://torch-tda.readthedocs.io/en/latest/",
      },
      license='MIT',
      # package_dir={"": "torch_tda"},
      # packages=setuptools.find_packages(where="torch_tda"),
      packages=['torch_tda', 'torch_tda.nn', 'torch_tda.nn.functional'],
      include_package_data=True,
      install_requires=[
        'numpy',
        'bats-tda',
        'persim>=0.3.1',
        'scipy',
        'hera-tda>=0.0.1',
      ],
      python_requires='>=3.7',
      classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Education',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Scientific/Engineering :: Mathematics',
        'License :: OSI Approved :: MIT License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
      ],
      keywords='topological data analysis, automatic differentiation, persistent homology'
     )
