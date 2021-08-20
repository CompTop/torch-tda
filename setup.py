#!/usr/bin/env python

from setuptools import setup


with open('README.md') as f:
    long_description = f.read()

setup(name='torch-tda',
      version='0.0.0',
      description='Automatic differentiation for topological data analysis',
      long_description=long_description,
      long_description_content_type="text/markdown",
      author='Brad Nelson, Yuan Luo',
      author_email='bradnelson@uchicago.edu, yuanluo@uchicago.edu',
      url='https://github.com/CompTop/torch-tda',
      license='MIT',
      packages=['torch_tda'],
      include_package_data=True,
      install_requires=[
        'numpy',
        'bats-tda',
        'persim',
        'scipy'
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
