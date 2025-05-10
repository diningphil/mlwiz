<p align="center">
  <img src="https://github.com/diningphil/MLWiz/blob/main/docs/_static/mlwiz-logo.png"  width="300"/>
</p>

# MLWiz: the Machine Learning Research Wizard 
[![License](https://img.shields.io/badge/License-BSD_3--Clause-gray.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Documentation Status](https://readthedocs.org/projects/mlwiz/badge/?version=latest)](https://mlwiz.readthedocs.io/en/latest/?badge=latest)
[![Publish Package](https://github.com/diningphil/mlwiz/actions/workflows/python-publish-package.yml/badge.svg)](https://github.com/diningphil/mlwiz/actions/workflows/python-publish-package.yml)
[![Downloads](https://static.pepy.tech/badge/mlwiz)](https://pepy.tech/project/mlwiz)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Interrogate](https://github.com/diningphil/MLWiz/blob/main/.badges/interrogate_badge.svg)](https://interrogate.readthedocs.io/en/latest/)
[![Coverage](https://github.com/diningphil/MLWiz/blob/main/.badges/coverage_badge.svg)]()

## [Documentation](https://mlwiz.readthedocs.io/en/stable/index.html)

MLWiz is a Python library that aids reproducible machine learning research.

It takes care of the boilerplate code to prepare and run experiments, by providing automatic management of data splitting, loading and common 
experimental settings. It especially handles both model selection and risk assessment procedures, by trying many different
configurations in parallel (CPU or GPU). It is a generalized version of [PyDGN](https://github.com/diningphil/PyDGN)
that can handle different kinds of data and models (vectors, images, time-series, graphs).

## Installation:

Requires at least Python 3.10. Simply run
    
    pip install mlwiz

## Quickstart:

#### Build dataset and data splits

    mlwiz-data --config-file examples/DATA_CONFIGS/config_MNIST.yml

#### Launch experiments

    mlwiz-exp  --config-file examples/MODEL_CONFIGS/config_MLP.yml [--debug]


#### Stop experiments
Use ``CTRL-C``, then type ``ray stop --force`` to stop **all** ray processes you have launched.

### Using the Trained Models

It's very easy to load the model from the experiments: see the end of the [Tutorial](https://mlwiz.readthedocs.io/en/stable/tutorial.html) for more information!
