<p align="center">
  <img src="https://github.com/diningphil/MLWiz/blob/main/docs/_static/mlwiz-logo.png"  width="300"/>
</p>

# MLWiz: the Machine Learning Research Wizard 
[![License](https://img.shields.io/badge/License-BSD_3--Clause-gray.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Documentation Status](https://readthedocs.org/projects/mlwiz/badge/?version=latest)](https://mlwiz.readthedocs.io/en/latest/?badge=latest)
[![Python Package](https://github.com/diningphil/MLWiz/actions/workflows/python-publish.yml/badge.svg)](https://github.com/diningphil/MLWiz/actions/workflows/python-publish.yml)
[![Downloads](https://static.pepy.tech/personalized-badge/mlwiz?period=total&units=international_system&left_color=grey&right_color=blue&left_text=Downloads)](https://pepy.tech/project/mlwiz)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Interrogate](https://github.com/diningphil/MLWiz/blob/main/.badges/interrogate_badge.svg)](https://interrogate.readthedocs.io/en/latest/)
[![Coverage](https://github.com/diningphil/MLWiz/blob/main/.badges/coverage_badge.svg)]()

## [Documentation](https://mlwiz.readthedocs.io/en/latest/index.html)

MLWiz is a Python library that fosters machine learning research by reducing the boilerplate code 
to run reproducible experiments. It provides automatic management of data splitting, loading and common 
experimental settings. It especially handles both model selection and risk assessment procedures, by trying many different
configurations in parallel (CPU or GPU). It is a generalized version of [PyDGN](https://github.com/diningphil/PyDGN)
that can handle different kinds of data and models (vectors, images, time-series, graphs).

## Installation:

Requires at least Python 3.10. Simply run
    
    pip install mlwiz

## Quickstart:

#### Build dataset and data splits

    mlwiz-data --config-file examples/DATA_CONFIGS/config_NCI1.yml [--debug]

#### Launch experiments

    mlwiz-exp  --config-file examples/MODEL_CONFIGS/config_SupToyDGN.yml [--debug]


#### Stop experiments
Use ``CTRL-C``, then type ``ray stop --force`` to stop **all** ray processes you have launched.

### Using the Trained Models

It's very easy to load the model from the experiments (see also the [Tutorial](https://mlwiz.readthedocs.io/en/latest/tutorial.html)):

    from mlwiz.evaluation.util import *

    config = retrieve_best_configuration('RESULTS/supervised_grid_search_toy_NCI1/MODEL_ASSESSMENT/OUTER_FOLD_1/MODEL_SELECTION/')
    splits_filepath = 'examples/DATA_SPLITS/CHEMICAL/NCI1/NCI1_outer10_inner1.splits'
    device = 'cpu'

    # instantiate dataset
    dataset = instantiate_dataset_from_config(config)

    # instantiate model
    model = instantiate_model_from_config(config, dataset, config_type="supervised_config")

    # load model's checkpoint, assuming the best configuration has been loaded
    checkpoint_location = 'RESULTS/supervised_grid_search_toy_NCI1/MODEL_ASSESSMENT/OUTER_FOLD_1/final_run1/best_checkpoint.pth'
    load_checkpoint(checkpoint_location, model, device=device)

    # you can now call the forward method of your model
    y, embeddings = model(dataset[0])