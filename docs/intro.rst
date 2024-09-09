Introduction
============

The Machine Learning Research Wizard (**MLWiz**) is a framework that you can use to easily train and evaluate machine learning models for tabular, image, timeseries, and graph data.

MLWiz helps you to:
 * automatize **model selection** and **risk assessment**,
 * foster reproducibility and robustness of results,
 * reduce the amount of boilerplate code to write,
 * make it flexible enough to encompass a wide range of use cases for research.
 * support a number of different hardware set ups, including a cluster of nodes (using `Ray <https://docs.ray.io/en/latest/>`_),

To run an experiment, you usually rely on 2 **YAML configuration files**:
  * one to pre-process the dataset and create the data splits,
  * another with information about the experiment itself and the hyper-parameters to try.

**MLWiz** is a minimal, but extended version of [PyDGN](https://github.com/diningphil/PyDGN).

Installation:
*******************

Automated tests passing on Linux and Windows.

The recommended way to install the library is to follow the steps to install ``torch`` and ``torch_geometric`` prior to installing MLWiz.

Then simply run

.. code-block:: python

    pip install mlwiz
