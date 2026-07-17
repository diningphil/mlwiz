Introduction
============

The Machine Learning Research Wizard (**MLWiz**) is a framework that you can use to easily train and evaluate machine learning models for tabular, image, timeseries, and graph data.

MLWiz helps you to:
 * automatize **model selection** and **risk assessment**,
 * foster reproducibility and robustness of results,
 * reduce the amount of boilerplate code to write,
 * make it flexible enough to encompass a wide range of use cases for research.
 * support a number of different hardware set ups, including a cluster of nodes (using `Ray <https://docs.ray.io/en/latest/>`_),
 * run single-GPU or multi-GPU training (DDP) transparently through configuration only, with or without mixed-precision

To run an experiment, you usually rely on 2 **YAML configuration files**:
  * one to pre-process the dataset and create the data splits,
  * another with information about the experiment itself and the hyper-parameters to try.

**MLWiz** is a minimal, but extended version of PyDGN (https://github.com/diningphil/PyDGN).

Installation:
*******************

Automated tests passing on Linux.

The recommended workflow uses `uv <https://docs.astral.sh/uv/>`_ to manage the
project environment and dependencies. Create a project and add MLWiz:

.. code-block:: bash

    uv init my-ml-project
    cd my-ml-project
    uv add mlwiz

If the project already has a ``pyproject.toml``, run only ``uv add mlwiz`` from
its root. For GPU or graph workloads, configure ``torch`` and
``torch_geometric`` following their official installation instructions before
adding MLWiz.
