<p align="center">
  <img src="https://raw.githubusercontent.com/diningphil/mlwiz/main/docs/_static/mlwiz-logo2-horizontal.png" width="360" alt="MLWiz logo"/>
</p>

# MLWiz
Machine Learning Research Wizard — reproducible experiments from YAML (model selection + risk assessment) for vectors, images, time-series, and graphs.

[![PyPI](https://img.shields.io/pypi/v/mlwiz.svg)](https://pypi.org/project/mlwiz/)
[![Python](https://img.shields.io/pypi/pyversions/mlwiz.svg)](https://pypi.org/project/mlwiz/)
[![CI](https://github.com/diningphil/mlwiz/actions/workflows/python-test-and-coverage.yml/badge.svg)](https://github.com/diningphil/mlwiz/actions/workflows/python-test-and-coverage.yml)
[![Docs](https://readthedocs.org/projects/mlwiz/badge/?version=stable)](https://mlwiz.readthedocs.io/en/stable/)
[![Coverage](https://raw.githubusercontent.com/diningphil/mlwiz/main/.badges/coverage_badge.svg)](https://github.com/diningphil/mlwiz/actions/workflows/python-test-and-coverage.yml)
[![Docstrings](https://raw.githubusercontent.com/diningphil/mlwiz/main/.badges/interrogate_badge.svg)](https://interrogate.readthedocs.io/en/latest/)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-gray.svg)](https://opensource.org/licenses/BSD-3-Clause)

## What it does
MLWiz helps you run end-to-end research experiments with minimal boilerplate:

- Build/prepare datasets and generate splits (hold-out or nested CV)
- Expand a hyperparameter search space (grid or random search)
- Run model selection + risk assessment in parallel with Ray (CPU/GPU or cluster)
- Log metrics, checkpoints, and TensorBoard traces in a consistent folder structure

Inspired by (and a generalized version of) [PyDGN](https://github.com/diningphil/PyDGN).

## Install
MLWiz supports Python 3.10+.

```bash
pip install mlwiz
```

Tip: for GPU / graph workloads, install PyTorch and PyG following their official instructions first, then `pip install mlwiz`.

## Quickstart
#### 1) Prepare dataset + splits:

```bash
mlwiz-data --config-file examples/DATA_CONFIGS/config_MNIST.yml
```

#### 2) Run an experiment (grid search):

```bash
mlwiz-exp --config-file examples/MODEL_CONFIGS/config_MLP.yml
# add --debug to run sequentially and print logs
```

#### 3) Inspect results:

```bash
cat RESULTS/mlp_MNIST/MODEL_ASSESSMENT/assessment_results.json
tensorboard --logdir RESULTS/mlp_MNIST
```

#### 4) Stop a running experiment:

```bash
# Ctrl-C
```

#### 5) Navigating the CLI (non-debug mode)

Example of the global view CLI.
<p align="center">
  <img src="https://raw.githubusercontent.com/diningphil/mlwiz/main/docs/_static/exp_gui.png" width="760" alt="MLWiz terminal progress UI"/>
</p>

Specific views can be accessed. For insance, you can visualize a specific model run by typing

```bash
:<outer_fold> <inner_fold> <config_id> <run_id>
```

or, analogously, a risk assessment run by typing

```bash
:<outer_fold> <run_id>
```

To get back to the global view, just type

```bash
:  # or :g or :global
```

To refresh the screen:
```bash
:r  # or :refresh
```

You can use **left-right arrows** to move across configurations, and **up-down arrows** to switch between model selection and risk assessment runs.

## Architecture (high-level)
MLWiz is built around two YAML files and a small set of composable components:

```text
data.yml ──► mlwiz-data ──► processed dataset + .splits
exp.yml  ──► mlwiz-exp  ──► Ray workers
                      ├─ inner folds: model selection (best hyperparams)
                      └─ outer folds: risk assessment (final scores)
```

- **Data pipeline**: `mlwiz-data` instantiates your dataset class and writes a `.splits` file for hold-out / (nested) CV.
- **Search space**: `grid:` and `random:` sections expand into concrete hyperparameter configurations.
- **Orchestration**: the evaluator schedules training runs with Ray across CPU/GPU (or a Ray cluster).
- **Execution**: each run builds a model + training engine from dotted paths, then logs artifacts and returns structured results.

## Configuration at a glance
MLWiz expects:

- one YAML for **data + splits**
- one YAML for **experiment + search space**

Minimal data config:

```yaml
splitter:
  splits_folder: DATA_SPLITS/
  class_name: mlwiz.data.splitter.Splitter
  args:
    n_outer_folds: 3
    n_inner_folds: 2
    seed: 42

dataset:
  class_name: mlwiz.data.dataset.MNIST
  args:
    storage_folder: DATA/
```

Minimal experiment config (grid search):

```yaml
storage_folder: DATA
dataset_class: mlwiz.data.dataset.MNIST
data_splits_file: DATA_SPLITS/MNIST/MNIST_outer3_inner2.splits

device: cpu
max_cpus: 8

dataset_getter: mlwiz.data.provider.DataProvider
data_loader:
  class_name: torch.utils.data.DataLoader
  args:
    num_workers : 0
    pin_memory: False

result_folder: RESULTS
exp_name: mlp
experiment: mlwiz.experiment.Experiment
higher_results_are_better: true
evaluate_every: 1
risk_assessment_training_runs: 3
model_selection_training_runs: 2

grid:
  model: mlwiz.model.MLP
  epochs: 400
  batch_size: 512
  dim_embedding: 5
  mlwiz_tests: True  # patch: allow reshaping of MNIST dataset
  optimizer:
    - class_name: mlwiz.training.callback.optimizer.Optimizer
      args:
        optimizer_class_name: torch.optim.Adam
        lr:
          - 0.01
          - 0.03
        weight_decay: 0.
  loss: mlwiz.training.callback.metric.MulticlassClassification
  scorer: mlwiz.training.callback.metric.MulticlassAccuracy
  engine: mlwiz.training.engine.TrainingEngine
```

See `examples/` for complete configs (including random search, schedulers, early stopping, and more).

### Custom code via dotted paths
Point YAML entries to your own classes (in your project). `mlwiz-data` and `mlwiz-exp` add the current working directory to `sys.path`, so this works out of the box:

```yaml
grid:
  model: my_project.models.MyModel

dataset:
  class_name: my_project.data.MyDataset
```

## Outputs
Runs are written under:

- `RESULTS/<exp_name>_<dataset>/MODEL_ASSESSMENT/assessment_results.json` (aggregated outer-fold results)
- `RESULTS/<exp_name>_<dataset>/MODEL_ASSESSMENT/OUTER_FOLD_k/outer_results.json` (per-fold summary)
- `.../MODEL_SELECTION/...` (inner-fold runs + winner config)
- `.../final_run*/` (re-trains with the selected hyperparameters)

Each training run also writes TensorBoard logs under `<run_dir>/tensorboard/`.

## Utilities
Duplicate a base experiment config across multiple datasets:

```bash
mlwiz-config-duplicator --base-exp-config base.yml --data-config-files data1.yml data2.yml
```

## Documentation / Tutorial
- Docs: https://mlwiz.readthedocs.io/en/stable/
- Tutorial **(recommended)**: https://mlwiz.readthedocs.io/en/stable/tutorial.html

## Contributing
See `CONTRIBUTING.md`.

## License
BSD-3-Clause. See `LICENSE`.
