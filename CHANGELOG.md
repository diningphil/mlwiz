# Changelog

## [1.2.4] Minor Fixes

## Changed

- No need to specify a metric name any longer. Now it is taken from the class name directly.

## Fixed

- Changed name of metric from MeanAverageLoss to MeanAbsoluteLoss

## [1.2.3] Minor Fixes

## Fixed

- Fixed bug when using `--execute-config-id` and `--skip-config-ids`


## [1.2.2] Aiding Plotting

## Added

- Function `mlwix.evaluation.util.create_dataframe` helps you to convert lists of results for each configuration into a Pandas DataFrame. Now MLWiz depends on Pandas.
- Jupyter notebook `Example Qualitative Analysis and Plots.ipynb` that shows how to use the above function.

## [1.2.1] Refactoring of Evaluator

## Changed

- Simplified the logic of Evaluator to facilitate understanding

## Fixed

- We explicitly list the package modules to allow different top-level folders to coexist.


## [1.2.0 YANKED] More Efficient Graph Dataset Storing/Loading and Minor Improvements

## Requirements

- Pytorch >= 2.5.0
- Pytorch-Geometric >= 2.6.0

## Added

- Automatic check that data splits created by any splitter do not overlap. You can skip this check by passing `--skip-data-splits-check` to `mlwiz-data`
- Added argument `eval_test_every_epoch` to TrainingEngine, which now defaults to False, to avoid evaluating on the test set at every epoch during risk assessment runs. Test metrics must not be checked during training, so this saves time. 
- Added argument `--skip-config-ids` to skip a list of model selection configurations. This might be useful when some configurations are not terminating. So far, we have not added an option to specify outer folds. It should be used sparingly. Cannot be used together with `--execute-config-id`.

## Fixed

- Loading and storing graphs still makes use of `torch.load` and `torch.save`. Relies on PyG defining safe_globals after the recent Pytorch update (2.4)

## [1.1.2] Random Search Fix

## Added

- Now you can execute an arbitrary configuration first in debug mode. Just pass the argument `--debug --execute-config-id [config_id]` to the `mlwiz-exp` command.


## [1.1.1] Random Search Fix

## Fixed

- Random search breaks when you need to pass args to a class


## [1.1.0] Data Augmentation Improvements

## Fixed

- Bug in pre-transform and transform_train/eval not being parsed correctly

## Changed

- Ray always creates a new local instance when initialized, except when working with a cluster of machines.
- DatasetInterface `__init__()` has changed to differentiate between runtime `transform_train` and `transform_eval`. 
  This helps when training vision models that need to perform data augmentation in the training set.
  Note that the `IterableDatasetInterface` has a slightly different logic here than `DatasetInterface`. 
  The latter relies on two custom `SubsetTrain` and `SubsetEval` classes used by the `DataProvider`.

  
## [1.0.1] Improvements

## Changed

- Substituted `torch.load` with `dill` in some places to resolve annoying warnings
- Replaced `data_root` attributes with `storage_folder` for consistency
- Added tests for post-training utilities
- Improvements to workflow files

## Fixed
- 
- Fixed post-training utilities


## [1.0.0] First Release
