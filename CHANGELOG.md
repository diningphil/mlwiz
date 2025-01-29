# Changelog

## [1.2.0] More Efficient Graph Dataset Storing/Loading

## Requirements

- Pytorch >= 2.5.0
- Pytorch-Geometric >= 2.6.0

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
