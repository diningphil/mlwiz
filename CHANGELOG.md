# Changelog

## [1.0.2] Improvements

## Changed

- Ray always creates a new local instance when initialized, except when working with a cluster of machines.


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
