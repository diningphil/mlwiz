# Changelog

## [1.3.4] Cached elapsed recomputation

## Changed

- When resuming cached model selection or final runs, recompute elapsed time from the experiment log (falling back to cached value if missing) so durations reflect restarts.

## [1.3.3] Final run resume fix

## Fixed

- Prevent duplicate Ray waitables when skipping cached final runs so restarts no longer fail with `ValueError: Wait requires a list of unique ray_waitables`.

## [1.3.2] Utilities improvements

## Changed

- `create_latex_table_from_assessment_results` does not break if it does not find an experiment's folder
- No need to launch a ray job when resuming only to check if an experimetn has finished. This dramatically accelerates the restart when many configurations are launched, avoid frustrating waitings.

## [1.3.1] Minor fixes

## Added

- more comments in the progress_manager cause the UI code is becoming challenging to parse

## Fixed

- atomic dill save of results
- fixed MLWIZ_RAY_NUM_GPUS_PER_TASK not found by the remote tasks, which caused using more memory than set in the config file
- fixed minor bug of skip-config-ids. The right best config was not selected in this case.
- fixed minor bug in handling best_config ids to display

## [1.3.0] Majorly Improved CLI and Added Statistical Significance Utilities 

## Added

- In non-debug mode, you can now switch between the "global" view and a specific configuration
- Computation of 95% confidence interval in addition to the standard deviation
- Utility computation of statistical significance between models using `mlwiz.evaluation.util.compare_statistical_significance`. See the tutorial for more info.
- Handling termination gracefully via CTRL-C

## Changed

- Handling errors more gracefully in the terminal, can check individual errors in non-debug mode already from terminal
- Using atomic torch save util to save tensorboard logs

## [1.2.7] Out-of-time (OOT) experiments

## Added

- You can now specify the max time (in seconds) for a single experiment run in the configuration file using the key `training_timeout_seconds`. By default, this is clearly disabled (set to -1). This also handles the case when experiments are stopped and resumed (of course, checkpointing has to be enabled otherwise the state is not carried over)

## Fixed

- Updated tutorial

## [1.2.6] Fixes

## Fixed

- Typos in tutorial
- Printing of config_id to be executed in debug mode
- fix in `create_dataframe` utility that was duplicating entries

## [1.2.5] More Utilities, experiment time improvement and fixed file extensions

## Fixed

- the total time taken to run experiments now reads from the `experiment.log` file, to account for multiple restarts 
- fixed the extensions of result files that now use dill rather than torch 

## Added

- New command `mlwiz-config-duplicator.py --base-exp-config <base_exp_config> --data-config-files <data_config_files>"` to duplicate the experiment configuration file of one model across many datasets, using the already defined configuration files.
- New function `mlwiz.evaluation.util.create_latex_table_from_assessment_results` that automatically produces a Latex Table with datasets as columns and models as rows, or viceversa.

## [1.2.4] Minor Fixes

## Added

- `pytest.ini` to resolve warnings at test time

## Changed

- No need to specify a metric name any longer. Now it is taken from the class name directly.

## Fixed

- Changed name of metric from MeanAverageError to MeanAbsoluteError

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
