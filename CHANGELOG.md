# Changelog

## [1.7.6]

## Changed

- hid dashboard HTTP request logs by default and added the opt-in
  `--show-logs` CLI option
- bumped the package version to 1.7.6

## [1.7.5]

## Added

- added the opt-in `resources.sync_batchnorm` setting for converting BatchNorm
  layers to synchronized BatchNorm before DDP wrapping; it defaults to `false`
  and is supported by grid, random, and Bayesian searches
- added `Min` and `Max` configuration-filter operators for selecting the
  current metric extrema independently within each outer fold, including ties
- added dashed, hoverable epoch-boundary markers to 2D step-mode plots and
  reproducible plot exports, with clearly labeled uniform estimates for legacy
  metric artifacts that predate exact epoch-boundary metadata

## Fixed

- made Hydra-style composition move overridden keys to their latest definition
  position, restoring bottom-to-top grid-variant priority for inline search
  parameters that override values imported through `defaults`

## Changed

- bumped the package version to 1.7.5

## [1.7.4]

## Added

- added shared curve smoothing to Model Selection Analysis trend and combined
  trend plots, including faint raw traces and reproducible plot exports
- added per-sample timestamps to new step histories and displayed them beside
  step IDs in Run Explorer and Model Selection Analysis trend hovers
- added repeatable rectangle zoom with stepwise zoom-out controls to 2D Run
  Explorer and Model Selection Analysis trends

## Fixed

- fixed Operators graph tracing for adaptive ESM models by keeping padding-mask
  handling and amino-acid pooling validation compatible with `torch.export`
- serialized operator-graph tracing and coalesced duplicate browser requests so
  auto-refresh cannot occupy PyTorch's proxy-dispatch tracing slot twice
- prioritized executable ATen operations over parameter placeholders when a
  large operator graph exceeds the dashboard's visible-node limit
- applied configuration filters to Model Selection Analysis plots and exports,
  and made long chart legends vertically scrollable and hover-focusable
- allowed the final configuration-filter condition to be removed without
  automatically recreating a default condition

## Changed

- moved Model Selection Analysis smoothing into a sticky plot-settings bar so
  it remains accessible while scrolling through analysis plots
- bumped the package version to 1.7.4

## [1.7.3]

## Added

- added exact-value configuration filtering for discovered, varying
  hyperparameters, including nested parameters, live run-manifest fallback,
  and mixed metric / hyperparameter AND/OR clauses
- added epoch/step unit selection to Model Selection Analysis trend and combined-trend plots, including sampled-step aggregation, axes, hover values, and reproducible Python exports
- added opt-in 1.5×IQR outlier removal for aggregate Run Explorer views and
  per-plot Model Selection Analysis trends, distributions, and tables, with
  removed-value counts and filtered reproducible exports
- added a default combined view for layer/component trend families, with a
  persistent per-card Together/Separate series control and complete legends
- added dedicated resolved-configuration inspectors when selecting a run or a
  model-selection configuration, including live manifest fallback and automatic
  first-view expansion of nested fields

## Fixed

- shaded and disabled the per-plot unit selector when a trend is available for
  epochs or steps only, while keeping the sole available unit visible
- fixed dashboard log scaling for metrics below one by using a conventional
  base-10 log axis for all-positive plots and an adaptive, data-scaled
  symmetric-log axis whenever zero or negative values are present; Python plot
  exports now use the same selection and threshold
- fixed adaptive/custom plotter integrations so sampled training-batch metrics are flushed before the current epoch finishes, allowing the dashboard to plot available step histories during long-running epochs
- fixed resumed step-wise logging so samples from an incomplete epoch after the last checkpoint are discarded and replaced when that epoch is replayed
- aligned model-selection step aggregates and combined trajectories by their recorded global step numbers instead of their positions in each sampled history

## Changed

- bumped the package version to 1.7.3

## [1.7.2]

## Added

- added opt-in training-step metric histories through `Plotter(store_every_N_steps=N)`; sampled losses and scores are stored with their global optimizer-step numbers while epoch-only logging remains the default
- added an automatic **Epoch / Step** selector to the dashboard run explorer when step histories are present, including true sampled-step axes, hover values, aggregation, and reproducible plot exports
- added a persistent **Smoothing** slider for run charts using TensorBoard-style bias-corrected exponential smoothing, with faint raw traces and raw/displayed data in reproducible plot exports

## Changed

- bumped the package version to 1.7.2

## [1.7.1]

## Added

- added a per-plot **</> Python** action to the dashboard that generates a standalone Matplotlib script containing the displayed plot data and reproducing the corresponding line, uncertainty-band, histogram, violin, heatmap-bar, or 3D visualization with `tueplots`
- added a plot-code configuration dialog with conference and journal bundles, single-column or full-width sizing, PDF/PNG/SVG/PGF output, optional LaTeX rendering, grid/title/legend controls, a live source preview, clipboard copy, and `.py` download
- added Paul Tol colorblind-safe palettes as the default plot-code export schemes, plus additional `tueplots` palettes; plot export preferences persist across browser sessions
- added persistent dashboard font-family and font-size controls that update both interface text and canvas-rendered plot labels dynamically
- all dashboard plot types now use one consistently named **Log scale** based on the sign-preserving log-modulus transform; it supports positive, zero, and negative values in run charts, 2D/3D trends, combined trajectories, histograms, heatmap bars and colors, violins, and generated Python code
- Combined Trends now provide a persistent per-card **Log scale** control that transforms both recorded-quantity axes
- 2D Trend and Metric-vs-Hyperparameter plots can now use **None — average all runs** instead of grouping by a hyperparameter; ungrouped aggregation is unavailable for Combined Trends and all 3D plots

## Changed

- bumped the package version to 1.7.1

## [1.7.0]

## Added

- added a live **Model Selection Analysis** dashboard tab with per-fold hyperparameter grouping, multi-plot trend and best-metric comparisons, automatic layer/component families, independent 2D/3D views, combined trajectories, histogram/heatmap and violin renderings, Markdown export, and persistent interactive plot controls
- added versioned, compressed `.mlwiz` dashboard snapshots for sharing normalized experiment metrics, result metadata, filters, and browser state without sharing the original result directory, checkpoints, or project-specific model classes
- added `mlwiz-dashboard-export` for exporting every experiment under a result root by default, with an optional dashboard path to limit the archive to one containing experiment
- added `mlwiz-dashboard-import` for validating a portable snapshot and serving it through an ad-hoc, read-only local dashboard without extracting files or deserializing Torch/pickle payloads
- added an **Export all** dashboard action and a tutorial guide covering sender/recipient workflows, archive inspection, privacy considerations, scoped exports, and immutable point-in-time behavior
- added Hydra-style modular YAML composition through ordered `defaults` lists, config-group directories, `_self_` ordering, recursive nested defaults, and package overrides (`@package`, `@_here_`, and `@_global_`)
- experiment configs now require structured `dataset`, `resources`, `reproducibility`, `data_loading`, and `experiment` sections plus exactly one `grid`, `random`, or `bayes` section
- `grid`, `random`, and `bayes` support their own local `defaults` lists, keeping model/optimizer/training choices separate from global runtime settings
- config-group files may contain multiple search alternatives; selecting multiple files concatenates all alternatives for grid search and exposes all of them as a categorical dimension for random and Bayesian search
- added modular optimizer examples under `examples/MODEL_CONFIGS/optimizer/`

## Changed

- last and best model checkpoints retain their existing names (`last_checkpoint.pth` and `best_checkpoint.pth`) but no longer contain optimizer, scheduler, or AMP scaler state
- optimizer-related state is now stored in `last_optimizer_checkpoint.pth` and `best_optimizer_checkpoint.pth`, preventing the dashboard and inference-only consumers from loading unnecessary optimizer tensors
- experiment, data-building, and config-duplication entrypoints now resolve modular configurations before use
- random-search and Bayesian-search budgets and Bayesian acquisition controls now live inside their corresponding search sections
- bumped the package version to 1.7.0

## Compatibility

- portable dashboard archives use snapshot format version 1; import validates the format and rejects unsupported versions instead of attempting a best-effort load
- when a separate optimizer checkpoint is absent during resume, MLWiz reads optimizer, scheduler, and scaler state from the model checkpoint, preserving compatibility with checkpoints created by releases up to 1.6.x
- flat pre-1.7.0 experiment configurations are intentionally unsupported and fail with an explicit schema-migration error; data-building configuration files are unaffected

## [1.6.3]

## Added

- added Architecture and Operators views to the Model graph panel; Operators lazily traces the selected checkpoint with `torch.export` and displays its ATen computational graph
- new runs persist a data-free first-forward tensor shape/dtype specification so operator graphs can be generated without storing training examples

## Changed

- Architecture view uses a Model Explorer-style collapsible hierarchy with expand/collapse-all, module search, a flattened leaf view, and block colors based on subtree parameters relative to total model parameters
- Operators view presents a directed module-level execution graph with arrowheads, recursively expandable module boundaries, persisted expansion/zoom/node placement, mouse-wheel zoom, background drag-to-pan, and freely draggable module/operator boxes instead of a flat operator list
- architecture and operator graphs use separate bounded cache entries, so switching views never returns a graph generated for the other mode

## Fixed

- the Model graph panel keeps an independent run selector while metric plots are grouped by inner fold

## [1.6.2]

## Added

- added interactive metric charts with epoch hover values and a symmetric logarithmic scale that supports positive, zero, and negative values
- added per-experiment configuration filters for training or validation metrics, configurable comparison operators, and combined `AND`/`OR` clauses
- added configurable dashboard metric-cache size and automatic refresh interval controls
- added a dashboard cache-reset button that clears in-memory metric histories and cache counters while preserving the configured memory limit
- added dark and day themes, with dark mode as the default
- added a collapsible overview of completed, running, and failed runs plus experiment timing statistics for the selected experiment
- added a collapsible configuration/result inspector with nested structured and raw JSON views
- added a plot mode for grouping model-selection curves by inner fold, with an optional mean ± standard deviation checkbox, plus an all-final-runs aggregation view
- added a sticky plot navigator with fold/run selectors, previous/next controls, focused-first defaults, and a `Show all` override for model-selection configurations
- added a lazy checkpoint-backed Model graph panel with module details, parameter shapes, dynamic run selection, and compatibility fallback for runs created before model manifests
- added per-run Auto/Best/Last checkpoint selection to the collapsed Model graph panel, which now appears directly below configuration statistics
- new runs persist `model_manifest.json` with resolved model reconstruction inputs for checkpoint architecture inspection

## Changed

- the dashboard now loads `metrics_data.torch` files on demand and retains them in a bounded LRU memory cache; a single oversized selection is still loaded but is not retained in the cache
- completed configuration filters use the experiment-selected best metric value, while running configurations use the latest available metric value
- opening an object or array nested beneath `config` in the dashboard inspector now expands its complete descendant tree in one action
- sibling final-run metrics are loaded on demand only when the all-final-runs aggregation view is selected
- running model graphs prefer `last_checkpoint.pth`, completed graphs prefer `best_checkpoint.pth`, and both fall back to the other existing checkpoint without introducing separate checkpoint files
- model graph loading is skipped when the selected checkpoint file exceeds the user-configured dashboard cache ceiling

## Fixed

- dashboard selections, sidebar disclosures, filters, and chart controls now remain stable across automatic refreshes; the day/dark theme choice also persists across browser sessions
- configuration inspector mode, nested disclosures, and scroll position now remain stable across automatic refreshes
- filters with no matching configurations now show an empty result instead of unfiltered configurations
- dashboard configuration counts no longer include final runs
- the central selection prompt is hidden after a configuration or run is opened
- the plot navigator now remains pinned to the top of the viewport while scrolling through charts
- changing the focused inner fold or run no longer resets the chart-page scroll position
- the sticky plot navigator gains a higher-contrast border and shadow while detached from its original page position

## [1.6.1]

## Changed

- `Plotter` now persists metrics every epoch by default to work with the dashboard (we abandoned tensorboard); configure `store_every_N_epochs` to change the interval or set it to `None` to write only on termination

## [1.6.0]

## Added

- new training callback event `on_termination`, dispatched when training ends regularly and when it is interrupted (e.g., CTRL-C / external termination signal)
- `Plotter` now supports `store_every_N_epochs` to flush `metrics_data.torch` every `N` epochs when `store_on_disk=True`
- added unit coverage to verify termination hooks run on both normal completion and interruption
- training logs are now buffered and flushed to `experiment.log` every `N` epochs via `engine.args.store_log_every_N_epochs` (default: `1`), with a forced flush on termination and at training end
- added CLI option `--detailed-gui` to toggle detailed per-run progress UI in non-debug mode; without it, MLWiz keeps the lightweight/global view (detailed GUI disabled). Use it when parallelism is low, otherwise it can spawn many threads
- added `mlwiz-dashboard`, a local web app for browsing model-selection configurations, final runs, epoch metrics, and result metadata

## Changed

- `Plotter` now keeps metrics in memory during epochs and persists them on `on_termination` (or every `N` epochs when configured), instead of writing to disk every epoch by default
- `Plotter` now persists `metrics_data.torch` for MLWiz Dashboard by default

## Removed

- removed the TensorBoard integration and runtime dependency in favor of `mlwiz-dashboard`

## [1.5.3]

## Fixed

- in DDP runs with dynamic architectures (e.g., adaptive width neural networks), epoch-end checkpoint collection now executes model `state_dict()` on all ranks (while keeping disk writes on main rank only) to preserve collective ordering and prevent cross-rank desynchronization/hangs. Indeed, calling `state_dict()` may cause the architecture to resize its width.

## [1.5.2]

## Fixed

- kept optimizer `param_names` metadata aligned when dynamic architectures add, replace, or remove parameters
- optimizer restore now fails clearly when named optimizer metadata is inconsistent, instead of silently falling back to unsafe state mapping

## [1.5.1]

## Added

- Bayesian Optimization search method

## Changed

- checkpoint snapshots now clone model/optimizer/scaler/scheduler payloads to CPU before serialization, reducing transient GPU memory spikes during epoch-end and best-model saves
- checkpoint loading now always maps tensors to CPU first (then moves model weights to the target device), avoiding large direct CUDA allocations during restore/load utilities
- in DDP mode, `batch_size` is now interpreted as the global batch size and divided by world size to build per-rank loaders; non-divisible values now raise a clear error
- optimizer checkpoints now include parameter-name metadata (via `named_parameters`) and optimizer-state restore now remaps by parameter name when possible, while keeping legacy order-based loading for checkpoints without names

## Fixed

- Badge creation in workflow (untested, done with Codex)
- DDP validation/test data loaders are now rank-sharded as well (not only training), by forwarding `ddp_rank`/`ddp_world_size` to eval loader creation
- with rank-sharded DDP evaluation, scalar epoch metrics are now reduced across ranks before returning inference results
- reduced DDP scalar metrics are now always stored on CPU after cross-rank reduction
- uncaught experiment exceptions are now persisted to `<run_folder>/experiment.err` so per-run failures are inspectable from disk in both debug and non-debug execution paths
- added unit tests covering both name-based optimizer-state loading and legacy order-based fallback behavior


## [1.5.0] Pytorch Distributed Data Parallel and Automatic Mixed-Precision Support

## Added

- added Distributed Data Parallel (DDP) training support that is automatically enabled when `device: cuda` and `gpus_per_task` is an integer greater than 1
- added DDP-aware data loading through `DistributedSampler` for training splits, so each rank processes a distinct shard
- added a DDP example configuration file at `examples/MODEL_CONFIGS/config_MLP_ddp.yml` and a lightweight smoke test script at `examples/toy_ddp_training_smoke.py`
- added optional AMP configuration (`engine.args.mixed_precision`, `engine.args.mixed_precision_dtype`) with dotted dtype paths (e.g. `torch.float16`) and autocast support on CUDA/CPU
- added configurable ordered model-selection criteria via `model_selection_criteria` (lexicographic comparison with per-criterion direction)


## Changed

- evaluator/experiment execution now forwards progress and termination signals correctly in both debug and non-debug modes when DDP is active
- distributed helper utilities are now centralized in `mlwiz.training.distributed` for reuse across engine/experiment code paths
- rank-0-only side effects are enforced for distributed runs (logging/progress/checkpoint-related writes), reducing file contention across ranks
- model selection now supports metrics from both loss and score aggregates, and non-main metrics must explicitly specify `source: loss|score`
- configuration validation now raises an error when both `model_selection_criteria` and `higher_results_are_better` are provided
- updated `MODEL_CONFIGS` examples/integration configs and templates to use `model_selection_criteria` as the default specification


## Removed

- Creation of data list at the end of the experiment, which was originally meant to support constructive models which are not frequently used. The overhead in the library is not justified. 

## [1.4.2] Version bump

## Changed

- bumped package version to 1.4.2 in `pyproject.toml`
- bumped docs release version to 1.4.2 in `docs/conf.py`
- improved plotter to allow disabling tensorboard logging and only store metrics on disk

## [1.4.1] Minor fix in toml

## Fixed

- added mlwiz.ui to TOML file
- fixed test due to Torch >= 1.9

## [1.4.0] Refactoring, adding Docstrings, more Tests, improvements

## Changed

- Better handling of terminal reshaping
- Minor refactoring of code in evaluator and progress manager to improve readability.
- Automatic addition of docstrings for all functions using Codex and GPT-5.2
- Replaced asserts with proper error raise
- Imported static variables that are strictly necessary in each file
- Automatic improvement of workflow according to standard practices using Codex and GPT-5.2
- When resuming cached model selection or final runs, recompute elapsed time from the experiment log (falling back to cached value if missing) so durations reflect restarts.

## Fixed

- Minor fix in evaluator for debug mode

## Added

- Automatic addition of tests to cover untested functions using Codex and GPT-5.2

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
