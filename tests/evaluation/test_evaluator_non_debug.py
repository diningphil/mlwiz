"""
Integration-style test for the non-debug (Ray-distributed) evaluator path.

The existing evaluator test suite mainly exercises the debug=True (sequential)
code path. This test validates that the distributed scheduling + result
aggregation works end-to-end with ``debug=False`` without running expensive
training loops.
"""

from __future__ import annotations

import json

import pytest
import ray
import sys

from mlwiz.evaluation.evaluator import RiskAssesser
from mlwiz.evaluation.grid import Grid
from mlwiz.ui.progress_manager import ProgressManager
from mlwiz.static import (
    AVG,
    DATASET_CLASS,
    DATASET_GETTER,
    DATA_LOADER,
    DEVICE,
    EXPERIMENT,
    EXP_NAME,
    GRID_SEARCH,
    HIGHER_RESULTS_ARE_BETTER,
    MAIN_LOSS,
    MAIN_SCORE,
    MODEL_ASSESSMENT,
    SCORE,
    STORAGE_FOLDER,
    TEST,
    evaluate_every,
    LOSS,
)


class FastExperiment:
    """
    Minimal experiment returning deterministic results without training.

    Supports the signatures used by the Ray wrappers in
    :mod:`mlwiz.evaluation.evaluator`.
    """

    def __init__(
        self, model_configuration: dict, exp_path: str, exp_seed: int
    ):
        """
        Initialize the fast experiment stub.

        Args:
            model_configuration: Configuration dict for the current run.
            exp_path: Experiment output folder.
            exp_seed: Experiment seed (stored for completeness).
        """
        self.model_configuration = model_configuration
        self.exp_path = exp_path
        self.exp_seed = exp_seed

    def run_valid(
        self,
        dataset_getter,
        training_timeout_seconds,
        logger,
        progress_callback=None,
        should_terminate=None,
    ):
        """
        Return deterministic train/validation metrics for model selection runs.

        The value depends only on the ``hp_id`` hyperparameter, so the expected
        winning configuration is known a priori.
        """
        hp_id = int(self.model_configuration.get("hp_id", 0))
        base = float(hp_id)
        train_res = {LOSS: {MAIN_LOSS: base}, SCORE: {MAIN_SCORE: base}}
        val_res = {
            LOSS: {MAIN_LOSS: base + 0.1},
            SCORE: {MAIN_SCORE: base + 0.1},
        }
        return train_res, val_res

    def run_test(
        self,
        dataset_getter,
        training_timeout_seconds,
        logger,
        progress_callback=None,
        should_terminate=None,
    ):
        """
        Return deterministic train/validation/test metrics for final runs.

        The test metric is offset from the train/val metric to validate result
        aggregation and file creation.
        """
        hp_id = int(self.model_configuration.get("hp_id", 0))
        base = float(hp_id)
        train_res = {LOSS: {MAIN_LOSS: base}, SCORE: {MAIN_SCORE: base}}
        val_res = {LOSS: {MAIN_LOSS: base}, SCORE: {MAIN_SCORE: base}}
        test_res = {
            LOSS: {MAIN_LOSS: base + 0.2},
            SCORE: {MAIN_SCORE: base + 0.2},
        }
        return train_res, val_res, test_res


def test_evaluator_non_debug_mode(tmp_path, monkeypatch):
    """
    Run ``RiskAssesser.risk_assessment`` with ``debug=False`` on a tiny grid.

    Assertions focus on correct orchestration and artifact generation:
    - model selection chooses the higher-scoring config (hp_id=1),
    - final-run aggregation produces assessment results.
    """

    # Avoid starting the interactive input listener (cbreak mode) in TTYs.
    class _NoTtyStdin:
        """stdin stand-in that always reports non-interactive mode."""

        def isatty(self) -> bool:  # pragma: no cover
            """Return ``False`` to disable interactive input handling."""
            return False

    monkeypatch.setattr(sys, "stdin", _NoTtyStdin())
    monkeypatch.setattr(
        ProgressManager, "_register_resize_handler", lambda _self: None
    )

    configs_dict = {
        EXP_NAME: "fast_non_debug",
        STORAGE_FOLDER: str(tmp_path / "DATA"),
        DATASET_CLASS: "builtins.list",
        DATASET_GETTER: "mlwiz.data.provider.DataProvider",
        DATA_LOADER: {
            "class_name": "torch.utils.data.DataLoader",
            "args": {"num_workers": 0, "pin_memory": False},
        },
        EXPERIMENT: f"{__name__}.FastExperiment",
        HIGHER_RESULTS_ARE_BETTER: True,
        evaluate_every: 1,
        DEVICE: "cpu",
        GRID_SEARCH: {"hp_id": [0, 1]},
    }
    search = Grid(configs_dict)
    search.telegram_config = None

    exp_path = tmp_path / "RESULTS"

    # Ensure tasks can schedule even if the evaluator module was imported with
    # a non-zero GPU request (tests may set MLWIZ_RAY_NUM_GPUS_PER_TASK).
    ray.init(ignore_reinit_error=True, num_cpus=2, num_gpus=1)
    try:
        evaluator = RiskAssesser(
            outer_folds=1,
            inner_folds=1,
            experiment_class=FastExperiment,
            exp_path=str(exp_path),
            splits_filepath=str(tmp_path / "dummy.splits"),
            model_configs=search,
            risk_assessment_training_runs=1,
            model_selection_training_runs=1,
            higher_is_better=True,
            gpus_per_task=0,
            base_seed=42,
        )
        evaluator.risk_assessment(debug=False)
    finally:
        ray.shutdown()

    winner_path = (
        exp_path
        / MODEL_ASSESSMENT
        / "OUTER_FOLD_1"
        / "MODEL_SELECTION"
        / "winner_config.json"
    )
    with open(winner_path, "r") as f:
        winner = json.load(f)
    assert winner["best_config_id"] == 2

    assessment_path = exp_path / MODEL_ASSESSMENT / "assessment_results.json"
    with open(assessment_path, "r") as f:
        assessment = json.load(f)
    assert assessment[f"{AVG}_{TEST}_{MAIN_SCORE}"] == pytest.approx(1.2)
