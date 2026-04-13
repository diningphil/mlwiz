"""
Tests for :mod:`mlwiz.evaluation.bayesian_search`.
"""

from __future__ import annotations

import json
import sys

import pytest
import ray

from mlwiz.evaluation.bayesian_search import BayesianSearch
from mlwiz.evaluation.evaluator import BayesOptRiskAssesser
from mlwiz.static import (
    AVG,
    BAYES_SEARCH,
    BUDGET,
    CANDIDATE_POOL_SIZE,
    DATASET_CLASS,
    DATASET_GETTER,
    DATA_LOADER,
    DEVICE,
    EI_XI,
    EXPERIMENT,
    EXP_NAME,
    HIGHER_RESULTS_ARE_BETTER,
    MAIN_LOSS,
    MAIN_SCORE,
    MODEL_ASSESSMENT,
    RANDOM_STARTS,
    SCORE,
    STORAGE_FOLDER,
    TEST,
    evaluate_every,
    LOSS,
)
from mlwiz.ui.progress_manager import ProgressManager


class _FastExperiment:
    """Deterministic experiment used to exercise evaluator orchestration."""

    def __init__(
        self, model_configuration: dict, exp_path: str, exp_seed: int
    ):
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
        hp_id = float(self.model_configuration.get("hp_id", 0.0))
        train_res = {LOSS: {MAIN_LOSS: hp_id}, SCORE: {MAIN_SCORE: hp_id}}
        val_res = {
            LOSS: {MAIN_LOSS: hp_id + 0.1},
            SCORE: {MAIN_SCORE: hp_id + 0.1},
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
        hp_id = float(self.model_configuration.get("hp_id", 0.0))
        train_res = {LOSS: {MAIN_LOSS: hp_id}, SCORE: {MAIN_SCORE: hp_id}}
        val_res = {LOSS: {MAIN_LOSS: hp_id}, SCORE: {MAIN_SCORE: hp_id}}
        test_res = {
            LOSS: {MAIN_LOSS: hp_id + 0.2},
            SCORE: {MAIN_SCORE: hp_id + 0.2},
        }
        return train_res, val_res, test_res


def _make_base_config(tmp_path):
    """Build a minimal valid experiment configuration dictionary."""
    return {
        EXP_NAME: "bayes_test",
        STORAGE_FOLDER: str(tmp_path / "DATA"),
        DATASET_CLASS: "builtins.list",
        DATASET_GETTER: "mlwiz.data.provider.DataProvider",
        DATA_LOADER: {
            "class_name": "torch.utils.data.DataLoader",
            "args": {"num_workers": 0, "pin_memory": False},
        },
        EXPERIMENT: f"{__name__}._FastExperiment",
        HIGHER_RESULTS_ARE_BETTER: True,
        evaluate_every: 1,
        DEVICE: "cpu",
        RANDOM_STARTS: 2,
        CANDIDATE_POOL_SIZE: 64,
        EI_XI: 1e-3,
    }


def _patch_non_interactive_progress(monkeypatch):
    class _NoTtyStdin:
        """stdin stand-in that always reports non-interactive mode."""

        def isatty(self) -> bool:  # pragma: no cover
            return False

    monkeypatch.setattr(sys, "stdin", _NoTtyStdin())
    monkeypatch.setattr(
        ProgressManager, "_register_resize_handler", lambda _self: None
    )


def _run_bayesopt_evaluator(
    tmp_path,
    *,
    budget: int,
    outer_folds: int,
    inner_folds: int,
    model_selection_training_runs: int,
    risk_assessment_training_runs: int,
):
    cfg = _make_base_config(tmp_path)
    cfg[BUDGET] = budget
    cfg[BAYES_SEARCH] = {
        "hp_id": {
            "sample_method": "mlwiz.evaluation.util.choice",
            "args": list(range(max(5, budget))),
        }
    }

    search = BayesianSearch(cfg)
    search.telegram_config = None

    exp_path = (
        tmp_path
        / (
            "RESULTS_"
            f"o{outer_folds}_i{inner_folds}_"
            f"ms{model_selection_training_runs}_"
            f"fr{risk_assessment_training_runs}_b{budget}"
        )
    )

    ray.init(
        ignore_reinit_error=True,
        num_cpus=max(2, outer_folds * inner_folds * model_selection_training_runs),
        num_gpus=1,
    )
    try:
        evaluator = BayesOptRiskAssesser(
            outer_folds=outer_folds,
            inner_folds=inner_folds,
            experiment_class=_FastExperiment,
            exp_path=str(exp_path),
            splits_filepath=str(tmp_path / "dummy.splits"),
            model_configs=search,
            risk_assessment_training_runs=risk_assessment_training_runs,
            model_selection_training_runs=model_selection_training_runs,
            higher_is_better=True,
            gpus_per_task=0,
            base_seed=42,
        )
        evaluator.risk_assessment(debug=False)
    finally:
        ray.shutdown()

    return exp_path


def test_bayesian_search_ask_tell_respects_budget(tmp_path):
    """Bayesian search should lazily generate at most ``budget`` configs."""
    cfg = _make_base_config(tmp_path)
    cfg[BUDGET] = 5
    cfg[BAYES_SEARCH] = {
        "hp_id": {
            "sample_method": "mlwiz.evaluation.util.randint",
            "args": [0, 5],
        },
        "lr": {
            "sample_method": "mlwiz.evaluation.util.loguniform",
            "args": [1e-4, 1e-2],
        },
    }

    search = BayesianSearch(cfg)
    assert len(search) == 5
    assert all(config is None for config in search.hparams)

    for expected_id in range(5):
        proposal = search.ask(outer_k=0)
        assert proposal is not None
        config_id, config = proposal
        assert config_id == expected_id
        assert "hp_id" in config
        search.tell(outer_k=0, config_id=config_id, objective=float(config_id))

    assert search.ask(outer_k=0) is None


@pytest.mark.parametrize(
    "missing_key",
    [RANDOM_STARTS, CANDIDATE_POOL_SIZE, EI_XI],
)
def test_bayesian_search_requires_bo_hyperparams(tmp_path, missing_key):
    """BayesianSearch should require explicit BO hyper-parameters."""
    cfg = _make_base_config(tmp_path)
    cfg[BUDGET] = 5
    cfg[BAYES_SEARCH] = {
        "hp_id": {
            "sample_method": "mlwiz.evaluation.util.choice",
            "args": [0, 1, 2, 3, 4],
        }
    }
    cfg.pop(missing_key)

    with pytest.raises(KeyError, match=missing_key):
        BayesianSearch(cfg)


@pytest.mark.parametrize(
    "outer_folds,inner_folds,model_selection_training_runs,risk_assessment_training_runs",
    [
        (2, 2, 2, 2),
        (1, 1, 1, 1),
    ],
)
def test_bayesian_search_evaluator_non_debug(
    tmp_path,
    monkeypatch,
    outer_folds,
    inner_folds,
    model_selection_training_runs,
    risk_assessment_training_runs,
):
    """Adaptive Bayesian search should run end-to-end in non-debug mode."""
    _patch_non_interactive_progress(monkeypatch)
    exp_path = _run_bayesopt_evaluator(
        tmp_path,
        budget=5,
        outer_folds=outer_folds,
        inner_folds=inner_folds,
        model_selection_training_runs=model_selection_training_runs,
        risk_assessment_training_runs=risk_assessment_training_runs,
    )

    for outer_k in range(outer_folds):
        winner_path = (
            exp_path
            / MODEL_ASSESSMENT
            / f"OUTER_FOLD_{outer_k + 1}"
            / "MODEL_SELECTION"
            / "winner_config.json"
        )
        with open(winner_path, "r") as f:
            winner = json.load(f)
        assert winner["best_config_id"] in {1, 2, 3, 4, 5}
        assert winner["config"] is not None

    assessment_path = exp_path / MODEL_ASSESSMENT / "assessment_results.json"
    with open(assessment_path, "r") as f:
        assessment = json.load(f)
    assert f"{AVG}_{TEST}_{MAIN_SCORE}" in assessment
    assert isinstance(assessment[f"{AVG}_{TEST}_{MAIN_SCORE}"], float)


def test_bayesian_search_evaluator_budget_one(tmp_path, monkeypatch):
    """Budget of 1 should still complete and always select config 1."""
    _patch_non_interactive_progress(monkeypatch)
    exp_path = _run_bayesopt_evaluator(
        tmp_path,
        budget=1,
        outer_folds=1,
        inner_folds=1,
        model_selection_training_runs=1,
        risk_assessment_training_runs=1,
    )

    winner_path = (
        exp_path
        / MODEL_ASSESSMENT
        / "OUTER_FOLD_1"
        / "MODEL_SELECTION"
        / "winner_config.json"
    )
    with open(winner_path, "r") as f:
        winner = json.load(f)
    assert winner["best_config_id"] == 1
    assert winner["config"] is not None


@pytest.mark.parametrize(
    "execute_config_id,skip_config_ids,expected_message",
    [
        (
            1,
            [],
            "execute_config_id is not supported by BayesOptRiskAssesser",
        ),
        (
            None,
            [0],
            "skip_config_ids is not supported by BayesOptRiskAssesser",
        ),
    ],
)
def test_bayesopt_evaluator_rejects_skip_and_execute_controls(
    tmp_path,
    execute_config_id,
    skip_config_ids,
    expected_message,
):
    """BayesOptRiskAssesser should reject skip/priority controls."""
    cfg = _make_base_config(tmp_path)
    cfg[BUDGET] = 5
    cfg[BAYES_SEARCH] = {
        "hp_id": {
            "sample_method": "mlwiz.evaluation.util.choice",
            "args": [0, 1, 2, 3, 4],
        }
    }

    search = BayesianSearch(cfg)
    search.telegram_config = None

    evaluator = BayesOptRiskAssesser(
        outer_folds=1,
        inner_folds=1,
        experiment_class=_FastExperiment,
        exp_path=str(tmp_path / "RESULTS"),
        splits_filepath=str(tmp_path / "dummy.splits"),
        model_configs=search,
        risk_assessment_training_runs=1,
        model_selection_training_runs=1,
        higher_is_better=True,
        gpus_per_task=0,
        base_seed=42,
    )

    with pytest.raises(ValueError, match=expected_message):
        evaluator.model_selection(
            kfold_folder=str(tmp_path / "OUTER_FOLD_1"),
            outer_k=0,
            debug=True,
            execute_config_id=execute_config_id,
            skip_config_ids=skip_config_ids,
        )
