"""
Unit tests for :mod:`mlwiz.evaluation.util`.

These tests focus on the pure/IO-light helper routines used to inspect
experiment artifacts (JSON/Dill), post-process results, and compute simple
statistics, avoiding any heavy training or UI logic.
"""

from __future__ import annotations

import json

import dill
import numpy as np
import pandas as pd
import pytest
import torch

from mlwiz.evaluation.util import (
    _collect_metric_samples,
    _list_outer_fold_ids,
    _load_final_run_metric_samples,
    create_dataframe,
    create_latex_table_from_assessment_results,
    filter_experiments,
    get_scores_from_assessment_results,
    get_scores_from_outer_results,
    instantiate_data_provider_from_config,
    instantiate_dataset_from_config,
    instantiate_model_from_config,
    load_checkpoint,
    retrieve_best_configuration,
    retrieve_experiments,
    statistical_significance,
)
from mlwiz.static import (
    CONFIG,
    DATASET_CLASS,
    DATASET_GETTER,
    DATA_LOADER,
    MODEL,
    MODEL_ASSESSMENT,
    MODEL_STATE,
    SCORE,
    STORAGE_FOLDER,
)


class DummyDataset:
    """Minimal dataset used for config instantiation tests."""

    def __init__(self, storage_folder: str):
        """
        Initialize the dataset stub.

        Args:
            storage_folder: Path passed through by config instantiation helpers.
        """
        self.storage_folder = storage_folder
        self.dim_input_features = 3
        self.dim_target = 2


def dummy_dataset_getter(**kwargs):
    """Return received kwargs to make call assertions easy."""

    return kwargs


class DummyModel:
    """Minimal model class used for config instantiation / checkpoint tests."""

    def __init__(self, dim_input_features: int, dim_target: int, config: dict):
        """
        Initialize the model stub.

        Args:
            dim_input_features: Input feature dimension.
            dim_target: Target dimension.
            config: Model configuration dictionary.
        """
        self.dim_input_features = dim_input_features
        self.dim_target = dim_target
        self.config = config
        self.loaded_state = None

    def load_state_dict(self, state_dict: dict):
        """
        Store a provided state dict for later assertions.

        Args:
            state_dict: Serialized model weights/state.
        """
        self.loaded_state = state_dict


def _write_outer_results(path, metric_key: str, train: float, val: float, test: float):
    """
    Write an ``outer_results.json`` file with a minimal metric payload.

    Args:
        path: Destination file path.
        metric_key: Metric name key to write.
        train: Training score.
        val: Validation score.
        test: Test score.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "outer_train": {metric_key: train, f"{metric_key}_std": 0.0},
        "outer_validation": {metric_key: val, f"{metric_key}_std": 0.0},
        "outer_test": {metric_key: test, f"{metric_key}_std": 0.0},
    }
    path.write_text(json.dumps(payload))


def _write_assessment_results(path, metric_key: str, train: float, val: float, test: float):
    """
    Write an ``assessment_results.json`` file with a minimal metric payload.

    Args:
        path: Destination file path.
        metric_key: Metric name key to write.
        train: Training score.
        val: Validation score.
        test: Test score.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        f"avg_training_{metric_key}": train,
        f"avg_validation_{metric_key}": val,
        f"avg_test_{metric_key}": test,
        f"std_training_{metric_key}": 0.0,
        f"std_validation_{metric_key}": 0.0,
        f"std_test_{metric_key}": 0.0,
    }
    path.write_text(json.dumps(payload))


def _write_final_run_results(path, metric_key: str, test_value: float):
    """
    Write a serialized final-run results tuple to a dill file.

    Args:
        path: Destination file path.
        metric_key: Score key to include.
        test_value: Test score value to store.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    training_res = {SCORE: {metric_key: 0.0}}
    validation_res = {SCORE: {metric_key: 0.0}}
    test_res = {SCORE: {metric_key: test_value}}
    with open(path, "wb") as f:
        dill.dump((training_res, validation_res, test_res, None), f)


def test_retrieve_experiments_and_best_config(tmp_path):
    """
    Retrieve config results from a model-selection folder and load the winner config.

    This covers the expected folder scanning behavior and the ``skip_results_not_found``
    option used while experiments are still running.
    """
    model_selection_folder = tmp_path / "MODEL_SELECTION"
    model_selection_folder.mkdir()

    # One valid config and one missing results file.
    config_1 = model_selection_folder / "config_1"
    config_1.mkdir()
    (config_1 / "config_results.json").write_text(json.dumps({"foo": "bar"}))

    config_2 = model_selection_folder / "config_2"
    config_2.mkdir()

    # A non-config folder should be ignored.
    (model_selection_folder / "notes").mkdir()

    results = retrieve_experiments(
        str(model_selection_folder), skip_results_not_found=True
    )
    assert len(results) == 1
    assert results[0]["foo"] == "bar"
    assert results[0]["exp_folder"] == str(config_1)

    winner = {"winner": True, "config_id": 1}
    (model_selection_folder / "winner_config.json").write_text(json.dumps(winner))
    assert retrieve_best_configuration(str(model_selection_folder)) == winner


def test_create_dataframe_and_filter_experiments_nested_keys():
    """
    Validate nested-key extraction and AND/OR filtering semantics.

    ``create_dataframe`` should tolerate missing keys (filled as ``None``),
    while ``filter_experiments`` is strict and raises when a requested key is
    absent in a configuration.
    """
    configs = [
        {
            "exp_folder": "/tmp/exp_a",
            "device": "cpu",
            "model": {"hidden": 32},
            "optim": {"lr": 0.1},
        },
        {
            "exp_folder": "/tmp/exp_b",
            "device": "cuda",
            "model": {"hidden": 64},
            "optim": {"lr": 0.01},
        },
    ]

    df = create_dataframe(
        configs,
        [
            ("device", str),
            ("hidden", int),
            ("lr", float),
            ("missing_hp", str),
        ],
    )
    assert list(df.columns) == ["device", "hidden", "lr", "missing_hp", "exp_folder"]
    assert df.loc[0, "hidden"] == 32
    assert df.loc[1, "lr"] == pytest.approx(0.01)
    assert pd.isna(df.loc[0, "missing_hp"])

    filtered = filter_experiments(
        configs, logic="AND", parameters={"device": "cpu", "hidden": [32, 64]}
    )
    assert [c["exp_folder"] for c in filtered] == ["/tmp/exp_a"]

    filtered_or = filter_experiments(
        configs, logic="OR", parameters={"device": "cuda", "hidden": 32}
    )
    assert {c["exp_folder"] for c in filtered_or} == {"/tmp/exp_a", "/tmp/exp_b"}

    with pytest.raises(ValueError, match="AND/OR"):
        filter_experiments(configs, logic="XOR", parameters={"device": "cpu"})

    with pytest.raises(ValueError, match="cannot be empty"):
        filter_experiments(configs, logic="AND", parameters={"device": []})

    with pytest.raises(ValueError, match="not found in the configuration"):
        filter_experiments(configs, logic="AND", parameters={"unknown": 1})


def test_scores_and_latex_table_generation(tmp_path):
    """
    Load assessment/outer-fold scores and render a simple LaTeX table.
    """
    exp_folder = tmp_path / "exp_1"
    metric_key = "main_score"

    _write_assessment_results(
        exp_folder / MODEL_ASSESSMENT / "assessment_results.json",
        metric_key,
        train=0.1,
        val=0.2,
        test=0.3,
    )
    _write_outer_results(
        exp_folder
        / MODEL_ASSESSMENT
        / "OUTER_FOLD_1"
        / "outer_results.json",
        metric_key,
        train=0.11,
        val=0.21,
        test=0.31,
    )

    scores = get_scores_from_assessment_results(str(exp_folder), metric_key)
    assert scores["test"] == pytest.approx(0.3)
    assert scores["test_std"] == pytest.approx(0.0)

    outer_scores = get_scores_from_outer_results(str(exp_folder), 1, metric_key)
    assert outer_scores["validation"] == pytest.approx(0.21)
    assert outer_scores["validation_std"] == pytest.approx(0.0)

    latex = create_latex_table_from_assessment_results(
        [(str(exp_folder), "DummyModel", "DummyDataset")],
        metric_key=metric_key,
        no_decimals=2,
        model_as_row=True,
    )
    assert isinstance(latex, str)
    assert "DummyModel" in latex
    assert "DummyDataset" in latex


def test_metric_samples_collection_and_statistical_significance(tmp_path):
    """
    Collect samples from either final runs or outer fold means and compare experiments.
    """
    metric_key = "main_score"

    # Single outer fold => use final run results as samples.
    exp_final_runs = tmp_path / "exp_final_runs"
    run_1 = (
        exp_final_runs
        / MODEL_ASSESSMENT
        / "OUTER_FOLD_1"
        / "final_run1"
        / "run_1_results.dill"
    )
    run_2 = (
        exp_final_runs
        / MODEL_ASSESSMENT
        / "OUTER_FOLD_1"
        / "final_run2"
        / "run_2_results.dill"
    )
    _write_final_run_results(run_1, metric_key, test_value=0.2)
    _write_final_run_results(run_2, metric_key, test_value=0.4)

    assert _list_outer_fold_ids(str(exp_final_runs)) == [1]
    samples, source = _collect_metric_samples(
        str(exp_final_runs), metric_key, set_key="TEST"
    )
    assert source == "final_runs"
    assert np.allclose(samples, np.array([0.2, 0.4], dtype=float))

    with pytest.raises(ValueError, match="set_key must be one of"):
        _collect_metric_samples(str(exp_final_runs), metric_key, set_key="oops")

    with pytest.raises(ValueError, match="No final run results found"):
        _load_final_run_metric_samples(
            str(tmp_path / "missing_runs"), 1, "test", metric_key
        )

    # Multiple outer folds => use outer fold means as samples.
    exp_outer_means = tmp_path / "exp_outer_means"
    for fold_id, test_score in enumerate([0.10, 0.11, 0.12], start=1):
        _write_outer_results(
            exp_outer_means
            / MODEL_ASSESSMENT
            / f"OUTER_FOLD_{fold_id}"
            / "outer_results.json",
            metric_key,
            train=0.0,
            val=0.0,
            test=test_score,
        )

    samples, source = _collect_metric_samples(
        str(exp_outer_means), metric_key, set_key="test"
    )
    assert source == "outer_fold_means"
    assert np.allclose(samples, np.array([0.10, 0.11, 0.12], dtype=float))

    exp_outer_means_comp = tmp_path / "exp_outer_means_comp"
    for fold_id, test_score in enumerate([0.40, 0.41, 0.42], start=1):
        _write_outer_results(
            exp_outer_means_comp
            / MODEL_ASSESSMENT
            / f"OUTER_FOLD_{fold_id}"
            / "outer_results.json",
            metric_key,
            train=0.0,
            val=0.0,
            test=test_score,
        )

    res_df = statistical_significance(
        (str(exp_outer_means), "RefModel", "D"),
        [(str(exp_outer_means_comp), "CompModel", "D")],
        metric_key=metric_key,
        set_key="Test",
        confidence_level=0.95,
    )
    assert len(res_df) == 1
    assert bool(res_df.loc[0, "statistically_significant"]) is True

    # Edge case: only one sample per model -> Welch t-test returns nan, which is mapped to p=1.0.
    exp_one_sample_ref = tmp_path / "exp_one_sample_ref"
    exp_one_sample_comp = tmp_path / "exp_one_sample_comp"
    _write_final_run_results(
        exp_one_sample_ref
        / MODEL_ASSESSMENT
        / "OUTER_FOLD_1"
        / "final_run1"
        / "run_1_results.dill",
        metric_key,
        test_value=0.1,
    )
    _write_final_run_results(
        exp_one_sample_comp
        / MODEL_ASSESSMENT
        / "OUTER_FOLD_1"
        / "final_run1"
        / "run_1_results.dill",
        metric_key,
        test_value=0.2,
    )
    nan_df = statistical_significance(
        (str(exp_one_sample_ref), "Ref", "D"),
        [(str(exp_one_sample_comp), "Comp", "D")],
        metric_key=metric_key,
        set_key="test",
        confidence_level=0.95,
    )
    assert nan_df.loc[0, "p_value"] == pytest.approx(1.0)
    assert bool(nan_df.loc[0, "statistically_significant"]) is False


def test_config_instantiation_and_checkpoint_loading(tmp_path):
    """
    Instantiate dataset/provider/model from dotted-path config and load a checkpoint.
    """
    config = {
        CONFIG: {
            STORAGE_FOLDER: str(tmp_path / "storage"),
            DATASET_CLASS: f"{__name__}.DummyDataset",
            DATASET_GETTER: f"{__name__}.dummy_dataset_getter",
            DATA_LOADER: "torch.utils.data.DataLoader",
            MODEL: f"{__name__}.DummyModel",
        }
    }

    dataset = instantiate_dataset_from_config(config)
    assert isinstance(dataset, DummyDataset)
    assert dataset.storage_folder == str(tmp_path / "storage")

    dp_kwargs = instantiate_data_provider_from_config(
        config,
        splits_filepath=str(tmp_path / "splits.splits"),
        n_outer_folds=2,
        n_inner_folds=3,
    )
    assert dp_kwargs["storage_folder"] == str(tmp_path / "storage")
    assert dp_kwargs["outer_folds"] == 2
    assert dp_kwargs["inner_folds"] == 3
    assert dp_kwargs["dataset_class"] is DummyDataset
    assert dp_kwargs["data_loader_args"] == {}

    model = instantiate_model_from_config(config, dataset)
    assert isinstance(model, DummyModel)
    assert model.dim_input_features == dataset.dim_input_features
    assert model.dim_target == dataset.dim_target

    ckpt_path = tmp_path / "ckpt.pth"
    torch.save({MODEL_STATE: {"w": torch.tensor([1.0, 2.0])}}, ckpt_path)
    load_checkpoint(str(ckpt_path), model, device="cpu")
    assert isinstance(model.loaded_state, dict)
    assert torch.equal(model.loaded_state["w"], torch.tensor([1.0, 2.0]))
