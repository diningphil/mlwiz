"""Tests for Hydra-style modular configuration composition."""

from __future__ import annotations

from copy import deepcopy

import pytest
import yaml

from mlwiz.config_loader import (
    ConfigCompositionError,
    load_config,
    load_experiment_config,
)
from mlwiz.evaluation.grid import Grid
from mlwiz.evaluation.random_search import RandomSearch
from mlwiz.static import ARGS, SAMPLE_METHOD


def _write_yaml(path, value):
    """Write one YAML fixture, creating config-group directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def test_multiple_group_files_concatenate_grid_configuration_sets(tmp_path):
    """Every alternative from every selected optimizer file enters the grid."""
    _write_yaml(
        tmp_path / "optimizer" / "adam.yml",
        [
            {"class_name": "Optimizer", "args": {"kind": "adam", "lr": [0.1, 0.01]}},
            {"class_name": "Optimizer", "args": {"kind": "adamw", "lr": [0.001]}},
        ],
    )
    _write_yaml(
        tmp_path / "optimizer" / "sgd.yml",
        [{"class_name": "Optimizer", "args": {"kind": "sgd", "lr": [1.0]}}],
    )
    _write_yaml(
        tmp_path / "config.yml",
        {
            "grid": {
                "defaults": [{"optimizer": ["adam", "sgd"]}, "_self_"],
                "model": "Model",
                "epochs": 2,
            },
        },
    )

    config = load_config(tmp_path / "config.yml")

    assert "defaults" not in config
    alternatives = config["grid"]["optimizer"]
    assert [item["args"]["kind"] for item in alternatives] == [
        "adam",
        "adamw",
        "sgd",
    ]
    assert alternatives[0]["args"]["lr"] == [0.1, 0.01]
    expanded = list(object.__new__(Grid)._gen_helper({"optimizer": alternatives}))
    assert [item["optimizer"]["args"]["kind"] for item in expanded] == [
        "adam",
        "adam",
        "adamw",
        "sgd",
    ]


def test_search_component_group_is_local_to_search_defaults(tmp_path):
    """A search-local defaults list packages components inside that search."""
    _write_yaml(
        tmp_path / "optimizer" / "adam.yml",
        {"class_name": "Optimizer", "args": {"kind": "adam"}},
    )
    _write_yaml(
        tmp_path / "config.yml",
        {
            "grid": {
                "defaults": [{"optimizer": "adam"}],
                "model": "Model",
            },
        },
    )

    config = load_config(tmp_path / "config.yml")

    assert "optimizer" not in config
    assert config["grid"]["optimizer"]["args"]["kind"] == "adam"


def test_self_order_applies_to_short_search_component_groups(tmp_path):
    """A short group before/after ``_self_`` follows normal override order."""
    _write_yaml(
        tmp_path / "optimizer" / "modular.yml",
        [{"kind": "first"}, {"kind": "second"}],
    )
    _write_yaml(
        tmp_path / "inline_wins.yml",
        {
            "grid": {
                "defaults": [{"optimizer": "modular"}, "_self_"],
                "model": "Model",
                "optimizer": {"kind": "inline"},
            }
        },
    )
    _write_yaml(
        tmp_path / "modular_wins.yml",
        {
            "grid": {
                "defaults": ["_self_", {"optimizer": "modular"}],
                "model": "Model",
                "optimizer": {"kind": "inline"},
            }
        },
    )

    assert load_config(tmp_path / "inline_wins.yml")["grid"]["optimizer"] == {
        "kind": "inline"
    }
    assert [
        value["kind"]
        for value in load_config(tmp_path / "modular_wins.yml")["grid"]["optimizer"]
    ] == ["first", "second"]


@pytest.mark.parametrize("search_section", ["random", "bayes"])
def test_multiple_group_files_become_categorical_sample_set(
    tmp_path, search_section, monkeypatch
):
    """Random/adaptive searches sample across all modular alternatives."""
    _write_yaml(
        tmp_path / "optimizer" / "adaptive.yml",
        [
            {
                "kind": "adam",
                "lr": {
                    SAMPLE_METHOD: "mlwiz.evaluation.util.choice",
                    ARGS: [0.1, 0.01],
                },
            },
            {"kind": "adamw", "lr": 0.01},
        ],
    )
    _write_yaml(tmp_path / "optimizer" / "sgd.yml", {"kind": "sgd", "lr": 1.0})
    _write_yaml(
        tmp_path / "config.yml",
        {
            search_section: {
                "defaults": [{"optimizer": ["adaptive", "sgd"]}],
                "model": "Model",
            },
        },
    )

    config = load_config(tmp_path / "config.yml")
    optimizer = config[search_section]["optimizer"]

    assert optimizer[SAMPLE_METHOD] == "mlwiz.evaluation.util.choice"
    assert [item["kind"] for item in optimizer[ARGS]] == [
        "adam",
        "adamw",
        "sgd",
    ]
    if search_section == "random":
        samples = iter([optimizer[ARGS][0], 0.1])
        monkeypatch.setattr(
            "mlwiz.evaluation.util.choice", lambda *_args: next(samples)
        )
        sampled = object.__new__(RandomSearch)._dict_helper(deepcopy(optimizer))
        assert sampled == {"kind": "adam", "lr": 0.1}


def test_global_package_and_self_order_follow_defaults_list(tmp_path):
    """Global fragments merge recursively and later ``_self_`` values win."""
    _write_yaml(
        tmp_path / "runtime" / "local.yml",
        {"device": "cpu", "data_loader": {"args": {"num_workers": 4}}},
    )
    _write_yaml(
        tmp_path / "config.yml",
        {
            "defaults": [{"runtime@_global_": "local"}, "_self_"],
            "data_loader": {"args": {"num_workers": 0, "pin_memory": False}},
        },
    )

    config = load_config(tmp_path / "config.yml")

    assert config == {
        "device": "cpu",
        "data_loader": {"args": {"num_workers": 0, "pin_memory": False}},
    }


def test_here_and_explicit_package_overrides_control_destination(tmp_path):
    """Package overrides merge locally or target an explicit dotted path."""
    _write_yaml(
        tmp_path / "search" / "shared.yml",
        {"loss": "Loss", "engine": "Engine"},
    )
    _write_yaml(tmp_path / "optimizer" / "adam.yml", {"kind": "adam"})
    _write_yaml(
        tmp_path / "config.yml",
        {
            "defaults": [{"optimizer@training.optimizer": "adam"}],
            "grid": {
                "defaults": ["search/shared@_here_"],
                "model": "Model",
            },
        },
    )

    config = load_config(tmp_path / "config.yml")

    assert config["training"]["optimizer"] == {"kind": "adam"}
    assert config["grid"] == {
        "loss": "Loss",
        "engine": "Engine",
        "model": "Model",
    }


def test_relative_nested_defaults_and_cycles_are_reported(tmp_path):
    """Nested group defaults resolve locally and recursive cycles fail clearly."""
    _write_yaml(tmp_path / "family" / "variant" / "small.yml", {"width": 8})
    _write_yaml(
        tmp_path / "family" / "base.yml",
        {"defaults": [{"variant": "small"}], "name": "base"},
    )
    _write_yaml(
        tmp_path / "config.yml",
        {"defaults": [{"family": "base"}]},
    )
    assert load_config(tmp_path / "config.yml")["family"] == {
        "variant": {"width": 8},
        "name": "base",
    }

    _write_yaml(tmp_path / "a.yml", {"defaults": ["b"]})
    _write_yaml(tmp_path / "b.yml", {"defaults": ["a"]})
    with pytest.raises(ConfigCompositionError, match="Cyclic config defaults"):
        load_config(tmp_path / "a.yml")


def test_experiment_loader_rejects_flat_pre_1_7_schema(tmp_path):
    """Old flat experiment files fail with an explicit migration message."""
    _write_yaml(
        tmp_path / "legacy.yml",
        {"storage_folder": "DATA", "device": "cpu", "grid": {}},
    )

    with pytest.raises(
        ConfigCompositionError,
        match=r"schema changed in MLWiz 1\.7\.0.*Flat pre-1\.7\.0",
    ):
        load_experiment_config(tmp_path / "legacy.yml")


def test_experiment_loader_accepts_required_structured_sections(tmp_path):
    """The new five global sections plus one search section form the schema."""
    _write_yaml(
        tmp_path / "config.yml",
        {
            "dataset": {},
            "resources": {},
            "reproducibility": {},
            "data_loading": {},
            "experiment": {},
            "grid": {},
        },
    )

    assert load_experiment_config(tmp_path / "config.yml")["grid"] == {}
