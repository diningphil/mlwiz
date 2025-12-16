"""
Lightweight unit tests for CLI entrypoints in :mod:`mlwiz`.

These tests avoid executing full training/evaluation; they only validate that
argument parsing and config plumbing behave as expected.
"""

from __future__ import annotations

import os

import pytest
import yaml

from mlwiz.build_dataset import get_args_dict as build_dataset_get_args_dict
from mlwiz.build_dataset import main as build_dataset_main
from mlwiz.config_duplicator import main as config_duplicator_main
from mlwiz.static import CONFIG_FILE, SKIP_SPLITS_CHECK


def test_build_dataset_get_args_dict_parses_flags(monkeypatch, tmp_path):
    """
    ``mlwiz-data`` should parse config path and boolean flags correctly.
    """
    cfg_path = tmp_path / "cfg.yml"
    monkeypatch.setattr(
        "sys.argv", ["mlwiz-data", "--config-file", str(cfg_path)]
    )
    args = build_dataset_get_args_dict()
    assert args[CONFIG_FILE] == str(cfg_path)
    assert args[SKIP_SPLITS_CHECK] is False

    monkeypatch.setattr(
        "sys.argv",
        [
            "mlwiz-data",
            "--config-file",
            str(cfg_path),
            "--skip-data-splits-check",
        ],
    )
    args = build_dataset_get_args_dict()
    assert args[SKIP_SPLITS_CHECK] is True


def test_build_dataset_main_calls_preprocess(monkeypatch, tmp_path):
    """
    ``mlwiz.build_dataset.main`` should load YAML and forward merged options.
    """
    cfg_path = tmp_path / "cfg.yml"
    cfg_path.write_text(yaml.safe_dump({"foo": "bar"}))

    captured = {}

    def _fake_preprocess(options):
        captured.update(options)

    monkeypatch.setattr("mlwiz.build_dataset.preprocess_data", _fake_preprocess)
    monkeypatch.setattr(
        "sys.argv",
        [
            "mlwiz-data",
            "--config-file",
            str(cfg_path),
            "--skip-data-splits-check",
        ],
    )

    build_dataset_main()
    assert captured["foo"] == "bar"
    assert captured[SKIP_SPLITS_CHECK] is True


def test_config_duplicator_creates_dataset_specific_files(monkeypatch, tmp_path):
    """
    ``mlwiz-config-duplicator`` should merge base and dataset configs into new YAMLs.
    """
    base_cfg = tmp_path / "base.yml"
    data_cfg = tmp_path / "data.yml"

    base_cfg.write_text(yaml.safe_dump({"exp_name": "exp", "keep": 123}))
    data_cfg.write_text(
        yaml.safe_dump(
            {
                "dataset": {
                    "class_name": "mlwiz.data.dataset.FakeMNIST",
                    "args": {"storage_folder": "/DATA"},
                },
                "splitter": {
                    "splits_folder": "/SPLITS",
                    "args": {"n_outer_folds": 3, "n_inner_folds": 2},
                },
            }
        )
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        [
            "mlwiz-config-duplicator",
            "--base-exp-config",
            str(base_cfg),
            "--data-config-files",
            str(data_cfg),
        ],
    )

    config_duplicator_main()

    out_path = tmp_path / "exp_FakeMNIST.yml"
    assert out_path.exists()
    merged = yaml.safe_load(out_path.read_text())

    assert merged["exp_name"] == "exp"
    assert merged["keep"] == 123
    assert merged["storage_folder"] == "/DATA"
    assert merged["dataset_class"] == "mlwiz.data.dataset.FakeMNIST"
    assert merged["data_splits_file"] == os.path.join(
        "/SPLITS", "FakeMNIST", "FakeMNIST_outer3_inner2.splits"
    )


def test_config_duplicator_exits_with_usage_on_missing_args(monkeypatch, capsys):
    """
    ``config_duplicator.main`` should exit with usage when invoked incorrectly.
    """
    monkeypatch.setattr("sys.argv", ["mlwiz-config-duplicator"])
    with pytest.raises(SystemExit) as exc_info:
        config_duplicator_main()
    assert exc_info.value.code == 1
    assert "Usage:" in capsys.readouterr().out

