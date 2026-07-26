"""Tests for dataset preprocessing metadata serialization."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

import mlwiz.data.util as data_util
from mlwiz.static import SKIP_SPLITS_CHECK
from mlwiz.util import atomic_dill_save


class _RecordingDataset:
    """Dataset stub that records its constructor arguments."""

    last_kwargs = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        type(self).last_kwargs = kwargs


class _RecordingSplitter:
    """Splitter stub sufficient for exercising preprocessing."""

    n_outer_folds = 1
    n_inner_folds = 1
    stratify = False

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def split(self, dataset, targets=None):
        self.dataset = dataset
        self.targets = targets

    def check_splits_overlap(self, skip_check=False):
        self.skip_check = skip_check

    def save(self, path):
        Path(path).touch()


class _LoadedDataset:
    """Dataset stub used to inspect arguments reconstructed by ``load_dataset``."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs


def test_preprocess_data_saves_declarative_transform_specs_as_json(
    tmp_path, monkeypatch
):
    """Transforms are instantiated for use but remain declarative on disk."""
    storage_folder = tmp_path / "data"
    splits_folder = tmp_path / "splits"
    transform_spec = {
        "class_name": "collections.Counter",
        "args": {"red": 2},
    }
    options = {
        SKIP_SPLITS_CHECK: False,
        "dataset": {
            "class_name": "test.RecordingDataset",
            "args": {
                "storage_folder": str(storage_folder),
                "transform_train": transform_spec,
            },
        },
        "splitter": {
            "splits_folder": str(splits_folder),
            "class_name": "test.RecordingSplitter",
            "args": {},
        },
    }
    classes = {
        "test.RecordingDataset": _RecordingDataset,
        "test.RecordingSplitter": _RecordingSplitter,
    }
    monkeypatch.setattr(data_util, "s2c", classes.__getitem__)

    data_util.preprocess_data(deepcopy(options))

    kwargs_path = storage_folder / "_RecordingDataset" / "dataset_kwargs.json"
    stored = json.loads(kwargs_path.read_text(encoding="utf-8"))
    assert stored["transform_train"] == transform_spec
    assert isinstance(_RecordingDataset.last_kwargs["transform_train"], Counter)
    assert not kwargs_path.with_suffix(".pt").exists()


def test_load_dataset_prefers_json_and_instantiates_transforms(tmp_path):
    """JSON metadata wins over legacy metadata and reconstructs transforms."""
    kwargs_folder = tmp_path / "_LoadedDataset"
    kwargs_folder.mkdir()
    (kwargs_folder / "dataset_kwargs.json").write_text(
        json.dumps(
            {
                "storage_folder": "old",
                "value": "json",
                "transform_eval": {
                    "class_name": "collections.Counter",
                    "args": {"blue": 3},
                },
            }
        ),
        encoding="utf-8",
    )
    atomic_dill_save(
        {"storage_folder": "old", "value": "legacy"},
        str(kwargs_folder / "dataset_kwargs.pt"),
    )

    dataset = data_util.load_dataset(str(tmp_path), _LoadedDataset)

    assert dataset.kwargs["value"] == "json"
    assert dataset.kwargs["storage_folder"] == str(tmp_path)
    assert dataset.kwargs["transform_eval"] == Counter(blue=3)


def test_load_dataset_falls_back_to_legacy_dill_metadata(tmp_path):
    """Existing ``dataset_kwargs.pt`` files continue to load unchanged."""
    kwargs_folder = tmp_path / "_LoadedDataset"
    kwargs_folder.mkdir()
    transform = Counter(green=4)
    atomic_dill_save(
        {
            "storage_folder": "old",
            "value": "legacy",
            "transform_train": transform,
        },
        str(kwargs_folder / "dataset_kwargs.pt"),
    )

    dataset = data_util.load_dataset(str(tmp_path), _LoadedDataset)

    assert dataset.kwargs["value"] == "legacy"
    assert dataset.kwargs["storage_folder"] == str(tmp_path)
    assert dataset.kwargs["transform_train"] == transform
