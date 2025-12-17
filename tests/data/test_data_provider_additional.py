"""
Additional unit tests for :mod:`mlwiz.data.provider`.

These tests focus on smaller helper behaviors and error/edge branches that are
not covered by the existing integration-style provider tests.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
from torch.utils.data import DataLoader, SequentialSampler

import mlwiz.data.provider as provider_mod
from mlwiz.data.provider import (
    DataProvider,
    SingleGraphDataProvider,
    SubsetTrainEval,
    _iterable_worker_init_fn,
)
from mlwiz.data.sampler import RandomSampler
from mlwiz.data.splitter import Splitter


class _DummyDataset(torch.utils.data.Dataset):
    """Small in-memory dataset with optional batched ``__getitems__`` support."""

    def __init__(self, values: list[int] | None = None):
        """
        Initialize the dataset.

        Args:
            values: Optional explicit sample values; defaults to ``range(10)``.

        Side effects:
            Initializes MLWiz dataset dimension attributes and transform hooks.
        """
        self.values = list(range(10)) if values is None else values
        self.dim_input_features = 7
        self.dim_target = 3
        self.transform_train = None
        self.transform_eval = None
        self.getitems_calls: list[list[int]] = []

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.values)

    def __getitem__(self, idx: int) -> int:
        """Return one sample value by index."""
        return self.values[idx]

    def __getitems__(self, indices: list[int]) -> list[int]:
        """
        Return a batch of samples and record the requested indices.

        Args:
            indices: Dataset indices to fetch.

        Returns:
            List of sample values in the requested order.
        """
        self.getitems_calls.append(list(indices))
        return [self.values[i] for i in indices]


def test_subset_train_eval_applies_transform_and_supports_batch_getitems():
    """Verify transform application and batched fetching via ``__getitems__``."""
    dataset = _DummyDataset(values=[10, 11, 12, 13, 14])
    dataset.transform_eval = lambda x: x + 100

    subset = SubsetTrainEval(dataset, indices=[1, 3, 4], is_eval=True)

    assert subset[0] == 111
    assert subset[[0, 2]] == [111, 114]

    batch = subset.__getitems__([0, 1])
    assert batch == [111, 113]
    assert dataset.getitems_calls == [[1, 3]]


def test_data_provider_get_dataset_caches_default_instance(monkeypatch):
    """Ensure ``_get_dataset`` caches the default (no-kwargs) dataset instance."""
    created = []

    def _fake_load_dataset(_storage_folder, _dataset_class, **_kwargs):
        """Return a new dummy dataset and record that it was instantiated."""
        ds = _DummyDataset()
        created.append(ds)
        return ds

    monkeypatch.setattr(provider_mod, "load_dataset", _fake_load_dataset)

    provider = DataProvider(
        storage_folder="unused",
        splits_filepath="unused",
        dataset_class=object,
        data_loader_class=DataLoader,
        data_loader_args={},
        outer_folds=1,
        inner_folds=1,
    )

    ds1 = provider._get_dataset()
    ds2 = provider._get_dataset()

    assert ds1 is ds2
    assert len(created) == 1
    assert provider.get_dim_input_features() == 7
    assert provider.get_dim_target() == 3


def test_data_provider_get_dataset_reloads_when_runtime_kwargs_present(monkeypatch):
    """Ensure runtime kwargs bypass provider-level dataset caching."""
    created = []

    def _fake_load_dataset(_storage_folder, _dataset_class, **_kwargs):
        """Return a new dummy dataset and record the received kwargs."""
        ds = _DummyDataset()
        created.append((_kwargs, ds))
        return ds

    monkeypatch.setattr(provider_mod, "load_dataset", _fake_load_dataset)

    provider = DataProvider(
        storage_folder="unused",
        splits_filepath="unused",
        dataset_class=object,
        data_loader_class=DataLoader,
        data_loader_args={},
        outer_folds=1,
        inner_folds=1,
    )

    base = provider._get_dataset()
    runtime = provider._get_dataset(url_indices=[0, 1, 2])

    assert base is provider.dataset
    assert runtime is not base
    assert created[0][0] == {}
    assert created[1][0] == {"url_indices": [0, 1, 2]}


def test_data_provider_get_dataset_rejects_shuffle_and_batch_size_kwargs():
    """Validate guardrails for subclasses: ``shuffle`` and ``batch_size`` are forbidden."""
    provider = DataProvider(
        storage_folder="unused",
        splits_filepath="unused",
        dataset_class=object,
        data_loader_class=DataLoader,
        data_loader_args={},
        outer_folds=1,
        inner_folds=1,
    )

    with pytest.raises(ValueError, match="remove `shuffle`"):
        provider._get_dataset(shuffle=True)

    with pytest.raises(ValueError, match="remove `batch_size`"):
        provider._get_dataset(batch_size=4)


def test_data_provider_requires_seed_and_fold_ids():
    """Verify fold/seed precondition errors for split loader getters."""
    provider = DataProvider(
        storage_folder="unused",
        splits_filepath="unused",
        dataset_class=object,
        data_loader_class=DataLoader,
        data_loader_args={},
        outer_folds=1,
        inner_folds=1,
    )

    with pytest.raises(RuntimeError, match="outer_k"):
        provider.get_outer_train()

    with pytest.raises(RuntimeError, match="outer_k"):
        provider.get_outer_val()

    with pytest.raises(RuntimeError, match="outer_k"):
        provider.get_outer_test()

    provider.set_outer_k(0)
    with pytest.raises(RuntimeError, match="outer_k"):
        provider.get_inner_train()


def test_data_provider_get_loader_uses_random_sampler_when_shuffling():
    """Ensure shuffle=True uses MLWiz ``RandomSampler`` (with stored permutation)."""
    provider = DataProvider(
        storage_folder="unused",
        splits_filepath="unused",
        dataset_class=object,
        data_loader_class=DataLoader,
        data_loader_args={},
        outer_folds=1,
        inner_folds=1,
    )
    provider.dataset = _DummyDataset()
    provider.set_exp_seed(0)

    loader_no_shuffle = provider._get_loader(
        indices=[0, 1, 2], is_eval=False, shuffle=False, batch_size=2
    )
    assert isinstance(loader_no_shuffle.sampler, SequentialSampler)

    loader_shuffle = provider._get_loader(
        indices=[0, 1, 2], is_eval=False, shuffle=True, batch_size=2
    )
    assert isinstance(loader_shuffle.sampler, RandomSampler)


def test_data_provider_dim_access_requires_loader_initialization():
    """Dim getters should fail until a dataset has been initialized via a loader."""
    provider = DataProvider(
        storage_folder="unused",
        splits_filepath="unused",
        dataset_class=object,
        data_loader_class=DataLoader,
        data_loader_args={},
        outer_folds=1,
        inner_folds=1,
    )
    with pytest.raises(Exception, match="initialize the dataset"):
        provider.get_dim_input_features()
    with pytest.raises(Exception, match="initialize the dataset"):
        provider.get_dim_target()


def test_iterable_worker_init_fn_raises_when_num_workers_zero(monkeypatch):
    """``_iterable_worker_init_fn`` should error if invoked without worker context."""
    @dataclass
    class _WorkerInfo:
        """Minimal worker-info stand-in for unit tests."""

        num_workers: int
        dataset: object

    monkeypatch.setattr(
        provider_mod.torch.utils.data,
        "get_worker_info",
        lambda: _WorkerInfo(num_workers=0, dataset=object()),
    )

    with pytest.raises(RuntimeError, match="num_workers > 0"):
        _iterable_worker_init_fn(worker_id=0, exp_seed=123)


def test_iterable_worker_init_fn_partitions_and_splices_dataset(monkeypatch):
    """Validate deterministic URL partition computation and dataset splicing."""
    calls = {"seed": [], "splice": []}

    class _Dataset:
        """Dataset stub that records splice boundaries."""

        def __len__(self) -> int:
            """Return a fixed dataset length."""
            return 10

        def splice(self, start: int, end: int):
            """Record the requested slice range."""
            calls["splice"].append((start, end))

    @dataclass
    class _WorkerInfo:
        """Minimal worker-info stand-in for unit tests."""

        num_workers: int
        dataset: object

    monkeypatch.setattr(
        provider_mod.torch.utils.data,
        "get_worker_info",
        lambda: _WorkerInfo(num_workers=3, dataset=_Dataset()),
    )
    monkeypatch.setattr(
        provider_mod,
        "seed_worker",
        lambda exp_seed, worker_id: calls["seed"].append((exp_seed, worker_id)),
    )

    _iterable_worker_init_fn(worker_id=1, exp_seed=42)

    assert calls["seed"] == [(1, 42)]
    assert calls["splice"] == [(4, 8)]


def test_single_graph_data_provider_rejects_non_single_graph_splitter():
    """SingleGraphDataProvider should raise if its loaded splitter type mismatches."""
    provider = SingleGraphDataProvider(
        storage_folder="unused",
        splits_filepath="unused",
        dataset_class=object,
        data_loader_class=DataLoader,
        data_loader_args={},
        outer_folds=1,
        inner_folds=1,
    )
    provider.splitter = Splitter(
        n_outer_folds=1,
        n_inner_folds=1,
        seed=0,
        stratify=False,
        shuffle=True,
    )

    with pytest.raises(TypeError, match="SingleGraphNodeSplitter"):
        provider._get_splitter()
