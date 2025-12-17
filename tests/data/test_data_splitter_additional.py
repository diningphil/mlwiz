"""
Additional unit tests for :mod:`mlwiz.data.splitter`.

These tests cover helper branches and error handling not exercised by the
existing split-overlap integration test.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    StratifiedShuffleSplit,
)

from mlwiz.data.splitter import (
    InnerFold,
    OuterFold,
    Splitter,
    SingleGraphSplitter,
    _NoShuffleTrainTestSplit,
)
from mlwiz.util import atomic_dill_save


class _TargetDataset:
    """Iterable dataset yielding ``(x, y)`` pairs for target extraction tests."""

    def __init__(self, ys):
        """
        Initialize the dataset with a list of targets.

        Args:
            ys: Iterable of target values (may include ``None``).
        """
        self._ys = list(ys)

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self._ys)

    def __iter__(self):
        """Yield ``(None, y)`` pairs to match DatasetInterface iteration."""
        for y in self._ys:
            yield None, y


class _LenOnlyDataset:
    """Minimal dataset implementing only ``__len__`` for split generation."""

    def __init__(self, n: int):
        """
        Initialize with a fixed size.

        Args:
            n: Dataset length.
        """
        self._n = int(n)

    def __len__(self) -> int:
        """Return the dataset length."""
        return self._n


def test_no_shuffle_train_test_split_is_deterministic():
    """_NoShuffleTrainTestSplit should split deterministically without shuffling."""
    splitter = _NoShuffleTrainTestSplit(test_ratio=0.2)
    (train_idxs, test_idxs), = splitter.split(list(range(10)))

    assert train_idxs.tolist() == list(range(8))
    assert test_idxs.tolist() == [8, 9]


def test_splitter_clears_seed_when_shuffle_is_false(capsys):
    """Splitter should warn and clear seed when ``shuffle=False``."""
    splitter = Splitter(
        n_outer_folds=1,
        n_inner_folds=1,
        seed=123,
        stratify=False,
        shuffle=False,
    )

    assert splitter.seed is None
    captured = capsys.readouterr()
    assert "seed set to None" in captured.out


def test_get_targets_returns_false_when_targets_missing():
    """If any sample target is ``None``, ``get_targets`` should return ``(False, None)``."""
    splitter = Splitter(
        n_outer_folds=1,
        n_inner_folds=1,
        seed=0,
        stratify=True,
        shuffle=True,
    )

    ok, targets = splitter.get_targets(_TargetDataset([None, 1]))
    assert ok is False
    assert targets is None


def test_get_targets_converts_scalar_targets_to_numpy_array():
    """Scalar targets should be converted to a 1D NumPy array."""
    splitter = Splitter(
        n_outer_folds=1,
        n_inner_folds=1,
        seed=0,
        stratify=True,
        shuffle=True,
    )

    ok, targets = splitter.get_targets(_TargetDataset([0, 1, 2]))
    assert ok is True
    np.testing.assert_array_equal(targets, np.array([0.0, 1.0, 2.0]))


def test_get_splitter_handles_all_branches():
    """Exercise the internal splitter factory for holdout/kfold + stratification."""
    splitter = Splitter(
        n_outer_folds=1,
        n_inner_folds=1,
        seed=0,
        stratify=False,
        shuffle=True,
    )

    with pytest.raises(ValueError, match="must be >=1"):
        splitter._get_splitter(n_splits=0, stratified=False, eval_ratio=0.1)

    splitter.shuffle = False
    no_shuffle = splitter._get_splitter(
        n_splits=1, stratified=False, eval_ratio=0.2
    )
    assert isinstance(no_shuffle, _NoShuffleTrainTestSplit)

    with pytest.raises(NotImplementedError, match="Stratified"):
        splitter._get_splitter(n_splits=1, stratified=True, eval_ratio=0.2)

    splitter.shuffle = True
    strat_holdout = splitter._get_splitter(
        n_splits=1, stratified=True, eval_ratio=0.2
    )
    assert isinstance(strat_holdout, StratifiedShuffleSplit)

    strat_kfold = splitter._get_splitter(
        n_splits=3, stratified=True, eval_ratio=0.2
    )
    assert isinstance(strat_kfold, StratifiedKFold)

    plain_kfold = splitter._get_splitter(
        n_splits=3, stratified=False, eval_ratio=0.2
    )
    assert isinstance(plain_kfold, KFold)


def test_check_splits_overlap_skip_check_allows_overlaps(capsys):
    """When requested, overlap checks should be skipped without raising."""
    splitter = Splitter(
        n_outer_folds=1,
        n_inner_folds=1,
        seed=0,
        stratify=False,
        shuffle=True,
    )
    splitter.outer_folds = [
        OuterFold(train_idxs=[0, 1], val_idxs=[1], test_idxs=[1])
    ]
    splitter.inner_folds = [[InnerFold(train_idxs=[0], val_idxs=[0])]]

    splitter.check_splits_overlap(skip_check=True)
    captured = capsys.readouterr()
    assert "skip data checking" in captured.out


def test_check_splits_overlap_raises_when_inner_fold_has_test_indices():
    """Inner folds must not contain test indices; otherwise ``check_splits_overlap`` raises."""
    splitter = Splitter(
        n_outer_folds=1,
        n_inner_folds=1,
        seed=0,
        stratify=False,
        shuffle=True,
    )
    splitter.outer_folds = [
        OuterFold(train_idxs=[0], val_idxs=[1], test_idxs=[2])
    ]
    splitter.inner_folds = [
        [InnerFold(train_idxs=[0], val_idxs=[1], test_idxs=[2])]
    ]

    with pytest.raises(RuntimeError, match="Test indices should not be present"):
        splitter.check_splits_overlap()


def test_save_and_load_roundtrip(tmp_path):
    """Persist splits to disk and reload them via :meth:`Splitter.load`."""
    dataset = _LenOnlyDataset(20)
    splitter = Splitter(
        n_outer_folds=2,
        n_inner_folds=2,
        seed=0,
        stratify=False,
        shuffle=True,
        inner_val_ratio=0.25,
        outer_val_ratio=0.2,
        test_ratio=0.2,
    )
    splitter.split(dataset, targets=None)

    splits_path = tmp_path / "demo.splits"
    splitter.save(str(splits_path))

    loaded = Splitter.load(str(splits_path))

    assert loaded.n_outer_folds == splitter.n_outer_folds
    assert loaded.n_inner_folds == splitter.n_inner_folds
    assert len(loaded.outer_folds) == len(splitter.outer_folds)
    assert len(loaded.inner_folds) == len(splitter.inner_folds)

    for outer_k in range(splitter.n_outer_folds):
        assert loaded.outer_folds[outer_k].train_idxs == splitter.outer_folds[
            outer_k
        ].train_idxs
        assert loaded.outer_folds[outer_k].val_idxs == splitter.outer_folds[
            outer_k
        ].val_idxs
        assert loaded.outer_folds[outer_k].test_idxs == splitter.outer_folds[
            outer_k
        ].test_idxs

        for inner_k in range(splitter.n_inner_folds):
            assert loaded.inner_folds[outer_k][inner_k].train_idxs == splitter.inner_folds[
                outer_k
            ][
                inner_k
            ].train_idxs
            assert loaded.inner_folds[outer_k][inner_k].val_idxs == splitter.inner_folds[
                outer_k
            ][
                inner_k
            ].val_idxs


def test_load_validates_fold_counts(tmp_path):
    """Loading should validate expected outer/inner fold counts from the metadata."""
    base_args = {
        "n_outer_folds": 2,
        "n_inner_folds": 1,
        "seed": 0,
        "stratify": False,
        "shuffle": True,
        "inner_val_ratio": 0.1,
        "outer_val_ratio": 0.1,
        "test_ratio": 0.1,
    }
    invalid_outer = {
        "splitter_class": "mlwiz.data.splitter.Splitter",
        "splitter_args": dict(base_args),
        "outer_folds": [{"train": [0], "val": [1], "test": [2]}],  # should be 2
        "inner_folds": [[{"train": [0], "val": [1]}]],
    }
    invalid_outer_path = tmp_path / "invalid_outer.splits"
    atomic_dill_save(invalid_outer, str(invalid_outer_path))

    with pytest.raises(ValueError, match="outer folds"):
        Splitter.load(str(invalid_outer_path))

    invalid_inner = {
        "splitter_class": "mlwiz.data.splitter.Splitter",
        "splitter_args": {"n_outer_folds": 1, **dict(base_args, n_outer_folds=1)},
        "outer_folds": [{"train": [0], "val": [1], "test": [2]}],
        "inner_folds": [[{"train": [0], "val": [1]}]],  # should be 1 inner fold? set to 1
    }
    # Force mismatch: expect 2 inner folds but provide 1.
    invalid_inner["splitter_args"]["n_inner_folds"] = 2
    invalid_inner_path = tmp_path / "invalid_inner.splits"
    atomic_dill_save(invalid_inner, str(invalid_inner_path))

    with pytest.raises(ValueError, match="inner folds"):
        Splitter.load(str(invalid_inner_path))


def test_single_graph_splitter_rejects_non_single_graph_datasets():
    """SingleGraphSplitter should refuse datasets with length != 1."""
    splitter = SingleGraphSplitter(
        n_outer_folds=1,
        n_inner_folds=1,
        seed=0,
        stratify=False,
        shuffle=True,
    )
    with pytest.raises(ValueError, match="single graph datasets"):
        splitter.split(_LenOnlyDataset(2), targets=torch.zeros(2).numpy())
