from copy import deepcopy
from shutil import rmtree
from unittest.mock import patch

import torch

from mlwiz.data.dataset import DatasetInterface, ToyIterableDataset
from mlwiz.data.provider import (
    DataProvider,
    SingleGraphDataProvider,
    IterableDataProvider,
)
from mlwiz.data.splitter import (
    Splitter,
    SingleGraphSplitter,
)
from tests.data.test_data_splitter import (
    datasets,
    single_graph_datasets,
)

def mock_get_dataset(cls, **kwargs):
    """
    Returns the dataset stored in the object (see main test)
    """
    return deepcopy(cls.dataset)


def mock_get_splitter(cls, **kwargs):
    """
    Instantiates a splitter and generates random splits
    """
    if cls.splitter is None:
        splitter = Splitter(
            n_outer_folds=cls.outer_folds,
            n_inner_folds=cls.inner_folds,
            seed=0,
            stratify=True,
            shuffle=True,
            outer_val_ratio=0.1,
        )
        dataset = cls._get_dataset()
        targets = splitter.get_targets(dataset)[1]
        splitter.split(dataset, targets)
        cls.splitter = splitter
        return cls.splitter

    return cls.splitter


def mock_get_singlegraphsplitter(cls, **kwargs):
    """
    Instantiates a splitter and generates random splits
    """
    splitter = SingleGraphSplitter(
        n_outer_folds=cls.outer_folds,
        n_inner_folds=cls.inner_folds,
        seed=0,
        stratify=True,
        shuffle=True,
        outer_val_ratio=0.1,
    )
    dataset = cls._get_dataset()
    splitter.split(dataset, splitter.get_targets(dataset)[1])

    return splitter


@patch.object(DataProvider, "_get_splitter", mock_get_splitter)
@patch.object(DataProvider, "_get_dataset", mock_get_dataset)
def test_DataProvider(datasets):
    """
    Check that the data provider returns the correct data associated
    with different data splits
    """

    for dataset_loader in datasets:
        dataset, loader = dataset_loader
        batch_size = 32
        for outer_folds in [1, 3]:
            for inner_folds in [1, 2]:
                for shuffle in [False, True]:
                    provider = DataProvider(
                        None,
                        None,
                        DatasetInterface,
                        loader,
                        {},
                        outer_folds=outer_folds,
                        inner_folds=inner_folds,
                    )
                    provider.dataset = dataset
                    provider.set_exp_seed(0)

                    for o in range(outer_folds):
                        provider.set_outer_k(o)
                        for i in range(inner_folds):
                            provider.set_inner_k(i)

                            inner_train_loader = provider.get_inner_train(
                                shuffle=shuffle, batch_size=batch_size
                            )

                            assert set(
                                inner_train_loader.dataset.indices
                            ) == set(
                                provider._get_splitter()
                                .inner_folds[o][i]
                                .train_idxs
                            )

                            inner_val_loader = provider.get_inner_val(
                                shuffle=shuffle, batch_size=batch_size
                            )

                            assert set(
                                inner_val_loader.dataset.indices
                            ) == set(
                                provider._get_splitter()
                                .inner_folds[o][i]
                                .val_idxs
                            )

                        provider.set_inner_k(None)

                        outer_train_loader = provider.get_outer_train(
                            shuffle=shuffle, batch_size=batch_size
                        )

                        assert set(outer_train_loader.dataset.indices) == set(
                            provider._get_splitter().outer_folds[o].train_idxs
                        )

                        outer_val_loader = provider.get_outer_val(
                            shuffle=shuffle, batch_size=batch_size
                        )

                        assert set(outer_val_loader.dataset.indices) == set(
                            provider._get_splitter().outer_folds[o].val_idxs
                        )

                        outer_test_loader = provider.get_outer_test(
                            shuffle=shuffle, batch_size=batch_size
                        )

                        assert set(outer_test_loader.dataset.indices) == set(
                            provider._get_splitter().outer_folds[o].test_idxs
                        )


@patch.object(
    SingleGraphDataProvider, "_get_splitter", mock_get_singlegraphsplitter
)
@patch.object(SingleGraphDataProvider, "_get_dataset", mock_get_dataset)
def test_SingleGraphDataProvider(single_graph_datasets):
    """
    Check that the data provider returns the correct data associated
    with different data splits
    """

    for dataset_loader in single_graph_datasets:
        dataset, loader = dataset_loader
        batch_size = 32
        for outer_folds in [1, 3]:
            for inner_folds in [1, 2]:
                for shuffle in [False, True]:
                    provider = SingleGraphDataProvider(
                        None,
                        None,
                        DatasetInterface,
                        loader,
                        {},
                        outer_folds=outer_folds,
                        inner_folds=inner_folds,
                    )
                    provider.dataset = dataset
                    provider.set_exp_seed(0)

                    for o in range(outer_folds):
                        provider.set_outer_k(o)
                        for i in range(inner_folds):
                            provider.set_inner_k(i)

                            inner_train_loader = provider.get_inner_train(
                                shuffle=shuffle, batch_size=batch_size
                            )

                            assert set(
                                inner_train_loader.dataset[0][
                                    0
                                ].training_indices.tolist()
                            ) == set(
                                provider._get_splitter()
                                .inner_folds[o][i]
                                .train_idxs
                            )

                            assert set(
                                inner_train_loader.dataset[0][
                                    0
                                ].eval_indices.tolist()
                            ) == set(
                                provider._get_splitter()
                                .inner_folds[o][i]
                                .train_idxs
                            )

                            inner_val_loader = provider.get_inner_val(
                                shuffle=shuffle, batch_size=batch_size
                            )

                            assert set(
                                inner_val_loader.dataset[0][
                                    0
                                ].training_indices.tolist()
                            ) == set(
                                provider._get_splitter()
                                .inner_folds[o][i]
                                .train_idxs
                            )

                            assert set(
                                inner_val_loader.dataset[0][
                                    0
                                ].eval_indices.tolist()
                            ) == set(
                                provider._get_splitter()
                                .inner_folds[o][i]
                                .val_idxs
                            )

                        provider.set_inner_k(None)

                        outer_train_loader = provider.get_outer_train(
                            shuffle=shuffle, batch_size=batch_size
                        )

                        assert set(
                            outer_train_loader.dataset[0][
                                0
                            ].training_indices.tolist()
                        ) == set(
                            provider._get_splitter().outer_folds[o].train_idxs
                        )

                        assert set(
                            outer_train_loader.dataset[0][
                                0
                            ].eval_indices.tolist()
                        ) == set(
                            provider._get_splitter().outer_folds[o].train_idxs
                        )

                        outer_val_loader = provider.get_outer_val(
                            shuffle=shuffle, batch_size=batch_size
                        )

                        assert set(
                            outer_val_loader.dataset[0][
                                0
                            ].training_indices.tolist()
                        ) == set(
                            provider._get_splitter().outer_folds[o].train_idxs
                        )

                        assert set(
                            outer_val_loader.dataset[0][
                                0
                            ].eval_indices.tolist()
                        ) == set(
                            provider._get_splitter().outer_folds[o].val_idxs
                        )

                        outer_test_loader = provider.get_outer_test(
                            shuffle=shuffle, batch_size=batch_size
                        )

                        assert set(
                            outer_test_loader.dataset[0][
                                0
                            ].training_indices.tolist()
                        ) == set(
                            provider._get_splitter().outer_folds[o].train_idxs
                        )

                        assert set(
                            outer_test_loader.dataset[0][
                                0
                            ].eval_indices.tolist()
                        ) == set(
                            provider._get_splitter().outer_folds[o].test_idxs
                        )


def mock_get_iterable_splitter(cls, **kwargs):
    """
    Instantiates a splitter and generates random splits
    """
    if cls.splitter is None:
        splitter = Splitter(
            n_outer_folds=cls.outer_folds,
            n_inner_folds=cls.inner_folds,
            seed=0,
            stratify=False,
            shuffle=True,
            outer_val_ratio=0.1,
        )
        dataset = cls._get_dataset()
        splitter.split(dataset, None)
        cls.splitter = splitter
        return cls.splitter

    return cls.splitter


def mock_get_iterable_dataset(cls, **kwargs):
    """
    Returns the dataset stored in the object (see main test)
    """
    storage_folder = "tests/tmp/DATA_ToyIterableDataset"

    if "url_indices" in kwargs:
        indices = kwargs["url_indices"]
        dataset = ToyIterableDataset(storage_folder)
        dataset.subset(indices)
        return dataset
    return ToyIterableDataset(storage_folder)


@patch.object(IterableDataProvider, "_get_dataset", mock_get_iterable_dataset)
@patch.object(
    IterableDataProvider, "_get_splitter", mock_get_iterable_splitter
)
def test_IterableDataProvider():
    """
    Check that the data provider returns the correct data associated
    with different data splits
    """

    storage_folder = "tests/tmp/DATA_ToyIterableDataset"
    batch_size = 32
    for outer_folds in [1, 3]:
        for inner_folds in [1, 3]:
            for shuffle in [False, True]:
                provider = IterableDataProvider(
                    storage_folder,
                    None,
                    DatasetInterface,
                    torch.utils.data.DataLoader,
                    {},
                    outer_folds=outer_folds,
                    inner_folds=inner_folds,
                )
                provider.dataset = ToyIterableDataset(storage_folder)
                provider.set_exp_seed(0)

                for o in range(outer_folds):
                    provider.set_outer_k(o)
                    for i in range(inner_folds):
                        provider.set_inner_k(i)

                        inner_train_loader = provider.get_inner_train(
                            shuffle=shuffle, batch_size=batch_size
                        )

                        assert len(inner_train_loader.dataset) == len(
                            set(
                                provider._get_splitter()
                                .inner_folds[o][i]
                                .train_idxs
                            )
                        )

                        inner_val_loader = provider.get_inner_val(
                            shuffle=shuffle, batch_size=batch_size
                        )

                        assert len(inner_val_loader.dataset) == len(
                            set(
                                provider._get_splitter()
                                .inner_folds[o][i]
                                .val_idxs
                            )
                        )

                    provider.set_inner_k(None)

                    outer_train_loader = provider.get_outer_train(
                        shuffle=shuffle, batch_size=batch_size
                    )

                    assert len(outer_train_loader.dataset) == len(
                        set(provider._get_splitter().outer_folds[o].train_idxs)
                    )

                    outer_val_loader = provider.get_outer_val(
                        shuffle=shuffle, batch_size=batch_size
                    )

                    assert len(outer_val_loader.dataset) == len(
                        set(provider._get_splitter().outer_folds[o].val_idxs)
                    )

                    outer_test_loader = provider.get_outer_test(
                        shuffle=shuffle, batch_size=batch_size
                    )

                    assert len(outer_test_loader.dataset) == len(
                        set(provider._get_splitter().outer_folds[o].test_idxs)
                    )
    rmtree("tests/tmp/")
