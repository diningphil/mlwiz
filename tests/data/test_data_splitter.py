import shutil

import pytest
import torch
import torch_geometric

from mlwiz.data.splitter import (
    Splitter,
    SingleGraphSplitter,
)
from tests.integration.fake_dataset import FakeMNIST, FakeMNISTTemporal, \
    FakeNCI1, FakeCora


@pytest.fixture
def datasets():
    return [
        (FakeMNIST("tests/tmp/DATA"), torch.utils.data.DataLoader),
        (FakeMNISTTemporal("tests/tmp/DATA"), torch.utils.data.DataLoader),
        (FakeNCI1("tests/tmp/DATA"), torch_geometric.loader.DataLoader),
    ]


@pytest.fixture
def single_graph_datasets():
    return [(FakeCora("tests/tmp/DATA"), torch_geometric.loader.DataLoader)]


# To each task its own splitter
@pytest.fixture
def node_and_graph_task_input(datasets, single_graph_datasets):
    """
    Returns tuples (dataset, splitter) to test for data splits overlap
    """
    return [
        (datasets, Splitter),
        (single_graph_datasets, SingleGraphSplitter),
    ]


def test_node_graph_split_overlap(node_and_graph_task_input):
    """
    Tests data splits overlap for node and graph prediction
    """
    for datasets, splitter_class in node_and_graph_task_input:
        for dataset_loader in datasets:
            dataset, loader = dataset_loader
            for n_outer_folds in [1, 3]:
                for n_inner_folds in [1, 2]:
                    for stratify in [False, True]:
                        splitter = splitter_class(
                            n_outer_folds=n_outer_folds,
                            n_inner_folds=n_inner_folds,
                            seed=0,
                            stratify=stratify,
                            shuffle=True,
                            outer_val_ratio=0.1,
                        )

                        _, targets = splitter.get_targets(dataset)
                        splitter.split(dataset, targets)

                        for outer in range(n_outer_folds):
                            outer_train_idxs = splitter.outer_folds[
                                outer
                            ].train_idxs
                            outer_val_idxs = splitter.outer_folds[
                                outer
                            ].val_idxs
                            outer_test_idxs = splitter.outer_folds[
                                outer
                            ].test_idxs

                            # False if empty
                            assert not bool(
                                set(outer_train_idxs)
                                & set(outer_val_idxs)
                                & set(outer_test_idxs)
                            )

                            for inner in range(n_inner_folds):
                                inner_train_idxs = splitter.inner_folds[outer][
                                    inner
                                ].train_idxs
                                inner_val_idxs = splitter.inner_folds[outer][
                                    inner
                                ].val_idxs

                                # False if empty
                                assert not bool(
                                    set(inner_train_idxs)
                                    & set(inner_val_idxs)
                                    & set(outer_test_idxs)
                                )

                                # Check length consistency
                                if len(dataset) == 1:
                                    assert (
                                        len(inner_train_idxs)
                                        + len(inner_val_idxs)
                                        + len(outer_test_idxs)
                                        == dataset[0][0].x.shape[0]
                                    )
                                else:
                                    assert len(inner_train_idxs) + len(
                                        inner_val_idxs
                                    ) + len(outer_test_idxs) == len(dataset)

                            # Check length consistency
                            if len(dataset) == 1:
                                assert (
                                    len(outer_train_idxs)
                                    + len(outer_val_idxs)
                                    + len(outer_test_idxs)
                                    == dataset[0][0].x.shape[0]
                                )
                            else:
                                assert len(outer_train_idxs) + len(
                                    outer_val_idxs
                                ) + len(outer_test_idxs) == len(dataset)
    shutil.rmtree("tests/tmp")
