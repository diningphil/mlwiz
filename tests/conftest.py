import pytest
import torch
import torch_geometric

from tests.integration.fake_dataset import (
    FakeMNIST,
    FakeMNISTTemporal,
    FakeNCI1,
    FakeCora,
)


@pytest.fixture
def datasets():
    """
    Provide a small set of dataset/loader pairs for parameterized tests.

    Returns:
        list[tuple[DatasetInterface, type]]: Tuples of ``(dataset, loader_class)``
        for classic tensor datasets and PyG graph datasets.

    Side effects:
        Instantiates fake datasets under ``tests/tmp/DATA`` (may create files).
    """
    return [
        (FakeMNIST("tests/tmp/DATA"), torch.utils.data.DataLoader),
        (FakeMNISTTemporal("tests/tmp/DATA"), torch.utils.data.DataLoader),
        (FakeNCI1("tests/tmp/DATA"), torch_geometric.loader.DataLoader),
    ]


@pytest.fixture
def single_graph_datasets():
    """
    Provide dataset/loader pairs for single-graph workflows.

    Returns:
        list[tuple[DatasetInterface, type]]: Tuples of ``(dataset, loader_class)``
        where the dataset represents a single graph.

    Side effects:
        Instantiates fake datasets under ``tests/tmp/DATA`` (may create files).
    """
    return [(FakeCora("tests/tmp/DATA"), torch_geometric.loader.DataLoader)]
