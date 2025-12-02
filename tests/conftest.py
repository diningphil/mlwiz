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
    return [
        (FakeMNIST("tests/tmp/DATA"), torch.utils.data.DataLoader),
        (FakeMNISTTemporal("tests/tmp/DATA"), torch.utils.data.DataLoader),
        (FakeNCI1("tests/tmp/DATA"), torch_geometric.loader.DataLoader),
    ]


@pytest.fixture
def single_graph_datasets():
    return [(FakeCora("tests/tmp/DATA"), torch_geometric.loader.DataLoader)]
