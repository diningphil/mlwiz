from pathlib import Path
from typing import Union, Tuple, List, Callable, Optional

import torch
from torch_geometric.data import Data

from mlwiz.data.dataset import DatasetInterface, IterableDatasetInterface
from mlwiz.util import dill_save


class FakeMNIST(DatasetInterface):
    @property
    def dim_input_features(self) -> Union[int, Tuple[int]]:
        return 28 * 28

    @property
    def dim_target(self) -> Union[int, Tuple[int]]:
        return 2

    def process_dataset(self) -> List[object]:
        dataset = []
        for _ in range(50):
            dataset.append(
                (torch.rand(1, 28, 28), torch.zeros(1, dtype=torch.long))
            )
        return dataset


class FakeMNISTTemporal(DatasetInterface):
    @property
    def dim_input_features(self) -> Union[int, Tuple[int]]:
        return 4

    @property
    def dim_target(self) -> Union[int, Tuple[int]]:
        return 2

    def process_dataset(self) -> List[object]:
        dataset = []
        for _ in range(50):
            dataset.append(
                (torch.rand((4, 4)), torch.zeros(1, dtype=torch.long))
            )
        return dataset


class FakeNCI1(DatasetInterface):
    @property
    def dim_input_features(self) -> Union[int, Tuple[int]]:
        return 4

    @property
    def dim_target(self) -> Union[int, Tuple[int]]:
        return 2

    def process_dataset(self) -> List[object]:
        dataset = []
        num_nodes = 5
        for _ in range(50):
            dataset.append(
                (
                    Data(
                        x=torch.rand((num_nodes, self.dim_input_features)),
                        edge_index=torch.tensor(
                            [[0, 1, 2, 3, 4], [3, 4, 2, 0, 1]]
                        ),
                    ),
                    torch.zeros(1, dtype=torch.long),
                )
            )
        return dataset


class FakeCora(DatasetInterface):
    @property
    def dim_input_features(self) -> Union[int, Tuple[int]]:
        return 4

    @property
    def dim_target(self) -> Union[int, Tuple[int]]:
        return 2

    def process_dataset(self) -> List[object]:
        dataset = []
        num_nodes = 5
        dataset.append(
            (
                Data(
                    x=torch.rand((num_nodes, self.dim_input_features)),
                    edge_index=torch.tensor(
                        [[0, 1, 2, 3, 4], [3, 4, 2, 0, 1]]
                    ),
                ),
                torch.zeros(num_nodes, dtype=torch.long),
            )
        )
        return dataset


class ToyIterableDataset(IterableDatasetInterface):
    @property
    def url_indices(self) -> List[Path]:
        r"""
        Specifies the list of file names where you plan to store
            portions of the large dataset
        """
        return [f"fake_path_{i}" for i in range(100)]

    @property
    def dim_input_features(self) -> Union[int, Tuple[int]]:
        r"""
        Specifies the number of node features (after pre-processing, but in
        the end it depends on the model that is implemented).
        """
        return 5

    @property
    def dim_target(self) -> Union[int, Tuple[int]]:
        r"""
        Specifies the dimension of each target vector.
        """
        return 2

    def process_dataset(self, pre_transform: Optional[Callable]):
        r"""
        Creates a fake dataset and stores it to the :obj:`self.processed_dir`
        folder. Each file will contain a list of 20 fake samples.
        """
        for i in range(len(self)):
            fake_samples = []
            for s in range(20):
                fake_sample = (
                    torch.zeros(5, self.dim_input_features),
                    torch.zeros(5, 2),
                )

                if pre_transform is not None:
                    fake_sample = pre_transform(fake_sample)

                fake_samples.append(fake_sample)

            dill_save(fake_samples, self.dataset_filepaths[i])
