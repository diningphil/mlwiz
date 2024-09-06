import os
from pathlib import Path
from random import shuffle
from typing import List, Union, Tuple, Optional, Callable

import torch
import torchvision
from torch_geometric.datasets import TUDataset, Planetoid

from mlwiz.data.util import get_or_create_dir


class DatasetInterface:
    r"""
    Class that defines a number of properties essential to all datasets
    implementations inside MLWiz. These properties are used by the training
    engine and forwarded to the model to be trained.

    Useful for small to medium datasets where a single file can contain the
    whole data. In case the dataset is too large and needs to be split into
    chunks, check :obj:`mlwiz.data.dataset.IterableDatasetInterface`

    Args:
        storage_folder (str): path to folder where to store the dataset
        raw_dataset_folder (Optional[str]): path to raw data folder where raw
            data is stored
        transform (Optional[Callable]): transformations to apply to each
            sample at run time
        pre_transform (Optional[Callable]): transformations to apply to each
            sample at dataset creation time
    """

    def __init__(
        self,
        storage_folder: str,
        raw_dataset_folder: Optional[str] = None,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        **kwargs,
    ):
        super().__init__()
        self._name = self.__class__.__name__
        self._storage_folder = storage_folder
        self._raw_dataset_folder = raw_dataset_folder
        self.transform = transform
        self.pre_transform = pre_transform
        self.dataset = None
        self._dataset_filename = f"{self.name}_processed_dataset.pt"

        # Create folders where to store processed dataset
        get_or_create_dir(self.dataset_folder)

        if self._raw_dataset_folder is not None and not os.path.exists(
            self.raw_dataset_folder
        ):
            raise FileNotFoundError(
                f"Folder {self._raw_dataset_folder} " f"not found"
            )

        # if any of the processed files is missing, process the dataset
        # and store the results in a file
        if not os.path.exists(Path(self.dataset_filepath)):
            print(
                f"File {self.dataset_filepath} from not found, "
                f"calling process_data()..."
            )
            dataset = self.process_dataset()

            # apply pre-processing if needed
            if self.pre_transform is not None:
                dataset = [self.pre_transform(d) for d in dataset]

            self.dataset = dataset

            # store dataset
            print(f"Storing into {self.dataset_filepath}...")
            torch.save(self.dataset, self.dataset_filepath)

        else:
            # Simply load the dataset in memory
            self.dataset = torch.load(self.dataset_filepath)

    @property
    def name(self) -> str:
        """
        Returns the name of the dataset
        Returns: a string
        """
        return self._name

    @property
    def dataset_folder(self) -> Path:
        """
        Returns the Path to folder where dataset is stored
        Returns: a Path object
        """
        return Path(self._storage_folder, self._name)

    @property
    def dataset_name(self) -> str:
        """
        Returns the name of the dataset
        """
        return self._name

    @property
    def raw_dataset_folder(self) -> Path:
        """
        Returns the Path to folder where the raw data is stored
        """
        return Path(self._raw_dataset_folder)

    @property
    def dataset_filename(self) -> str:
        """
        Returns the name of the single file where the dataset is stored
        """
        return self._dataset_filename

    @property
    def dataset_filepath(self) -> Path:
        """
        Returns the full Path to the single file where the dataset is stored
        """
        return Path(self.dataset_folder, self._dataset_filename)

    def process_dataset(self) -> List[object]:
        r"""
        Processes the dataset to the :obj:`self.dataset_folder` folder. It
        should generate files according to the obj:`self.dataset_file_names`
        list.
        """
        raise NotImplementedError(
            "You should subclass DatasetInterface and implement this method"
        )

    @property
    def dim_input_features(self) -> Union[int, Tuple[int]]:
        r"""
        Specifies the number of input features or a tuple if there are more,
        for instance node and edge features in graphs.
        """
        raise NotImplementedError(
            "You should subclass DatasetInterface and implement this method"
        )

    @property
    def dim_target(self) -> Union[int, Tuple[int]]:
        r"""
        Specifies the dimension of each target vector or a tuple if there are
        more.
        """
        raise NotImplementedError(
            "You should subclass DatasetInterface and implement this method"
        )

    def __getitem__(self, idx: int) -> object:
        r"""
        Returns sample ``idx`` of the dataset.

        Args:
            idx (int): the sample's index

        Returns: the i-th sample of the dataset

        """
        sample = self.dataset[idx]

        # apply runtime preprocessing if needed
        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def __iter__(self):
        return iter(self.dataset)

    def __len__(self) -> int:
        r"""
        Returns the number of samples stored in the dataset.
        """
        return len(self.dataset)


class MNIST(DatasetInterface):
    @property
    def dim_input_features(self) -> Union[int, Tuple[int]]:
        return 28 * 28

    @property
    def dim_target(self) -> Union[int, Tuple[int]]:
        return 10

    def process_dataset(self) -> List[object]:
        r"""
        Processes the dataset to the :obj:`self.dataset_folder` folder. It
        should generate files according to the obj:`self.dataset_file_names`
        list.
        """
        train = torchvision.datasets.MNIST(
            self.dataset_folder,
            train=True,
            download=True,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                ]
            ),
        )
        test = torchvision.datasets.MNIST(
            self.dataset_folder,
            train=False,
            download=True,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                ]
            ),
        )
        self.dataset = train + test
        return self.dataset


class NCI1(DatasetInterface):
    @property
    def dim_input_features(self) -> Union[int, Tuple[int]]:
        return 37

    @property
    def dim_target(self) -> Union[int, Tuple[int]]:
        return 2

    def process_dataset(self) -> List[object]:
        r"""
        Processes the dataset to the :obj:`self.dataset_folder` folder. It
        should generate files according to the obj:`self.dataset_file_names`
        list.
        """
        self.dataset = TUDataset(root=self.dataset_folder, name="NCI1")
        # casting class to int will allow PyG collater to create a tensor of
        # size (batch_size) instead of (batch_size, 1), making it consistent
        # with other non-graph datasets
        return [(g, int(g.y)) for g in self.dataset]


class Cora(DatasetInterface):
    @property
    def dim_input_features(self) -> Union[int, Tuple[int]]:
        return 1433

    @property
    def dim_target(self) -> Union[int, Tuple[int]]:
        return 7

    def process_dataset(self) -> List[object]:
        r"""
        Processes the dataset to the :obj:`self.dataset_folder` folder. It
        should generate files according to the obj:`self.dataset_file_names`
        list.
        """
        self.dataset = Planetoid(root=self.dataset_folder, name="Cora")

        g = self.dataset[0]
        y = g.y
        del g.train_mask
        del g.val_mask
        del g.test_mask

        # TODO PyG collater will add a dummy dimension in front of y
        #   it might be nice to create another collater to remove it, but it
        #   might require the user to know about this..
        return [(g, y)]


class _ReshapeMNISTTemporal(torch.nn.Module):
    def __call__(self, img: torch.Tensor):
        """
        Will reshape the image into a timeseries of vectors
        Args:
            img: the MNIST image

        Returns: a reshaped MNIST image into a timeseries
        """
        img = torch.reshape(img, (28, 28))
        return img  # (length, n_features)


class MNISTTemporal(DatasetInterface):
    @property
    def dim_input_features(self) -> Union[int, Tuple[int]]:
        return 28

    @property
    def dim_target(self) -> Union[int, Tuple[int]]:
        return 10

    def process_dataset(self) -> List[object]:
        r"""
        Processes the dataset to the :obj:`self.dataset_folder` folder. It
        should generate files according to the obj:`self.dataset_file_names`
        list.
        """

        train = torchvision.datasets.MNIST(
            self.dataset_folder,
            train=True,
            download=True,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                    _ReshapeMNISTTemporal(),
                ]
            ),
        )
        test = torchvision.datasets.MNIST(
            self.dataset_folder,
            train=False,
            download=True,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                    _ReshapeMNISTTemporal(),
                ]
            ),
        )
        self.dataset = train + test
        return self.dataset


class IterableDatasetInterface(torch.utils.data.IterableDataset):
    r"""
    Class that implements the Iterable-style dataset, including multi-process
    data loading
    (https://pytorch.org/docs/stable/data.html#iterable-style-datasets).
    Useful when the dataset is too big and split in chunks of files to be
    stored on disk. Each chunk can hold a single sample or a set of samples,
    and there is the chance to shuffle sample-wise or chunk-wise.
    Must be combined with an
    appropriate :class:`mlwiz.data.provider.IterableDataProvider`.

    NOTE 1: We assume the splitter will split the dataset with respect to
    the number of files stored on disk, so be sure that the length of your
    dataset reflects that number. Then, examples will be provided sequentially,
    so if each file holds more than one sample, we will still be able to create
    a batch of samples from one or multiple files.

    NOTE 2: NEVER override the __len__() method, as it varies dynamically with
    the ``url_indices`` argument.

    Args:
        storage_folder (str): path to root folder where to store the dataset
        raw_dataset_folder (Optional[str]): path to raw data folder where raw
            data is stored
        transform (Optional[Callable]): transformations to apply to each
            sample atrun time
        pre_transform (Optional[Callable]): transformations to apply to each
            sample atdataset creation time
    """

    def __init__(
        self,
        storage_folder: str,
        raw_dataset_folder: Optional[str] = None,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        **kwargs,
    ):
        super().__init__()
        self._name = self.__class__.__name__
        self._storage_folder = storage_folder
        self._raw_dataset_folder = raw_dataset_folder
        self._dataset_folder = Path(self._storage_folder, self._name)

        # Create folders where to store processed dataset
        get_or_create_dir(self._dataset_folder)

        if self._raw_dataset_folder is not None and not os.path.exists(
            self.raw_dataset_folder
        ):
            raise FileNotFoundError(
                f"Folder {self._raw_dataset_folder} " f"not found"
            )

        self.shuffled_urls = self.dataset_filepaths

        self.start_index = 0
        self.end_index = len(self.shuffled_urls)

        # This information allows us to shuffle between urls and sub-patches
        # inside each url
        self._shuffle_urls = False
        self._shuffle_subpatches = False

        self.pre_transform = pre_transform
        self.transform = transform

        for u in self.dataset_filepaths:
            if not os.path.exists(u):
                print(f"File {u} from not found, calling process_data()...")
                self.process_dataset(pre_transform=pre_transform)

    @property
    def dataset_folder(self) -> Path:
        """
        Returns the Path to folder where dataset is stored
        Returns: a Path object
        """
        return Path(self._storage_folder, self._name)

    @property
    def dataset_name(self) -> str:
        """
        Returns the name of the dataset
        """
        return self._name

    @property
    def raw_dataset_folder(self) -> Path:
        """
        Returns the Path to folder where the raw data is stored
        """
        return Path(self._raw_dataset_folder)

    @property
    def dataset_filepaths(self) -> List[Path]:
        """
        Returns the full Path to the single file where the dataset is stored
        """
        return [Path(self.dataset_folder, u) for u in self.url_indices]

    @property
    def url_indices(self) -> List[Path]:
        r"""
        Specifies the ist of file names where you plan to store
            portions of the large dataset
        """
        raise NotImplementedError(
            "You should subclass "
            "IterableDatasetInterface and implement "
            "this method"
        )

    def shuffle_urls_elements(self, value: bool):
        r"""
        Shuffles elements contained in each file (associated with an url).
        Use this method when a single file stores multiple samples and you want
        to provide them in shuffled order.
        IMPORTANT: in this case we assume that each file contains a
        list of Data objects!

        Args:
            value (bool): whether to shuffle urls
        """
        self._shuffle_subpatches = value

    def shuffle_urls(self, value: bool):
        r"""
        Shuffles urls associated to individual files stored on disk

        Args:
            value (bool): whether to shuffle urls
        """
        self._shuffle_urls = value

        # Needed for multiple dataloader workers
        if self._shuffle_urls:
            shuffle(self.shuffled_urls)

    def splice(self, start: int, end: int):
        r"""
        Use this method to assign portions of the dataset to load to different
        workers, otherwise they will load the same samples.

        Args:
            start (int): the index where to start
            end (int): the index where to stop
        """
        self.start_index = start
        self.end_index = end

    def __iter__(self):
        r"""
        Generator that returns individual Data objects. If each files contains
        a list of data objects, these can be shuffled using the
        method :func:`shuffle_urls_elements`.

        Returns:
            a Data object with the next element to process
        """
        end_index = (
            self.end_index
            if self.end_index <= len(self.shuffled_urls)
            else len(self.shuffled_urls)
        )
        for url in self.shuffled_urls[self.start_index : end_index]:
            url_data = torch.load(url)

            if not isinstance(url_data, list):
                url_data = [url_data]

            if self._shuffle_subpatches:
                shuffle(url_data)

            for i in range(len(url_data)):
                yield (
                    self.transform(url_data[i])
                    if self.transform is not None
                    else url_data[i]
                )

    def process_dataset(self, pre_transform: Optional[Callable]):
        r"""
        Processes the dataset to the :obj:`self.dataset_folder` folder. It
        should generate and store files according to the obj:`self.url_indices`
        list.

        Args:
            pre_transform (Optional[Callable]): transformations to apply to
                each sample at dataset creation time
        """
        raise NotImplementedError(
            "You should subclass IterableDatasetInterface "
            "and implement this method"
        )

    @property
    def dim_input_features(self) -> Union[int, Tuple[int]]:
        r"""
        Specifies the number of input features or a tuple if there are more,
        for instance node and edge features in graphs.
        """
        raise NotImplementedError(
            "You should subclass IterableDatasetInterface "
            "and implement this method"
        )

    @property
    def dim_target(self) -> Union[int, Tuple[int]]:
        r"""
        Specifies the dimension of each target vector or a tuple if there are
        more.
        """
        raise NotImplementedError(
            "You should subclass IterableDatasetInterface "
            "and implement this method"
        )

    def __len__(self):
        r"""
        Returns the number of graphs stored in the dataset.
        Note: we need to implement both `len` and `__len__` to comply
        with PyG interface
        """
        # It's important it stays dynamic, because
        # self.urls depends on url_indices
        return len(self.url_indices)


class ToyIterableDataset(IterableDatasetInterface):
    def __init__(
        self,
        storage_folder: str,
        raw_dataset_folder: Optional[str] = None,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        **kwargs,
    ):
        super().__init__(
            storage_folder,
            raw_dataset_folder,
            transform,
            pre_transform,
            **kwargs,
        )

    @property
    def url_indices(self) -> List[Path]:
        r"""
        Specifies the ist of file names where you plan to store
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
        return 1

    def process_dataset(self, pre_transform: Optional[Callable]):
        r"""
        Creates a fake dataset and stores it to the :obj:`self.processed_dir`
        folder. Each file will contain a list of 10 fake graphs.
        """
        for i in range(len(self)):
            fake_samples = []
            for s in range(100):
                fake_sample = (
                    torch.zeros(20, self.dim_input_features),
                    torch.zeros(1, 1),
                )

                if pre_transform is not None:
                    fake_sample = pre_transform(fake_sample)

                fake_samples.append(fake_sample)

            torch.save(fake_samples, self.dataset_filepaths[i])