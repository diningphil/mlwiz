import os
from pathlib import Path
from random import shuffle
from typing import List, Union, Tuple, Optional, Callable

import torch
import torchvision
from torch_geometric.datasets import TUDataset, Planetoid

from mlwiz.data.util import get_or_create_dir
from mlwiz.util import dill_save, dill_load, s2c


class DatasetInterface:
    r"""
    Class that defines a number of properties essential to all datasets
    implementations inside MLWiz. These properties are used by the training
    engine and forwarded to the model to be trained.

    Useful for small to medium datasets where a single file can contain the
    whole data. In case the dataset is too large and needs to be split into
    chunks, check :obj:`mlwiz.data.dataset.IterableDatasetInterface`

    Please note that in order to use transformations you need to use classes
    like :obj:`mlwiz.data.provider.SubsetTrainEval`

    Args:
        storage_folder (str): path to folder where to store the dataset
        raw_dataset_folder (Optional[str]): path to raw data folder where raw
            data is stored
        transform_train (Optional[Callable]): transformations to apply to each
            sample at training time
        transform_eval (Optional[Callable]): transformations to apply to each
            sample at eval time
        pre_transform (Optional[Callable]): transformations to apply to each
            sample at dataset creation time
    """

    def __init__(
        self,
        storage_folder: str,
        raw_dataset_folder: Optional[str] = None,
        transform_train: Optional[Callable] = None,
        transform_eval: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        **kwargs,
    ):
        super().__init__()
        self._name = self.__class__.__name__
        self._storage_folder = storage_folder
        self._raw_dataset_folder = raw_dataset_folder
        self.transform_train = transform_train
        self.transform_eval = transform_eval
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
            self._save_dataset(self.dataset, self.dataset_filepath)

        else:
            # Simply load the dataset in memory
            self.dataset = self._load_dataset(self.dataset_filepath)

    @staticmethod
    def _save_dataset(dataset, dataset_filepath):
        dill_save(dataset, dataset_filepath)

    @staticmethod
    def _load_dataset(dataset_filepath):
        return dill_load(dataset_filepath)

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

        # NOTE runtime preprocessing is handled by DataProvider
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
    """
    Note: For graph datasets we still use torch.save/load, since
    PyG >=2.6.0 specifies the safe globals (Pytorch 2.5) and torch.save/load
    is much faster and more efficient (space/time) at storing Data objects.
    """

    @staticmethod
    def _save_dataset(dataset, dataset_filepath):
        torch.save(dataset, dataset_filepath)

    @staticmethod
    def _load_dataset(dataset_filepath):
        return torch.load(dataset_filepath, weights_only=False)

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
        return [(g, g.y.item()) for g in self.dataset]


class Cora(DatasetInterface):
    """
    Note: For graph datasets we still use torch.save/load, since
    PyG >=2.6.0 specifies the safe globals (Pytorch 2.5) and torch.save/load
    is much faster and more efficient (space/time) at storing Data objects.
    """

    @staticmethod
    def _save_dataset(dataset, dataset_filepath):
        torch.save(dataset, dataset_filepath)

    @staticmethod
    def _load_dataset(dataset_filepath):
        return torch.load(dataset_filepath, weights_only=False)

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
        transform_train (Optional[Callable]): transformations to apply to each
            sample at training time
        transform_eval (Optional[Callable]): transformations to apply to each
            sample at eval time
        pre_transform (Optional[Callable]): transformations to apply to each
            sample at dataset creation time
    """

    def __init__(
        self,
        storage_folder: str,
        raw_dataset_folder: Optional[str] = None,
        transform_train: Optional[Callable] = None,
        transform_eval: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        **kwargs,
    ):
        super().__init__()
        self._name = self.__class__.__name__
        self._storage_folder = storage_folder
        self._raw_dataset_folder = raw_dataset_folder
        self._dataset_folder = Path(self._storage_folder, self._name)
        self._eval = None

        # Create folders where to store processed dataset
        get_or_create_dir(self._dataset_folder)

        if self._raw_dataset_folder is not None and not os.path.exists(
            self.raw_dataset_folder
        ):
            raise FileNotFoundError(
                f"Folder {self._raw_dataset_folder} " f"not found"
            )

        self._url_indices = self.url_indices
        self.shuffled_urls = self.dataset_filepaths

        self.start_index = 0
        self.end_index = len(self.shuffled_urls)

        # This information allows us to shuffle between urls and sub-patches
        # inside each url
        self._shuffle_urls = False
        self._shuffle_subpatches = False

        self.pre_transform = pre_transform
        self.transform_train = transform_train
        self.transform_eval = transform_eval

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
        return [Path(self.dataset_folder, u) for u in self._url_indices]

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

    def set_eval(self, is_eval: bool):
        """
        Informs the object that we should either apply transform_train or
            transform_eval depending on the samples
        :param is_eval: whether the subset refers to training or evaluation
            stages, so we can apply the proper transformations.
            Set to false if training, true otherwise
        :return:
        """
        self._eval = is_eval

    def subset(self, indices: List[int]):
        r"""
        Use this method to modify the dataset by taking a subset of samples.
        WARNING: It PERMANENTLY changes the object URLs, so you have to create
        a copy of the original object before calling this method. It is not
        a memory intensive process to create a copy since this dataset works as
        iterable and loads data from disk on the fly.

        Args:
            indices (List[int]): the indices to keep
        """
        self._url_indices = [self._url_indices[i] for i in indices]
        self.shuffled_urls = [self.shuffled_urls[i] for i in indices]

    def __iter__(self):
        r"""
        Generator that returns individual Data objects. If each file contains
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
            url_data = dill_load(url)

            if not isinstance(url_data, list):
                url_data = [url_data]

            if self._shuffle_subpatches:
                shuffle(url_data)

            # this is the case where the user forgets to set the _eval field,
            # e.g., in a notebook, but it's not a problem because transforms
            # have not been used at all
            if self._eval is None:
                if (
                    self.transform_train is not None
                    or self.transform_eval is not None
                ):
                    raise Exception(
                        "You specified some transforms but you "
                        "have not called set_eval() first."
                    )

                for i in range(len(url_data)):
                    yield url_data[i]

            else:
                transform = (
                    self.transform_eval if self._eval else self.transform_train
                )

                for i in range(len(url_data)):
                    yield (
                        transform(url_data[i])
                        if transform is not None
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
        Returns the number of samples stored in the dataset.
        """
        # It's important it stays dynamic
        return len(self.shuffled_urls[self.start_index : self.end_index])


class ToyIterableDataset(IterableDatasetInterface):
    def __init__(
        self,
        storage_folder: str,
        raw_dataset_folder: Optional[str] = None,
        transform_train: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        **kwargs,
    ):
        super().__init__(
            storage_folder,
            raw_dataset_folder,
            transform_train,
            pre_transform,
            **kwargs,
        )

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
            for s in range(100):
                fake_sample = (
                    torch.zeros(20, self.dim_input_features),
                    torch.zeros(20, 2),
                )

                if pre_transform is not None:
                    fake_sample = pre_transform(fake_sample)

                fake_samples.append(fake_sample)

            dill_save(fake_samples, self.dataset_filepaths[i])
