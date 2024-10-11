import math
import random
from copy import deepcopy
from typing import Union, Callable, Sequence, List, T_co

import numpy as np
import torch
import torch_geometric.loader
from torch.utils.data import Subset
from torch_geometric.loader.dataloader import Collater

import mlwiz.data.dataset
from mlwiz.data.dataset import DatasetInterface
from mlwiz.data.sampler import RandomSampler
from mlwiz.data.splitter import Splitter, SingleGraphSplitter
from mlwiz.data.util import load_dataset, single_graph_collate
from mlwiz.util import s2c


def seed_worker(exp_seed, worker_id):
    r"""
    Used to set a different, but reproducible, seed for all data-retriever
    workers. Without this, all workers will retrieve the data in the same
    order (important for Iterable-style datasets).

    Args:
        exp_seed (int): base seed to be used for reproducibility
        worker_id (int): id number of the worker
    """
    np.random.seed(exp_seed + worker_id)
    random.seed(exp_seed + worker_id)
    torch.manual_seed(exp_seed + worker_id)
    torch.cuda.manual_seed(exp_seed + worker_id)


class SubsetTrainEval(Subset):
    r"""
    Extension of Pytorch Subset to differentiate between training and
    evaluation subsets.

    Args:
        dataset (DatasetInterface): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
        is_eval (bool): false if training true otherwise
    """

    def __init__(
        self, dataset: DatasetInterface, indices: Sequence[int], is_eval: bool
    ):
        super().__init__(dataset, indices)
        self.is_eval = is_eval
        self._t = (
            self.dataset.transform_eval
            if is_eval
            else self.dataset.transform_train
        )

    def __getitem__(self, idx):
        if self._t is None:
            super().__getitem__(idx)
        else:
            if isinstance(idx, list):
                return [self._t(self.dataset[self.indices[i]]) for i in idx]
            return self._t(self.dataset[self.indices[idx]])

    def __getitems__(self, indices: List[int]) -> List[T_co]:
        if self._t is None:
            return super().__getitems__(indices)
        else:
            # add batched sampling support when parent dataset supports it.
            # see torch.utils.data._utils.fetch._MapDatasetFetcher
            if callable(getattr(self.dataset, "__getitems__", None)):
                samples = self.dataset.__getitems__(
                    [self.indices[idx] for idx in indices]
                )
            else:
                samples = [self.dataset[self.indices[idx]] for idx in indices]
            return [self._t(s) for s in samples]


class DataProvider:
    r"""
    A DataProvider object retrieves the correct data according to the external
    and internal data splits. It can be additionally used to augment the data,
    or to create a specific type of data loader. The base class does nothing
    special, but here is where the i-th element of a dataset could be
    pre-processed before constructing the mini-batches.

    IMPORTANT: if the dataset is to be shuffled, you MUST use a
    :class:`mlwiz.data.sampler.RandomSampler` object to determine the
    permutation.

    Args:
        storage_folder (str): the path of the root folder in which data is stored
        splits_filepath (str): the filepath of the splits. with additional
            metadata
        dataset_class
            (Callable[...,:class:`mlwiz.data.dataset.DatasetInterface`]):
            the class of the dataset
        data_loader_class
            (Union[Callable[...,:class:`torch.utils.data.DataLoader`],
            Callable[...,:class:`torch_geometric.loader.DataLoader`]]):
            the class of the data loader to use
        data_loader_args (dict): the arguments of the data loader
        outer_folds (int): the number of outer folds for risk assessment.
            1 means hold-out, >1 means k-fold
        inner_folds (int): the number of outer folds for model selection.
            1 means hold-out, >1 means k-fold

    """

    def __init__(
        self,
        storage_folder: str,
        splits_filepath: str,
        dataset_class: Callable[..., mlwiz.data.dataset.DatasetInterface],
        data_loader_class: Union[
            Callable[..., torch.utils.data.DataLoader],
            Callable[..., torch_geometric.loader.DataLoader],
        ],
        data_loader_args: dict,
        outer_folds: int,
        inner_folds: int,
    ) -> object:
        self.exp_seed = None

        self.storage_folder = storage_folder
        self.dataset_class = dataset_class

        self.data_loader_class = data_loader_class
        self.data_loader_args = data_loader_args

        self.outer_folds = outer_folds
        self.inner_folds = inner_folds

        self.outer_k = None
        self.inner_k = None

        self.splits_filepath = splits_filepath
        self.splitter = None

        # use this to avoid instantiating multiple versions of the same
        # dataset when no run-time
        # specific arguments are needed
        self.dataset = None

        self.dim_input_features = None
        self.dim_target = None

    def set_exp_seed(self, seed: int):
        r"""
        Sets the experiment seed to give to the DataLoader.
        Helps with reproducibility.

        Args:
            seed (int): id of the seed
        """
        self.exp_seed = seed

    def set_outer_k(self, k: int):
        r"""
        Sets the parameter k of the `risk assessment` procedure.
        Called by the evaluation modules to load the correct data subset.

        Args:
            k (int): the id of the fold, ranging from 0 to K-1.
        """
        self.outer_k = k

    def set_inner_k(self, k):
        r"""
        Sets the parameter k of the `model selection` procedure. Called by the
        evaluation modules to load the correct subset of the data.

        Args:
            k (int): the id of the fold, ranging from 0 to K-1.
        """
        self.inner_k = k

    def _get_splitter(self) -> Splitter:
        """
        Instantiates the splitter with the parameters stored
        in the file ``self.splits_filepath``

        Returns:
            a :class:`~mlwiz.data.splitter.Splitter` object
        """
        if self.splitter is None:
            self.splitter = Splitter.load(self.splits_filepath)
        return self.splitter

    def _get_dataset(self, **kwargs: dict) -> DatasetInterface:
        """
        Instantiates the dataset. Relies on the parameters stored in
        the ``dataset_kwargs.pt`` file.

        Args:
            kwargs (dict): a dictionary of additional arguments to be passed
                to the dataset. Not used in the base version

        Returns:
            a :class:`~mlwiz.data.dataset.DatasetInterface` object
        """
        assert "shuffle" not in kwargs, (
            "Your implementation of _get_loader should remove `shuffle` "
            "from kwargs before calling _get_dataset"
        )
        assert "batch_size" not in kwargs, (
            "Your implementation of _get_loader should remove `batch_size` "
            "from kwargs before calling _get_dataset"
        )

        if kwargs is not None and len(kwargs) != 0:
            # we probably need to pass run-time specific parameters,
            # so load the dataset in memory again
            # an example is the subset of urls in Iterable style datasets
            dataset = load_dataset(
                self.storage_folder, self.dataset_class, **kwargs
            )
        else:
            if self.dataset is None:
                dataset = load_dataset(self.storage_folder, self.dataset_class)
                self.dataset = dataset
            else:
                dataset = self.dataset

        self.dim_input_features = dataset.dim_input_features
        self.dim_target = dataset.dim_target

        return dataset

    def _get_loader(
        self, indices: list, is_eval: bool, **kwargs: dict
    ) -> Union[torch.utils.data.DataLoader, torch_geometric.loader.DataLoader]:
        r"""
        Instantiates the data loader.

        Args:
            indices (sequence): Indices in the whole set selected for subset
            is_eval: false if training, true otherwise
            kwargs (dict): a dictionary of additional arguments to be passed
                to the dataset being loaded. Not used in the base version

        Returns:
            a Union[:class:`torch.utils.data.DataLoader`,
            :class:`torch_geometric.loader.DataLoader`] object
        """
        shuffle = kwargs.pop("shuffle", False)
        batch_size = kwargs.pop("batch_size", 1)

        dataset: DatasetInterface = self._get_dataset(**kwargs)
        dataset = SubsetTrainEval(dataset, indices, is_eval)

        assert (
            self.exp_seed is not None
        ), "DataLoader's seed has not been specified! Is this a bug?"

        # no need to set worker seed in map-stye dataset, see pytorch doc about reproducibility
        # this would also cause transforms based on random sampling to behave in the same way every time
        # kwargs["worker_init_fn"] = lambda worker_id: seed_worker(
        #     worker_id, self.exp_seed
        # )

        kwargs.update(self.data_loader_args)

        if shuffle is True:
            sampler = RandomSampler(dataset)
            dataloader = self.data_loader_class(
                dataset, sampler=sampler, batch_size=batch_size, **kwargs
            )
        else:
            dataloader = self.data_loader_class(
                dataset, shuffle=False, batch_size=batch_size, **kwargs
            )

        return dataloader

    def get_inner_train(
        self, **kwargs: dict
    ) -> Union[torch.utils.data.DataLoader, torch_geometric.loader.DataLoader]:
        r"""
        Returns the training set for model selection associated with specific
        outer and inner folds

        Args:
            kwargs (dict): a dictionary of additional arguments to be passed
                to the dataset being loaded. Not used in the base version

        Returns:
            a Union[:class:`torch.utils.data.DataLoader`,
            :class:`torch_geometric.loader.DataLoader`] object

        """
        is_eval = False
        assert self.outer_k is not None and self.inner_k is not None
        splitter = self._get_splitter()
        indices = splitter.inner_folds[self.outer_k][self.inner_k].train_idxs
        return self._get_loader(indices, is_eval, **kwargs)

    def get_inner_val(
        self, **kwargs: dict
    ) -> Union[torch.utils.data.DataLoader, torch_geometric.loader.DataLoader]:
        r"""
        Returns the validation set for model selection associated with
        specific outer and inner folds

        Args:
            kwargs (dict): a dictionary of additional arguments to be passed
                to the dataset being loaded. Not used in the base version

        Returns:
            a Union[:class:`torch.utils.data.DataLoader`,
            :class:`torch_geometric.loader.DataLoader`] object

        """
        is_eval = True
        assert self.outer_k is not None and self.inner_k is not None
        splitter = self._get_splitter()
        indices = splitter.inner_folds[self.outer_k][self.inner_k].val_idxs
        return self._get_loader(indices, is_eval, **kwargs)

    def get_outer_train(
        self, **kwargs: dict
    ) -> Union[torch.utils.data.DataLoader, torch_geometric.loader.DataLoader]:
        r"""
        Returns the training set for risk assessment associated with specific
        outer and inner folds

        Args:
            kwargs (dict): a dictionary of additional arguments to be passed
                to the dataset being loaded. Not used in the base version

        Returns:
            a Union[:class:`torch.utils.data.DataLoader`,
            :class:`torch_geometric.loader.DataLoader`] object

        """
        is_eval = False
        assert self.outer_k is not None
        splitter = self._get_splitter()
        train_indices = splitter.outer_folds[self.outer_k].train_idxs
        return self._get_loader(train_indices, is_eval, **kwargs)

    def get_outer_val(
        self, **kwargs: dict
    ) -> Union[torch.utils.data.DataLoader, torch_geometric.loader.DataLoader]:
        r"""
        Returns the validation set for risk assessment associated with
        specific outer and inner folds

        Args:
            kwargs (dict): a dictionary of additional arguments to be passed
                to the dataset being loaded. Not used in the base version

        Returns:
            a Union[:class:`torch.utils.data.DataLoader`,
            :class:`torch_geometric.loader.DataLoader`] object

        """
        is_eval = True
        assert self.outer_k is not None
        splitter = self._get_splitter()
        val_indices = splitter.outer_folds[self.outer_k].val_idxs
        return self._get_loader(val_indices, is_eval, **kwargs)

    def get_outer_test(
        self, **kwargs: dict
    ) -> Union[torch.utils.data.DataLoader, torch_geometric.loader.DataLoader]:
        r"""
        Returns the test set for risk assessment associated with specific
        outer and inner folds

        Args:
            kwargs (dict): a dictionary of additional arguments to be passed
                to the dataset being loaded. Not used in the base version

        Returns:
            a Union[:class:`torch.utils.data.DataLoader`,
            :class:`torch_geometric.loader.DataLoader`] object

        """
        is_eval = True
        assert self.outer_k is not None
        splitter = self._get_splitter()
        indices = splitter.outer_folds[self.outer_k].test_idxs
        return self._get_loader(indices, is_eval, **kwargs)

    def get_dim_input_features(self) -> int:
        r"""
        Returns the number of node features of the dataset

        Returns:
            the value of the property ``dim_input_features`` in the dataset

        """
        if self.dim_input_features is None:
            raise Exception(
                "You should first initialize the dataset "
                "by creating a data loader!"
            )
        return self.dim_input_features

    def get_dim_target(self) -> int:
        r"""
        Returns the dimension of the target for the task

        Returns:
            the value of the property ``dim_target`` in the dataset

        """
        if self.dim_target is None:
            raise Exception(
                "You should first initialize the dataset "
                "by creating a data loader!"
            )
        return self.dim_target


class IterableDataProvider(DataProvider):
    r"""
    A DataProvider object that allows to fetch data from an Iterable-style
    Dataset (see :class:`mlwiz.data.dataset.IterableDatasetInterface`).
    """

    def _get_loader(
        self, indices: list, is_eval: bool, **kwargs: dict
    ) -> Union[torch.utils.data.DataLoader, torch_geometric.loader.DataLoader]:
        r"""
        Instantiates the data loader, passing to the dataset an additional
        `url_indices` argument with the indices to fetch. This is because
        each time this method is called with different indices a separate
        instance of the dataset is called.

        Args:
            indices (sequence): Indices in the whole set selected for subset
            is_eval: false if training, true otherwise
            kwargs (dict): a dictionary of additional arguments to be passed
                to the dataset being loaded. Not used in the base version

        Returns:
            a Union[:class:`torch.utils.data.DataLoader`,
            :class:`torch_geometric.loader.DataLoader`] object
        """
        shuffle = kwargs.pop("shuffle", False)
        batch_size = kwargs.pop("batch_size", 1)

        # we can deepcopy the dataset each time the loader is called,
        # because iterable datasets are not supposed to keep all data in memory
        dataset = deepcopy(self._get_dataset())
        dataset.set_eval(is_eval)
        dataset.subset(indices)

        if shuffle:
            dataset.shuffle_urls(True)
            dataset.shuffle_urls_elements(True)

        assert (
            self.exp_seed is not None
        ), "DataLoader's seed has not been specified! Is this a bug?"

        # Define a worker_init_fn that configures each dataset copy differently
        # this is called only when num_workers is set to a value > 0
        def worker_init_fn(worker_id: int):
            """
            Set the seeds for the worker and computes the range of
            samples ids to fetch.
            """
            worker_info = torch.utils.data.get_worker_info()
            num_workers = worker_info.num_workers
            assert num_workers > 0

            # Set the random seed
            seed_worker(worker_id, self.exp_seed)

            # Get the dataset and overall length
            dataset = (
                worker_info.dataset
            )  # the dataset copy in this worker process
            dataset_length = len(
                dataset
            )  # dynamic, already refers to the subset of urls!

            per_worker = int(
                math.ceil((dataset_length) / float(worker_info.num_workers))
            )

            start = worker_id * per_worker
            end = worker_id * per_worker + per_worker

            # configure the dataset to only process the split workload
            dataset.splice(start, end)

        kwargs.update(self.data_loader_args)

        dataloader = self.data_loader_class(
            dataset,
            sampler=None,
            collate_fn=Collater(None, None),
            worker_init_fn=worker_init_fn,
            batch_size=batch_size,
            **kwargs,
        )
        return dataloader


class SingleGraphDataProvider(DataProvider):
    r"""
    A DataProvider subclass that only works with
    :class:`mlwiz.data.splitter.SingleGraphSplitter`.

    Args:
        storage_folder (str): the path of the root folder in which data is stored
        splits_filepath (str): the filepath of the splits. with additional
            metadata
        dataset_class
            (Callable[...,:class:`mlwiz.data.dataset.DatasetInterface`]):
            the class of the dataset
        data_loader_class
            (Union[Callable[...,:class:`torch.utils.data.DataLoader`],
            Callable[...,:class:`torch_geometric.loader.DataLoader`]]):
            the class of the data loader to use
        data_loader_args (dict): the arguments of the data loader
        outer_folds (int): the number of outer folds for risk assessment.
            1 means hold-out, >1 means k-fold
        inner_folds (int): the number of outer folds for model selection.
            1 means hold-out, >1 means k-fold

    """

    def _get_splitter(self):
        """
        Instantiates the splitter with the parameters stored in the file
        ``self.splits_filepath``. Only works
        with `~mlwiz.data.splitter.SingleGraphSplitter`.

        Returns:
            a :class:`~mlwiz.data.splitter.Splitter` object
        """
        super()._get_splitter()  # loads splitter into self.splitter
        assert isinstance(
            self.splitter, SingleGraphSplitter
        ), "This class only works with a SingleGraphNodeSplitter splitter."
        return self.splitter

    def _get_dataset(self, **kwargs: dict) -> DatasetInterface:
        """
        Compared to superclass method, this always returns a new instance of
        the dataset, optionally passing extra arguments specified at runtime.

        Args:
            kwargs (dict): a dictionary of additional arguments to be
                passed to the dataset. Not used in the base version

        Returns:
            a :class:`~mlwiz.data.dataset.DatasetInterface` object
        """
        assert "shuffle" not in kwargs, (
            "Your implementation of _get_loader should remove `shuffle` "
            "from kwargs before calling _get_dataset"
        )
        assert "batch_size" not in kwargs, (
            "Your implementation of _get_loader should remove `batch_size` "
            "from kwargs before calling _get_dataset"
        )

        # we probably need to pass run-time specific parameters, so load the
        # dataset in memory again
        # an example is the subset of urls in Iterable style datasets
        dataset = load_dataset(
            self.storage_folder, self.dataset_class, **kwargs
        )

        self.dim_input_features = dataset.dim_input_features
        self.dim_target = dataset.dim_target
        return dataset

    def _get_loader(
        self, eval_indices: list, training_indices: list, **kwargs: dict
    ) -> Union[torch.utils.data.DataLoader, torch_geometric.loader.DataLoader]:
        r"""
        Compared to superclass method, returns a dataloader with the single
        graph augmented with additional fields. These are `training_indices`
        with the indices that refer to training nodes (usually always
        available) and `eval_indices`, which specify which are the indices
        on which to evaluate (can be validation or test).

        Args:
            indices (sequence): Indices in the whole set selected for subset
            eval_set (bool): whether or not indices refer to eval set
                (validation or test) or to training
            kwargs (dict): a dictionary of additional arguments to be passed
                to the dataset being loaded. Not used in the base version
        Returns:
            a Union[:class:`torch.utils.data.DataLoader`,
            :class:`torch_geometric.loader.DataLoader`] object
        """
        shuffle = kwargs.pop("shuffle", False)
        batch_size = kwargs.pop("batch_size", 1)

        # TODO is there a way to refactor the code to avoid that we have
        #  to duplicate the dataset for every loader just to set different
        #  training and evaluation indices?
        dataset: DatasetInterface = self._get_dataset(**kwargs)

        # single graph means there is only one sample
        dataset[0][0].training_indices = torch.tensor(training_indices)
        dataset[0][0].eval_indices = torch.tensor(eval_indices)

        assert (
            self.exp_seed is not None
        ), "DataLoader's seed has not been specified! Is this a bug?"

        # no need to set worker seed in map-stye dataset, see pytorch doc about reproducibility
        # this would also cause transforms based on random sampling to behave in the same way every time
        # kwargs["worker_init_fn"] = lambda worker_id: seed_worker(
        #     worker_id, self.exp_seed
        # )

        kwargs.update(self.data_loader_args)

        if shuffle is True:
            sampler = RandomSampler(dataset)
            dataloader = self.data_loader_class(
                dataset,
                sampler=sampler,
                batch_size=batch_size,
                collate_fn=single_graph_collate,
                **kwargs,
            )
        else:
            dataloader = self.data_loader_class(
                dataset,
                shuffle=False,
                batch_size=batch_size,
                collate_fn=single_graph_collate,
                **kwargs,
            )

        return dataloader

    def get_inner_train(
        self, **kwargs: dict
    ) -> Union[torch.utils.data.DataLoader, torch_geometric.loader.DataLoader]:
        r"""
        Returns the training set for model selection associated with specific
        outer and inner folds

        Args:
            kwargs (dict): a dictionary of additional arguments to be passed
                to the dataset being loaded. Not used in the base version

        Returns:
            a Union[:class:`torch.utils.data.DataLoader`,
            :class:`torch_geometric.loader.DataLoader`] object

        """
        assert self.outer_k is not None and self.inner_k is not None
        splitter = self._get_splitter()
        train_indices = splitter.inner_folds[self.outer_k][
            self.inner_k
        ].train_idxs
        return self._get_loader(
            train_indices, training_indices=train_indices, **kwargs
        )

    def get_inner_val(
        self, **kwargs: dict
    ) -> Union[torch.utils.data.DataLoader, torch_geometric.loader.DataLoader]:
        r"""
        Returns the validation set for model selection associated with
        specific outer and inner folds

        Args:
            kwargs (dict): a dictionary of additional arguments to be passed
                to the dataset being loaded. Not used in the base version

        Returns:
            a Union[:class:`torch.utils.data.DataLoader`,
            :class:`torch_geometric.loader.DataLoader`] object

        """
        assert self.outer_k is not None and self.inner_k is not None
        splitter = self._get_splitter()
        train_indices = splitter.inner_folds[self.outer_k][
            self.inner_k
        ].train_idxs
        val_indices = splitter.inner_folds[self.outer_k][self.inner_k].val_idxs
        return self._get_loader(
            val_indices, training_indices=train_indices, **kwargs
        )

    def get_outer_train(
        self, **kwargs: dict
    ) -> Union[torch.utils.data.DataLoader, torch_geometric.loader.DataLoader]:
        r"""
        Returns the training set for risk assessment associated with specific
        outer and inner folds

        Args:
            kwargs (dict): a dictionary of additional arguments to be passed
                to the dataset being loaded. Not used in the base version

        Returns:
            a Union[:class:`torch.utils.data.DataLoader`,
            :class:`torch_geometric.loader.DataLoader`] object

        """
        assert self.outer_k is not None
        splitter = self._get_splitter()

        train_indices = splitter.outer_folds[self.outer_k].train_idxs
        return self._get_loader(
            train_indices, training_indices=train_indices, **kwargs
        )

    def get_outer_val(
        self, **kwargs: dict
    ) -> Union[torch.utils.data.DataLoader, torch_geometric.loader.DataLoader]:
        r"""
        Returns the validation set for risk assessment associated with
        specific outer and inner folds

        Args:
            kwargs (dict): a dictionary of additional arguments to be passed
                to the dataset being loaded.Not used in the base version

        Returns:
            a Union[:class:`torch.utils.data.DataLoader`,
            :class:`torch_geometric.loader.DataLoader`] object

        """
        assert self.outer_k is not None
        splitter = self._get_splitter()
        train_indices = splitter.outer_folds[self.outer_k].train_idxs
        val_indices = splitter.outer_folds[self.outer_k].val_idxs
        return self._get_loader(
            val_indices, training_indices=train_indices, **kwargs
        )

    def get_outer_test(
        self, **kwargs: dict
    ) -> Union[torch.utils.data.DataLoader, torch_geometric.loader.DataLoader]:
        r"""
        Returns the test set for risk assessment associated with specific
        outer and inner folds

        Args:
            kwargs (dict): a dictionary of additional arguments to be passed
                to the dataset being loaded. Not used in the base version

        Returns:
            a Union[:class:`torch.utils.data.DataLoader`,
            :class:`torch_geometric.loader.DataLoader`] object

        """
        assert self.outer_k is not None
        splitter = self._get_splitter()
        train_indices = splitter.outer_folds[self.outer_k].train_idxs
        test_indices = splitter.outer_folds[self.outer_k].test_idxs
        return self._get_loader(
            test_indices, training_indices=train_indices, **kwargs
        )
