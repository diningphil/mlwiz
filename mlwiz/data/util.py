"""Dataset preprocessing and loading helpers.

Implements config-driven dataset/splitter instantiation and utilities like :func:`preprocess_data` and :func:`load_dataset`.
"""

import inspect
import json
import os
import os.path as osp
import warnings
from typing import Callable

from mlwiz.util import s2c, dill_load, return_class_and_args
from mlwiz.static import (
    ATOMIC_SAVE_EXTENSION,
    STORAGE_FOLDER,
    SKIP_SPLITS_CHECK,
)


_TRANSFORM_ARGUMENTS = ("pre_transform", "transform_train", "transform_eval")


def _instantiate_transforms(dataset_args: dict) -> dict:
    """Return dataset arguments with declarative transform specs instantiated."""
    instantiated_args = dataset_args.copy()
    for key in _TRANSFORM_ARGUMENTS:
        transform_class, transform_args = return_class_and_args(dataset_args, key)
        if transform_class is not None:
            instantiated_args[key] = transform_class(**transform_args)
    return instantiated_args


def _atomic_json_save(data: dict, filepath: str) -> None:
    """Atomically save a JSON dictionary."""
    tmp_path = filepath + ATOMIC_SAVE_EXTENSION
    try:
        with open(tmp_path, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=4)
        os.replace(tmp_path, filepath)
    finally:
        try:
            os.remove(tmp_path)
        except FileNotFoundError:
            pass


def get_or_create_dir(path: str) -> str:
    r"""
    Creates directories associated to the specified path if they are missing,
    and it returns the path string.

    Args:
        path (str): the path

    Returns:
        the same path as the given argument
    """
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def check_argument(cls: object, arg_name: str) -> bool:
    r"""
    Checks whether ``arg_name`` is in the signature of a method or class.

    Args:
        cls (object): the class to inspect
        arg_name (str): the name to look for

    Returns:
        ``True`` if the name was found, ``False`` otherwise
    """
    sign = inspect.signature(cls)
    return arg_name in sign.parameters.keys()


def preprocess_data(options: dict) -> dict:
    r"""
    One of the main functions of the MLWiz library. Used to create the dataset
    and its associated files that ensure the correct functioning of the
    data loading steps.

    Args:
        options (dict): a dictionary of dataset/splitter arguments as
            defined in the data configuration file used.

    """
    skip_splits_check = options.pop(SKIP_SPLITS_CHECK)
    data_info = options.pop("dataset")
    if "class_name" not in data_info:
        raise ValueError("You must specify 'class_name' in your dataset.")
    dataset_class = s2c(data_info.pop("class_name"))
    dataset_args_specification = data_info.pop("args")
    storage_folder = dataset_args_specification.get(STORAGE_FOLDER)
    dataset_args = _instantiate_transforms(dataset_args_specification)

    dataset = dataset_class(**dataset_args)
    dataset_name = dataset.__class__.__name__

    # Store dataset additional arguments in a separate file
    kwargs_folder = osp.join(storage_folder, dataset_name)
    kwargs_path = osp.join(kwargs_folder, "dataset_kwargs.json")

    get_or_create_dir(kwargs_folder)
    _atomic_json_save(dataset_args_specification, kwargs_path)

    # Process data splits

    splits_info = options.pop("splitter")
    splits_folder = splits_info.pop("splits_folder")
    if "class_name" not in splits_info:
        raise ValueError("You must specify 'class_name' in your splitter.")
    splitter_class = s2c(splits_info.pop("class_name"))
    splitter_args = splits_info.pop("args")
    splitter = splitter_class(**splitter_args)

    splits_dir = get_or_create_dir(osp.join(splits_folder, dataset_name))
    splits_path = osp.join(
        splits_dir,
        f"{dataset_name}_outer{splitter.n_outer_folds}"
        f"_inner{splitter.n_inner_folds}.splits",
    )

    if not os.path.exists(splits_path):
        if splitter.stratify:
            has_targets, targets = splitter.get_targets(dataset)
        else:
            print("No stratification required, skipping targets extraction...")
            has_targets, targets = False, None

        # The splitter is in charge of eventual stratifications
        splitter.split(dataset, targets=targets if has_targets else None)
        splitter.check_splits_overlap(skip_check=skip_splits_check)
        splitter.save(splits_path)
    else:
        print("Data splits are already present, I will not overwrite them.")


def load_dataset(
    storage_folder: str,
    dataset_class: Callable,
    **kwargs: dict,
) -> object:
    r"""
    Loads the dataset using the ``dataset_kwargs.json`` file created when parsing
    the data config file. Legacy ``dataset_kwargs.pt`` files remain supported.

    Args:
        storage_folder (str): path of the folder that contains the dataset folder
        dataset_class
            (Callable):
            the class of the dataset to instantiate with the parameters
            stored in the dataset kwargs file.
        kwargs (dict): additional arguments to be passed to the
            dataset (potentially provided by a DataProvider)

    Returns:
        a dataset object
    """
    # Load arguments
    dataset_name = dataset_class.__name__
    kwargs_folder = osp.join(storage_folder, dataset_name)
    kwargs_path = osp.join(kwargs_folder, "dataset_kwargs.json")
    if not os.path.exists(kwargs_path):
        kwargs_path = osp.join(kwargs_folder, "processed", "dataset_kwargs.json")

    if os.path.exists(kwargs_path):
        with open(kwargs_path, "r", encoding="utf-8") as file:
            dataset_args = _instantiate_transforms(json.load(file))
    else:  # backward compatibility
        kwargs_path = osp.join(kwargs_folder, "dataset_kwargs.pt")
        if not os.path.exists(kwargs_path):
            kwargs_path = osp.join(kwargs_folder, "processed", "dataset_kwargs.pt")
        dataset_args = dill_load(kwargs_path)

    # Overwrite original storage_folder field, which may have changed
    dataset_args["storage_folder"] = storage_folder

    # pass extra arguments to dataset
    dataset_args.update(kwargs)

    with warnings.catch_warnings():
        # suppress PyG warnings
        warnings.simplefilter("ignore")
        dataset = dataset_class(**dataset_args)

    return dataset


def single_graph_collate(batch):
    """
    Collate function for single-graph datasets.

    PyTorch/PyG data loaders build a list of samples for each batch. For
    single-graph workflows, the loader is typically configured with
    ``batch_size=1`` and each sample already contains all needed information.
    This collate function returns the single element in the batch list.

    Args:
        batch (list): Batch list produced by a DataLoader.

    Returns:
        object: The first (and expected only) element of ``batch``.
    """
    return batch[0]
