import inspect
import os
import os.path as osp
import warnings
from typing import Callable

from mlwiz.util import s2c, dill_load, dill_save, return_class_and_args
from mlwiz.static import STORAGE_FOLDER, SKIP_SPLITS_CHECK


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
    dataset_args = data_info.pop("args")
    storage_folder = dataset_args.get(STORAGE_FOLDER)

    dataset_kwargs = {}

    pre_transform_class, pre_transform_args = return_class_and_args(
        dataset_args, "pre_transform"
    )
    if pre_transform_class is not None:
        pre_transform_args = (
            {} if pre_transform_args is None else pre_transform_args
        )
        dataset_kwargs.update(
            pre_transform=pre_transform_class(**pre_transform_args)
        )

    transform_tr_class, transform_tr_args = return_class_and_args(
        dataset_args, "transform_train"
    )
    if transform_tr_class is not None:
        transform_tr_args = (
            {} if transform_tr_args is None else transform_tr_args
        )
        dataset_kwargs.update(
            transform_train=transform_tr_class(**transform_tr_args)
        )

    transform_ev_class, transform_ev_args = return_class_and_args(
        dataset_args, "transform_eval"
    )

    if transform_ev_class is not None:
        transform_ev_class = (
            {} if transform_ev_class is None else transform_ev_class
        )
        dataset_kwargs.update(
            transform_eval=transform_ev_class(**transform_ev_args)
        )

    dataset_args.update(dataset_kwargs)

    dataset = dataset_class(**dataset_args)
    dataset_name = dataset.__class__.__name__

    # Store dataset additional arguments in a separate file
    kwargs_folder = osp.join(storage_folder, dataset_name)
    kwargs_path = osp.join(kwargs_folder, "dataset_kwargs.pt")

    get_or_create_dir(kwargs_folder)
    dill_save(dataset_args, kwargs_path)

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
    Loads the dataset using the ``dataset_kwargs.pt`` file created when parsing
    the data config file.

    Args:
        storage_folder (str): path of the folder that contains the dataset folder
        dataset_class
            (Callable):
            the class of the dataset to instantiate with the parameters
            stored in the ``dataset_kwargs.pt`` file.
        kwargs (dict): additional arguments to be passed to the
            dataset (potentially provided by a DataProvider)

    Returns:
        a dataset object
    """
    # Load arguments
    dataset_name = dataset_class.__name__
    kwargs_path = osp.join(storage_folder, dataset_name, "dataset_kwargs.pt")
    if not os.path.exists(kwargs_path):  # backward compatibility
        kwargs_path = osp.join(
            storage_folder, dataset_name, "processed", "dataset_kwargs.pt"
        )

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
    return batch[0]
