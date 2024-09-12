import inspect
import os
import os.path as osp
import warnings
from typing import Callable

from torchvision.transforms import Compose

from mlwiz.util import s2c, dill_load, dill_save
from mlwiz.static import STORAGE_FOLDER


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


def preprocess_data(options: dict):
    r"""
    One of the main functions of the MLWiz library. Used to create the dataset
    and its associated files that ensure the correct functioning of the
    data loading steps.

    Args:
        options (dict): a dictionary of dataset/splitter arguments as
            defined in the data configuration file used.

    """
    data_info = options.pop("dataset")
    if "class_name" not in data_info:
        raise ValueError("You must specify 'class_name' in your dataset.")
    dataset_class = s2c(data_info.pop("class_name"))
    dataset_args = data_info.pop("args")
    storage_folder = dataset_args.get(STORAGE_FOLDER)

    ################################

    dataset_kwargs = {}

    pre_transforms_opt = data_info.pop("pre_transform", None)
    if pre_transforms_opt is not None:
        pre_transforms = []
        for pre_transform in pre_transforms_opt:
            pre_transform_class = s2c(pre_transform["class_name"])
            args = pre_transform.pop("args", {})
            pre_transforms.append(pre_transform_class(**args))
        dataset_kwargs.update(pre_transform=Compose(pre_transforms))

    transforms_opt = data_info.pop("transform", None)
    if transforms_opt is not None:
        transforms = []
        for transform in transforms_opt:
            transform_class = s2c(transform["class_name"])
            args = transform.pop("args", {})
            transforms.append(transform_class(**args))
        dataset_kwargs.update(transform=Compose(transforms))

    dataset_args.update(dataset_kwargs)

    ################################

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
