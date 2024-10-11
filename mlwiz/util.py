from pydoc import locate
from typing import Tuple, Callable

import dill


def return_class_and_args(
    config: dict, key: str, return_class_name: bool = False
) -> Tuple[Callable[..., object], dict]:
    r"""
    Returns the class and arguments associated to a specific key in the
    configuration file.

    Args:
        config (dict): the configuration dictionary
        key (str): a string representing a particular class in the
            configuration dictionary
        return_class_name (bool): if ``True``, returns the class name as a
            string rather than the class object

    Returns:
        a tuple (class, dict of arguments), or (None, None) if the key
        is not present in the config dictionary
    """
    if key not in config or config[key] is None:
        return None, None
    elif isinstance(config[key], str):
        return s2c(config[key]), {}
    elif isinstance(config[key], dict):
        return (
            (
                s2c(config[key]["class_name"])
                if not return_class_name
                else config[key]["class_name"]
            ),
            config[key]["args"] if "args" in config[key] else {},
        )
    else:
        raise NotImplementedError(
            f"Parameter {key} " f"has not been formatted properly"
        )


def s2c(class_name: str) -> Callable[..., object]:
    r"""
    Converts a dotted path to the corresponding class

    Args:
         class_name (str): dotted path to class name

    Returns:
        the class to be used
    """
    result = locate(class_name)
    if result is None:
        raise ImportError(
            f"The (dotted) path '{class_name}' is unknown. "
            f"Check your configuration."
        )
    return result


def dill_save(data: object, filepath: str) -> object:
    """
    Saves a dill object to a file.
    :param data: the dill object to save
    :param filepath: the path to the dill object to save
    """
    with open(filepath, "wb") as file:
        return dill.dump(data, file)


def dill_load(filepath: str) -> object:
    """
    Loads a dill file.
    :param filepath: the path to the dill file
    :return: an object
    """
    with open(filepath, "rb") as file:
        return dill.load(file)
