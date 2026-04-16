"""Training utility helpers.

Includes helper functions for safe and memory-aware checkpoint serialization.
"""

import copy
import os

import torch

from mlwiz.static import ATOMIC_SAVE_EXTENSION


def atomic_torch_save(data: dict, filepath: str):
    r"""
    Atomically stores a dictionary that can be serialized by
    :func:`torch.save`, exploiting the atomic :func:`os.replace`.

    Args:
        data (dict): the dictionary to be stored
        filepath (str): the absolute filepath where to store the dictionary
    """
    try:
        tmp_path = str(filepath) + ATOMIC_SAVE_EXTENSION
        torch.save(data, tmp_path)
        os.replace(tmp_path, filepath)
    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise e


def clone_to_cpu(obj):
    r"""
    Recursively clone tensors to CPU for checkpoint serialization.

    This avoids transient GPU-memory spikes caused by deep-copying large
    model/optimizer state dictionaries on CUDA.
    """
    if torch.is_tensor(obj):
        return obj.detach().cpu().clone()
    if isinstance(obj, dict):
        return {k: clone_to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clone_to_cpu(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(clone_to_cpu(v) for v in obj)
    return copy.deepcopy(obj)
