"""Training utility helpers.

Includes helper functions for safe and memory-aware checkpoint serialization.
"""

import copy
import json
import os
from pathlib import Path
from typing import Any

import torch

from mlwiz.static import (
    ATOMIC_SAVE_EXTENSION,
    MODEL_GRAPH_INPUT_SPEC_FILENAME,
)


MODEL_GRAPH_INPUT_SPEC_VERSION = 1


def record_model_graph_input_spec(
    exp_path: str | os.PathLike | None, batch_input: Any
) -> bool:
    """Atomically persist a data-free example-input description for a run.

    The dashboard can use this file to create a synthetic input for
    :func:`torch.export.export` while inspecting a checkpoint.  It deliberately
    records only a tensor's shape and dtype, never values, device information,
    or other batch contents.  A non-tensor input is written as an unsupported
    marker so graph inspection can explain why an operator graph is unavailable.

    The first successfully observed input wins: reusing an experiment folder
    does not overwrite its representative shape.  This helper is intentionally
    best-effort because the optional dashboard metadata must not interrupt
    training when a run directory cannot be written.

    Args:
        exp_path: Run directory where the metadata belongs.  ``None`` disables
            recording, as in engine-only/unit-test usage.
        batch_input: Value passed to ``model.forward``.

    Returns:
        ``True`` if a new specification was written, otherwise ``False``.
    """
    if exp_path is None:
        return False

    destination = Path(exp_path) / MODEL_GRAPH_INPUT_SPEC_FILENAME
    if destination.exists():
        return False

    if torch.is_tensor(batch_input):
        try:
            specification = {
                "version": MODEL_GRAPH_INPUT_SPEC_VERSION,
                "kind": "tensor",
                "shape": [int(dimension) for dimension in batch_input.shape],
                "dtype": str(batch_input.dtype).removeprefix("torch."),
            }
        except (OverflowError, TypeError, ValueError):
            specification = _unsupported_model_graph_input_spec(batch_input)
    else:
        specification = _unsupported_model_graph_input_spec(batch_input)

    temporary = destination.with_name(
        destination.name + ATOMIC_SAVE_EXTENSION
    )
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(specification, stream, ensure_ascii=False)
        # Avoid replacing a valid first observation when the destination was
        # created after the initial existence check (for example by a resumed
        # process).  Main-process-only invocation normally prevents this race.
        if destination.exists():
            return False
        os.replace(temporary, destination)
        return True
    except (OSError, TypeError, ValueError):
        return False
    finally:
        try:
            if temporary.exists():
                temporary.unlink()
        except OSError:
            # This is optional dashboard metadata: cleanup errors must not
            # interrupt model training.
            pass


def _unsupported_model_graph_input_spec(batch_input: Any) -> dict[str, Any]:
    """Return a data-free marker for input types unsupported by ``torch.export``."""
    input_type = type(batch_input)
    return {
        "version": MODEL_GRAPH_INPUT_SPEC_VERSION,
        "kind": "unsupported",
        "type": f"{input_type.__module__}.{input_type.__qualname__}",
    }


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
