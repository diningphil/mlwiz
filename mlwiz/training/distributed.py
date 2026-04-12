"""Small helpers shared by distributed training code."""

import torch


def dist_is_initialized() -> bool:
    """Return True when torch.distributed is initialized."""
    return (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
    )


def is_main_process() -> bool:
    """Return True for rank 0 (or for non-distributed execution)."""
    if not dist_is_initialized():
        return True
    return torch.distributed.get_rank() == 0


def unwrap_model(model):
    """Return the wrapped module when model is DDP."""
    return model.module if hasattr(model, "module") else model
