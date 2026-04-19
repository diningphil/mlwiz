"""Unit tests for :mod:`mlwiz.training.callback.optimizer` state loading."""

from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from mlwiz.training.callback.optimizer import Optimizer


class _TwoScalarsAB(torch.nn.Module):
    """Model declaring parameters in ``p1, p2`` order."""

    def __init__(self):
        """Initialize two scalar trainable parameters."""
        super().__init__()
        self.p1 = torch.nn.Parameter(torch.tensor([0.0]))
        self.p2 = torch.nn.Parameter(torch.tensor([0.0]))


class _TwoScalarsBA(torch.nn.Module):
    """Model declaring parameters in ``p2, p1`` order."""

    def __init__(self):
        """Initialize two scalar trainable parameters with reversed order."""
        super().__init__()
        self.p2 = torch.nn.Parameter(torch.tensor([0.0]))
        self.p1 = torch.nn.Parameter(torch.tensor([0.0]))


class _TwoDifferentShapesAB(torch.nn.Module):
    """Model declaring ``p1`` then ``p2`` with different tensor shapes."""

    def __init__(self):
        """Initialize trainable parameters with non-matching shapes."""
        super().__init__()
        self.p1 = torch.nn.Parameter(torch.zeros(2))
        self.p2 = torch.nn.Parameter(torch.zeros(3))


class _TwoDifferentShapesBA(torch.nn.Module):
    """Model declaring ``p2`` then ``p1`` with different tensor shapes."""

    def __init__(self):
        """Initialize trainable parameters with reversed declaration order."""
        super().__init__()
        self.p2 = torch.nn.Parameter(torch.zeros(3))
        self.p1 = torch.nn.Parameter(torch.zeros(2))


def _make_optimizer_with_state(model: torch.nn.Module) -> Optimizer:
    """
    Build an SGD optimizer and populate momentum buffers with distinct values.

    Returns:
        Optimizer: Callback wrapping a stateful SGD optimizer.
    """
    optimizer_cb = Optimizer(
        model,
        "torch.optim.SGD",
        lr=0.1,
        momentum=0.9,
    )
    grads = {"p1": torch.tensor([1.0]), "p2": torch.tensor([2.0])}
    for name, param in model.named_parameters():
        param.grad = grads[name].clone()
    optimizer_cb.optimizer.step()
    return optimizer_cb


def _momentum_by_name(model: torch.nn.Module, optimizer_cb: Optimizer) -> dict:
    """
    Collect momentum buffers keyed by parameter name.

    Returns:
        dict: Mapping ``param_name -> momentum_buffer``.
    """
    momentum = {}
    for name, param in model.named_parameters():
        momentum[name] = optimizer_cb.optimizer.state[param][
            "momentum_buffer"
        ].clone()
    return momentum


def test_optimizer_state_dict_includes_param_names():
    """
    Optimizer state dicts should include ``param_names`` metadata.
    """
    model = _TwoScalarsAB()
    optimizer_cb = _make_optimizer_with_state(model)

    state_dict = optimizer_cb.optimizer.state_dict()
    assert "param_names" in state_dict["param_groups"][0]
    assert state_dict["param_groups"][0]["param_names"] == ["p1", "p2"]


def test_load_state_dict_matches_optimizer_state_by_param_name():
    """
    State loading should match buffers by parameter name when names are present.
    """
    source_model = _TwoScalarsAB()
    source_optimizer = _make_optimizer_with_state(source_model)
    source_momentum = _momentum_by_name(source_model, source_optimizer)
    assert not torch.equal(source_momentum["p1"], source_momentum["p2"])

    target_model = _TwoScalarsBA()
    target_optimizer = Optimizer(
        target_model,
        "torch.optim.SGD",
        lr=7.0,
        momentum=0.0,
    )
    target_optimizer.load_state_dict(source_optimizer.optimizer.state_dict())

    loaded_momentum = _momentum_by_name(target_model, target_optimizer)
    assert torch.equal(loaded_momentum["p1"], source_momentum["p1"])
    assert torch.equal(loaded_momentum["p2"], source_momentum["p2"])
    assert (
        target_optimizer.optimizer.param_groups[0]["lr"]
        == source_optimizer.optimizer.param_groups[0]["lr"]
    )
    assert (
        target_optimizer.optimizer.param_groups[0]["momentum"]
        == source_optimizer.optimizer.param_groups[0]["momentum"]
    )


def test_legacy_order_mapping_fails_with_swapped_different_shapes():
    """
    Missing ``param_names`` falls back to order-based loading and breaks
    state/parameter shape alignment for swapped parameters.
    """
    source_model = _TwoDifferentShapesAB()
    source_optimizer = Optimizer(
        source_model,
        "torch.optim.SGD",
        lr=0.1,
        momentum=0.9,
    )
    for _, param in source_model.named_parameters():
        param.grad = torch.ones_like(param)
    source_optimizer.optimizer.step()

    legacy_state_dict = deepcopy(source_optimizer.optimizer.state_dict())
    for group in legacy_state_dict["param_groups"]:
        group.pop("param_names", None)

    target_model = _TwoDifferentShapesBA()
    target_optimizer = Optimizer(
        target_model,
        "torch.optim.SGD",
        lr=0.3,
        momentum=0.1,
    )
    target_optimizer.load_state_dict(legacy_state_dict)

    assert (
        target_optimizer.optimizer.state[target_model.p1][
            "momentum_buffer"
        ].shape
        == target_model.p2.shape
    )
    assert (
        target_optimizer.optimizer.state[target_model.p2][
            "momentum_buffer"
        ].shape
        == target_model.p1.shape
    )

    for _, param in target_model.named_parameters():
        param.grad = torch.ones_like(param)
    with pytest.raises(RuntimeError):
        target_optimizer.optimizer.step()


def test_named_mapping_succeeds_with_swapped_different_shapes():
    """
    Name-based loading should remap state correctly for swapped shape-mismatched
    parameters.
    """
    source_model = _TwoDifferentShapesAB()
    source_optimizer = Optimizer(
        source_model,
        "torch.optim.SGD",
        lr=0.1,
        momentum=0.9,
    )
    for name, param in source_model.named_parameters():
        if name == "p1":
            param.grad = torch.ones_like(param)
        else:
            param.grad = torch.full_like(param, 2.0)
    source_optimizer.optimizer.step()
    source_momentum = _momentum_by_name(source_model, source_optimizer)
    assert not torch.equal(source_momentum["p1"], source_momentum["p2"])

    target_model = _TwoDifferentShapesBA()
    target_optimizer = Optimizer(
        target_model,
        "torch.optim.SGD",
        lr=0.3,
        momentum=0.1,
    )
    target_optimizer.load_state_dict(source_optimizer.optimizer.state_dict())

    loaded_momentum = _momentum_by_name(target_model, target_optimizer)
    assert not torch.equal(loaded_momentum["p1"], loaded_momentum["p2"])
    assert torch.equal(loaded_momentum["p1"], source_momentum["p1"])
    assert torch.equal(loaded_momentum["p2"], source_momentum["p2"])
    assert (
        target_optimizer.optimizer.param_groups[0]["lr"]
        == source_optimizer.optimizer.param_groups[0]["lr"]
    )
    assert (
        target_optimizer.optimizer.param_groups[0]["momentum"]
        == source_optimizer.optimizer.param_groups[0]["momentum"]
    )

    for _, param in target_model.named_parameters():
        param.grad = torch.ones_like(param)
    target_optimizer.optimizer.step()
