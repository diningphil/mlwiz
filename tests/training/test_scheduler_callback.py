"""
Unit tests for :mod:`mlwiz.training.callback.scheduler`.

These focus on the callback logic (metric lookup, state persistence/loading),
not on specific PyTorch scheduler behavior.
"""

from __future__ import annotations

import pytest
import torch

from mlwiz.static import LOSSES, SCORES
from mlwiz.training.callback.scheduler import EpochScheduler, MetricScheduler, Scheduler
from mlwiz.training.event.state import State


class DummyTorchScheduler:
    """A tiny scheduler stand-in used to exercise the callback wrappers."""

    def __init__(self, _optimizer, **kwargs):
        self.kwargs = kwargs
        self.loaded_state = None
        self.step_calls = 0
        self.last_metric = None

    def step(self, metric=None):
        self.step_calls += 1
        self.last_metric = metric

    def state_dict(self):
        return {
            "loaded_state": self.loaded_state,
            "step_calls": self.step_calls,
            "last_metric": self.last_metric,
            "kwargs": dict(self.kwargs),
        }

    def load_state_dict(self, state_dict):
        self.loaded_state = state_dict


def _make_optimizer():
    param = torch.nn.Parameter(torch.tensor([1.0]))
    return torch.optim.SGD([param], lr=0.1)


def test_scheduler_loads_state_on_fit_start():
    """
    ``Scheduler.on_fit_start`` should restore state when a checkpoint state exists.
    """
    sched = Scheduler(f"{__name__}.DummyTorchScheduler", _make_optimizer(), foo=1)
    state = State(model=None, optimizer=None, device="cpu")
    state.scheduler_state = {"restored": True}

    sched.on_fit_start(state)
    assert sched.scheduler.loaded_state == {"restored": True}


def test_epoch_scheduler_steps_and_persists_state():
    """
    ``EpochScheduler`` should call ``step()`` and store the scheduler state on epoch end.
    """
    sched = EpochScheduler(f"{__name__}.DummyTorchScheduler", _make_optimizer())
    state = State(model=None, optimizer=None, device="cpu")

    sched.on_training_epoch_end(state)
    assert sched.scheduler.step_calls == 1
    assert sched.scheduler.last_metric is None

    sched.on_epoch_end(state)
    assert isinstance(state.scheduler_state, dict)
    assert state.scheduler_state["step_calls"] == 1


def test_metric_scheduler_validates_monitor_and_steps():
    """
    ``MetricScheduler`` should validate the monitored metric key, step with its value,
    and persist the scheduler state.
    """
    sched = MetricScheduler(
        f"{__name__}.DummyTorchScheduler",
        use_loss=False,
        monitor="val_score",
        optimizer=_make_optimizer(),
    )
    state = State(model=None, optimizer=None, device="cpu")
    state.epoch_results = {LOSSES: {}, SCORES: {"val_score": 0.5}}

    sched.on_epoch_end(state)
    assert sched.scheduler.last_metric == 0.5
    assert state.scheduler_state["last_metric"] == 0.5

    state.epoch_results = {LOSSES: {}, SCORES: {}}
    with pytest.raises(ValueError, match="not found in epoch_results"):
        sched.on_epoch_end(state)

