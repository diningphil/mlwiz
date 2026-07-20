"""Tests for dashboard metric persistence in :mod:`mlwiz` training."""

from types import SimpleNamespace

import pytest
import torch

from mlwiz.static import LOSSES, SCORES
from mlwiz.training.callback.plotter import Plotter, WidthPlotter
from mlwiz.training.event.state import State


def _state(epoch: int = 0):
    """Build a minimal epoch state consumed by :class:`Plotter`."""
    return SimpleNamespace(
        epoch=epoch,
        epoch_results={
            LOSSES: {
                "training_main_loss": torch.tensor(0.8 - epoch * 0.1),
                "validation_main_loss": torch.tensor(0.9 - epoch * 0.1),
            },
            SCORES: {
                "training_main_score": torch.tensor(0.6 + epoch * 0.1),
                "validation_main_score": torch.tensor(0.5 + epoch * 0.1),
            },
        },
    )


def test_plotter_stores_dashboard_metrics_by_default(tmp_path):
    """Plotter should produce dashboard data without an opt-in flag."""
    plotter = Plotter(str(tmp_path))
    state = _state()

    plotter.on_epoch_end(state)

    metrics_path = tmp_path / "metrics_data.torch"
    assert metrics_path.is_file()
    assert not (tmp_path / "tensorboard").exists()
    metrics = torch.load(metrics_path, weights_only=True)
    assert metrics["losses"]["training_main_loss"] == pytest.approx([0.8])
    assert metrics["scores"]["validation_main_score"] == pytest.approx([0.5])
    assert "step" not in metrics


def test_plotter_stores_and_resumes_sampled_training_steps(tmp_path, monkeypatch):
    """Step histories should retain global steps, timestamps, and resumes."""
    recorded_times = iter((100.25, 200.5))
    monkeypatch.setattr(
        "mlwiz.training.callback.plotter.time.time", lambda: next(recorded_times)
    )
    plotter = Plotter(str(tmp_path), store_every_N_steps=2)
    state = SimpleNamespace(
        batch_loss={"main_loss": torch.tensor(0.8)},
        batch_score={"main_score": torch.tensor(0.6)},
    )

    for value in (0.8, 0.7, 0.6):
        state.batch_loss["main_loss"] = torch.tensor(value)
        plotter.on_training_batch_end(state)
    plotter.on_termination(state)

    resumed = Plotter(str(tmp_path), store_every_N_steps=2)
    state.batch_loss["main_loss"] = torch.tensor(0.5)
    state.batch_score["main_score"] = torch.tensor(0.75)
    resumed.on_training_batch_end(state)

    metrics = torch.load(tmp_path / "metrics_data.torch", weights_only=True)
    assert metrics["step"]["steps"] == [2, 4]
    assert metrics["step"]["timestamps"] == pytest.approx([100.25, 200.5])
    assert metrics["step"]["last_step"] == 4
    assert metrics["step"]["losses"]["training_main_loss"] == pytest.approx(
        [0.7, 0.5]
    )
    assert metrics["step"]["scores"]["training_main_score"] == pytest.approx(
        [0.6, 0.75]
    )


def test_plotter_replaces_steps_after_resumed_epoch_checkpoint(tmp_path):
    """A resume should discard samples newer than the restored checkpoint."""
    plotter = Plotter(str(tmp_path), store_every_N_steps=2)
    state = _state(0)
    state.batch_loss = {"main_loss": torch.tensor(0.0)}
    state.batch_score = {"main_score": torch.tensor(0.0)}

    for loss in (0.9, 0.8, 0.7, 0.6, 0.5):
        state.batch_loss["main_loss"] = torch.tensor(loss)
        plotter.on_training_batch_end(state)
    plotter.on_epoch_end(state)

    # These samples belong to epoch 1, for which no checkpoint was written.
    state.epoch = 1
    for loss in (0.4, 0.3, 0.2):
        state.batch_loss["main_loss"] = torch.tensor(loss)
        plotter.on_training_batch_end(state)
    plotter.on_termination(state)

    resumed = Plotter(str(tmp_path), store_every_N_steps=2)
    resumed.on_fit_start(SimpleNamespace(initial_epoch=1))

    metrics_path = tmp_path / "metrics_data.torch"
    resumed_metrics = torch.load(metrics_path, weights_only=True)
    assert resumed_metrics["step"]["steps"] == [2, 4]
    assert len(resumed_metrics["step"]["timestamps"]) == 2
    assert resumed_metrics["step"]["last_step"] == 5
    assert resumed_metrics["step"]["epoch_last_steps"] == {0: 5}
    assert resumed_metrics["step"]["losses"][
        "training_main_loss"
    ] == pytest.approx([0.8, 0.6])

    # Replaying epoch 1 overwrites the discarded step positions instead of
    # continuing after the partial run's step 8.
    for loss in (0.35, 0.25, 0.15):
        state.batch_loss["main_loss"] = torch.tensor(loss)
        resumed.on_training_batch_end(state)

    metrics = torch.load(metrics_path, weights_only=True)
    assert metrics["step"]["steps"] == [2, 4, 6, 8]
    assert len(metrics["step"]["timestamps"]) == 4
    assert metrics["step"]["last_step"] == 8
    assert metrics["step"]["losses"]["training_main_loss"] == pytest.approx(
        [0.8, 0.6, 0.35, 0.15]
    )


def test_plotter_requests_batch_scores_only_on_sampled_steps(tmp_path):
    """Accumulated scorers should run only when the Plotter will store them."""
    plotter = Plotter(str(tmp_path), store_every_N_steps=2)
    state = State(model=None, optimizer=None, device="cpu")
    state.update(batch_loss={}, batch_score={})
    plotter.on_epoch_start(state)

    plotter.on_training_batch_start(state)
    assert state.log_step_metrics is False
    plotter.on_training_batch_end(state)

    plotter.on_training_batch_start(state)
    assert state.log_step_metrics is True
    plotter.on_eval_batch_start(state)
    assert state.log_step_metrics is False


def test_plotter_periodically_flushes_and_resumes_histories(tmp_path):
    """Periodic writes and resumed runs should preserve earlier epochs."""
    plotter = Plotter(str(tmp_path), store_every_N_epochs=2)
    plotter.on_epoch_end(_state(0))
    assert not (tmp_path / "metrics_data.torch").exists()

    plotter.on_epoch_end(_state(1))
    metrics_path = tmp_path / "metrics_data.torch"
    assert metrics_path.is_file()

    resumed = Plotter(str(tmp_path))
    resumed.on_epoch_end(_state(2))
    resumed.on_termination(_state(2))
    metrics = torch.load(metrics_path, weights_only=True)
    assert metrics["scores"]["training_main_score"] == pytest.approx([0.6, 0.7, 0.8])


def test_width_plotter_records_one_curve_per_learnable_layer(tmp_path):
    """WidthPlotter should persist an epochs-by-layers history matrix."""
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 8),
        torch.nn.ReLU(),
        torch.nn.Linear(8, 3),
    )
    plotter = WidthPlotter(str(tmp_path))
    first = _state(0)
    first.model = model
    second = _state(1)
    second.model = model

    plotter.on_epoch_end(first)
    plotter.on_epoch_end(second)

    metrics = torch.load(tmp_path / "metrics_data.torch", weights_only=True)
    assert metrics["model_widths"] == [[8, 3], [8, 3]]


@pytest.mark.parametrize("value", [0, -1, 1.5, "2"])
def test_plotter_rejects_invalid_flush_interval(tmp_path, value):
    """Flush intervals must remain positive integers."""
    with pytest.raises(ValueError, match="positive integer"):
        Plotter(str(tmp_path), store_every_N_epochs=value)


@pytest.mark.parametrize("value", [0, -1, 1.5, "2", True])
def test_plotter_rejects_invalid_step_interval(tmp_path, value):
    """Step intervals must be positive integers when enabled."""
    with pytest.raises(ValueError, match="store_every_N_steps"):
        Plotter(str(tmp_path), store_every_N_steps=value)
