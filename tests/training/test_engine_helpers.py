"""
Additional unit tests for :mod:`mlwiz.training.engine`.

This module covers small helpers and additional branches in
inference/termination handling.
"""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from mlwiz.exceptions import TerminationRequested
from mlwiz.model.interface import ModelInterface
from mlwiz.static import (
    BEST_CHECKPOINT_FILENAME,
    BEST_EPOCH,
    BEST_OPTIMIZER_CHECKPOINT_FILENAME,
    EPOCH,
    LAST_CHECKPOINT_FILENAME,
    LAST_OPTIMIZER_CHECKPOINT_FILENAME,
    LAST_RUN_ELAPSED_TIME,
    MAIN_LOSS,
    MAIN_SCORE,
    MODEL_STATE,
    OPTIMIZER_STATE,
    SCALER_STATE,
    SCHEDULER_STATE,
    STOP_TRAINING,
    TRAINING,
)
from mlwiz.training.callback.engine_callback import EngineCallback
from mlwiz.training.callback.metric import ToyMetric
from mlwiz.training.callback.optimizer import Optimizer
from mlwiz.training.engine import TrainingEngine, fmt
from mlwiz.training.event.handler import EventHandler
from mlwiz.training.util import atomic_save_split_checkpoint


class _EmbeddingModel(ModelInterface):
    """Minimal model that returns the input tensor as its embedding."""

    def __init__(self):
        """Initialize a minimal linear model returning embeddings."""
        super().__init__(dim_input_features=1, dim_target=1, config={})
        self.linear = torch.nn.Linear(1, 1, bias=False)

    def forward(self, x):
        """
        Forward pass returning both prediction and embedding.

        Args:
            x: Input tensor.

        Returns:
            Tuple ``(out, embeddings)`` where embeddings are the inputs.
        """
        out = self.linear(x)
        return out, x


class _TerminationRecorder(EventHandler):
    """Callback used to count fit-end and termination hook invocations."""

    def __init__(self):
        """Initialize invocation counters."""
        self.fit_end_calls = 0
        self.termination_calls = 0

    def on_fit_end(self, state):
        """Count ``on_fit_end`` invocations."""
        self.fit_end_calls += 1

    def on_termination(self, state):
        """Count ``on_termination`` invocations."""
        self.termination_calls += 1


def _make_engine(tmp_path=None, plotter=None) -> TrainingEngine:
    """
    Construct a TrainingEngine instance for unit tests.

    Args:
        tmp_path: Optional path used as experiment folder for checkpointing/logging.

    Returns:
        Configured :class:`~mlwiz.training.engine.TrainingEngine` instance.
    """
    model = _EmbeddingModel()
    loss = ToyMetric(use_as_loss=True)
    scorer = ToyMetric(use_as_loss=False)
    optimizer = Optimizer(model, "torch.optim.SGD", lr=0.1)
    return TrainingEngine(
        EngineCallback,
        model,
        loss,
        optimizer,
        scorer,
        plotter=plotter,
        device="cpu",
        exp_path=str(tmp_path) if tmp_path is not None else None,
    )


def _noop_progress(*_args, **_kwargs):
    """No-op progress callback used by TrainingEngine methods in unit tests."""
    return None


def test_fmt_formats_zero_small_and_regular_numbers():
    """fmt() should format 0, small values (scientific), and regular values."""
    assert fmt(0.0, decimals=2) == "0.00"
    assert fmt(0.0005, decimals=2, sci_decimals=2) == "5.00e-04"
    assert fmt(1.2345, decimals=2) == "1.23"


def test_check_termination_behaviors(tmp_path):
    """_check_termination should handle None/False/True and exception paths."""
    engine = _make_engine(tmp_path)

    engine._should_terminate = None
    engine._check_termination()

    engine._should_terminate = lambda: False
    engine._check_termination()

    engine._should_terminate = lambda: True
    with pytest.raises(TerminationRequested, match="Termination requested"):
        engine._check_termination()

    def _raise_other():
        """Raise an arbitrary exception (non-termination) to test swallowing behavior."""
        raise RuntimeError("boom")

    engine._should_terminate = _raise_other
    engine._check_termination()

    def _raise_termination():
        """Raise :class:`~mlwiz.exceptions.TerminationRequested` to test propagation."""
        raise TerminationRequested("stop")

    engine._should_terminate = _raise_termination
    with pytest.raises(TerminationRequested, match="stop"):
        engine._check_termination()


def test_infer_sets_main_keys(
    tmp_path,
):
    """infer() should set MAIN_* keys for downstream consumers."""
    x = torch.arange(5, dtype=torch.float32).unsqueeze(1)
    y = torch.arange(5, dtype=torch.float32).unsqueeze(1)
    loader = DataLoader(TensorDataset(x, y), batch_size=2, shuffle=False)

    engine = _make_engine(tmp_path)
    loss, score = engine.infer(loader, TRAINING, _noop_progress)

    assert MAIN_LOSS in loss
    assert MAIN_SCORE in score


def test_on_termination_called_on_normal_end_and_interrupt(tmp_path):
    """``on_termination`` should run on both normal completion and interrupt."""
    x = torch.arange(4, dtype=torch.float32).unsqueeze(1)
    y = torch.arange(4, dtype=torch.float32).unsqueeze(1)
    loader = DataLoader(TensorDataset(x, y), batch_size=2, shuffle=False)

    normal_recorder = _TerminationRecorder()
    normal_engine = _make_engine(
        tmp_path / "normal",
        plotter=normal_recorder,
    )
    normal_engine.train(
        train_loader=loader,
        validation_loader=None,
        test_loader=None,
        max_epochs=0,
        progress_callback=_noop_progress,
        should_terminate=lambda: False,
    )
    assert normal_recorder.fit_end_calls == 1
    assert normal_recorder.termination_calls == 1

    interrupt_recorder = _TerminationRecorder()
    interrupt_engine = _make_engine(
        tmp_path / "interrupt",
        plotter=interrupt_recorder,
    )

    def _raise_keyboard_interrupt():
        """Simulate a user interrupt request callback."""
        raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        interrupt_engine.train(
            train_loader=loader,
            validation_loader=None,
            test_loader=None,
            max_epochs=0,
            progress_callback=_noop_progress,
            should_terminate=_raise_keyboard_interrupt,
        )

    assert interrupt_recorder.fit_end_calls == 0
    assert interrupt_recorder.termination_calls == 1


def test_split_checkpoint_keeps_optimizer_state_out_of_model_file(tmp_path):
    """Model checkpoint filenames stay stable while training state is separate."""
    checkpoint = {
        EPOCH: 3,
        MODEL_STATE: {"linear.weight": torch.ones(1, 1)},
        OPTIMIZER_STATE: {"state": {0: {"momentum_buffer": torch.ones(1)}}},
        SCHEDULER_STATE: {"last_epoch": 3},
        SCALER_STATE: {"scale": 1024.0},
        STOP_TRAINING: False,
    }
    model_path = tmp_path / LAST_CHECKPOINT_FILENAME
    optimizer_path = tmp_path / LAST_OPTIMIZER_CHECKPOINT_FILENAME

    atomic_save_split_checkpoint(checkpoint, model_path, optimizer_path)

    model_payload = torch.load(model_path, weights_only=True)
    optimizer_payload = torch.load(optimizer_path, weights_only=True)
    assert model_payload[MODEL_STATE]["linear.weight"].item() == 1.0
    assert OPTIMIZER_STATE not in model_payload
    assert SCHEDULER_STATE not in model_payload
    assert SCALER_STATE not in model_payload
    assert optimizer_payload[EPOCH] == 3
    assert optimizer_payload[OPTIMIZER_STATE]["state"][0][
        "momentum_buffer"
    ].item() == 1.0


def test_engine_callback_writes_parallel_last_checkpoint_files(tmp_path):
    """Epoch-end checkpointing uses the split artifacts in normal training."""
    engine = _make_engine(tmp_path)
    state = engine.state
    state.update(
        epoch=0,
        epoch_results={"losses": {}, "scores": {}},
        optimizer_state=engine.optimizer.optimizer.state_dict(),
        scheduler_state=None,
        scaler_state=None,
        current_elapsed_time=1.25,
    )

    EngineCallback(store_last_checkpoint=True).on_epoch_end(state)

    assert (tmp_path / LAST_CHECKPOINT_FILENAME).is_file()
    assert (tmp_path / LAST_OPTIMIZER_CHECKPOINT_FILENAME).is_file()
    model_payload = torch.load(
        tmp_path / LAST_CHECKPOINT_FILENAME, weights_only=True
    )
    assert MODEL_STATE in model_payload
    assert OPTIMIZER_STATE not in model_payload


def test_engine_restores_split_and_legacy_optimizer_checkpoints(tmp_path):
    """New split artifacts and old bundled checkpoints resume identically."""
    for legacy in (False, True):
        run_path = tmp_path / ("legacy" if legacy else "split")
        run_path.mkdir()
        engine = _make_engine(run_path)
        model_state = {
            key: torch.full_like(value, 2.0)
            for key, value in engine.model.state_dict().items()
        }
        optimizer_state = {
            "state": {},
            "param_groups": engine.optimizer.optimizer.state_dict()[
                "param_groups"
            ],
        }
        checkpoint = {
            EPOCH: 4,
            MODEL_STATE: model_state,
            OPTIMIZER_STATE: optimizer_state,
            SCHEDULER_STATE: None,
            SCALER_STATE: {"scale": 8.0},
            STOP_TRAINING: False,
            LAST_RUN_ELAPSED_TIME: 12.5,
        }
        last_path = run_path / LAST_CHECKPOINT_FILENAME
        if legacy:
            torch.save(checkpoint, last_path)
        else:
            atomic_save_split_checkpoint(
                checkpoint,
                last_path,
                run_path / LAST_OPTIMIZER_CHECKPOINT_FILENAME,
            )

        best_checkpoint = {
            BEST_EPOCH: 2,
            MODEL_STATE: model_state,
            OPTIMIZER_STATE: optimizer_state,
            SCHEDULER_STATE: None,
            SCALER_STATE: None,
        }
        best_path = run_path / BEST_CHECKPOINT_FILENAME
        if legacy:
            torch.save(best_checkpoint, best_path)
        else:
            atomic_save_split_checkpoint(
                best_checkpoint,
                best_path,
                run_path / BEST_OPTIMIZER_CHECKPOINT_FILENAME,
            )

        engine._restore_checkpoint_and_best_results(
            last_path, best_path, zero_epoch=False
        )

        assert engine.state.initial_epoch == 5
        assert engine.state.current_elapsed_time == pytest.approx(12.5)
        assert engine.state.optimizer_state == optimizer_state
        assert engine.state.scaler_state == {"scale": 8.0}
        assert engine.state.best_epoch_results[OPTIMIZER_STATE] == optimizer_state
        assert all(
            torch.equal(value, model_state[key])
            for key, value in engine.model.state_dict().items()
        )
