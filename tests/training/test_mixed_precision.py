"""Unit tests for mixed-precision plumbing in training callbacks/engine."""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from mlwiz.model.interface import ModelInterface
from mlwiz.static import MAIN_LOSS, MAIN_SCORE, TRAINING
from mlwiz.training.callback.engine_callback import EngineCallback
from mlwiz.training.callback.gradient_clipping import GradientClipper
from mlwiz.training.callback.metric import ToyMetric
from mlwiz.training.callback.optimizer import Optimizer
from mlwiz.training.engine import TrainingEngine
from mlwiz.training.event.state import State


class _EmbeddingModel(ModelInterface):
    """Minimal model that returns both output and embedding tensors."""

    def __init__(self):
        """Initialize the linear layer used in tests."""
        super().__init__(dim_input_features=1, dim_target=1, config={})
        self.linear = torch.nn.Linear(1, 1, bias=False)

    def forward(self, x):
        """
        Return output plus embedding.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: ``(out, emb)``.
        """
        out = self.linear(x)
        return out, x


class _ScaledLoss:
    """Simple object returned by a fake scaler ``scale`` call."""

    def __init__(self, loss: torch.Tensor, scaler):
        """
        Store references used by ``backward``.

        Args:
            loss (torch.Tensor): Loss tensor.
            scaler: Fake scaler collecting call counters.
        """
        self.loss = loss
        self.scaler = scaler

    def backward(self):
        """Record that scaled backward was requested."""
        self.scaler.backward_calls += 1


class _DummyScaler:
    """Minimal fake scaler used to test callback integration points."""

    def __init__(self):
        """Initialize call counters and placeholders."""
        self.scale_calls = 0
        self.step_calls = 0
        self.update_calls = 0
        self.unscale_calls = 0
        self.backward_calls = 0
        self.last_unscaled_optimizer = None

    def scale(self, loss: torch.Tensor):
        """
        Mimic scaler.scale(loss).

        Args:
            loss (torch.Tensor): Loss tensor.

        Returns:
            _ScaledLoss: Backward-capable wrapper.
        """
        self.scale_calls += 1
        return _ScaledLoss(loss, self)

    def step(self, optimizer):
        """
        Mimic scaler.step(optimizer).

        Args:
            optimizer: Wrapped torch optimizer.
        """
        self.step_calls += 1

    def update(self):
        """Mimic scaler.update()."""
        self.update_calls += 1

    def unscale_(self, optimizer):
        """
        Mimic scaler.unscale_(optimizer).

        Args:
            optimizer: Wrapped torch optimizer.
        """
        self.unscale_calls += 1
        self.last_unscaled_optimizer = optimizer

    def state_dict(self):
        """Return a serializable fake scaler state."""
        return {"dummy_scaler": True}


def _noop_progress(*_args, **_kwargs):
    """No-op callback used when calling ``TrainingEngine.infer``."""
    return None


def test_metric_backward_uses_scaler_when_available():
    """Loss callback should use scaler.scale(...).backward() when present."""
    metric = ToyMetric(use_as_loss=True)
    loss = torch.tensor(1.0, requires_grad=True)
    scaler = _DummyScaler()

    state = State(model=None, optimizer=None, device="cpu")
    state.update(batch_loss={metric.name: loss}, grad_scaler=scaler)

    metric.on_backward(state)

    assert scaler.scale_calls == 1
    assert scaler.backward_calls == 1


def test_optimizer_and_gradient_clipper_use_scaler_hooks():
    """Optimizer/clipper callbacks should route through scaler step/unscale."""
    model = torch.nn.Linear(1, 1)
    optimizer_cb = Optimizer(model, "torch.optim.SGD", lr=0.1)
    scaler = _DummyScaler()

    state = State(model=model, optimizer=optimizer_cb, device="cpu")
    state.update(grad_scaler=scaler)

    for param in model.parameters():
        param.grad = torch.ones_like(param)

    clipper = GradientClipper(clip_value=0.1)
    clipper.on_backward(state)
    assert scaler.unscale_calls == 1
    assert scaler.last_unscaled_optimizer is optimizer_cb.optimizer

    optimizer_cb.on_training_batch_end(state)
    assert scaler.step_calls == 1
    assert scaler.update_calls == 1

    optimizer_cb.on_epoch_end(state)
    assert state.scaler_state == {"dummy_scaler": True}
    assert isinstance(state.optimizer_state, dict)


@pytest.mark.parametrize(
    ("dtype_path", "expected_dtype"),
    [
        ("torch.float16", torch.float16),
        ("torch.bfloat16", torch.bfloat16),
    ],
)
def test_engine_mixed_precision_uses_cpu_autocast_without_scaler(
    tmp_path, dtype_path, expected_dtype
):
    """AMP on CPU should use autocast and skip GradScaler for supported dtypes."""
    model = _EmbeddingModel()
    loss = ToyMetric(use_as_loss=True)
    scorer = ToyMetric(use_as_loss=False)
    optimizer = Optimizer(model, "torch.optim.SGD", lr=0.1)

    engine = TrainingEngine(
        engine_callback=EngineCallback,
        model=model,
        loss=loss,
        optimizer=optimizer,
        scorer=scorer,
        device="cpu",
        exp_path=str(tmp_path),
        mixed_precision=True,
        mixed_precision_dtype=dtype_path,
    )

    assert engine.use_mixed_precision is True
    assert engine.amp_dtype == expected_dtype
    assert engine.grad_scaler is None
    assert engine.state.grad_scaler is None

    x = torch.arange(4, dtype=torch.float32).unsqueeze(1)
    y = torch.arange(4, dtype=torch.float32).unsqueeze(1)
    loader = DataLoader(TensorDataset(x, y), batch_size=2, shuffle=False)
    loss_dict, score_dict, _ = engine.infer(loader, TRAINING, _noop_progress)
    assert MAIN_LOSS in loss_dict
    assert MAIN_SCORE in score_dict


def test_engine_mixed_precision_bfloat16_cuda_device_skips_scaler(tmp_path):
    """CUDA+bfloat16 should enable AMP but not instantiate GradScaler."""
    model = _EmbeddingModel()
    loss = ToyMetric(use_as_loss=True)
    scorer = ToyMetric(use_as_loss=False)
    optimizer = Optimizer(model, "torch.optim.SGD", lr=0.1)

    engine = TrainingEngine(
        engine_callback=EngineCallback,
        model=model,
        loss=loss,
        optimizer=optimizer,
        scorer=scorer,
        device="cuda",
        exp_path=str(tmp_path),
        mixed_precision=True,
        mixed_precision_dtype="torch.bfloat16",
    )

    assert engine.use_mixed_precision is True
    assert engine.amp_dtype == torch.bfloat16
    assert engine.grad_scaler is None
    assert engine.state.grad_scaler is None


def test_engine_mixed_precision_dtype_must_be_valid_dotted_path(tmp_path):
    """An invalid dtype dotted path should raise an import error."""
    model = _EmbeddingModel()
    loss = ToyMetric(use_as_loss=True)
    scorer = ToyMetric(use_as_loss=False)
    optimizer = Optimizer(model, "torch.optim.SGD", lr=0.1)

    with pytest.raises(ImportError):
        TrainingEngine(
            engine_callback=EngineCallback,
            model=model,
            loss=loss,
            optimizer=optimizer,
            scorer=scorer,
            device="cpu",
            exp_path=str(tmp_path),
            mixed_precision=True,
            mixed_precision_dtype="float16",
        )
