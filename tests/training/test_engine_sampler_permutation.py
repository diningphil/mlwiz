"""
Tests for :meth:`mlwiz.training.engine.TrainingEngine.infer` sampler-permutation handling.

When returning embeddings, the training engine must be able to restore the
original (unshuffled) sample order. These tests enforce that requirement by
checking that inference:

- Raises when the DataLoader sampler does not expose a usable ``permutation``.
- Reorders returned embeddings when a sampler provides a fixed permutation.
"""

import pytest
import torch
from torch.utils.data import DataLoader, Sampler, TensorDataset

from mlwiz.model.interface import ModelInterface
from mlwiz.static import TRAINING
from mlwiz.training.callback.engine_callback import EngineCallback
from mlwiz.training.callback.metric import ToyMetric
from mlwiz.training.callback.optimizer import Optimizer
from mlwiz.training.engine import TrainingEngine


class _EmbeddingModel(ModelInterface):
    """
    Minimal model that returns inputs as embeddings.

    The engine stores embeddings per-sample and later reorders them based on
    the sampler permutation. Using the input tensor as the embedding makes the
    expected order easy to assert.
    """

    def __init__(self):
        """Initialize a simple linear model used to produce outputs."""
        super().__init__(dim_input_features=1, dim_target=1, config={})
        self.linear = torch.nn.Linear(1, 1, bias=False)

    def forward(self, x):
        """
        Forward pass returning both output and embedding.

        Args:
            x (torch.Tensor): Batch input tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: ``(output, embeddings)`` where
            embeddings are the original inputs.
        """
        out = self.linear(x)
        return out, x


class _FixedPermutationSampler(Sampler[int]):
    """
    Sampler that iterates according to a fixed permutation.

    Exposes the permutation via the ``permutation`` attribute so the training
    engine can restore the original order during inference.
    """

    def __init__(self, data_source, permutation):
        """
        Args:
            data_source: Dataset being sampled.
            permutation (Sequence[int]): Fixed index order for iteration.
        """
        super().__init__()
        self.data_source = data_source
        self.permutation = permutation

    def __iter__(self):
        """Iterate over the stored permutation."""
        return iter(self.permutation)

    def __len__(self):
        """Return the number of samples in the data source."""
        return len(self.data_source)


def _noop_progress(*_args, **_kwargs):
    """Progress callback stub used by :meth:`TrainingEngine.infer`."""
    return None


def _make_engine():
    """
    Construct a :class:`~mlwiz.training.engine.TrainingEngine` for unit tests.

    Uses :class:`~mlwiz.training.callback.metric.ToyMetric` for both loss and
    score, and a simple SGD optimizer.
    """
    model = _EmbeddingModel()
    loss = ToyMetric(use_as_loss=True)  # required by engine API (loss metric)
    scorer = ToyMetric(
        use_as_loss=False
    )  # required by engine API (score metric)
    optimizer = Optimizer(
        model, "torch.optim.SGD", lr=0.1
    )  # minimal optimizer wrapper
    return TrainingEngine(
        EngineCallback,
        model,
        loss,
        optimizer,
        scorer,
        device="cpu",
    )


def test_infer_raises_without_sampler_permutation():
    """
    Inference with embeddings requires a sampler permutation.

    PyTorch's default random sampler (used by ``shuffle=True``) does not expose
    a stable permutation attribute, so the engine must raise.
    """
    x = torch.arange(5, dtype=torch.float32).unsqueeze(
        1
    )  # 5 samples shaped (N, 1)
    y = torch.arange(5, dtype=torch.float32).unsqueeze(
        1
    )  # dummy targets (same shape)
    loader = DataLoader(
        TensorDataset(x, y), batch_size=1, shuffle=True
    )  # => RandomSampler

    engine = _make_engine()
    engine.state.update(
        return_embeddings=True
    )  # ask engine to return per-sample embeddings

    with pytest.raises(ValueError, match="permutation"):
        engine.infer(
            loader, TRAINING, _noop_progress
        )  # should fail: no sampler.permutation


def test_infer_reorders_with_sampler_permutation():
    """
    Inference restores original order when a sampler provides ``permutation``.

    The DataLoader returns samples according to the sampler order, while the
    engine should reorder the embedding list back to the original dataset
    index order.
    """
    x = torch.arange(5, dtype=torch.float32).unsqueeze(
        1
    )  # sample i has value i
    y = torch.arange(5, dtype=torch.float32).unsqueeze(1)  # dummy targets
    dataset = TensorDataset(x, y)
    sampler = _FixedPermutationSampler(
        dataset, [2, 0, 4, 1, 3]
    )  # iteration order
    loader = DataLoader(
        dataset, batch_size=1, sampler=sampler
    )  # sampler order != dataset order

    engine = _make_engine()
    engine.state.update(
        return_embeddings=True
    )  # enables `epoch_data_list` collection

    _loss, _score, data_list = engine.infer(loader, TRAINING, _noop_progress)

    # `data_list` is returned in original dataset-index order (0..N-1), not sampling order.
    assert [emb.item() for emb, _target in data_list] == [0, 1, 2, 3, 4]
