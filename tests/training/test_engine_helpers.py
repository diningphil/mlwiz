"""
Additional unit tests for :mod:`mlwiz.training.engine`.

The existing engine tests focus on sampler-permutation behavior during
inference. This module covers small helpers and additional branches in
inference/termination handling.
"""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.data import Data

from mlwiz.exceptions import TerminationRequested
from mlwiz.model.interface import ModelInterface
from mlwiz.static import MAIN_LOSS, MAIN_SCORE, TRAINING
from mlwiz.training.callback.engine_callback import EngineCallback
from mlwiz.training.callback.metric import ToyMetric
from mlwiz.training.callback.optimizer import Optimizer
from mlwiz.training.engine import TrainingEngine, fmt, reorder


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


def _make_engine(tmp_path=None) -> TrainingEngine:
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
        device="cpu",
        exp_path=str(tmp_path) if tmp_path is not None else None,
    )


def _noop_progress(*_args, **_kwargs):
    """No-op progress callback used by TrainingEngine methods in unit tests."""
    return None


def test_reorder_raises_on_invalid_inputs():
    """reorder() should validate inputs and raise for invalid lengths."""
    with pytest.raises(ValueError, match="non-zero length"):
        reorder([], [])
    with pytest.raises(ValueError, match="same non-zero length"):
        reorder([1, 2], [0])


def test_reorder_sorts_by_permutation_indices():
    """reorder() should restore original order given a permutation list."""
    assert reorder(["a", "b", "c"], [2, 0, 1]) == ["b", "c", "a"]


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


def test_infer_sequential_sampler_requires_no_permutation_and_sets_main_keys(tmp_path):
    """infer() with SequentialSampler should not require a permutation and should set MAIN_* keys."""
    x = torch.arange(5, dtype=torch.float32).unsqueeze(1)
    y = torch.arange(5, dtype=torch.float32).unsqueeze(1)
    loader = DataLoader(TensorDataset(x, y), batch_size=2, shuffle=False)

    engine = _make_engine(tmp_path)
    engine.state.update(return_embeddings=True)

    loss, score, data_list = engine.infer(loader, TRAINING, _noop_progress)

    assert MAIN_LOSS in loss
    assert MAIN_SCORE in score
    assert [emb.item() for emb, _target in data_list] == [0, 1, 2, 3, 4]


def test_to_list_rejects_unknown_embedding_types(tmp_path):
    """_to_list should raise for non-tensor embedding inputs."""
    engine = _make_engine(tmp_path)
    with pytest.raises(NotImplementedError, match="Embeddings not understood"):
        engine._to_list(
            data_list=[],
            embeddings="not-a-tensor",
            batch=torch.tensor([0]),
            y=None,
        )


def test_to_data_list_splits_graph_and_node_targets(tmp_path):
    """_to_data_list should split embeddings by graph and shape targets appropriately."""
    engine = _make_engine(tmp_path)

    embeddings = torch.arange(10, dtype=torch.float32).reshape(5, 2)
    batch = torch.tensor([0, 0, 0, 1, 1])

    graph_targets = torch.tensor([1, 2], dtype=torch.long)
    graph_list = engine._to_data_list(embeddings, batch, graph_targets)
    assert len(graph_list) == 2
    g0, y0 = graph_list[0]
    g1, y1 = graph_list[1]
    assert isinstance(g0, Data)
    assert g0.x.shape[0] == 3
    assert y0.shape == (1, 1)
    assert g1.x.shape[0] == 2
    assert y1.shape == (1, 1)

    node_targets = torch.arange(5, dtype=torch.long)
    node_list = engine._to_data_list(embeddings, batch, node_targets)
    assert len(node_list) == 2
    g0, y0 = node_list[0]
    g1, y1 = node_list[1]
    assert y0.shape == (3, 1)
    assert y1.shape == (2, 1)

    none_list = engine._to_data_list(embeddings, batch, None)
    assert isinstance(none_list[0], Data)
    assert none_list[1][1] is None
