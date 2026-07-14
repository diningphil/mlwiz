"""Tests for safe representative-input metadata used by the dashboard."""

from __future__ import annotations

import json

import torch

from mlwiz.static import MODEL_GRAPH_INPUT_SPEC_FILENAME
from mlwiz.training import engine as training_engine
from mlwiz.training.engine import TrainingEngine
from mlwiz.training.event.handler import EventHandler
from mlwiz.training.event.state import State
from mlwiz.training.util import record_model_graph_input_spec


def _input_spec_path(tmp_path):
    """Return the input-spec path for a temporary experiment directory."""
    return tmp_path / MODEL_GRAPH_INPUT_SPEC_FILENAME


def test_record_model_graph_input_spec_stores_only_tensor_metadata(tmp_path):
    """A tensor input should create an atomic, data-free shape/dtype spec."""
    batch = torch.arange(6, dtype=torch.float32).reshape(2, 3)

    assert record_model_graph_input_spec(tmp_path, batch)

    specification = json.loads(_input_spec_path(tmp_path).read_text())
    assert specification == {
        "version": 1,
        "kind": "tensor",
        "shape": [2, 3],
        "dtype": "float32",
    }
    assert "values" not in specification
    assert "data" not in specification


def test_record_model_graph_input_spec_preserves_first_observation(tmp_path):
    """Later batches must not overwrite the input shape used for export."""
    assert record_model_graph_input_spec(tmp_path, torch.zeros(2, 3))
    assert not record_model_graph_input_spec(tmp_path, torch.zeros(9, 4))

    specification = json.loads(_input_spec_path(tmp_path).read_text())
    assert specification["shape"] == [2, 3]


def test_record_model_graph_input_spec_marks_non_tensor_inputs(tmp_path):
    """Unsupported inputs should be identifiable without serializing contents."""
    private_batch = {"sensitive": torch.tensor([1.0])}

    assert record_model_graph_input_spec(tmp_path, private_batch)

    specification = json.loads(_input_spec_path(tmp_path).read_text())
    assert specification == {
        "version": 1,
        "kind": "unsupported",
        "type": "builtins.dict",
    }


def test_record_model_graph_input_spec_uses_model_adapter(tmp_path):
    """Custom models may safely describe non-tensor inputs for the dashboard."""

    class GraphModel:
        @staticmethod
        def model_graph_input_spec(batch_input):
            return {
                "version": 1,
                "kind": "graph",
                "num_nodes": int(batch_input["nodes"].shape[0]),
            }

    batch = {"nodes": torch.ones(5, 3)}

    assert record_model_graph_input_spec(tmp_path, batch, model=GraphModel())
    assert json.loads(_input_spec_path(tmp_path).read_text()) == {
        "version": 1,
        "kind": "graph",
        "num_nodes": 5,
    }


def _bare_engine(tmp_path):
    """Build the minimum engine state needed to exercise ``_loop_helper``."""
    engine = object.__new__(TrainingEngine)
    engine.device = "cpu"
    engine.training = False
    engine.use_mixed_precision = False
    engine.state = State(model=torch.nn.Identity(), optimizer=None, device="cpu")
    engine.state.update(
        exp_path=str(tmp_path),
        batch_input=(torch.ones(4, 2), torch.zeros(4, 2)),
    )
    return engine


def test_loop_helper_records_input_before_forward_on_main_process(tmp_path):
    """All engine modes should record input metadata before ``ON_FORWARD``."""
    engine = _bare_engine(tmp_path)
    spec_exists_during_forward = []

    def dispatch(event, state):
        """Record when the forward event is reached without full callbacks."""
        if event == EventHandler.ON_FORWARD:
            spec_exists_during_forward.append(_input_spec_path(tmp_path).exists())

    engine._dispatch = dispatch
    engine._loop_helper()

    specification = json.loads(_input_spec_path(tmp_path).read_text())
    assert specification["shape"] == [4, 2]
    assert spec_exists_during_forward == [True]


def test_loop_helper_skips_input_recording_off_main_process(tmp_path, monkeypatch):
    """Non-main DDP ranks must never create dashboard metadata files."""
    monkeypatch.setattr(training_engine, "_is_main_process", lambda: False)
    engine = _bare_engine(tmp_path)
    engine._dispatch = lambda _event, _state: None

    engine._loop_helper()

    assert not _input_spec_path(tmp_path).exists()
