"""Unit tests for Experiment batch-size handling in DDP and non-DDP runs."""

import pytest

from mlwiz.experiment import Experiment
from mlwiz.static import EXPERIMENT_ERRFILE


class _DatasetGetterStub:
    """Dataset-getter stub that records loader kwargs for assertions."""

    def __init__(self):
        """Initialize call recorder."""
        self.calls = []

    def _record(self, name: str, **kwargs):
        """Store call metadata and return a placeholder loader object."""
        self.calls.append((name, kwargs))
        return object()

    def get_inner_train(self, **kwargs):
        """Record inner-train loader construction."""
        return self._record("get_inner_train", **kwargs)

    def get_inner_val(self, **kwargs):
        """Record inner-val loader construction."""
        return self._record("get_inner_val", **kwargs)

    def get_outer_train(self, **kwargs):
        """Record outer-train loader construction."""
        return self._record("get_outer_train", **kwargs)

    def get_outer_val(self, **kwargs):
        """Record outer-val loader construction."""
        return self._record("get_outer_val", **kwargs)

    def get_outer_test(self, **kwargs):
        """Record outer-test loader construction."""
        return self._record("get_outer_test", **kwargs)

    def get_dim_input_features(self):
        """Return a fixed input-dimension placeholder."""
        return 4

    def get_dim_target(self):
        """Return a fixed output-dimension placeholder."""
        return 2


class _EngineStub:
    """Training engine stub returning fixed metric payloads."""

    def train(self, **_kwargs):
        """Return shape-compatible outputs for both valid and test paths."""
        return {}, {}, {}, {}, {}, {}


def _make_experiment(monkeypatch, tmp_path, batch_size):
    """Create an Experiment instance patched with lightweight stubs."""
    config = {
        "model": "unused",
        "device": "cpu",
        "epochs": 1,
        "shuffle": True,
        "batch_size": batch_size,
    }
    experiment = Experiment(config, str(tmp_path), exp_seed=0)

    monkeypatch.setattr(
        Experiment,
        "create_model",
        lambda self, dim_input_features, dim_target, config: object(),
    )
    monkeypatch.setattr(
        Experiment,
        "_wrap_ddp_model",
        lambda self, model, ddp_rank: model,
    )
    monkeypatch.setattr(
        Experiment,
        "create_engine",
        lambda self, config, model: _EngineStub(),
    )

    return experiment


def test_run_valid_keeps_batch_size_in_single_process(monkeypatch, tmp_path):
    """Single-process runs should pass the configured batch size unchanged."""
    experiment = _make_experiment(monkeypatch, tmp_path, batch_size=512)
    dataset_getter = _DatasetGetterStub()

    experiment._run_valid_impl(
        dataset_getter=dataset_getter,
        training_timeout_seconds=1,
        logger=None,
        ddp_rank=None,
        ddp_world_size=1,
    )

    assert dataset_getter.calls[0][0] == "get_inner_train"
    assert dataset_getter.calls[1][0] == "get_inner_val"
    assert dataset_getter.calls[0][1]["batch_size"] == 512
    assert dataset_getter.calls[1][1]["batch_size"] == 512


def test_run_valid_divides_batch_size_across_ddp_ranks(monkeypatch, tmp_path):
    """DDP validation path should divide global batch size by world size."""
    experiment = _make_experiment(monkeypatch, tmp_path, batch_size=512)
    dataset_getter = _DatasetGetterStub()

    experiment._run_valid_impl(
        dataset_getter=dataset_getter,
        training_timeout_seconds=1,
        logger=None,
        ddp_rank=0,
        ddp_world_size=2,
    )

    assert dataset_getter.calls[0][1]["batch_size"] == 256
    assert dataset_getter.calls[1][1]["batch_size"] == 256


def test_run_test_divides_batch_size_across_ddp_ranks(monkeypatch, tmp_path):
    """DDP final-run path should use per-rank batch size for all loaders."""
    experiment = _make_experiment(monkeypatch, tmp_path, batch_size=512)
    dataset_getter = _DatasetGetterStub()

    experiment._run_test_impl(
        dataset_getter=dataset_getter,
        training_timeout_seconds=1,
        logger=None,
        ddp_rank=1,
        ddp_world_size=4,
    )

    assert dataset_getter.calls[0][0] == "get_outer_train"
    assert dataset_getter.calls[1][0] == "get_outer_val"
    assert dataset_getter.calls[2][0] == "get_outer_test"
    assert dataset_getter.calls[0][1]["batch_size"] == 128
    assert dataset_getter.calls[1][1]["batch_size"] == 128
    assert dataset_getter.calls[2][1]["batch_size"] == 128


def test_ddp_raises_when_global_batch_is_not_divisible(monkeypatch, tmp_path):
    """DDP runs should fail fast when global batch cannot be split evenly."""
    experiment = _make_experiment(monkeypatch, tmp_path, batch_size=513)
    dataset_getter = _DatasetGetterStub()

    with pytest.raises(ValueError, match="must be divisible by world size"):
        experiment._run_valid_impl(
            dataset_getter=dataset_getter,
            training_timeout_seconds=1,
            logger=None,
            ddp_rank=0,
            ddp_world_size=2,
        )


def test_run_valid_writes_experiment_err_on_exception(monkeypatch, tmp_path):
    """Validation failures should be recorded in experiment.err and re-raised."""
    experiment = _make_experiment(monkeypatch, tmp_path, batch_size=32)

    def _boom(self, *args, **kwargs):
        raise RuntimeError("validation exploded")

    monkeypatch.setattr(Experiment, "_should_use_ddp", lambda self: False)
    monkeypatch.setattr(Experiment, "_run_valid_impl", _boom)

    with pytest.raises(RuntimeError, match="validation exploded"):
        experiment.run_valid(
            dataset_getter=_DatasetGetterStub(),
            training_timeout_seconds=1,
            logger=None,
        )

    err_path = tmp_path / EXPERIMENT_ERRFILE
    assert err_path.exists()
    content = err_path.read_text()
    assert "validation exploded" in content
    assert "RuntimeError" in content


def test_run_test_writes_experiment_err_on_exception(monkeypatch, tmp_path):
    """Final-run failures should be recorded in experiment.err and re-raised."""
    experiment = _make_experiment(monkeypatch, tmp_path, batch_size=32)

    def _boom(self, *args, **kwargs):
        raise RuntimeError("test exploded")

    monkeypatch.setattr(Experiment, "_should_use_ddp", lambda self: False)
    monkeypatch.setattr(Experiment, "_run_test_impl", _boom)

    with pytest.raises(RuntimeError, match="test exploded"):
        experiment.run_test(
            dataset_getter=_DatasetGetterStub(),
            training_timeout_seconds=1,
            logger=None,
        )

    err_path = tmp_path / EXPERIMENT_ERRFILE
    assert err_path.exists()
    content = err_path.read_text()
    assert "test exploded" in content
    assert "RuntimeError" in content
