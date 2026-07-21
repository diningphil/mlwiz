"""Tests for optional synchronized BatchNorm in DDP experiments."""

import pytest
import torch

import mlwiz.experiment.experiment as experiment_module
from mlwiz.experiment import Experiment
from mlwiz.static import SYNC_BATCHNORM


class _BatchNormModel(torch.nn.Module):
    """Small model containing a BatchNorm layer eligible for conversion."""

    def __init__(self):
        super().__init__()
        self.norm = torch.nn.BatchNorm1d(4)

    def forward(self, inputs):
        return self.norm(inputs)


class _DDPStub:
    """Record the module and device arguments passed to DDP."""

    def __init__(self, module, **kwargs):
        self.module = module
        self.kwargs = kwargs


def _make_experiment(tmp_path, sync_batchnorm=None):
    """Build an experiment with an optional explicit SyncBatchNorm setting."""
    config = {"device": "cpu"}
    if sync_batchnorm is not None:
        config[SYNC_BATCHNORM] = sync_batchnorm
    return Experiment(config, str(tmp_path), exp_seed=0)


def test_sync_batchnorm_defaults_to_disabled(monkeypatch, tmp_path):
    """Ordinary BatchNorm should remain unchanged when the option is absent."""
    monkeypatch.setattr(experiment_module, "DDP", _DDPStub)
    experiment = _make_experiment(tmp_path)

    wrapped = experiment._wrap_ddp_model(_BatchNormModel(), ddp_rank=1)

    assert type(wrapped.module.norm) is torch.nn.BatchNorm1d
    assert wrapped.kwargs == {"device_ids": [1], "output_device": 1}


def test_sync_batchnorm_converts_before_ddp_wrapping(monkeypatch, tmp_path):
    """Enabled BatchNorm layers should reach DDP as SyncBatchNorm layers."""
    monkeypatch.setattr(experiment_module, "DDP", _DDPStub)
    experiment = _make_experiment(tmp_path, sync_batchnorm=True)

    wrapped = experiment._wrap_ddp_model(_BatchNormModel(), ddp_rank=0)

    assert isinstance(wrapped.module.norm, torch.nn.SyncBatchNorm)
    assert wrapped.kwargs == {"device_ids": [0], "output_device": 0}


def test_sync_batchnorm_requires_active_ddp(tmp_path):
    """An enabled option must not be silently ignored outside DDP."""
    experiment = _make_experiment(tmp_path, sync_batchnorm=True)

    with pytest.raises(ValueError, match="requires active DDP"):
        experiment._wrap_ddp_model(_BatchNormModel(), ddp_rank=None)


def test_sync_batchnorm_rejects_model_parallel_ddp(tmp_path):
    """SyncBatchNorm supports MLWiz's one-GPU-per-process DDP path only."""

    class _FakeParameter:
        def __init__(self, device):
            self.is_cuda = True
            self.device = torch.device(device)

    class _ModelParallelStub:
        def parameters(self):
            return iter(
                [_FakeParameter("cuda:0"), _FakeParameter("cuda:1")]
            )

    experiment = _make_experiment(tmp_path, sync_batchnorm=True)

    with pytest.raises(ValueError, match="multiple CUDA devices"):
        experiment._wrap_ddp_model(_ModelParallelStub(), ddp_rank=0)


def test_sync_batchnorm_requires_a_boolean(tmp_path):
    """String-like truthy values should not accidentally enable conversion."""
    experiment = _make_experiment(tmp_path, sync_batchnorm="false")

    with pytest.raises(ValueError, match="must be a boolean"):
        experiment._wrap_ddp_model(_BatchNormModel(), ddp_rank=None)
