"""
Unit tests for small helpers in :mod:`mlwiz.evaluation.evaluator`.

The evaluator module contains several IO-light helpers and Ray wrappers that
are difficult to cover via the distributed code paths (remote tasks run in
separate processes). These tests exercise those helpers directly in-process.
"""

from __future__ import annotations

import os
import types

import numpy as np
import pytest

import mlwiz.evaluation.evaluator as evaluator
from mlwiz.static import (
    CONFIG,
    EXPERIMENT_LOGFILE,
    LOSS,
    MAIN_LOSS,
    MAIN_SCORE,
    MLWIZ_RAY_NUM_GPUS_PER_TASK,
    SCORE,
)
from mlwiz.util import dill_load


def _unwrap_ray_remote(remote_obj):
    """
    Return the underlying Python function for a Ray-remote wrapper.

    Ray's API has changed attribute names across versions; this helper keeps
    the tests compatible across the supported Ray range.
    """
    for attr in ("python_func", "_function", "func", "__wrapped__"):
        candidate = getattr(remote_obj, attr, None)
        if callable(candidate):
            return candidate
    raise AssertionError(f"Cannot unwrap Ray remote function: {remote_obj!r}")


def test_send_telegram_update_builds_url_and_returns_json(monkeypatch):
    """Ensure telegram helper calls requests.get and returns decoded JSON."""
    captured = {}

    class _Resp:
        """Response stub returning a fixed JSON payload."""

        def json(self):
            """Return a deterministic JSON-like dict."""
            return {"ok": True}

    def _fake_get(url):
        """Capture the URL and return a stub response."""
        captured["url"] = url
        return _Resp()

    monkeypatch.setattr(evaluator.requests, "get", _fake_get)

    res = evaluator.send_telegram_update("TOKEN", "CHAT", "Hello")
    assert res == {"ok": True}
    assert "TOKEN" in captured["url"]
    assert "chat_id=CHAT" in captured["url"]
    assert "text=Hello" in captured["url"]


def test_extract_and_sum_elapsed_seconds_sums_all_matches(tmp_path):
    """Verify elapsed seconds extraction and summation from a log file."""
    log_path = tmp_path / "experiment.log"
    log_path.write_text(
        "\n".join(
            [
                "noise",
                "Total time of the experiment in seconds: 1.5 ",
                "noise",
                "Total time of the experiment in seconds: 2 \n",
            ]
        )
    )

    assert evaluator.extract_and_sum_elapsed_seconds(
        str(log_path)
    ) == pytest.approx(3.5)


def test_mean_std_ci_matches_numpy():
    """Mean/std/CI helper should match the corresponding NumPy computations."""
    values = np.array([0.0, 1.0, 2.0, 3.0], dtype=float)
    mean, std, ci = evaluator._mean_std_ci(values)

    assert mean == pytest.approx(values.mean())
    assert std == pytest.approx(values.std())
    assert ci == pytest.approx(1.96 * (values.std() / np.sqrt(len(values))))


def test_push_progress_update_deepcopies_payload():
    """Progress updates should deepcopy payloads before forwarding."""

    class _Recorder:
        """Actor method stub that records received payloads."""

        def __init__(self):
            """Initialize the payload capture list."""
            self.payloads = []

        def remote(self, payload):
            """Record a payload as if called via Ray actor handle."""
            self.payloads.append(payload)

    actor = types.SimpleNamespace(push=_Recorder())

    payload = {"nested": [1, 2]}
    evaluator._push_progress_update(actor, payload)
    payload["nested"].append(3)

    assert actor.push.payloads == [{"nested": [1, 2]}]


def test_get_ray_num_gpus_per_task_parses_env(monkeypatch):
    """GPU env parsing should handle unset/invalid/negative values."""
    monkeypatch.delenv(MLWIZ_RAY_NUM_GPUS_PER_TASK, raising=False)
    assert evaluator._get_ray_num_gpus_per_task(default=0.3) == pytest.approx(
        0.3
    )

    monkeypatch.setenv(MLWIZ_RAY_NUM_GPUS_PER_TASK, "0.5")
    assert evaluator._get_ray_num_gpus_per_task(default=0.0) == pytest.approx(
        0.5
    )

    monkeypatch.setenv(MLWIZ_RAY_NUM_GPUS_PER_TASK, "-1")
    assert evaluator._get_ray_num_gpus_per_task(default=0.7) == pytest.approx(
        0.7
    )

    monkeypatch.setenv(MLWIZ_RAY_NUM_GPUS_PER_TASK, "not-a-float")
    assert evaluator._get_ray_num_gpus_per_task(default=0.9) == pytest.approx(
        0.9
    )


def test_set_cuda_memory_limit_from_env_calls_torch_cuda(monkeypatch):
    """When CUDA is available, the memory fraction should be set per visible GPU."""
    monkeypatch.setenv(MLWIZ_RAY_NUM_GPUS_PER_TASK, "0.25")

    calls = []

    class _Mem:
        """Stub for ``torch.cuda.memory`` module."""

        @staticmethod
        def set_per_process_memory_fraction(value, device=None):
            """Record the requested memory fraction and device."""
            calls.append((value, str(device)))

    class _Cuda:
        """Minimal torch.cuda stand-in implementing only used methods."""

        memory = _Mem

        @staticmethod
        def is_available():
            """Report CUDA availability."""
            return True

        @staticmethod
        def init():
            """No-op CUDA initialization."""
            return None

        @staticmethod
        def device_count():
            """Return a fixed number of visible devices."""
            return 2

    monkeypatch.setattr(evaluator.torch, "cuda", _Cuda)
    evaluator._set_cuda_memory_limit_from_env()

    assert calls == [
        (pytest.approx(0.25), "cuda:0"),
        (pytest.approx(0.25), "cuda:1"),
    ]


def test_make_termination_checker_errs_on_safe_side(monkeypatch):
    """If the actor cannot be queried, the checker should return True."""

    class _Terminated:
        """Actor method stub returning an opaque reference."""

        def remote(self):
            """Return a dummy reference."""
            return object()

    actor = types.SimpleNamespace(is_terminated=_Terminated())

    monkeypatch.setattr(evaluator.time, "time", lambda: 1.0)
    monkeypatch.setattr(
        evaluator.ray,
        "get",
        lambda _obj: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    should_terminate = evaluator._make_termination_checker(
        actor, min_interval=0.0
    )
    assert should_terminate() is True


def test_run_valid_and_run_test_execute_in_process(tmp_path, monkeypatch):
    """Execute Ray remote wrappers in-process and assert saved artifacts."""
    monkeypatch.setenv(MLWIZ_RAY_NUM_GPUS_PER_TASK, "0.0")
    monkeypatch.setattr(evaluator.ray, "get", lambda obj: obj)

    pushed = []

    class _Push:
        """Actor push method stub that records payloads."""

        def remote(self, payload):
            """Record a pushed payload."""
            pushed.append(payload)

    class _Terminated:
        """Actor termination flag method stub."""

        def remote(self):
            """Return a fixed termination status."""
            return False

    progress_actor = types.SimpleNamespace(
        push=_Push(), is_terminated=_Terminated()
    )

    class _DatasetGetter:
        """Minimal dataset getter stub exposing ``outer_k`` and ``inner_k`` fields."""

        def __init__(self, outer_k, inner_k):
            """
            Initialize dataset getter stand-in.

            Args:
                outer_k: Outer fold id.
                inner_k: Inner fold id or ``None``.
            """
            self.outer_k = outer_k
            self.inner_k = inner_k

    class _Experiment:
        """Experiment stub implementing ``run_valid``/``run_test`` and writing logs."""

        def __init__(self, _config, exp_path, _seed):
            """
            Initialize experiment stub.

            Args:
                _config: Experiment configuration (unused).
                exp_path: Folder where the experiment writes its log file.
                _seed: Experiment seed (unused).
            """
            self.exp_path = exp_path

        def run_valid(
            self,
            dataset_getter,
            training_timeout_seconds,
            logger,
            progress_callback=None,
            should_terminate=None,
        ):
            """Write an experiment log and return deterministic train/val results."""
            os.makedirs(self.exp_path, exist_ok=True)
            with open(
                os.path.join(self.exp_path, EXPERIMENT_LOGFILE), "w"
            ) as f:
                f.write("Total time of the experiment in seconds: 1.5 \n")
                f.write("Total time of the experiment in seconds: 2 \n")
            if progress_callback is not None:
                progress_callback(
                    {"type": "progress", "outer": dataset_getter.outer_k}
                )
            return (
                {LOSS: {MAIN_LOSS: 1.0}, SCORE: {MAIN_SCORE: 2.0}},
                {LOSS: {MAIN_LOSS: 3.0}, SCORE: {MAIN_SCORE: 4.0}},
            )

        def run_test(
            self,
            dataset_getter,
            training_timeout_seconds,
            logger,
            progress_callback=None,
            should_terminate=None,
        ):
            """Write an experiment log and return deterministic train/val/test results."""
            os.makedirs(self.exp_path, exist_ok=True)
            with open(
                os.path.join(self.exp_path, EXPERIMENT_LOGFILE), "w"
            ) as f:
                f.write("Total time of the experiment in seconds: 1 \n")
            if progress_callback is not None:
                progress_callback(
                    {"type": "progress", "outer": dataset_getter.outer_k}
                )
            return (
                {LOSS: {MAIN_LOSS: 1.0}, SCORE: {MAIN_SCORE: 2.0}},
                {LOSS: {MAIN_LOSS: 3.0}, SCORE: {MAIN_SCORE: 4.0}},
                {LOSS: {MAIN_LOSS: 5.0}, SCORE: {MAIN_SCORE: 6.0}},
            )

    run_valid_py = _unwrap_ray_remote(evaluator.run_valid)
    run_test_py = _unwrap_ray_remote(evaluator.run_test)

    fold_folder = tmp_path / "fold"
    fold_folder.mkdir()
    valid_out = tmp_path / "valid_results.dill"

    dataset_getter = _DatasetGetter(outer_k=0, inner_k=1)

    result = run_valid_py(
        _Experiment,
        dataset_getter,
        config={},
        config_id=0,
        run_id=0,
        fold_run_exp_folder=str(fold_folder),
        fold_run_results_torch_path=str(valid_out),
        exp_seed=0,
        training_timeout_seconds=-1,
        logger=None,
        progress_actor=progress_actor,
    )

    assert result[:4] == (0, 1, 0, 0)
    train_res, val_res, elapsed = dill_load(str(valid_out))
    assert train_res[LOSS][MAIN_LOSS] == pytest.approx(1.0)
    assert val_res[SCORE][MAIN_SCORE] == pytest.approx(4.0)
    assert elapsed == pytest.approx(3.5)

    final_folder = tmp_path / "final"
    final_folder.mkdir()
    test_out = tmp_path / "test_results.dill"

    dataset_getter = _DatasetGetter(outer_k=0, inner_k=None)
    outer_k, run_id, elapsed = run_test_py(
        _Experiment,
        dataset_getter,
        best_config={CONFIG: {}, "best_config_id": 1},
        outer_k=0,
        run_id=0,
        final_run_exp_path=str(final_folder),
        final_run_torch_path=str(test_out),
        exp_seed=0,
        training_timeout_seconds=-1,
        logger=None,
        progress_actor=progress_actor,
    )
    assert (outer_k, run_id) == (0, 0)
    assert elapsed == pytest.approx(1.0)

    train_res, val_res, test_res, elapsed = dill_load(str(test_out))
    assert test_res[SCORE][MAIN_SCORE] == pytest.approx(6.0)
    assert elapsed == pytest.approx(1.0)
    assert pushed  # progress updates were forwarded
