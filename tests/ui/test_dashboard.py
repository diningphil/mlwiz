"""Tests for the local MLWiz metrics dashboard."""

from __future__ import annotations

import json
import threading
from urllib.request import Request, urlopen

import pytest
import torch

from mlwiz.ui.dashboard import (
    MetricsCache,
    ResultsRepository,
    create_server,
    get_args,
)


def _write_fixture_results(tmp_path):
    """Create a compact model-selection and final-run result hierarchy."""
    experiment = tmp_path / "RESULTS" / "mlp_MNIST"
    assessment = experiment / "MODEL_ASSESSMENT"
    outer = assessment / "OUTER_FOLD_1"
    selection = outer / "MODEL_SELECTION"
    config = selection / "config_2"
    selection_run = config / "INNER_FOLD_1" / "run_1"
    final_run = outer / "final_run1"
    selection_run.mkdir(parents=True)
    final_run.mkdir(parents=True)

    metrics = {
        "losses": {
            "training_main_loss": [1.0, 0.6, 0.3],
            "validation_main_loss": [1.1, 0.7, 0.4],
        },
        "scores": {
            "training_main_score": [0.3, 0.6, 0.8],
            "validation_main_score": [0.2, 0.5, 0.75],
        },
    }
    torch.save(metrics, selection_run / "metrics_data.torch")
    torch.save(metrics, final_run / "metrics_data.torch")

    (selection / "winner_config.json").write_text(
        json.dumps({"best_config_id": 2, "config": {"lr": 0.01}})
    )
    (config / "config_results.json").write_text(
        json.dumps(
            {
                "config": {"lr": 0.01},
                "avg_validation_score": 0.75,
            }
        )
    )
    (outer / "outer_results.json").write_text(
        json.dumps({"outer_test": {"main_score": 0.72}})
    )
    (assessment / "assessment_results.json").write_text(
        json.dumps({"avg_test_main_score": 0.72})
    )
    return experiment, config, selection_run, final_run


def test_repository_builds_run_tree_and_marks_winner(tmp_path):
    """The sidebar payload should mirror MLWiz's nested result layout."""
    experiment, _, _, _ = _write_fixture_results(tmp_path)
    repository = ResultsRepository(experiment.parent)

    tree = repository.tree()

    assert tree["experiment_count"] == 1
    exp = tree["experiments"][0]
    assert exp["name"] == "mlp_MNIST"
    assert exp["run_count"] == 2
    fold = exp["outer_folds"][0]
    assert fold["model_selection"][0]["number"] == 2
    assert fold["model_selection"][0]["is_winner"] is True
    assert fold["model_selection"][0]["has_metrics"] is True
    assert fold["final_runs"][0]["has_metrics"] is True


def test_details_loads_run_metrics_and_context(tmp_path):
    """A run selection should expose numeric series plus fold metadata."""
    experiment, _, selection_run, _ = _write_fixture_results(tmp_path)
    repository = ResultsRepository(experiment.parent)
    relative_run = selection_run.relative_to(experiment.parent).as_posix()

    details = repository.details(relative_run)

    assert details["selection"]["kind"] == "Model-selection run"
    assert details["metrics_file_count"] == 1
    assert len(details["series"]) == 4
    validation_score = next(
        item for item in details["series"] if item["name"] == "validation_main_score"
    )
    assert validation_score["values"] == pytest.approx([0.2, 0.5, 0.75])
    assert [item["label"] for item in details["metadata"]] == [
        "Configuration results",
        "Selected configuration",
        "Outer-fold results",
        "Assessment results",
    ]


def test_details_loads_oversized_selection_without_caching(tmp_path):
    """The cache ceiling must never prevent an active selection from loading."""
    experiment, _, selection_run, _ = _write_fixture_results(tmp_path)
    repository = ResultsRepository(experiment.parent, cache_max_bytes=0)

    details = repository.details(
        selection_run.relative_to(experiment.parent).as_posix()
    )

    assert len(details["series"]) == 4
    assert details["cache"]["entries"] == 0
    assert details["cache"]["skipped"] == 1


def test_metrics_cache_hits_and_invalidates_changed_file(tmp_path):
    """Unchanged files should hit the cache and rewritten files should reload."""
    experiment, _, selection_run, _ = _write_fixture_results(tmp_path)
    repository = ResultsRepository(experiment.parent)
    relative_run = selection_run.relative_to(experiment.parent).as_posix()

    repository.details(relative_run)
    first = repository.cache_status()
    repository.details(relative_run)
    second = repository.cache_status()
    assert first["misses"] == 1
    assert second["hits"] == 1

    torch.save(
        {
            "losses": {},
            "scores": {"validation_main_score": [0.2, 0.5, 0.75, 0.9]},
        },
        selection_run / "metrics_data.torch",
    )
    details = repository.details(relative_run)
    assert details["cache"]["invalidations"] == 1
    assert details["cache"]["misses"] == 2
    assert details["series"][0]["values"][-1] == pytest.approx(0.9)


def test_metrics_cache_evicts_least_recently_used_entry(tmp_path):
    """Reducing the ceiling should retain only the most recent fitting data."""
    cache = MetricsCache(max_bytes=1024 * 1024)
    series = [
        {
            "group": "scores",
            "name": "validation_main_score",
            "values": [float(index) for index in range(100)],
        }
    ]
    first_path = tmp_path / "first.torch"
    second_path = tmp_path / "second.torch"
    assert cache.put(first_path, (1, 1), series)
    one_entry_bytes = cache.stats()["used_bytes"]
    cache.configure(one_entry_bytes)

    assert cache.put(second_path, (1, 1), series)

    status = cache.stats()
    assert status["entries"] == 1
    assert status["evictions"] == 1
    assert cache.get(first_path, (1, 1)) is None


def test_details_aggregates_all_runs_below_configuration(tmp_path):
    """Clicking a configuration should collect histories from its child runs."""
    experiment, config, _, _ = _write_fixture_results(tmp_path)
    second_run = config / "INNER_FOLD_2" / "run_1"
    second_run.mkdir(parents=True)
    torch.save(
        {"losses": {}, "scores": {"validation_main_score": [0.1, 0.9]}},
        second_run / "metrics_data.torch",
    )
    repository = ResultsRepository(experiment.parent)

    details = repository.details(config.relative_to(experiment.parent).as_posix())

    assert details["selection"]["kind"] == "Model-selection configuration"
    assert details["metrics_file_count"] == 2
    assert {item["source"] for item in details["series"]} == {
        "INNER_FOLD_1/run_1",
        "INNER_FOLD_2/run_1",
    }


def test_details_reports_unreadable_metric_file(tmp_path):
    """A damaged artifact should not take down the dashboard API."""
    experiment, _, _, final_run = _write_fixture_results(tmp_path)
    (final_run / "metrics_data.torch").write_bytes(b"not a torch file")
    repository = ResultsRepository(experiment.parent)

    details = repository.details(final_run.relative_to(experiment.parent).as_posix())

    assert details["series"] == []
    assert details["metrics_file_count"] == 1
    assert len(details["errors"]) == 1


def test_repository_rejects_path_traversal(tmp_path):
    """The details API must never expose files outside the chosen logdir."""
    experiment, _, _, _ = _write_fixture_results(tmp_path)
    repository = ResultsRepository(experiment.parent)

    with pytest.raises(ValueError, match="outside"):
        repository.resolve("../../")


def test_cli_parses_dashboard_options(tmp_path):
    """Dashboard CLI flags should be usable without starting the server."""
    args = get_args(
        [
            "--logdir",
            str(tmp_path),
            "--host",
            "0.0.0.0",
            "--port",
            "6010",
            "--open",
        ]
    )
    assert args.host == "0.0.0.0"
    assert args.port == 6010
    assert args.open_browser is True


def test_http_server_serves_frontend_and_api(tmp_path):
    """A startable server should expose both the page and tree API."""
    experiment, _, _, _ = _write_fixture_results(tmp_path)
    server = create_server(experiment.parent, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base_url = f"http://127.0.0.1:{server.server_address[1]}"
        with urlopen(f"{base_url}/", timeout=3) as response:
            page = response.read().decode("utf-8")
        with urlopen(f"{base_url}/assets/app.js", timeout=3) as response:
            app_script = response.read().decode("utf-8")
        with urlopen(f"{base_url}/assets/styles.css", timeout=3) as response:
            stylesheet = response.read().decode("utf-8")
        with urlopen(f"{base_url}/assets/mlwiz-logo.png", timeout=3) as response:
            logo = response.read()
        with urlopen(f"{base_url}/api/tree", timeout=3) as response:
            tree = json.loads(response.read())
        with urlopen(f"{base_url}/api/cache", timeout=3) as response:
            initial_cache = json.loads(response.read())
        cache_request = Request(
            f"{base_url}/api/cache",
            data=json.dumps({"max_mb": 64}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(cache_request, timeout=3) as response:
            updated_cache = json.loads(response.read())
        assert "MLWiz Dashboard" in page
        assert "/assets/mlwiz-logo.png" in page
        assert "sessionStorage" in app_script
        assert "openNodes" in app_script
        assert 'postJson("/api/cache"' in app_script
        assert "[hidden] { display: none !important; }" in stylesheet
        assert logo.startswith(b"\x89PNG\r\n\x1a\n")
        assert tree["experiments"][0]["name"] == "mlp_MNIST"
        assert initial_cache["max_mb"] == 256
        assert updated_cache["max_mb"] == 64
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
