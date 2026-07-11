"""Tests for the local MLWiz metrics dashboard."""

from __future__ import annotations

import json
import threading
from urllib.request import Request, urlopen

import pytest
import torch

from mlwiz.static import MODEL_GRAPH_INPUT_SPEC_FILENAME
from mlwiz.ui.dashboard import (
    MetricsCache,
    ResultsRepository,
    create_server,
    get_args,
)


class _TinyGraphModel(torch.nn.Module):
    """Small reconstructable model used by checkpoint-graph tests."""

    def __init__(self, dim_input_features, dim_target, config):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(dim_input_features, 3),
            torch.nn.ReLU(),
        )
        self.output = torch.nn.Linear(3, dim_target)

    def forward(self, inputs):
        return self.output(self.encoder(inputs))


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
    assert details["selection"]["plot_scope"] == "single_run"
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


def test_metrics_cache_reset_clears_entries_and_counters(tmp_path):
    """Resetting the cache should retain its limit but clear all cache state."""
    cache = MetricsCache(max_bytes=1024 * 1024)
    metrics_path = tmp_path / "metrics_data.torch"
    assert cache.put(
        metrics_path,
        (1, 1),
        [{"group": "scores", "name": "main_score", "values": [0.5]}],
    )
    assert cache.get(metrics_path, (1, 1)) is not None

    status = cache.clear()

    assert status["entries"] == 0
    assert status["used_bytes"] == 0
    assert status["max_mb"] == 1
    assert status["hits"] == 0
    assert status["misses"] == 0


def test_completed_experiment_filter_uses_aggregated_results(tmp_path):
    """Finished experiments should filter on config-level validation results."""
    experiment, config, _, _ = _write_fixture_results(tmp_path)
    results_path = config / "config_results.json"
    results = json.loads(results_path.read_text())
    results["avg_validation_aux_score"] = 0.63
    results["avg_training_score"] = 0.81
    results_path.write_text(json.dumps(results))
    repository = ResultsRepository(experiment.parent)

    data = repository.experiment_filter_data(experiment.name)

    config_path = next(iter(data["configurations"]))
    assert data["complete"] is True
    assert data["default_metric"] == "scores:main_score"
    assert data["value_source"] == "aggregated training/validation result"
    assert data["configurations"][config_path]["values"][
        "validation:scores:main_score"
    ] == pytest.approx(0.75)
    assert data["configurations"][config_path]["values"][
        "validation:scores:aux"
    ] == pytest.approx(0.63)
    assert data["configurations"][config_path]["values"][
        "training:scores:main_score"
    ] == pytest.approx(0.81)
    assert data["splits"] == ["validation", "training"]
    assert "scores:aux" in {metric["id"] for metric in data["metrics"]}
    aux_metric = next(
        metric for metric in data["metrics"] if metric["id"] == "scores:aux"
    )
    assert aux_metric["label"] == "Aux"


def test_running_experiment_filter_uses_last_recorded_metrics(tmp_path):
    """In-progress experiments should use the latest validation history value."""
    experiment, _, selection_run, _ = _write_fixture_results(tmp_path)
    (experiment / "MODEL_ASSESSMENT" / "assessment_results.json").unlink()
    metrics_path = selection_run / "metrics_data.torch"
    metrics = torch.load(metrics_path, weights_only=True)
    metrics["scores"]["validation_aux"] = [0.1, 0.4, 0.7]
    torch.save(metrics, metrics_path)
    repository = ResultsRepository(experiment.parent)

    data = repository.experiment_filter_data(experiment.name)

    config_values = next(iter(data["configurations"].values()))["values"]
    assert data["complete"] is False
    assert data["value_source"] == "last recorded training/validation epoch"
    assert config_values["validation:scores:main_score"] == pytest.approx(0.75)
    assert config_values["validation:scores:aux"] == pytest.approx(0.7)
    assert config_values["validation:losses:main_loss"] == pytest.approx(0.4)
    assert config_values["training:scores:main_score"] == pytest.approx(0.8)


def test_selected_experiment_overview_summarizes_run_times(tmp_path):
    """Details should include timing statistics only for their parent experiment."""
    experiment, _, selection_run, final_run = _write_fixture_results(tmp_path)
    (selection_run / "experiment.log").write_text(
        "Total time of the experiment in seconds: 10.0 \n"
        "Total time of the experiment in seconds: 2.5 \n"
    )
    (final_run / "experiment.log").write_text(
        "Total time of the experiment in seconds: 17.5 \n"
    )
    repository = ResultsRepository(experiment.parent)

    details = repository.details(
        selection_run.relative_to(experiment.parent).as_posix()
    )
    overview = details["overview"]

    assert overview["name"] == "mlp_MNIST"
    assert overview["state"] == "completed"
    assert overview["runs"] == {
        "total": 2,
        "completed": 2,
        "running": 0,
        "queued": 0,
        "failed": 0,
    }
    assert overview["configurations"] == {"total": 1, "completed": 1}
    assert overview["timing"]["recorded_total_seconds"] == pytest.approx(30.0)
    assert overview["timing"]["average_run_seconds"] == pytest.approx(15.0)
    assert overview["timing"]["median_run_seconds"] == pytest.approx(15.0)


def test_running_overview_estimates_remaining_compute_time(tmp_path):
    """Running experiments should estimate unfinished compute from timed runs."""
    experiment, _, selection_run, _ = _write_fixture_results(tmp_path)
    (experiment / "MODEL_ASSESSMENT" / "assessment_results.json").unlink()
    (selection_run / "experiment.log").write_text(
        "Total time of the experiment in seconds: 12.0 \n"
    )
    repository = ResultsRepository(experiment.parent)

    overview = repository.details(
        selection_run.relative_to(experiment.parent).as_posix()
    )["overview"]

    assert overview["state"] == "running"
    assert overview["runs"]["completed"] == 1
    assert overview["runs"]["running"] == 1
    assert overview["timing"]["estimated_remaining_compute_seconds"] == pytest.approx(
        12.0
    )


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
    assert details["selection"]["plot_scope"] == "model_selection_configuration"
    assert details["metrics_file_count"] == 2
    assert {item["source"] for item in details["series"]} == {
        "INNER_FOLD_1/run_1",
        "INNER_FOLD_2/run_1",
    }


def test_final_run_siblings_are_loaded_only_for_aggregation(tmp_path):
    """Final-run aggregation should load siblings lazily, not on first click."""
    experiment, _, _, final_run = _write_fixture_results(tmp_path)
    second_final = final_run.parent / "final_run2"
    second_final.mkdir()
    torch.save(
        {"losses": {}, "scores": {"validation_main_score": [0.4, 0.8]}},
        second_final / "metrics_data.torch",
    )
    repository = ResultsRepository(experiment.parent)
    relative = final_run.relative_to(experiment.parent).as_posix()

    selected = repository.details(relative)
    aggregated = repository.details(relative, include_final_siblings=True)

    assert selected["metrics_file_count"] == 1
    assert selected["selection"]["plot_scope"] == "final_runs"
    assert selected["selection"]["final_runs_included"] is False
    assert aggregated["metrics_file_count"] == 2
    assert aggregated["selection"]["final_runs_included"] is True
    assert {item["source"] for item in aggregated["series"]} == {
        "final_run1",
        "final_run2",
    }


def test_running_model_graph_uses_last_checkpoint_and_manifest(tmp_path):
    """A live run should reconstruct its latest checkpoint module graph."""
    experiment, _, selection_run, _ = _write_fixture_results(tmp_path)
    model = _TinyGraphModel(2, 1, {})
    torch.save(
        {"epoch": 2, "model_state": model.state_dict()},
        selection_run / "last_checkpoint.pth",
    )
    torch.save(
        {"best_epoch": 0, "model_state": model.state_dict()},
        selection_run / "best_checkpoint.pth",
    )
    (selection_run / "model_manifest.json").write_text(
        json.dumps(
            {
                "version": 1,
                "model": f"{__name__}._TinyGraphModel",
                "config": {},
                "dim_input_features": 2,
                "dim_target": 1,
            }
        )
    )
    (selection_run / MODEL_GRAPH_INPUT_SPEC_FILENAME).write_text(
        json.dumps(
            {
                "version": 1,
                "kind": "tensor",
                "shape": [2, 2],
                "dtype": "float32",
            }
        )
    )
    repository = ResultsRepository(experiment.parent)
    relative = selection_run.relative_to(experiment.parent).as_posix()

    graph = repository.model_graph(relative)
    cached = repository.model_graph(relative)
    explicit_best = repository.model_graph(relative, checkpoint_kind="best")
    operators = repository.model_graph(relative, graph_mode="operators")
    cached_operators = repository.model_graph(
        relative, graph_mode="operators"
    )
    info = repository.model_graph_info(relative)

    assert graph["checkpoint"]["kind"] == "last"
    assert graph["checkpoint"]["requested"] == "auto"
    assert set(graph["checkpoint"]["available"]) == {"best", "last"}
    assert graph["epoch"] == 3
    assert graph["graph_kind"] == "module"
    assert graph["warning"] is None
    assert {node["id"] for node in graph["nodes"]} >= {
        "__root__",
        "encoder",
        "encoder.0",
        "output",
    }
    assert cached["checkpoint"]["cache_hit"] is True
    assert explicit_best["checkpoint"]["kind"] == "best"
    assert explicit_best["epoch"] == 1
    assert operators["graph_mode"] == "operators"
    assert operators["graph_kind"] == "torch.export ATen operators"
    assert operators["summary"]["operators"] >= 3
    assert operators["edges"]
    assert any(
        "aten." in node["target"]
        for node in operators["nodes"]
        if node["op"] == "call_function"
    )
    assert cached_operators["checkpoint"]["cache_hit"] is True
    assert info["checkpoint"]["kind"] == "last"
    assert set(info["checkpoint"]["loadable"]) == {"best", "last"}
    assert info["modes"]["operators"]["available"] is True


def test_operator_graph_requires_a_recorded_tensor_input(tmp_path):
    """Old and custom-input runs should retain Architecture with a clear reason."""
    experiment, _, selection_run, _ = _write_fixture_results(tmp_path)
    model = _TinyGraphModel(2, 1, {})
    torch.save(
        {"epoch": 0, "model_state": model.state_dict()},
        selection_run / "last_checkpoint.pth",
    )
    (selection_run / "model_manifest.json").write_text(
        json.dumps(
            {
                "version": 1,
                "model": f"{__name__}._TinyGraphModel",
                "config": {},
                "dim_input_features": 2,
                "dim_target": 1,
            }
        )
    )
    repository = ResultsRepository(experiment.parent)
    relative = selection_run.relative_to(experiment.parent).as_posix()

    architecture = repository.model_graph(relative)
    info = repository.model_graph_info(relative)

    assert architecture["graph_mode"] == "architecture"
    assert info["modes"]["operators"]["available"] is False
    assert "input shape" in info["modes"]["operators"]["reason"]
    with pytest.raises(ValueError, match="input shape"):
        repository.model_graph(relative, graph_mode="operators")


def test_completed_model_graph_prefers_best_checkpoint_with_fallback(tmp_path):
    """Completed runs prefer best state and old runs retain hierarchy support."""
    experiment, _, _, final_run = _write_fixture_results(tmp_path)
    (final_run / "experiment.log").write_text(
        "Total time of the experiment in seconds: 4.0\n"
    )
    (final_run / "run_1_results.dill").write_bytes(b"completed")
    torch.save(
        {
            "best_epoch": 4,
            "model_state": {
                "encoder.weight": torch.ones(3, 2),
                "encoder.bias": torch.ones(3),
            },
        },
        final_run / "best_checkpoint.pth",
    )
    torch.save(
        {"epoch": 8, "model_state": {"other.weight": torch.ones(1, 1)}},
        final_run / "last_checkpoint.pth",
    )
    repository = ResultsRepository(experiment.parent)

    graph = repository.model_graph(
        final_run.relative_to(experiment.parent).as_posix()
    )
    explicit_last = repository.model_graph(
        final_run.relative_to(experiment.parent).as_posix(),
        checkpoint_kind="last",
    )

    assert graph["run_state"] == "completed"
    assert graph["checkpoint"]["kind"] == "best"
    assert graph["epoch"] == 5
    assert graph["graph_kind"] == "checkpoint hierarchy"
    assert "predates model manifests" in graph["warning"]
    assert any(node["id"] == "encoder" for node in graph["nodes"])
    assert explicit_last["checkpoint"]["kind"] == "last"
    assert explicit_last["epoch"] == 9


def test_model_graph_skips_checkpoint_larger_than_cache(tmp_path):
    """Checkpoint graph loading must respect the cache memory ceiling."""
    experiment, _, selection_run, _ = _write_fixture_results(tmp_path)
    torch.save(
        {"epoch": 1, "model_state": {"layer.weight": torch.ones(2, 2)}},
        selection_run / "last_checkpoint.pth",
    )
    repository = ResultsRepository(experiment.parent, cache_max_bytes=1)
    relative = selection_run.relative_to(experiment.parent).as_posix()

    with pytest.raises(ValueError, match="exceeding the 0.00 MB cache limit"):
        repository.model_graph(relative)

    info = repository.model_graph_info(relative)
    assert info["checkpoint"]["available"] == ["last"]
    assert info["checkpoint"]["loadable"] == []


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
    experiment, _, selection_run, _ = _write_fixture_results(tmp_path)
    model = _TinyGraphModel(2, 1, {})
    torch.save(
        {"epoch": 1, "model_state": model.state_dict()},
        selection_run / "last_checkpoint.pth",
    )
    (selection_run / "model_manifest.json").write_text(
        json.dumps(
            {
                "version": 1,
                "model": f"{__name__}._TinyGraphModel",
                "config": {},
                "dim_input_features": 2,
                "dim_target": 1,
            }
        )
    )
    (selection_run / MODEL_GRAPH_INPUT_SPEC_FILENAME).write_text(
        json.dumps(
            {
                "version": 1,
                "kind": "tensor",
                "shape": [2, 2],
                "dtype": "float32",
            }
        )
    )
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
        with urlopen(
            f"{base_url}/api/experiment-filter?path=mlp_MNIST", timeout=3
        ) as response:
            filter_data = json.loads(response.read())
        graph_path = selection_run.relative_to(experiment.parent).as_posix()
        with urlopen(
            f"{base_url}/api/model-graph?path={graph_path}", timeout=3
        ) as response:
            graph_data = json.loads(response.read())
        with urlopen(
            f"{base_url}/api/model-graph-info?path={graph_path}", timeout=3
        ) as response:
            graph_info = json.loads(response.read())
        with urlopen(
            f"{base_url}/api/model-graph?path={graph_path}&mode=operators",
            timeout=3,
        ) as response:
            operator_graph = json.loads(response.read())
        cache_request = Request(
            f"{base_url}/api/cache",
            data=json.dumps({"max_mb": 64}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(cache_request, timeout=3) as response:
            updated_cache = json.loads(response.read())
        reset_request = Request(
            f"{base_url}/api/cache/reset",
            data=b"{}",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(reset_request, timeout=3) as response:
            reset_cache = json.loads(response.read())
        assert "MLWiz Dashboard" in page
        assert "/assets/mlwiz-logo.png" in page
        assert 'id="scale-toggle"' in page
        assert 'id="cache-reset"' in page
        assert 'id="plot-mode-select"' in page
        assert 'id="inner-fold-aggregate"' in page
        assert 'id="plot-navigator"' in page
        assert 'id="show-all-plots"' in page
        assert 'id="refresh-interval"' in page
        assert 'id="theme-toggle"' in page
        assert 'id="experiment-overview"' in page
        assert 'id="model-graph-section"' in page
        assert 'id="model-graph-checkpoint-select"' in page
        assert 'id="model-graph-run-select"' in page
        assert 'id="model-graph-mode-select"' in page
        assert 'id="graph-expand-all"' in page
        assert 'id="graph-collapse-all"' in page
        assert 'id="graph-view-toggle"' in page
        assert 'id="graph-search"' in page
        assert page.index('id="summary-grid"') < page.index(
            'id="model-graph-section"'
        ) < page.index('id="chart-toolbar"')
        assert 'data-theme="dark"' in page
        assert 'id="tree-search"' not in page
        assert "sessionStorage" in app_script
        assert "localStorage.setItem(themeStorageKey" in app_script
        assert "openNodes" in app_script
        assert 'postJson("/api/cache"' in app_script
        assert 'postJson("/api/cache/reset"' in app_script
        assert "/api/experiment-filter" in app_script
        assert "/api/model-graph" in app_script
        assert "/api/model-graph-info" in app_script
        assert "renderModelGraph" in app_script
        assert "graphCheckpointChoices" in app_script
        assert "graphFocusedRuns" in app_script
        assert "renderModelGraphRunSelector" in app_script
        assert "renderOperatorGraphCanvas" in app_script
        assert "renderGraphModeSelect" in app_script
        assert "buildGraphExplorerModel" in app_script
        assert "graphParameterColor" in app_script
        assert "toggleGraphBlock" in app_script
        assert "setAllGraphBlocks" in app_script
        assert 'addEventListener("pointermove"' in app_script
        assert "createValueScale" in app_script
        assert "aggregateMetricLines" in app_script
        assert "renderInnerFoldAggregation" in app_script
        assert "renderPlotNavigator" in app_script
        assert "moveNavigatorSelection" in app_script
        assert "renderChartsPreservingScroll" in app_script
        assert "observeStickyPlotNavigator" in app_script
        assert "drawMetricBand" in app_script
        assert "aggregate_final_runs=1" in app_script
        assert "configurationPassesFilter" in app_script
        assert "scheduleRefresh" in app_script
        assert "applyTheme" in app_script
        assert "metadataViewer" in app_script
        assert '"Raw JSON"' in app_script
        assert "metadata-json:" in app_script
        assert "metadataModes" in app_script
        assert "metadataScrolls" in app_script
        assert "restoreScroll" in app_script
        assert "expandJsonDescendants" in app_script
        assert ".json-inspector" in stylesheet
        assert ".model-graph-section" in stylesheet
        assert ".parameter-legend" in stylesheet
        assert ".graph-node-card" in stylesheet
        assert ".plot-navigator { position: sticky" in stylesheet
        assert ".plot-navigator.is-stuck" in stylesheet
        assert ".content { min-width: 0;" in stylesheet
        assert "overflow: visible;" in stylesheet
        assert "[hidden] { display: none !important; }" in stylesheet
        assert logo.startswith(b"\x89PNG\r\n\x1a\n")
        assert tree["experiments"][0]["name"] == "mlp_MNIST"
        assert initial_cache["max_mb"] == 256
        assert updated_cache["max_mb"] == 64
        assert reset_cache["entries"] == 0
        assert reset_cache["max_mb"] == 64
        assert graph_data["checkpoint"]["kind"] == "last"
        assert graph_data["epoch"] == 2
        assert graph_info["checkpoint"]["kind"] == "last"
        assert graph_info["checkpoint"]["loadable"] == ["last"]
        assert graph_info["modes"]["operators"]["available"] is True
        assert operator_graph["graph_mode"] == "operators"
        assert operator_graph["summary"]["operators"] >= 3
        assert filter_data["default_metric"] == "scores:main_score"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
