"""Pareto computation, configuration compatibility, reporting, and integration."""

from copy import deepcopy
import json
import operator
from unittest.mock import Mock

import pytest

from mlwiz.evaluation.bayesian_search import BayesianSearch
from mlwiz.evaluation.evaluator import RiskAssesser
from mlwiz.evaluation.grid import Grid
from mlwiz.evaluation.random_search import RandomSearch
from mlwiz.evaluation.util import (
    normalize_selection_criteria,
    pareto_frontier,
    select_pareto_configuration,
    write_pareto_report,
)
from tests.evaluation.test_evaluator_helpers import _structured_grid_config


CRITERIA = [("avg_validation_score", operator.gt), ("avg_validation_loss", operator.lt)]
RAW_CRITERIA = [
    {"metric": "main_score", "direction": "max"},
    {"metric": "main_loss", "direction": "min"},
]


def configurations():
    """Three tradeoffs and two dominated points in deliberately unordered ID order."""
    return [
        (5, {"avg_validation_score": -2, "avg_validation_loss": 12}),
        (3, {"avg_validation_score": 10, "avg_validation_loss": 10}),
        (2, {"avg_validation_score": 6, "avg_validation_loss": 4}),
        (4, {"avg_validation_score": -1, "avg_validation_loss": 11}),
        (1, {"avg_validation_score": 0, "avg_validation_loss": 0}),
    ]


def test_frontier_and_compromise_do_not_mutate_inputs():
    configs = configurations()
    original = deepcopy(configs)
    assert [i for i, _ in pareto_frontier(configs, CRITERIA)] == [3, 2, 1]
    selection = select_pareto_configuration(configs, CRITERIA)
    assert selection.winner[0] == 2
    assert [i for i, _ in selection.frontier] == [2, 1, 3]
    assert configs == original


def test_order_and_positive_scale_invariance():
    configs = configurations()
    assert select_pareto_configuration(configs[::-1], CRITERIA[::-1]).winner[0] == 2
    for _, results in configs:
        results["avg_validation_loss"] = results["avg_validation_loss"] * 100 + 50
    assert select_pareto_configuration(configs, CRITERIA).winner[0] == 2


@pytest.mark.parametrize("dimension", [1, 2, 3, 5])
def test_arbitrary_dimensions_duplicates_and_constant_objectives(dimension):
    criteria = [
        (str(i), operator.gt if i % 2 == 0 else operator.lt) for i in range(dimension)
    ]
    best = {key: 1 if comparator is operator.gt else -1 for key, comparator in criteria}
    configs = [(3, dict(best)), (2, dict(best)), (1, {key: 0 for key, _ in criteria})]
    selection = select_pareto_configuration(configs, criteria)
    assert [i for i, _ in selection.frontier] == [2, 3]
    assert selection.distances == {3: 0, 2: 0}
    assert select_pareto_configuration(configs[:1], criteria).winner[0] == 3


def test_tolerance_and_standard_deviation_tiebreaks():
    criteria = [("x", operator.gt)]
    configs = [(2, {"x": 1}), (1, {"x": 1 + 5e-13})]
    assert len(pareto_frontier(configs, criteria)) == 2
    assert [i for i, _ in pareto_frontier(configs, criteria, atol=0)] == [1]
    # Equal objective vectors: lower std wins; missing std loses to finite std.
    configs = [
        (3, {"x": 1}),
        (2, {"x": 1, "std_validation_score": 0.2}),
        (1, {"x": 1, "std_validation_score": 0.1}),
    ]
    assert [i for i, _ in select_pareto_configuration(configs, criteria).frontier] == [
        1,
        2,
        3,
    ]


def test_large_finite_objectives():
    configs = [
        (1, {"x": -1e308, "y": 1e308}),
        (2, {"x": 0, "y": 0}),
        (3, {"x": 1e308, "y": -1e308}),
    ]
    assert (
        select_pareto_configuration(
            configs, [("x", operator.gt), ("y", operator.gt)]
        ).winner[0]
        == 2
    )


@pytest.mark.parametrize(
    "bad", [None, "bad", float("nan"), float("inf"), -float("inf")]
)
def test_invalid_objective_values(bad):
    for utility in (pareto_frontier, select_pareto_configuration):
        with pytest.raises(ValueError, match="Configuration 7.*'x'"):
            utility([(7, {"x": bad})], [("x", operator.gt)])


@pytest.mark.parametrize(
    "configs,criteria",
    [([], [("x", operator.gt)]), ([(1, {})], []), ([(1, {})], [("x", operator.gt)])],
)
def test_empty_or_missing_objectives(configs, criteria):
    with pytest.raises(ValueError):
        select_pareto_configuration(configs, criteria)


def test_markdown_report_order_values_and_replacement(tmp_path):
    configs = configurations()
    configs[2][1]["avg_validation_score"] = 6.123456789012345
    selection = select_pareto_configuration(configs, CRITERIA)
    write_pareto_report(tmp_path, configs, CRITERIA, selection)
    path = tmp_path / "pareto_results.md"
    report = path.read_text()
    rows = [line for line in report.splitlines() if "](config_" in line]
    assert [next(i for i in range(1, 6) if f"[config_{i}]" in row) for row in rows] == [
        2,
        1,
        3,
        4,
        5,
    ]
    assert "**Winner · Pareto**" in rows[0]
    assert all("**Pareto**" in row for row in rows[1:3])
    assert all("Dominated" in row and "**" not in row for row in rows[3:])
    assert "config_2/config_results.json" in report
    assert "avg\\_validation\\_score (max)" in report
    assert "avg\\_validation\\_loss (min)" in report
    assert "6.12345678901" in rows[0]
    assert "not a Pareto rank" in report
    path.write_text(report + "stale content")
    write_pareto_report(tmp_path, configs, CRITERIA, selection)
    assert path.read_text() == report


def test_markdown_escapes_metric_names(tmp_path):
    key = "odd|[metric]*_`<tag>\\\nnext"
    configs = [(1, {key: 1})]
    criteria = [(key, operator.gt)]
    write_pareto_report(
        tmp_path, configs, criteria, select_pareto_configuration(configs, criteria)
    )
    report = (tmp_path / "pareto_results.md").read_text()
    assert "odd&#124;\\[metric\\]\\*\\_\\`&lt;tag&gt;\\\\ next" in report


@pytest.mark.parametrize("mode", ["pareto", "lexicographic", "classic"])
@pytest.mark.parametrize("search_class", [Grid, RandomSearch])
def test_structured_modes_and_generated_configurations(tmp_path, mode, search_class):
    setting = {
        "mode": mode,
        **({"direction": "min"} if mode == "classic" else {"criteria": RAW_CRITERIA}),
    }
    config = _structured_grid_config(tmp_path, "modes", setting)
    if search_class is RandomSearch:
        config["random"] = {"budget": 2, "hp_id": 0}
        del config["grid"]
    search = search_class(config)
    assert search.model_selection_mode == mode
    assert search.model_selection_criteria == (
        [{"metric": "main_score", "direction": "min"}]
        if mode == "classic"
        else RAW_CRITERIA
    )
    assert all(
        candidate["model_selection_criteria"] == setting for candidate in search.hparams
    )


def test_legacy_and_list_configuration():
    assert normalize_selection_criteria(None, False) == ("classic", None)
    assert normalize_selection_criteria(RAW_CRITERIA, None) == (
        "lexicographic",
        RAW_CRITERIA,
    )


@pytest.mark.parametrize(
    "setting,legacy",
    [
        (None, None),
        ([], None),
        ({"mode": "invalid"}, None),
        ({}, None),
        ({"mode": "classic"}, None),
        ({"mode": "pareto", "criteria": []}, None),
        ({"mode": "classic", "direction": "max", "criteria": RAW_CRITERIA}, None),
        (RAW_CRITERIA, True),
        ({"mode": "pareto", "criteria": RAW_CRITERIA}, False),
        ([{"metric": "aux", "direction": "max"}], None),
        ([{"metric": "main_score", "source": "loss", "direction": "max"}], None),
        ([{"metric": "main_score", "direction": "invalid"}], None),
    ],
)
def test_invalid_configuration(setting, legacy):
    with pytest.raises(ValueError):
        normalize_selection_criteria(setting, legacy)


def test_pareto_bayesian_search_rejected(tmp_path):
    config = _structured_grid_config(
        tmp_path, "bayes", {"mode": "pareto", "criteria": RAW_CRITERIA}
    )
    del config["grid"]
    config["bayes"] = {
        "budget": 2,
        "random_starts": 1,
        "candidate_pool_size": 4,
        "ei_xi": 0.01,
        "hp_id": 0,
    }
    with pytest.raises(ValueError, match="not supported with Bayesian"):
        BayesianSearch(config)


def make_assesser(tmp_path, mode):
    """Build an evaluator without training or reading a dataset."""
    search = Grid(
        _structured_grid_config(
            tmp_path, "selection", {"mode": mode, "criteria": RAW_CRITERIA}
        )
    )
    search.telegram_config = None
    return RiskAssesser(
        1, 1, object, str(tmp_path), "dummy.splits", search, 1, 1, None, 0
    )


def test_evaluator_pareto_winner_report_and_skips(tmp_path):
    configs = configurations() + [
        (6, {"avg_validation_score": 100, "avg_validation_loss": -100})
    ]
    for config_id, results in configs:
        folder = tmp_path / f"config_{config_id}"
        folder.mkdir()
        (folder / "config_results.json").write_text(
            json.dumps({"config": {"hp_id": config_id}, **results})
        )
    assesser = make_assesser(tmp_path, "pareto")
    assesser.compute_best_hyperparameters(str(tmp_path), 0, 6, [5])
    winner = json.loads((tmp_path / "winner_config.json").read_text())
    assert winner["best_config_id"] == 2
    assert winner["config"] == {"hp_id": 2}
    report = (tmp_path / "pareto_results.md").read_text()
    assert "config_6" not in report
    assert report.count("](config_") == 5
    lexicographic = make_assesser(tmp_path, "lexicographic")
    lexicographic.compute_best_hyperparameters(str(tmp_path), 0, 6, [5])
    assert (
        json.loads((tmp_path / "winner_config.json").read_text())["best_config_id"] == 3
    )


def test_single_configuration_shortcut_has_no_report(tmp_path, monkeypatch):
    assesser = make_assesser(tmp_path, "pareto")
    assesser.model_configs.hparams = assesser.model_configs.hparams[:1]
    assesser.progress_manager = Mock()
    monkeypatch.setattr(assesser, "_create_dataset_getter", Mock(return_value=None))
    assesser.model_selection(str(tmp_path), 0, True, None, [])
    folder = tmp_path / "MODEL_SELECTION"
    assert (
        json.loads((folder / "winner_config.json").read_text())["best_config_id"] == 1
    )
    assert not (folder / "pareto_results.md").exists()


def test_third_objective_changes_frontier():
    configs = [(1, {"x": 2, "y": 2, "z": 0}), (2, {"x": 1, "y": 1, "z": 3})]
    criteria = [(key, operator.gt) for key in ("x", "y", "z")]
    assert [i for i, _ in pareto_frontier(configs, criteria[:2])] == [1]
    assert [i for i, _ in pareto_frontier(configs, criteria)] == [1, 2]


def test_compromise_distance_ties_use_std_then_id():
    configs = [
        (3, {"x": 0, "y": 1, "std_validation_score": 0.1}),
        (2, {"x": 1, "y": 0, "std_validation_score": 0.2}),
    ]
    criteria = [("x", operator.gt), ("y", operator.gt)]
    assert select_pareto_configuration(configs, criteria).winner[0] == 3
    configs[1][1]["std_validation_score"] = 0.1
    assert select_pareto_configuration(configs, criteria).winner[0] == 2
