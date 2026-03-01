"""Tests for grid and random-search configuration generation.

Ensures the YAML-defined search space expands to the expected number of configurations.
"""

import pytest
import yaml

from mlwiz.evaluation.grid import Grid
from mlwiz.evaluation.random_search import RandomSearch


@pytest.fixture
def search_method_config_length():
    """
    Provide search method classes and expected configuration counts.

    Returns:
        list[tuple[type, str, int]]: Tuples of ``(search_class, yaml_path, expected_len)``.
    """
    return [
        (Grid, "tests/evaluation/grid_search.yml", 6),
        (RandomSearch, "tests/evaluation/random_search.yml", 10),
    ]


def test_search_method(search_method_config_length):
    """
    Validate grid/random-search expand to the expected number of configurations.

    This test also checks that distinct indices yield distinct configurations
    unless the YAML explicitly produces duplicates.
    """
    for search_method, filepath, num_of_configs in search_method_config_length:
        search = search_method(
            yaml.load(open(filepath, "r"), Loader=yaml.FullLoader)
        )
        # Check the amount of configurations expected and those produced
        # are the same
        assert len(search) == num_of_configs

        # No two configurations should be equal
        # (unless it's intended from the config file)
        for i in range(len(search)):
            for j in range(i + 1, len(search)):
                assert search[i] != search[j]


def test_grid_rejects_conflicting_model_selection_keys():
    """
    ``Grid`` should reject configs that define both legacy and new selectors.
    """
    with pytest.raises(ValueError, match="cannot define both"):
        Grid(
            {
                "exp_name": "conflict",
                "storage_folder": "DATA",
                "dataset_class": "builtins.list",
                "data_splits_file": "dummy.splits",
                "device": "cpu",
                "max_cpus": 1,
                "max_gpus": 0,
                "gpu_memory": 0,
                "dataset_getter": "mlwiz.data.provider.DataProvider",
                "data_loader": {
                    "class_name": "torch.utils.data.DataLoader",
                    "args": {"num_workers": 0, "pin_memory": False},
                },
                "experiment": "mlwiz.experiment.Experiment",
                "higher_results_are_better": True,
                "model_selection_criteria": [
                    {"metric": "main_score", "direction": "max"}
                ],
                "evaluate_every": 1,
                "risk_assessment_training_runs": 1,
                "model_selection_training_runs": 1,
                "grid": {"hp_id": [0, 1]},
            }
        )
