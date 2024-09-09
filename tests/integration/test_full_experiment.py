from shutil import rmtree

import pytest
import yaml

from mlwiz.data.util import preprocess_data
from mlwiz.launch_experiment import evaluation
from mlwiz.static import DEBUG, CONFIG_FILE


def test_datasets_creation():
    yaml_files = [
        "tests/integration/DATA_CONFIGS/config_FakeMNIST.yml",
        "tests/integration/DATA_CONFIGS/config_FakeMNISTTemporal.yml",
        "tests/integration/DATA_CONFIGS/config_FakeNCI1.yml",
        "tests/integration/DATA_CONFIGS/config_FakeCora.yml",
        "tests/integration/DATA_CONFIGS/config_FakeToyIterableDataset.yml",
    ]
    for y in yaml_files:
        config = yaml.load(
            open(y, "r"),
            Loader=yaml.FullLoader,
        )
        preprocess_data(config)


@pytest.mark.dependency(depends=["test_datasets_creation"])
def test_experiments():
    class MockConfig:
        def __init__(self, d):
            for key in d.keys():
                setattr(self, key, d[key])

    config_files = [
        "tests/integration/MODEL_CONFIGS/config_MLP.yml",
        "tests/integration/MODEL_CONFIGS/config_CNN.yml",
        "tests/integration/MODEL_CONFIGS/config_GRU.yml",
        "tests/integration/MODEL_CONFIGS/config_DGN.yml",
        "tests/integration/MODEL_CONFIGS/config_DGN_SingleGraph.yml",
        "tests/integration/MODEL_CONFIGS/config_MLP_IterableDataset.yml",
    ]
    for config_file in config_files:
        config = {}
        config[CONFIG_FILE] = config_file
        config[DEBUG] = True
        config = MockConfig(config)
        evaluation(config)


@pytest.mark.dependency(depends=["test_experiments"])
def test_cleanup():
    rmtree("tests/tmp/")