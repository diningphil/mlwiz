from shutil import rmtree

import pytest
import yaml

from mlwiz.data.util import preprocess_data
from mlwiz.launch_experiment import evaluation
from mlwiz.static import DEBUG, CONFIG_FILE


def test_datasets_creation():
    yaml_files = [
        "examples/DATA_CONFIGS/config_MNIST.yml",
        "examples/DATA_CONFIGS/config_MNISTTemporal.yml",
        "examples/DATA_CONFIGS/config_NCI1.yml",
        "examples/DATA_CONFIGS/config_Cora.yml",
        "examples/DATA_CONFIGS/config_ToyIterableDataset.yml",
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
        "examples/MODEL_CONFIGS/config_MLP.yml",
        "examples/MODEL_CONFIGS/config_CNN.yml",
        "examples/MODEL_CONFIGS/config_GRU.yml",
        "examples/MODEL_CONFIGS/config_DGN.yml",
        "examples/MODEL_CONFIGS/config_DGN_SingleGraph.yml",
        "examples/MODEL_CONFIGS/config_MLP_IterableDataset.yml",
    ]
    for config_file in config_files:
        config = {}
        config[CONFIG_FILE] = config_file
        config[DEBUG] = True
        config = MockConfig(config)
        evaluation(config)


@pytest.mark.dependency(depends=["test_experiments"])
def test_cleanup():
    rmtree("RESULTS")
    rmtree("DATA")
