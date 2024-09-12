from shutil import rmtree

import pytest
import yaml

from mlwiz.data.util import preprocess_data
from mlwiz.evaluation.util import retrieve_best_configuration, \
    instantiate_dataset_from_config, instantiate_model_from_config, \
    load_checkpoint, instantiate_data_provider_from_config
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
    for i, config_file in enumerate(config_files):
        config = {}
        config[CONFIG_FILE] = config_file
        config[DEBUG] = True
        config = MockConfig(config)
        evaluation(config)


    config = retrieve_best_configuration('tests/tmp/RESULTS/mlp_FakeMNIST/MODEL_ASSESSMENT/OUTER_FOLD_1/MODEL_SELECTION/')
    splits_filepath = 'tests/tmp/DATA_SPLITS/FakeMNIST/FakeMNIST_outer3_inner2.splits'
    device = 'cpu'

    # instantiate dataset
    dataset = instantiate_dataset_from_config(config)

    # instantiate model
    model = instantiate_model_from_config(config, dataset)

    # load model's checkpoint, assuming the best configuration has been loaded
    checkpoint_location = 'tests/tmp/RESULTS/mlp_FakeMNIST/MODEL_ASSESSMENT/OUTER_FOLD_1/final_run1/best_checkpoint.pth'
    load_checkpoint(checkpoint_location, model, device=device)

    # you can now call the forward method of your model
    y, embeddings = model(dataset[0][0])

    # ------------------------------------------------------------------ #
    # OPTIONAL: you can also instantiate a DataProvider to load TR/VL/TE splits specific to each fold

    data_provider = instantiate_data_provider_from_config(config,
                                                          splits_filepath,
                                                          3,
                                                          2)
    # select outer fold 1 (indices start from 0)
    data_provider.set_outer_k(0)
    # select inner fold 1 (indices start from 0)
    data_provider.set_inner_k(0)

    # set exp seet for workers (does not affect inference)
    data_provider.set_exp_seed(42)  # any seed

    # load loaders associated with final runs of outer 1 split
    train_loader = data_provider.get_outer_train()
    val_loader = data_provider.get_outer_train()
    test_loader = data_provider.get_outer_train()

    # Please refer to the DataProvider documentation to use it properly.
    # ------------------------------------------------------------------ #

@pytest.mark.dependency(depends=["test_experiments"])
def test_cleanup():
    rmtree("tests/tmp/")