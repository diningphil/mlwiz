"""CLI utility to generate per-dataset experiment configs from a base YAML file.

Writes derived experiment configuration files by merging dataset-specific settings.
"""

import os
import sys
import yaml
from copy import deepcopy

from mlwiz.config_loader import load_config, load_experiment_config


def main():
    """
    Generate per-dataset experiment config files from a base YAML config.

    This script is a small CLI utility that takes a base experiment config and
    one or more dataset config files. For each dataset config, it creates a new
    YAML file named ``<exp_name>_<dataset_name>.yml`` that merges the base
    config with dataset-specific values (storage folder, dataset class, and the
    derived splits file path).

    Expected CLI usage (parsed via ``sys.argv``):
        ``mlwiz-config-duplicator.py --base-exp-config <base_exp_config> --data-config-files <data_config_files...>``

    Side effects:
        Reads YAML files from disk, writes new YAML files to the current working
        directory, prints progress/errors to stdout, and may call ``sys.exit``.
    """
    if len(sys.argv) < 5:
        print(
            "Usage: mlwiz-config-duplicator.py --base-exp-config <base_exp_config> --data-config-files <data_config_files>"
        )
        sys.exit(1)

    try:
        base_exp_config = sys.argv[2]
        data_config_files = sys.argv[4:]

        # Resolve modular defaults before applying dataset-specific values.
        base_exp_config_data = load_experiment_config(base_exp_config)

        exp_name = base_exp_config_data["experiment"]["exp_name"]

        # Data configs can use the same modular defaults system.
        data_config_files_data = [
            load_config(file) for file in data_config_files
        ]

        # Extract dataset class name and storage folder from each data config file
        for data_config in data_config_files_data:
            storage_folder = data_config["dataset"]["args"]["storage_folder"]
            dataset_class_name = data_config["dataset"]["class_name"]
            dataset_name = dataset_class_name.split(".")[-1]
            splits_folder = data_config["splitter"]["splits_folder"]
            outer_folds = data_config["splitter"]["args"]["n_outer_folds"]
            inner_folds = data_config["splitter"]["args"]["n_inner_folds"]

            # Create a new config file for each dataset
            new_config = deepcopy(base_exp_config_data)
            new_config["dataset"]["storage_folder"] = storage_folder
            new_config["dataset"]["dataset_class"] = dataset_class_name
            new_config["dataset"]["data_splits_file"] = os.path.join(
                splits_folder,
                dataset_name,
                f"{dataset_name}_outer{outer_folds}_inner{inner_folds}.splits",
            )

            # Store yaml into a new file
            new_config_file = os.path.join(f"{exp_name}_{dataset_name}.yml")
            print(f"Creating new config file: {new_config_file}")
            with open(new_config_file, "w") as f:
                yaml.safe_dump(new_config, f, sort_keys=False)

    except Exception as e:
        print(f"An error occurred: {e}")
        print(
            "Usage: mlwiz-config-duplicator.py --base-exp-config <base_exp_config> --data-config-files <data_config_files>"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
