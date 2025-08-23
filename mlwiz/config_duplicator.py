import os
import sys
from pathlib import Path
import yaml
from copy import deepcopy


def main():

    if len(sys.argv) < 5:
        print(
            "Usage: mlwiz-config-duplicator.py --base-exp-config <base_exp_config> --data-config-files <data_config_files>"
        )
        sys.exit(1)

    try:
        base_exp_config = sys.argv[2]
        data_config_files = sys.argv[4:]

        # Load YAML base_exp_config file
        with open(base_exp_config, "r") as f:
            base_exp_config_data = yaml.safe_load(f)

        exp_name = base_exp_config_data["exp_name"]

        # Load YAML data_config_files
        data_config_files_data = []
        for file in data_config_files:
            with open(file, "r") as f:
                data_config_files_data.append(yaml.safe_load(f))

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
            new_config["storage_folder"] = storage_folder
            new_config["dataset_class"] = dataset_class_name
            new_config["data_splits_file"] = os.path.join(
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
