import argparse
import os
import sys

import yaml

from mlwiz.data.util import preprocess_data
from mlwiz.static import (
    CONFIG_FILE_CLI_ARGUMENT,
    CONFIG_FILE,
    SKIP_SPLITS_CHECK_CLI_ARGUMENT,
    SKIP_SPLITS_CHECK,
)


def get_args_dict() -> dict:
    """
    Processes CLI arguments (i.e., the config file location) and returns
    a dictionary.

    Returns:
        a dictionary with the name of the configuration file in the
        :obj:`mlwiz.static.CONFIG_FILE` field.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        CONFIG_FILE_CLI_ARGUMENT,
        dest=CONFIG_FILE,
        help="config file to parse the data",
    )
    parser.add_argument(
        SKIP_SPLITS_CHECK_CLI_ARGUMENT,
        dest=SKIP_SPLITS_CHECK,
        action="store_true",
        default=False,
        help="whether to skip automatic data splits check",
    )
    return vars(parser.parse_args())


def main():
    """
    Launches the data preparation pipeline.
    """
    # Necessary to locate dotted paths in projects that use MLWiz
    sys.path.append(os.getcwd())

    args = get_args_dict()

    options = yaml.load(open(args[CONFIG_FILE], "r"), Loader=yaml.FullLoader)
    options.update({SKIP_SPLITS_CHECK: args[SKIP_SPLITS_CHECK]})

    preprocess_data(options)
