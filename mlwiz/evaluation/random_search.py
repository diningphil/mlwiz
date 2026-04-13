"""Random-search configuration sampling.

The :class:`~mlwiz.evaluation.random_search.RandomSearch` class samples configurations from a YAML-defined search space.
"""

from copy import deepcopy

from mlwiz.evaluation.grid import Grid
from mlwiz.util import s2c
from mlwiz.static import (
    ARGS,
    BUDGET,
    RANDOM_SEARCH,
    SAMPLE_METHOD,
)

from typing import Iterator, Dict, Any


class RandomSearch(Grid):
    r"""
    Class that implements random-search. It computes all possible
    configurations starting from a suitable config file.

    Args:
        configs_dict (dict): the configuration dictionary specifying the
            different configurations to try
    """

    __search_type__ = RANDOM_SEARCH

    def __init__(self, configs_dict: dict):
        r"""
        Initialize a random-search configuration generator.

        Args:
            configs_dict (dict): Configuration dictionary specifying the search
                space and shared experiment settings. It must include
                ``budget`` (see :mod:`mlwiz.static`) to control how many
                configurations are sampled.

        Raises:
            KeyError: If ``budget`` is missing.

        Side effects:
            Stores ``budget`` and initializes the base :class:`~Grid`
            fields.
        """
        raw_budget = configs_dict.get(BUDGET, None)
        if raw_budget is None:
            raise KeyError(
                f"Missing required '{BUDGET}' key in configuration."
            )
        self.budget = int(raw_budget)
        if self.budget <= 0:
            raise ValueError(
                f"'{BUDGET}' must be > 0, got {self.budget}."
            )
        super().__init__(configs_dict)

    def _gen_helper(self, cfgs_dict: dict) -> Iterator[Dict[str, Any]]:
        r"""
        Takes a dictionary of key:list pairs and computes all possible
        combinations.

        Returns:
            A list of all possible configurations in the form of dictionaries
        """
        keys = cfgs_dict.keys()
        param = list(keys)[0]

        for _ in range(self.budget):
            result = {}
            for key, values in cfgs_dict.items():
                # BASE CASE: key is associated to an atomic value
                if type(values) in [str, int, float, bool, None]:
                    result[key] = values
                # DICT CASE: call _dict_helper on this dict
                elif isinstance(values, dict):
                    result[key] = self._dict_helper(deepcopy(values))
                # LIST CASE: you should call _list_helper recursively on each
                # element
                elif isinstance(values, list):
                    if len(values) != 1:
                        raise ValueError(
                            f"Only one {key} value per "
                            "configuration if you do not "
                            "specify a sampling method"
                        )
                    result[key] = self._dict_helper(deepcopy(values[0]))

            yield deepcopy(result)

    def _dict_helper(self, configs: dict):
        r"""
        Recursively parses a dictionary

        Returns:
            A dictionary
        """
        if type(configs) in [str, int, float, bool, None]:
            return configs

        if SAMPLE_METHOD in configs:
            return self._sampler_helper(configs)

        for key, values in configs.items():
            # BASE CASE: key is associated to an atomic value
            if type(values) in [str, int, float, bool, None]:
                configs[key] = values
            elif isinstance(values, dict):
                configs[key] = self._dict_helper(configs[key])
            elif isinstance(values, list):
                if len(values) != 1:
                    raise ValueError(
                        f"Only one {key} value per "
                        "configuration if you do not "
                        "specify a sampling method"
                    )
                configs[key] = self._dict_helper(deepcopy(values[0]))

        return configs

    def _sampler_helper(self, configs: dict):
        r"""
        Samples possible hyperparameter(s) and returns it
        (them, in this case as a dict)

         Returns:
             A dictionary
        """
        method, args = configs[SAMPLE_METHOD], configs[ARGS]
        sampler = s2c(method)
        sample = sampler(*args)

        if isinstance(sample, dict):
            return self._dict_helper(sample)

        return sample

    def __iter__(self):
        """
        Iterates over all hyper-parameter configurations (generated just once)
        """
        if self.hparams is None:
            self.hparams = self._gen_configs()
        return iter(self.hparams)

    def __len__(self):
        """
        Computes the number of hyper-parameter configurations to try
        """
        if self.hparams is None:
            self.hparams = self._gen_configs()
        return len(self.hparams)

    def __getitem__(self, index: int):
        """
        Gets a specific configuration indexed by an id
        """
        if self.hparams is None:
            self.hparams = self._gen_configs()
        return self.hparams[index]
