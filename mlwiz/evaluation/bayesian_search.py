"""Bayesian optimization search-space generation.

The :class:`~mlwiz.evaluation.bayesian_search.BayesianSearch` class exposes an
``ask/tell`` interface used by the evaluator to adaptively propose the next
configuration based on previously observed validation results.
"""

from __future__ import annotations

import random
import warnings
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from mlwiz.evaluation.grid import Grid
from mlwiz.static import (
    ARGS,
    BAYES_SEARCH,
    BUDGET,
    CANDIDATE_POOL_SIZE,
    DATASET_CLASS,
    DATASET_GETTER,
    DATA_LOADER,
    DATA_LOADER_ARGS,
    DEVICE,
    EI_XI,
    EXPERIMENT,
    HIGHER_RESULTS_ARE_BETTER,
    MODEL_SELECTION_CRITERIA,
    RANDOM_STARTS,
    SAMPLE_METHOD,
    STORAGE_FOLDER,
    evaluate_every,
)


def _is_atomic(value: Any) -> bool:
    """Return ``True`` for scalar config values handled as leaves."""
    return isinstance(value, (str, int, float, bool)) or value is None


@dataclass
class _Dimension:
    """Internal representation of one optimizable search dimension."""

    path: Tuple[Any, ...]
    kind: str
    args: Tuple[Any, ...]


class BayesianSearch(Grid):
    r"""
    Bayesian optimization search.

    It reuses the same configuration syntax as random search for sampled
    parameters (``sample_method`` + ``args``), but proposes points
    sequentially through ``ask`` and receives feedback through ``tell``.
    """

    __search_type__ = BAYES_SEARCH

    def __init__(self, configs_dict: dict):
        """
        Initialize the search object.

        Args:
            configs_dict (dict): Root experiment configuration. It must define
                ``budget``, ``random_starts``, ``candidate_pool_size``,
                ``ei_xi`` and a ``bayes`` section.
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
        self._search_space_template = deepcopy(configs_dict[self.__search_type__])
        self._dimensions: List[_Dimension] = []
        self._collect_dimensions(self._search_space_template, path=())

        # BO hyper-parameters are explicitly user-configurable.
        self._random_starts = self._read_positive_int(
            configs_dict, RANDOM_STARTS
        )
        self._candidate_pool_size = self._read_positive_int(
            configs_dict, CANDIDATE_POOL_SIZE
        )
        self._ei_xi = self._read_non_negative_float(configs_dict, EI_XI)

        # Per-outer-fold adaptive state.
        self._outer_states: Dict[int, dict] = {}

        super().__init__(configs_dict)

    @staticmethod
    def _read_positive_int(configs_dict: dict, key: str) -> int:
        """Read and validate a required positive integer config value."""
        raw_value = configs_dict.get(key, None)
        if raw_value is None:
            raise KeyError(f"Missing required '{key}' key in configuration.")
        try:
            value = int(raw_value)
        except (TypeError, ValueError):
            raise ValueError(f"'{key}' must be an integer, got {raw_value!r}.")
        if value <= 0:
            raise ValueError(f"'{key}' must be > 0, got {value}.")
        return value

    @staticmethod
    def _read_non_negative_float(configs_dict: dict, key: str) -> float:
        """Read and validate a required non-negative float config value."""
        raw_value = configs_dict.get(key, None)
        if raw_value is None:
            raise KeyError(f"Missing required '{key}' key in configuration.")
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            raise ValueError(f"'{key}' must be a float, got {raw_value!r}.")
        if value < 0.0:
            raise ValueError(f"'{key}' must be >= 0, got {value}.")
        return value

    def _gen_configs(self) -> List[Optional[dict]]:
        """
        Allocate a fixed-size config list.

        For adaptive search, configs are materialized lazily via ``ask``.
        """
        return [None for _ in range(self.budget)]

    def start_outer(self, outer_k: int):
        """
        Reset adaptive state for one outer fold.

        The evaluator calls this before scheduling model selection for
        ``outer_k``.
        """
        base_seed = int(self.seed if self.seed is not None else 42)
        rng_seed = base_seed + (outer_k + 1) * 7919
        self._outer_states[outer_k] = {
            "rng": random.Random(rng_seed),
            "asked_ids": set(),
            "told_ids": set(),
            "x": [],
            "y": [],
            "features_by_id": {},
        }
        self.hparams = [None for _ in range(self.budget)]

    def ask(self, outer_k: int) -> Optional[Tuple[int, dict]]:
        """
        Propose the next configuration for ``outer_k``.

        Args:
            outer_k (int): Outer fold id.

        Returns:
            tuple[int, dict] | None: ``(config_id, config)`` or ``None`` when
            no further configuration is available.
        """
        if outer_k not in self._outer_states:
            self.start_outer(outer_k)
        state = self._outer_states[outer_k]

        config_id = None
        for idx in range(self.budget):
            if idx in state["asked_ids"]:
                continue
            config_id = idx
            break
        if config_id is None:
            return None

        if len(state["x"]) < self._random_starts:
            assignment = self._sample_random_assignment(state["rng"])
        else:
            assignment = self._propose_with_gp(state)

        feature = self._assignment_to_feature(assignment)
        state["features_by_id"][config_id] = feature
        state["asked_ids"].add(config_id)

        config = self._build_config(assignment)
        self.hparams[config_id] = deepcopy(config)
        return config_id, deepcopy(config)

    def tell(self, outer_k: int, config_id: int, objective: float):
        """
        Feed back the observed objective value of one configuration.

        Args:
            outer_k (int): Outer fold id.
            config_id (int): Config id returned by :meth:`ask`.
            objective (float): Scalar objective to maximize.
        """
        if outer_k not in self._outer_states:
            raise ValueError(f"Unknown outer fold state: {outer_k}")
        state = self._outer_states[outer_k]

        if config_id not in state["asked_ids"]:
            raise ValueError(
                f"Config id {config_id} was not proposed for outer fold {outer_k}."
            )
        if config_id in state["told_ids"]:
            return

        state["x"].append(state["features_by_id"][config_id])
        state["y"].append(float(objective))
        state["told_ids"].add(config_id)

    def _build_config(self, assignment: Dict[Tuple[Any, ...], Any]) -> dict:
        """Materialize one full experiment config from sampled parameter values."""
        config = self._materialize(self._search_space_template, (), assignment)
        shared_cfg = {
            DATASET_GETTER: self.dataset_getter,
            DATA_LOADER: self.data_loader_class,
            DATA_LOADER_ARGS: self.data_loader_args,
            DATASET_CLASS: self.dataset_class,
            STORAGE_FOLDER: self.storage_folder,
            DEVICE: self.device,
            EXPERIMENT: self.experiment,
            evaluate_every: self.evaluate_every,
        }
        if self.higher_results_are_better is not None:
            shared_cfg[HIGHER_RESULTS_ARE_BETTER] = self.higher_results_are_better
        if self.model_selection_criteria is not None:
            shared_cfg[MODEL_SELECTION_CRITERIA] = self.model_selection_criteria
        config.update(shared_cfg)
        return config

    def _materialize(
        self,
        node: Any,
        path: Tuple[Any, ...],
        assignment: Dict[Tuple[Any, ...], Any],
    ) -> Any:
        """Recursively replace ``sample_method`` nodes with sampled values."""
        if _is_atomic(node):
            return deepcopy(node)
        if isinstance(node, dict):
            if SAMPLE_METHOD in node:
                return deepcopy(assignment[path])
            return {
                key: self._materialize(value, path + (key,), assignment)
                for key, value in node.items()
            }
        if isinstance(node, list):
            if len(node) != 1:
                raise ValueError(
                    "Bayesian search expects lists used in search spaces to have exactly one element."
                )
            # Keep compatibility with RandomSearch semantics: single-item lists
            # are wrappers around nested structures, not literal list values.
            return self._materialize(node[0], path + (0,), assignment)
        raise TypeError(f"Unsupported search-space node type: {type(node)}")

    def _collect_dimensions(self, node: Any, path: Tuple[Any, ...]):
        """Walk the template and extract optimizable dimensions."""
        if _is_atomic(node):
            return
        if isinstance(node, dict):
            if SAMPLE_METHOD in node:
                self._dimensions.append(self._make_dimension(path, node))
                return
            for key, value in node.items():
                self._collect_dimensions(value, path + (key,))
            return
        if isinstance(node, list):
            if len(node) != 1:
                raise ValueError(
                    "Bayesian search expects lists used in search spaces to have exactly one element."
                )
            self._collect_dimensions(node[0], path + (0,))
            return
        raise TypeError(f"Unsupported search-space node type: {type(node)}")

    def _make_dimension(
        self, path: Tuple[Any, ...], node: Dict[str, Any]
    ) -> _Dimension:
        """Parse one ``sample_method`` block into an internal dimension spec."""
        if ARGS not in node:
            raise KeyError(
                f"Missing '{ARGS}' for Bayesian search dimension at path {path}."
            )
        method = str(node[SAMPLE_METHOD]).split(".")[-1]
        args = tuple(node[ARGS])

        if method == "choice":
            if len(args) == 0:
                raise ValueError(f"'choice' requires at least one value at {path}.")
            return _Dimension(path=path, kind=method, args=args)
        if method in {"uniform", "loguniform", "randint"}:
            if len(args) < 2:
                raise ValueError(
                    f"'{method}' requires at least two args at path {path}."
                )
            return _Dimension(path=path, kind=method, args=args)
        if method == "normal":
            if len(args) < 2:
                raise ValueError("'normal' requires (mu, sigma) arguments.")
            return _Dimension(path=path, kind=method, args=args)

        raise ValueError(
            "Unsupported Bayesian search sample method "
            f"'{node[SAMPLE_METHOD]}'. Supported: choice, randint, uniform, "
            "loguniform, normal."
        )

    def _sample_random_assignment(
        self, rng: random.Random
    ) -> Dict[Tuple[Any, ...], Any]:
        """Sample one random assignment over all dimensions."""
        assignment = {}
        for dim in self._dimensions:
            assignment[dim.path] = self._sample_dimension(dim, rng)
        return assignment

    def _sample_dimension(self, dim: _Dimension, rng: random.Random) -> Any:
        """Sample one value for a single dimension."""
        if dim.kind == "choice":
            return dim.args[rng.randrange(len(dim.args))]

        if dim.kind == "randint":
            low, high = int(dim.args[0]), int(dim.args[1])
            return int(rng.randint(low, high))

        if dim.kind == "uniform":
            low, high = float(dim.args[0]), float(dim.args[1])
            return float(rng.uniform(low, high))

        if dim.kind == "loguniform":
            low, high = float(dim.args[0]), float(dim.args[1])
            base = float(dim.args[2]) if len(dim.args) > 2 else 10.0
            low_log = np.log(low) / np.log(base)
            high_log = np.log(high) / np.log(base)
            return float(base ** rng.uniform(low_log, high_log))

        if dim.kind == "normal":
            mu, sigma = float(dim.args[0]), float(dim.args[1])
            return float(rng.normalvariate(mu, sigma))

        raise ValueError(f"Unknown dimension kind: {dim.kind}")

    def _assignment_to_feature(self, assignment: Dict[Tuple[Any, ...], Any]) -> List[float]:
        """Encode an assignment into a numeric feature vector for the GP."""
        if len(self._dimensions) == 0:
            return [0.0]

        feature = []
        for dim in self._dimensions:
            value = assignment[dim.path]
            if dim.kind == "choice":
                # Encode category as index to keep the GP input numeric.
                try:
                    idx = dim.args.index(value)
                except ValueError:
                    idx = 0
                feature.append(float(idx))
            elif dim.kind == "randint":
                feature.append(float(int(value)))
            else:
                feature.append(float(value))
        return feature

    def _feature_to_assignment(self, feature: Sequence[float]) -> Dict[Tuple[Any, ...], Any]:
        """Decode a numeric feature vector into a valid assignment."""
        assignment = {}
        for idx, dim in enumerate(self._dimensions):
            raw_value = float(feature[idx])
            if dim.kind == "choice":
                max_idx = len(dim.args) - 1
                chosen_idx = int(np.clip(np.rint(raw_value), 0, max_idx))
                assignment[dim.path] = dim.args[chosen_idx]
            elif dim.kind == "randint":
                low, high = int(dim.args[0]), int(dim.args[1])
                assignment[dim.path] = int(np.clip(np.rint(raw_value), low, high))
            elif dim.kind == "uniform":
                low, high = float(dim.args[0]), float(dim.args[1])
                assignment[dim.path] = float(np.clip(raw_value, low, high))
            elif dim.kind == "loguniform":
                low, high = float(dim.args[0]), float(dim.args[1])
                assignment[dim.path] = float(np.clip(raw_value, low, high))
            elif dim.kind == "normal":
                assignment[dim.path] = float(raw_value)
            else:
                raise ValueError(f"Unknown dimension kind: {dim.kind}")
        return assignment

    def _propose_with_gp(self, state: dict) -> Dict[Tuple[Any, ...], Any]:
        """
        Fit a GP surrogate and pick the candidate maximizing Expected Improvement.

        If fitting fails for numerical reasons, it falls back to random sampling.
        """
        if len(self._dimensions) == 0:
            return {}
        if len(state["x"]) == 0:
            return self._sample_random_assignment(state["rng"])

        x_obs = np.asarray(state["x"], dtype=float)
        y_obs = np.asarray(state["y"], dtype=float)
        if x_obs.ndim == 1:
            x_obs = x_obs.reshape(-1, 1)

        kernel = (
            ConstantKernel(1.0, (1e-3, 1e3)) * Matern(nu=2.5)
            + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-8, 1e1))
        )
        gp = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            n_restarts_optimizer=0,
        )

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gp.fit(x_obs, y_obs)
        except Exception:
            return self._sample_random_assignment(state["rng"])

        candidate_assignments = [
            self._sample_random_assignment(state["rng"])
            for _ in range(self._candidate_pool_size)
        ]
        x_cand = np.asarray(
            [self._assignment_to_feature(cand) for cand in candidate_assignments],
            dtype=float,
        )
        if x_cand.ndim == 1:
            x_cand = x_cand.reshape(-1, 1)

        mu, std = gp.predict(x_cand, return_std=True)
        std = np.asarray(std, dtype=float)
        best = float(np.max(y_obs))

        improvement = mu - best - self._ei_xi
        safe_std = np.maximum(std, 1e-12)
        z = improvement / safe_std
        ei = improvement * norm.cdf(z) + safe_std * norm.pdf(z)
        ei[std <= 1e-12] = 0.0

        best_idx = int(np.argmax(ei))
        return self._feature_to_assignment(x_cand[best_idx])
