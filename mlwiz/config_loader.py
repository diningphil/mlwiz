"""Hydra-style YAML configuration composition for MLWiz.

The loader intentionally implements the small, file-based part of Hydra that
is useful to MLWiz: ``defaults`` lists, config groups, package overrides and
``_self_`` composition ordering.  It does not require Hydra/OmegaConf and keeps
plain dictionaries as the public configuration type used by the rest of the
project.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from mlwiz.static import (
    ARGS,
    BAYES_SEARCH,
    GRID_SEARCH,
    RANDOM_SEARCH,
    SAMPLE_METHOD,
)


DEFAULTS = "defaults"
SELF = "_self_"
GLOBAL_PACKAGE = "_global_"
HERE_PACKAGE = "_here_"
_CONFIG_EXTENSIONS = (".yml", ".yaml")
_SEARCH_SECTIONS = (GRID_SEARCH, RANDOM_SEARCH, BAYES_SEARCH)
_REQUIRED_EXPERIMENT_SECTIONS = {
    "dataset",
    "resources",
    "reproducibility",
    "data_loading",
    "experiment",
}


class ConfigCompositionError(ValueError):
    """Raised when a modular YAML configuration cannot be composed."""


def load_config(config_file: str | Path) -> dict:
    """Load and compose an MLWiz YAML configuration.

    ``defaults`` entries follow Hydra's familiar forms::

        defaults:
          - runtime@_global_: local
          - dataset: mnist
          - _self_

    Search sections can contain their own ``defaults`` list. A selected search
    config-group file may contain one mapping or a list of mappings. Multiple
    mappings are combined as alternatives: grid search enumerates them, while
    random and Bayesian search treat them as a categorical ``choice`` dimension.

    Args:
        config_file: Main YAML file to compose.

    Returns:
        A plain, fully composed configuration dictionary.
    """
    path = Path(config_file).expanduser().resolve()
    composed, _ = _compose_file(path, root_dir=path.parent, stack=())
    if not isinstance(composed, dict):
        raise ConfigCompositionError(
            f"Main configuration '{path}' must contain a YAML mapping."
        )
    return composed


def load_experiment_config(config_file: str | Path) -> dict:
    """Compose and validate a version-1.7 experiment configuration."""
    return validate_experiment_config(load_config(config_file))


def validate_experiment_config(config: dict) -> dict:
    """Require the structured experiment schema introduced in MLWiz 1.7.0."""
    missing = sorted(_REQUIRED_EXPERIMENT_SECTIONS.difference(config))
    search_sections = [section for section in _SEARCH_SECTIONS if section in config]
    if missing or len(search_sections) != 1:
        details = []
        if missing:
            details.append("missing required top-level sections: " + ", ".join(missing))
        if len(search_sections) == 0:
            details.append(
                "missing one model-selection section: grid, random, or bayes"
            )
        elif len(search_sections) > 1:
            details.append(
                "define only one model-selection section: grid, random, or bayes"
            )
        raise ConfigCompositionError(
            "The experiment configuration schema changed in MLWiz 1.7.0; "
            + "; ".join(details)
            + ". Flat pre-1.7.0 experiment configurations are not supported."
        )

    non_mappings = [
        section
        for section in (
            *sorted(_REQUIRED_EXPERIMENT_SECTIONS),
            search_sections[0],
        )
        if not isinstance(config[section], dict)
    ]
    if non_mappings:
        raise ConfigCompositionError(
            "The experiment configuration schema changed in MLWiz 1.7.0; "
            "these sections must be mappings: " + ", ".join(non_mappings) + "."
        )
    return config


def _compose_file(
    path: Path, root_dir: Path, stack: tuple[Path, ...]
) -> tuple[Any, set[tuple[str, ...]]]:
    """Compose one file and return its value plus modular package metadata."""
    path = _existing_config_path(path)
    if path in stack:
        cycle = " -> ".join(str(item) for item in (*stack, path))
        raise ConfigCompositionError(f"Cyclic config defaults detected: {cycle}")

    try:
        with path.open("r", encoding="utf-8") as stream:
            document = yaml.safe_load(stream)
    except yaml.YAMLError as error:
        raise ConfigCompositionError(
            f"Invalid YAML in configuration '{path}': {error}"
        ) from error

    if document is None:
        document = {}
    if not isinstance(document, dict):
        return document, set()

    defaults = document.get(DEFAULTS, [])
    if defaults is None:
        defaults = []
    if not isinstance(defaults, list):
        raise ConfigCompositionError(f"'{DEFAULTS}' in '{path}' must be a list.")

    own_content = {
        key: deepcopy(value) for key, value in document.items() if key != DEFAULTS
    }
    own_content = _compose_search_defaults(
        own_content,
        containing_file=path,
        root_dir=root_dir,
        stack=(*stack, path),
    )
    entries = list(defaults)
    if SELF not in entries:
        entries.append(SELF)

    result: dict[str, Any] = {}
    modular_packages: set[tuple[str, ...]] = set()
    for entry in entries:
        if entry == SELF:
            modular_packages = {
                package
                for package in modular_packages
                if not _path_exists(own_content, package)
            }
            result = _deep_merge(result, own_content)
            continue

        contribution, packages = _compose_default(
            entry,
            containing_file=path,
            root_dir=root_dir,
            stack=(*stack, path),
        )
        result = _deep_merge(result, contribution)
        modular_packages.update(packages)

    return result, modular_packages


def _compose_default(
    entry: Any,
    containing_file: Path,
    root_dir: Path,
    stack: tuple[Path, ...],
) -> tuple[dict, set[tuple[str, ...]]]:
    """Resolve one ``defaults`` entry into a packaged contribution."""
    if isinstance(entry, str):
        reference, package = _split_package(entry)
        selected_path = _resolve_reference(reference, containing_file, root_dir)
        value, nested_packages = _compose_file(selected_path, root_dir, stack)
        if package is None:
            group_parts = _reference_group_parts(reference)
            package_parts = group_parts
        else:
            package_parts = _package_parts(package)
        return (
            _package_value(value, package_parts),
            {_prefix_package(package_parts, item) for item in nested_packages},
        )

    if not isinstance(entry, dict) or len(entry) != 1:
        raise ConfigCompositionError(
            f"Invalid defaults entry in '{containing_file}': {entry!r}. "
            "Expected '_self_', a config path, or a one-key config-group mapping."
        )

    raw_group, selection = next(iter(entry.items()))
    if not isinstance(raw_group, str):
        raise ConfigCompositionError(
            f"Config group names must be strings in '{containing_file}'."
        )
    group, package = _split_package(raw_group)
    if selection is None:
        return {}, set()
    selections = selection if isinstance(selection, list) else [selection]
    if not selections or not all(isinstance(item, str) for item in selections):
        raise ConfigCompositionError(
            f"Config group '{group}' in '{containing_file}' must select one "
            "or more config names."
        )

    values = []
    nested_package_sets = []
    for name in selections:
        reference = f"{group.rstrip('/')}/{name}"
        selected_path = _resolve_reference(reference, containing_file, root_dir)
        value, nested_packages = _compose_file(selected_path, root_dir, stack)
        values.append(value)
        nested_package_sets.append(nested_packages)

    package_parts = (
        _package_parts(package)
        if package is not None
        else tuple(part for part in group.strip("/").split("/") if part)
    )
    is_config_set = len(values) > 1 or any(isinstance(value, list) for value in values)
    if is_config_set and package_parts:
        value = _flatten_config_set(values)
    elif is_config_set:
        value = {}
        for selected in values:
            if not isinstance(selected, dict):
                raise ConfigCompositionError(
                    "Multiple configs placed at @_global_ or @_here_ must all "
                    "contain YAML mappings."
                )
            value = _deep_merge(value, selected)
    else:
        value = values[0]

    packages = {package_parts} if is_config_set and package_parts else set()
    for nested_packages in nested_package_sets:
        packages.update(
            _prefix_package(package_parts, item) for item in nested_packages
        )
    return _package_value(value, package_parts), packages


def _compose_search_defaults(
    content: dict,
    containing_file: Path,
    root_dir: Path,
    stack: tuple[Path, ...],
) -> dict:
    """Compose a local defaults list inside ``grid``, ``random``, or ``bayes``."""
    result = deepcopy(content)
    for search_section in _SEARCH_SECTIONS:
        search_content = result.get(search_section)
        if not isinstance(search_content, dict) or DEFAULTS not in search_content:
            continue
        defaults = search_content[DEFAULTS]
        if not isinstance(defaults, list):
            raise ConfigCompositionError(
                f"'{search_section}.{DEFAULTS}' in '{containing_file}' must be a list."
            )
        own_search_content = {
            key: deepcopy(value)
            for key, value in search_content.items()
            if key != DEFAULTS
        }
        entries = list(defaults)
        if SELF not in entries:
            entries.append(SELF)

        composed_search: dict[str, Any] = {}
        config_set_packages: set[tuple[str, ...]] = set()
        for entry in entries:
            if entry == SELF:
                config_set_packages = {
                    package
                    for package in config_set_packages
                    if not _path_exists(own_search_content, package)
                }
                composed_search = _deep_merge(composed_search, own_search_content)
                continue
            contribution, packages = _compose_default(
                entry,
                containing_file=containing_file,
                root_dir=root_dir,
                stack=stack,
            )
            composed_search = _deep_merge(composed_search, contribution)
            config_set_packages.update(packages)

        if search_section in (RANDOM_SEARCH, BAYES_SEARCH):
            for package in config_set_packages:
                if len(package) != 1:
                    continue
                value = composed_search.get(package[0])
                if isinstance(value, list):
                    composed_search[package[0]] = {
                        SAMPLE_METHOD: "mlwiz.evaluation.util.choice",
                        ARGS: value,
                    }
        result[search_section] = composed_search
    return result


def _resolve_reference(reference: str, containing_file: Path, root_dir: Path) -> Path:
    """Resolve a Hydra-style relative or absolute config reference."""
    if reference.startswith("/"):
        return root_dir / reference.lstrip("/")
    return containing_file.parent / reference


def _existing_config_path(path: Path) -> Path:
    """Find a config path, accepting extension-less defaults entries."""
    candidates = [path]
    if path.suffix not in _CONFIG_EXTENSIONS:
        candidates = [path.with_suffix(ext) for ext in _CONFIG_EXTENSIONS]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    shown = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Configuration file not found (tried: {shown}).")


def _split_package(reference: str) -> tuple[str, str | None]:
    """Split ``group@package`` syntax."""
    if "@" not in reference:
        return reference, None
    group, package = reference.rsplit("@", 1)
    if not group or not package:
        raise ConfigCompositionError(f"Invalid package override '{reference}'.")
    return group, package


def _reference_group_parts(reference: str) -> tuple[str, ...]:
    """Return the default package for a direct config reference."""
    clean = reference.strip("/")
    parts = tuple(part for part in clean.split("/") if part)
    return parts[:-1]


def _package_parts(package: str) -> tuple[str, ...]:
    """Convert a dotted package to a dictionary path."""
    if package in (GLOBAL_PACKAGE, HERE_PACKAGE):
        return ()
    parts = tuple(part for part in package.split(".") if part)
    if not parts:
        raise ConfigCompositionError(f"Invalid empty package '{package}'.")
    return parts


def _prefix_package(
    prefix: tuple[str, ...], package: tuple[str, ...]
) -> tuple[str, ...]:
    """Prefix nested package metadata when wrapping a composed config."""
    return (*prefix, *package)


def _package_value(value: Any, package: tuple[str, ...]) -> dict:
    """Wrap a value below its output package."""
    if not package:
        if not isinstance(value, dict):
            raise ConfigCompositionError(
                "A config placed at @_global_ or @_here_ must contain a YAML mapping."
            )
        return deepcopy(value)
    wrapped = deepcopy(value)
    for part in reversed(package):
        wrapped = {part: wrapped}
    return wrapped


def _flatten_config_set(values: list[Any]) -> list[Any]:
    """Concatenate configuration lists without flattening nested parameters."""
    flattened = []
    for value in values:
        if isinstance(value, list):
            flattened.extend(deepcopy(value))
        else:
            flattened.append(deepcopy(value))
    return flattened


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge mappings, with later values taking precedence."""
    result = deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _path_exists(value: Any, path: tuple[str, ...]) -> bool:
    """Return whether a mapping explicitly defines a package path."""
    current = value
    for part in path:
        if not isinstance(current, dict) or part not in current:
            return False
        current = current[part]
    return True
