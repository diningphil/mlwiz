"""Browser dashboard for inspecting MLWiz experiment metrics.

The dashboard mirrors MLWiz's result hierarchy and reads the
``metrics_data.torch`` artifacts produced by :class:`~mlwiz.training.callback.plotter.Plotter`.
It intentionally uses Python's standard HTTP server so it can ship with MLWiz
without adding a web-framework dependency.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import mimetypes
import pickle
import re
import statistics
import sys
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from pydoc import locate
from typing import Any, Iterable, Optional
from urllib.parse import parse_qs, unquote, urlparse

import torch

from mlwiz.evaluation.config import Config
from mlwiz.static import (
    BEST_CHECKPOINT_FILENAME,
    BEST_EPOCH,
    EPOCH,
    LAST_CHECKPOINT_FILENAME,
    MODEL_GRAPH_INPUT_SPEC_FILENAME,
    MODEL_ASSESSMENT,
    MODEL_MANIFEST_FILENAME,
    MODEL_STATE,
)


_OUTER_FOLD_PATTERN = re.compile(r"OUTER_FOLD_(\d+)$")
_CONFIG_PATTERN = re.compile(r"config_(\d+)$")
_INNER_FOLD_PATTERN = re.compile(r"INNER_FOLD_(\d+)$")
_RUN_PATTERN = re.compile(r"run_?(\d+)$")
_FINAL_RUN_PATTERN = re.compile(r"final_run_?(\d+)$")
_ASSET_DIRECTORY = Path(__file__).with_name("web_assets")
_MAX_METRICS_FILES_PER_SELECTION = 256
_MAX_GRAPH_NODES = 600
_MAX_GRAPH_INPUT_SPEC_BYTES = 64 * 1024
_MEBIBYTE = 1024 * 1024
_DEFAULT_CACHE_BYTES = 256 * _MEBIBYTE
_MAX_CACHE_BYTES = 64 * 1024 * _MEBIBYTE
_GRAPH_MODES = {"architecture", "operators"}
_FILTER_RESULT_PATTERN = re.compile(r"^avg_(training|validation)_(.+)$")
_ELAPSED_PATTERN = re.compile(
    r"Total time of the experiment in seconds: (\d+(?:\.\d+)?)"
)
_EXPORT_DTYPES = {
    name: dtype
    for name, dtype in (
        ("bool", torch.bool),
        ("uint8", torch.uint8),
        ("int8", torch.int8),
        ("int16", torch.int16),
        ("int32", torch.int32),
        ("int64", torch.int64),
        ("float16", torch.float16),
        ("bfloat16", torch.bfloat16),
        ("float32", torch.float32),
        ("float64", torch.float64),
        ("complex64", torch.complex64),
        ("complex128", torch.complex128),
    )
}


def _numbered_directories(
    folder: Path, pattern: re.Pattern[str]
) -> list[tuple[int, Path]]:
    """Return matching direct child directories sorted by numeric suffix."""
    if not folder.is_dir():
        return []
    matches = []
    for path in folder.iterdir():
        match = pattern.fullmatch(path.name)
        if path.is_dir() and match:
            matches.append((int(match.group(1)), path))
    return sorted(matches, key=lambda item: item[0])


def _read_json(path: Path) -> Optional[Any]:
    """Read a JSON artifact, returning ``None`` while it is absent or partial."""
    try:
        with path.open("r", encoding="utf-8") as stream:
            return _json_safe(json.load(stream))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def _file_signature(path: Path) -> Optional[tuple[int, int]]:
    """Return the change signature of an optional dashboard artifact."""
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None
    return stat.st_mtime_ns, stat.st_size


def _json_safe(value: Any) -> Any:
    """Convert common metric and metadata values to JSON-compatible objects."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, torch.Tensor):
        return _json_safe(value.detach().cpu().tolist())
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except (TypeError, ValueError, RuntimeError):
            pass
    return str(value)


def _numeric_series(value: Any) -> Optional[list[Optional[float]]]:
    """Normalize a stored metric history to finite floats and gaps."""
    value = _json_safe(value)
    if not isinstance(value, list):
        value = [value]

    output: list[Optional[float]] = []
    for item in value:
        if item is None:
            output.append(None)
            continue
        if isinstance(item, bool):
            output.append(float(item))
            continue
        try:
            number = float(item)
        except (TypeError, ValueError):
            return None
        output.append(number if math.isfinite(number) else None)
    return output


def _deep_size(value: Any, seen: Optional[set[int]] = None) -> int:
    """Estimate the resident size of a normalized cache value in bytes."""
    if seen is None:
        seen = set()
    identifier = id(value)
    if identifier in seen:
        return 0
    seen.add(identifier)
    size = sys.getsizeof(value)
    if isinstance(value, dict):
        size += sum(
            _deep_size(key, seen) + _deep_size(item, seen)
            for key, item in value.items()
        )
    elif isinstance(value, (list, tuple, set, frozenset)):
        size += sum(_deep_size(item, seen) for item in value)
    return size


class MetricsCache:
    """Thread-safe, memory-bounded LRU cache for normalized metric series."""

    def __init__(self, max_bytes: int = _DEFAULT_CACHE_BYTES):
        """Initialize an empty cache with a byte ceiling."""
        self._max_bytes = max_bytes
        self._used_bytes = 0
        self._entries: collections.OrderedDict[Path, dict[str, Any]] = (
            collections.OrderedDict()
        )
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._invalidations = 0
        self._skipped = 0

    def get(
        self, path: Path, signature: tuple[Any, ...]
    ) -> Optional[Any]:
        """Return a fresh cached entry and update LRU order."""
        with self._lock:
            entry = self._entries.get(path)
            if entry is None:
                self._misses += 1
                return None
            if entry["signature"] != signature:
                self._remove(path)
                self._invalidations += 1
                self._misses += 1
                return None
            self._entries.move_to_end(path)
            self._hits += 1
            return entry["series"]

    def put(
        self,
        path: Path,
        signature: tuple[Any, ...],
        series: Any,
    ) -> bool:
        """Cache normalized series unless the entry exceeds the memory limit."""
        size = _deep_size(series) + _deep_size(path) + _deep_size(signature)
        with self._lock:
            if self._max_bytes == 0 or size > self._max_bytes:
                self._skipped += 1
                return False
            self._remove(path)
            self._entries[path] = {
                "signature": signature,
                "series": series,
                "size": size,
            }
            self._used_bytes += size
            self._evict_to_limit()
            return path in self._entries

    def configure(self, max_bytes: int) -> dict[str, Any]:
        """Set the memory ceiling and immediately evict excess entries."""
        if not isinstance(max_bytes, int) or not 0 <= max_bytes <= _MAX_CACHE_BYTES:
            raise ValueError(
                f"Cache limit must be between 0 and {_MAX_CACHE_BYTES // _MEBIBYTE} MB."
            )
        with self._lock:
            self._max_bytes = max_bytes
            self._evict_to_limit()
            return self.stats()

    def clear(self) -> dict[str, Any]:
        """Remove every cached metric entry and reset lifecycle counters."""
        with self._lock:
            self._entries.clear()
            self._used_bytes = 0
            self._hits = 0
            self._misses = 0
            self._evictions = 0
            self._invalidations = 0
            self._skipped = 0
            return self.stats()

    def stats(self) -> dict[str, Any]:
        """Return cache capacity, usage, and lifecycle counters."""
        with self._lock:
            return {
                "entries": len(self._entries),
                "used_bytes": self._used_bytes,
                "max_bytes": self._max_bytes,
                "used_mb": round(self._used_bytes / _MEBIBYTE, 2),
                "max_mb": round(self._max_bytes / _MEBIBYTE, 2),
                "hits": self._hits,
                "misses": self._misses,
                "evictions": self._evictions,
                "invalidations": self._invalidations,
                "skipped": self._skipped,
            }

    def _remove(self, path: Path) -> None:
        """Remove one entry; the caller must hold ``_lock``."""
        entry = self._entries.pop(path, None)
        if entry is not None:
            self._used_bytes -= entry["size"]

    def _evict_to_limit(self) -> None:
        """Evict least-recently-used entries; the caller must hold ``_lock``."""
        while self._used_bytes > self._max_bytes and self._entries:
            _, entry = self._entries.popitem(last=False)
            self._used_bytes -= entry["size"]
            self._evictions += 1


class ResultsRepository:
    """Read-only view over one experiment or a directory of experiments."""

    def __init__(
        self,
        logdir: str | Path,
        cache_max_bytes: int = _DEFAULT_CACHE_BYTES,
    ):
        """Create a repository rooted at ``logdir``.

        Passing a ``MODEL_ASSESSMENT`` directory is treated as passing its
        parent experiment directory, which keeps all API paths inside the root.
        """
        root = Path(logdir).expanduser().resolve()
        self.root = root.parent if root.name == MODEL_ASSESSMENT else root
        self.metrics_cache = MetricsCache(cache_max_bytes)

    def cache_status(self) -> dict[str, Any]:
        """Return current metric-cache statistics."""
        return self.metrics_cache.stats()

    def configure_cache(self, max_mb: float) -> dict[str, Any]:
        """Set the metric-cache ceiling in mebibytes."""
        if isinstance(max_mb, bool) or not isinstance(max_mb, (int, float)):
            raise ValueError("Cache limit must be a number of MB.")
        if not math.isfinite(max_mb):
            raise ValueError("Cache limit must be finite.")
        max_bytes = int(max_mb * _MEBIBYTE)
        return self.metrics_cache.configure(max_bytes)

    def reset_cache(self) -> dict[str, Any]:
        """Clear normalized metric histories while preserving the size limit."""
        return self.metrics_cache.clear()

    def model_graph(
        self,
        relative_path: str,
        checkpoint_kind: str = "auto",
        graph_mode: str = "architecture",
    ) -> dict[str, Any]:
        """Lazily build an architecture or operator graph from a checkpoint."""
        (
            run_folder,
            status,
            checkpoint_paths,
            available_checkpoints,
            loadable_checkpoints,
            cache_limit,
        ) = self._model_graph_context(relative_path)
        if checkpoint_kind not in {"auto", "best", "last"}:
            raise ValueError("Checkpoint must be 'auto', 'best', or 'last'.")
        if graph_mode not in _GRAPH_MODES:
            raise ValueError("Graph mode must be 'architecture' or 'operators'.")
        preferred = (
            [BEST_CHECKPOINT_FILENAME, LAST_CHECKPOINT_FILENAME]
            if status == "completed"
            else [LAST_CHECKPOINT_FILENAME, BEST_CHECKPOINT_FILENAME]
        )
        if checkpoint_kind == "auto":
            checkpoint = next(
                (
                    run_folder / name
                    for name in preferred
                    if (run_folder / name).is_file()
                ),
                None,
            )
        else:
            checkpoint = checkpoint_paths[checkpoint_kind]
            if not checkpoint.is_file():
                raise ValueError(
                    f"The {checkpoint_kind} checkpoint is not available for this run."
                )
        if checkpoint is None:
            raise ValueError("No best or last checkpoint is available for this run.")

        manifest_path = run_folder / MODEL_MANIFEST_FILENAME
        checkpoint_stat = checkpoint.stat()
        if checkpoint_stat.st_size > cache_limit:
            checkpoint_mb = checkpoint_stat.st_size / _MEBIBYTE
            limit_mb = cache_limit / _MEBIBYTE
            raise ValueError(
                f"Checkpoint is {checkpoint_mb:.2f} MB, exceeding the "
                f"{limit_mb:.2f} MB cache limit; model graph loading was skipped."
            )
        input_spec_path = run_folder / MODEL_GRAPH_INPUT_SPEC_FILENAME
        manifest_signature = _file_signature(manifest_path)
        input_spec_signature = (
            _file_signature(input_spec_path)
            if graph_mode == "operators"
            else None
        )
        signature = (
            graph_mode,
            checkpoint_stat.st_mtime_ns,
            checkpoint_stat.st_size,
            manifest_signature,
            input_spec_signature,
        )
        cache_key = self._model_graph_cache_key(checkpoint, graph_mode)
        graph = self.metrics_cache.get(cache_key, signature)
        cache_hit = graph is not None
        if graph is None:
            try:
                graph = self._load_checkpoint_graph(
                    checkpoint,
                    manifest_path,
                    input_spec_path,
                    graph_mode,
                    cache_limit,
                )
            except (
                RuntimeError,
                TypeError,
                EOFError,
                pickle.UnpicklingError,
            ) as error:
                raise ValueError(f"Could not load checkpoint: {error}") from error
            final_stat = checkpoint.stat()
            if (
                graph_mode,
                final_stat.st_mtime_ns,
                final_stat.st_size,
                _file_signature(manifest_path),
                (
                    _file_signature(input_spec_path)
                    if graph_mode == "operators"
                    else None
                ),
            ) == signature:
                self.metrics_cache.put(cache_key, signature, graph)

        return {
            **graph,
            "graph_mode": graph_mode,
            "run": self._relative(run_folder),
            "run_state": status,
            "checkpoint": {
                "kind": "best"
                if checkpoint.name == BEST_CHECKPOINT_FILENAME
                else "last",
                "path": self._relative(checkpoint),
                "modified_at": checkpoint_stat.st_mtime,
                "cache_hit": cache_hit,
                "requested": checkpoint_kind,
                "available": available_checkpoints,
                "loadable": loadable_checkpoints,
            },
            "cache": self.cache_status(),
        }

    def model_graph_info(self, relative_path: str) -> dict[str, Any]:
        """Return checkpoint availability without loading checkpoint contents."""
        (
            run_folder,
            status,
            checkpoint_paths,
            available,
            loadable,
            cache_limit,
        ) = self._model_graph_context(relative_path)
        preference = ["best", "last"] if status == "completed" else ["last", "best"]
        default_kind = next((kind for kind in preference if kind in available), None)
        operators_available, operators_reason = self._operator_graph_available(
            run_folder, cache_limit
        )
        return {
            "run": self._relative(run_folder),
            "run_state": status,
            "checkpoint": {
                "kind": default_kind,
                "requested": "auto",
                "available": available,
                "loadable": loadable,
                "sizes_mb": {
                    kind: round(checkpoint_paths[kind].stat().st_size / _MEBIBYTE, 2)
                    for kind in available
                },
            },
            "modes": {
                "architecture": {"available": True},
                "operators": {
                    "available": operators_available,
                    "reason": operators_reason,
                },
            },
            "cache_max_mb": round(cache_limit / _MEBIBYTE, 2),
        }

    @staticmethod
    def _model_graph_cache_key(checkpoint: Path, graph_mode: str) -> Path:
        """Keep architecture and exported-operator cache entries independent."""
        return checkpoint.with_name(
            f".{checkpoint.name}.{graph_mode}.model-graph-cache"
        )

    def _operator_graph_available(
        self, run_folder: Path, cache_limit: int
    ) -> tuple[bool, Optional[str]]:
        """Check whether a run has the safe artifacts needed for export."""
        manifest_path = run_folder / MODEL_MANIFEST_FILENAME
        if not isinstance(_read_json(manifest_path), dict):
            return False, "Operators view needs the run's model manifest."
        try:
            self._export_input_spec_details(
                run_folder / MODEL_GRAPH_INPUT_SPEC_FILENAME, cache_limit
            )
        except ValueError as error:
            return False, str(error)
        return True, None

    def _model_graph_context(
        self, relative_path: str
    ) -> tuple[Path, str, dict[str, Path], list[str], list[str], int]:
        """Resolve one run and its checkpoint availability policy."""
        run_folder = self.resolve(relative_path)
        if not run_folder.is_dir() or not (
            _RUN_PATTERN.fullmatch(run_folder.name)
            or _FINAL_RUN_PATTERN.fullmatch(run_folder.name)
        ):
            raise ValueError("Select a model-selection or final run.")
        observed_status = self._run_status(run_folder)["state"]
        if any(run_folder.glob("run_*_results.dill")):
            status = "completed"
        elif observed_status == "failed":
            status = "failed"
        else:
            status = "running"
        checkpoint_paths = {
            "best": run_folder / BEST_CHECKPOINT_FILENAME,
            "last": run_folder / LAST_CHECKPOINT_FILENAME,
        }
        available = [
            kind for kind in ("best", "last") if checkpoint_paths[kind].is_file()
        ]
        cache_limit = self.metrics_cache.stats()["max_bytes"]
        loadable = [
            kind
            for kind in available
            if checkpoint_paths[kind].stat().st_size <= cache_limit
        ]
        return (
            run_folder,
            status,
            checkpoint_paths,
            available,
            loadable,
            cache_limit,
        )

    def _load_checkpoint_graph(
        self,
        checkpoint: Path,
        manifest_path: Path,
        input_spec_path: Path,
        graph_mode: str,
        cache_limit: int,
    ) -> dict[str, Any]:
        """Load one checkpoint and build the requested graph on CPU."""
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or not isinstance(
            payload.get(MODEL_STATE), dict
        ):
            raise ValueError("Checkpoint does not contain a model state dictionary.")
        model_state = payload[MODEL_STATE]
        manifest = _read_json(manifest_path)
        warning = None
        if graph_mode == "operators":
            if not isinstance(manifest, dict):
                raise ValueError(
                    "Operators view needs a model manifest; use Architecture "
                    "for runs created before model manifests were recorded."
                )
            try:
                model = self._reconstruct_model(manifest, model_state)
            except Exception as error:
                raise ValueError(
                    "Operators view could not reconstruct the checkpoint model: "
                    f"{error}"
                ) from error
            example_input, input_summary = self._export_example_input(
                input_spec_path, cache_limit
            )
            try:
                graph = self._operator_graph(model, example_input, input_summary)
            except Exception as error:
                raise ValueError(
                    "torch.export could not capture this model with the recorded "
                    f"input shape: {error}"
                ) from error
        elif isinstance(manifest, dict):
            try:
                model = self._reconstruct_model(manifest, model_state)
                graph = self._module_graph(model)
            except Exception as error:  # custom reconstruction is best-effort
                graph = self._state_dict_graph(model_state, manifest)
                warning = (
                    "The model class could not be reconstructed; showing the "
                    f"checkpoint parameter hierarchy instead ({error})."
                )
        else:
            graph = self._state_dict_graph(model_state, None)
            warning = (
                "This run predates model manifests; showing the checkpoint "
                "parameter hierarchy."
            )

        epoch = payload.get(
            BEST_EPOCH if checkpoint.name == BEST_CHECKPOINT_FILENAME else EPOCH
        )
        try:
            display_epoch = int(epoch) + 1 if epoch is not None else None
        except (TypeError, ValueError):
            display_epoch = None
        graph.update(
            {
                "epoch": display_epoch,
                "warning": warning,
            }
        )
        return graph

    @staticmethod
    def _export_input_spec_details(
        input_spec_path: Path, cache_limit: int
    ) -> tuple[tuple[int, ...], torch.dtype, int]:
        """Validate a data-free input specification without allocating it."""
        try:
            spec_size = input_spec_path.stat().st_size
        except FileNotFoundError as error:
            raise ValueError(
                "Operators view needs an input shape recorded by a new MLWiz run."
            ) from error
        if spec_size > _MAX_GRAPH_INPUT_SPEC_BYTES:
            raise ValueError("The recorded operator-input specification is invalid.")
        specification = _read_json(input_spec_path)
        if not isinstance(specification, dict):
            raise ValueError("The recorded operator-input specification is invalid.")
        if specification.get("kind") != "tensor":
            input_type = specification.get("type")
            suffix = f" ({input_type})" if isinstance(input_type, str) else ""
            raise ValueError(
                "Operators view currently requires a tensor model input"
                f"{suffix}; this run needs a custom export adapter."
            )

        shape_payload = specification.get("shape")
        if not isinstance(shape_payload, list) or len(shape_payload) > 32:
            raise ValueError("The recorded operator-input shape is invalid.")
        shape = []
        number_elements = 1
        for dimension in shape_payload:
            if (
                isinstance(dimension, bool)
                or not isinstance(dimension, int)
                or dimension < 0
            ):
                raise ValueError("The recorded operator-input shape is invalid.")
            shape.append(dimension)
            number_elements *= dimension

        dtype_name = specification.get("dtype")
        if isinstance(dtype_name, str):
            dtype_name = dtype_name.removeprefix("torch.")
        dtype = _EXPORT_DTYPES.get(dtype_name)
        if dtype is None:
            raise ValueError("The recorded operator-input dtype is unsupported.")
        byte_count = number_elements * torch.empty((), dtype=dtype).element_size()
        if byte_count > cache_limit:
            size_mb = byte_count / _MEBIBYTE
            limit_mb = cache_limit / _MEBIBYTE
            raise ValueError(
                f"The synthetic export input is {size_mb:.2f} MB, exceeding "
                f"the {limit_mb:.2f} MB cache limit."
            )
        return tuple(shape), dtype, byte_count

    def _export_example_input(
        self, input_spec_path: Path, cache_limit: int
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Create a CPU example tensor from safe shape and dtype metadata."""
        shape, dtype, byte_count = self._export_input_spec_details(
            input_spec_path, cache_limit
        )
        return torch.zeros(shape, dtype=dtype, device="cpu"), {
            "shape": list(shape),
            "dtype": str(dtype),
            "bytes": byte_count,
        }

    def _operator_graph(
        self,
        model: torch.nn.Module,
        example_input: torch.Tensor,
        input_summary: dict[str, Any],
    ) -> dict[str, Any]:
        """Trace one model execution into a low-level ``torch.export`` DAG."""
        model.eval()
        with torch.no_grad():
            exported = torch.export.export(
                model, (example_input,), strict=False
            )
        graph_nodes = list(exported.graph_module.graph.nodes)
        visible_graph_nodes = graph_nodes[:_MAX_GRAPH_NODES]
        visible_ids = {item.name for item in visible_graph_nodes}
        input_metadata = self._export_graph_input_metadata(exported)
        parameters = {
            name: (value.numel(), value.requires_grad)
            for name, value in model.named_parameters()
        }
        nodes = []
        for item in visible_graph_nodes:
            target = str(item.target)
            metadata = input_metadata.get(item.name, {})
            parameter_name = metadata.get("target")
            parameter_count, trainable = parameters.get(
                parameter_name, (0, False)
            )
            nodes.append(
                {
                    "id": item.name,
                    "label": self._operator_node_label(item.op, target),
                    "type": self._operator_node_type(
                        item.op, target, metadata.get("kind")
                    ),
                    "op": item.op,
                    "target": target,
                    "module_path": self._operator_module_path(item.meta),
                    "parameters": parameter_count,
                    "trainable_parameters": (
                        parameter_count if trainable else 0
                    ),
                    "tensors": self._operator_tensor_records(
                        item.meta.get("val")
                    ),
                }
            )
        edges = [
            {"source": dependency.name, "target": item.name}
            for item in visible_graph_nodes
            for dependency in getattr(item, "all_input_nodes", ())
            if dependency.name in visible_ids
        ]
        total_parameters = sum(value.numel() for value in model.parameters())
        return {
            "graph_kind": "torch.export ATen operators",
            "nodes": nodes,
            "edges": edges,
            "summary": {
                "model_class": type(model).__name__,
                "parameters": total_parameters,
                "nodes": len(graph_nodes),
                "visible_nodes": len(nodes),
                "operators": sum(
                    item.op in {"call_function", "call_method", "call_module"}
                    for item in graph_nodes
                ),
                "truncated": len(graph_nodes) > len(nodes),
                "input": input_summary,
            },
        }

    @staticmethod
    def _export_graph_input_metadata(exported: Any) -> dict[str, dict[str, str]]:
        """Map exported placeholder names to parameter/user-input metadata."""
        metadata = {}
        signature = getattr(exported, "graph_signature", None)
        for specification in getattr(signature, "input_specs", ()):
            argument = getattr(specification, "arg", None)
            name = getattr(argument, "name", None)
            if not isinstance(name, str):
                continue
            kind = str(getattr(specification, "kind", "")).rsplit(".", 1)[-1]
            target = getattr(specification, "target", None)
            metadata[name] = {
                "kind": kind.lower(),
                "target": str(target) if target is not None else "",
            }
        return metadata

    @staticmethod
    def _operator_node_label(operation: str, target: str) -> str:
        """Create a compact, stable label for an exported graph node."""
        if operation == "placeholder":
            return target.removeprefix("p_").removeprefix("b_")
        if operation == "output":
            return "output"
        return target.removeprefix("torch.ops.")

    @staticmethod
    def _operator_node_type(
        operation: str, target: str, input_kind: Optional[str]
    ) -> str:
        """Describe the role of an exported node for the inspector."""
        if input_kind:
            return input_kind.replace("_", " ").title()
        if operation == "output":
            return "Graph output"
        if target.startswith("aten.") or target.startswith("torch.ops.aten."):
            return "ATen operator"
        return operation.replace("_", " ").title()

    @staticmethod
    def _operator_module_path(metadata: dict[str, Any]) -> Optional[str]:
        """Return the most specific originating module recorded by export."""
        module_stack = metadata.get("nn_module_stack")
        if not isinstance(module_stack, dict):
            return None
        for value in reversed(list(module_stack.values())):
            if isinstance(value, (tuple, list)) and value and value[0]:
                return str(value[0])
        return None

    @staticmethod
    def _operator_tensor_records(value: Any) -> list[dict[str, Any]]:
        """Normalize exported fake-tensor metadata without tensor allocation."""
        records = []

        def visit(item: Any, name: str) -> None:
            if len(records) >= 16:
                return
            if isinstance(item, (tuple, list)):
                for index, nested in enumerate(item):
                    visit(nested, f"{name}.{index}")
                return
            shape = getattr(item, "shape", None)
            if shape is None:
                return
            try:
                normalized_shape = [int(dimension) for dimension in shape]
            except (TypeError, ValueError):
                return
            records.append(
                {
                    "name": name,
                    "shape": normalized_shape,
                    "dtype": str(getattr(item, "dtype", "")),
                }
            )

        visit(value, "output")
        return records

    def _reconstruct_model(
        self, manifest: dict[str, Any], model_state: dict[str, Any]
    ) -> torch.nn.Module:
        """Instantiate a manifest model on CPU and restore checkpoint weights."""
        model_path = manifest.get("model")
        config_payload = manifest.get("config")
        if not isinstance(model_path, str) or not isinstance(config_payload, dict):
            raise ValueError("Model manifest is incomplete.")
        config_payload = {**config_payload, "device": "cpu"}
        dim_input = manifest.get("dim_input_features")
        if isinstance(dim_input, list):
            dim_input = tuple(dim_input)
        dim_target = manifest.get("dim_target")
        model_class = locate(model_path)
        if model_class is None:
            raise ImportError(f"Unknown model class '{model_path}'.")
        model = model_class(
            dim_input_features=dim_input,
            dim_target=dim_target,
            config=Config(config_payload),
        )
        model.load_state_dict(model_state)
        model.eval()
        return model

    def _module_graph(self, model: torch.nn.Module) -> dict[str, Any]:
        """Build a hierarchy graph from the reconstructed module topology."""
        nodes = []
        edges = []
        total_parameters = sum(
            parameter.numel() for parameter in model.parameters()
        )
        named_modules = list(model.named_modules())
        total_nodes = len(named_modules)
        for name, module in named_modules[:_MAX_GRAPH_NODES]:
            node_id = name or "__root__"
            parent_name = name.rpartition(".")[0] if name else ""
            parent_id = parent_name or "__root__"
            direct_parameters = list(module.named_parameters(recurse=False))
            direct_buffers = list(module.named_buffers(recurse=False))
            nodes.append(
                {
                    "id": node_id,
                    "label": name.rsplit(".", 1)[-1] if name else type(module).__name__,
                    "type": type(module).__name__,
                    "depth": name.count(".") + (1 if name else 0),
                    "parameters": sum(item.numel() for _, item in direct_parameters),
                    "trainable_parameters": sum(
                        item.numel()
                        for _, item in direct_parameters
                        if item.requires_grad
                    ),
                    "tensors": [
                        {"name": key, "shape": list(value.shape)}
                        for key, value in (direct_parameters + direct_buffers)[:16]
                    ],
                }
            )
            if name:
                edges.append({"source": parent_id, "target": node_id})
        visible = {node["id"] for node in nodes}
        edges = [
            edge
            for edge in edges
            if edge["source"] in visible and edge["target"] in visible
        ]
        return {
            "graph_kind": "module",
            "nodes": nodes,
            "edges": edges,
            "summary": {
                "model_class": type(model).__name__,
                "parameters": total_parameters,
                "nodes": total_nodes,
                "visible_nodes": len(nodes),
                "truncated": total_nodes > len(nodes),
            },
        }

    def _state_dict_graph(
        self,
        model_state: dict[str, Any],
        manifest: Optional[dict[str, Any]],
    ) -> dict[str, Any]:
        """Build a backward-compatible hierarchy directly from state keys."""
        model_name = (
            str(manifest.get("model", "Model")).rsplit(".", 1)[-1]
            if manifest
            else "Model"
        )
        records: dict[str, dict[str, Any]] = {
            "__root__": {
                "id": "__root__",
                "label": model_name,
                "type": "Checkpoint model",
                "depth": 0,
                "parameters": 0,
                "trainable_parameters": 0,
                "tensors": [],
            }
        }
        total_parameters = 0
        for key, value in model_state.items():
            if not torch.is_tensor(value):
                continue
            parts = str(key).split(".")
            module_parts = parts[:-1]
            parent = "__root__"
            for depth in range(1, len(module_parts) + 1):
                node_id = ".".join(module_parts[:depth])
                records.setdefault(
                    node_id,
                    {
                        "id": node_id,
                        "label": module_parts[depth - 1],
                        "type": "Checkpoint module",
                        "depth": depth,
                        "parameters": 0,
                        "trainable_parameters": 0,
                        "tensors": [],
                        "parent": parent,
                    },
                )
                parent = node_id
            record = records[parent]
            count = value.numel()
            record["parameters"] += count
            record["trainable_parameters"] += count
            if len(record["tensors"]) < 16:
                record["tensors"].append(
                    {"name": parts[-1], "shape": list(value.shape)}
                )
            total_parameters += count

        all_nodes = list(records.values())
        nodes = all_nodes[:_MAX_GRAPH_NODES]
        visible = {node["id"] for node in nodes}
        edges = [
            {"source": node["parent"], "target": node["id"]}
            for node in nodes
            if node.get("parent") in visible
        ]
        for node in nodes:
            node.pop("parent", None)
        return {
            "graph_kind": "checkpoint hierarchy",
            "nodes": nodes,
            "edges": edges,
            "summary": {
                "model_class": model_name,
                "parameters": total_parameters,
                "nodes": len(all_nodes),
                "visible_nodes": len(nodes),
                "truncated": len(all_nodes) > len(nodes),
            },
        }

    def experiment_filter_data(self, relative_path: str) -> dict[str, Any]:
        """Return lazy configuration-filter values for one experiment."""
        experiment = self.resolve(relative_path)
        assessment = experiment / MODEL_ASSESSMENT
        if not experiment.is_dir() or not assessment.is_dir():
            raise ValueError("Select an experiment containing MODEL_ASSESSMENT.")

        experiment_complete = (assessment / "assessment_results.json").is_file()
        options: dict[str, dict[str, str]] = {}
        configurations = {}
        for outer_number, outer_folder in _numbered_directories(
            assessment, _OUTER_FOLD_PATTERN
        ):
            selection = outer_folder / "MODEL_SELECTION"
            for config_number, config_folder in _numbered_directories(
                selection, _CONFIG_PATTERN
            ):
                if experiment_complete:
                    values = self._completed_filter_values(config_folder, options)
                    value_source = "aggregated training/validation result"
                else:
                    values = self._running_filter_values(config_folder, options)
                    value_source = "last recorded training/validation epoch"
                configurations[self._relative(config_folder)] = {
                    "outer_fold": outer_number,
                    "configuration": config_number,
                    "values": values,
                }

        sorted_options = sorted(
            options.values(),
            key=lambda item: (
                item["id"] != "scores:main_score",
                item["id"] != "losses:main_loss",
                item["label"].lower(),
            ),
        )
        option_ids = {item["id"] for item in sorted_options}
        if "scores:main_score" in option_ids:
            default_metric = "scores:main_score"
        elif "losses:main_loss" in option_ids:
            default_metric = "losses:main_loss"
        else:
            default_metric = sorted_options[0]["id"] if sorted_options else None
        return {
            "experiment": self._relative(experiment),
            "complete": experiment_complete,
            "value_source": value_source if configurations else None,
            "default_metric": default_metric,
            "default_split": "validation",
            "splits": ["validation", "training"],
            "metrics": sorted_options,
            "configurations": configurations,
            "cache": self.cache_status(),
        }

    def _completed_filter_values(
        self, config_folder: Path, options: dict[str, dict[str, str]]
    ) -> dict[str, float]:
        """Extract aggregated training and validation metrics from a config."""
        results = _read_json(config_folder / "config_results.json")
        if not isinstance(results, dict):
            return {}
        values = {}
        for key, value in results.items():
            parsed = self._result_metric_descriptor(key)
            if parsed is None or isinstance(value, bool):
                continue
            descriptor, split = parsed
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(number):
                continue
            options[descriptor["id"]] = descriptor
            values[f"{split}:{descriptor['id']}"] = number
        return values

    def _running_filter_values(
        self, config_folder: Path, options: dict[str, dict[str, str]]
    ) -> dict[str, float]:
        """Average latest training and validation values across active runs."""
        samples: dict[str, list[float]] = {}
        for metrics_file in sorted(config_folder.rglob("metrics_data.torch")):
            try:
                file_series, _ = self._metrics_file_series(metrics_file)
            except (
                OSError,
                RuntimeError,
                ValueError,
                TypeError,
                EOFError,
                pickle.UnpicklingError,
            ):
                continue
            for item in file_series:
                descriptor, split = self._series_metric_descriptor(
                    item["group"], item["name"]
                )
                if split not in {"training", "validation"}:
                    continue
                options[descriptor["id"]] = descriptor
                last_value = next(
                    (value for value in reversed(item["values"]) if value is not None),
                    None,
                )
                if last_value is None:
                    continue
                value_key = f"{split}:{descriptor['id']}"
                samples.setdefault(value_key, []).append(float(last_value))
        return {
            metric_id: sum(metric_samples) / len(metric_samples)
            for metric_id, metric_samples in samples.items()
            if metric_samples
        }

    @staticmethod
    def _result_metric_descriptor(
        key: str,
    ) -> Optional[tuple[dict[str, str], str]]:
        """Map one ``config_results.json`` key to a filter metric."""
        match = _FILTER_RESULT_PATTERN.fullmatch(key)
        if match is None:
            return None
        split, remainder = match.groups()
        if remainder == "score":
            return (
                ResultsRepository._metric_descriptor("scores", "main_score"),
                split,
            )
        if remainder == "loss":
            return (
                ResultsRepository._metric_descriptor("losses", "main_loss"),
                split,
            )
        for suffix, group in (("_score", "scores"), ("_loss", "losses")):
            if remainder.endswith(suffix):
                return (
                    ResultsRepository._metric_descriptor(
                        group, remainder[: -len(suffix)]
                    ),
                    split,
                )
        return None

    @staticmethod
    def _series_metric_descriptor(group: str, name: str) -> tuple[dict[str, str], str]:
        """Map a stored series name to a filter metric and data split."""
        split = "other"
        metric_name = name
        for candidate in ("validation", "training", "test"):
            prefix = f"{candidate}_"
            if name.startswith(prefix):
                split = candidate
                metric_name = name[len(prefix) :]
                break
        return ResultsRepository._metric_descriptor(group, metric_name), split

    @staticmethod
    def _metric_descriptor(group: str, metric_name: str) -> dict[str, str]:
        """Build a stable filter id and readable label."""
        kind = "score" if group == "scores" else "loss"
        label = metric_name.replace("_", " ").strip().title()
        return {
            "id": f"{group}:{metric_name}",
            "label": label,
            "kind": kind,
        }

    def _relative(self, path: Path) -> str:
        """Return a POSIX path relative to the configured result root."""
        return path.resolve().relative_to(self.root).as_posix()

    def resolve(self, relative_path: str) -> Path:
        """Resolve an API path and reject traversal outside ``logdir``."""
        candidate = (self.root / unquote(relative_path)).resolve()
        if not candidate.is_relative_to(self.root):
            raise ValueError("The requested path is outside the dashboard root.")
        if not candidate.exists():
            raise FileNotFoundError(relative_path)
        return candidate

    def _assessment_directories(self) -> list[Path]:
        """Find MLWiz assessment directories below the configured root."""
        direct = self.root / MODEL_ASSESSMENT
        if direct.is_dir():
            return [direct]
        return sorted(
            (path for path in self.root.rglob(MODEL_ASSESSMENT) if path.is_dir()),
            key=lambda path: path.as_posix().lower(),
        )

    def tree(self) -> dict[str, Any]:
        """Return the live experiment/run hierarchy for the sidebar."""
        experiments = []
        for assessment in self._assessment_directories():
            experiment = assessment.parent
            outer_folds = [
                self._outer_fold_node(number, path)
                for number, path in _numbered_directories(
                    assessment, _OUTER_FOLD_PATTERN
                )
            ]
            run_count = sum(
                len(fold["final_runs"])
                + sum(
                    len(inner["runs"])
                    for config in fold["model_selection"]
                    for inner in config["inner_folds"]
                )
                for fold in outer_folds
            )
            experiments.append(
                {
                    "id": self._relative(experiment),
                    "name": experiment.name,
                    "path": self._relative(experiment),
                    "assessment": _read_json(assessment / "assessment_results.json"),
                    "outer_folds": outer_folds,
                    "run_count": run_count,
                }
            )
        return {
            "root": str(self.root),
            "experiments": experiments,
            "experiment_count": len(experiments),
        }

    def _outer_fold_node(self, number: int, folder: Path) -> dict[str, Any]:
        """Build one outer-fold node including selection and final runs."""
        selection_folder = folder / "MODEL_SELECTION"
        winner = _read_json(selection_folder / "winner_config.json")
        winner_id = winner.get("best_config_id") if isinstance(winner, dict) else None
        configs = []
        for config_number, config_folder in _numbered_directories(
            selection_folder, _CONFIG_PATTERN
        ):
            inner_folds = []
            for inner_number, inner_folder in _numbered_directories(
                config_folder, _INNER_FOLD_PATTERN
            ):
                runs = [
                    self._run_node(run_number, run_folder, "model_selection")
                    for run_number, run_folder in _numbered_directories(
                        inner_folder, _RUN_PATTERN
                    )
                ]
                inner_folds.append(
                    {
                        "number": inner_number,
                        "path": self._relative(inner_folder),
                        "runs": runs,
                    }
                )
            configs.append(
                {
                    "number": config_number,
                    "path": self._relative(config_folder),
                    "has_metrics": any(
                        run["has_metrics"]
                        for inner in inner_folds
                        for run in inner["runs"]
                    ),
                    "is_winner": config_number == winner_id,
                    "results": _read_json(config_folder / "config_results.json"),
                    "inner_folds": inner_folds,
                }
            )
        final_runs = [
            self._run_node(run_number, run_folder, "final")
            for run_number, run_folder in _numbered_directories(
                folder, _FINAL_RUN_PATTERN
            )
        ]
        return {
            "number": number,
            "path": self._relative(folder),
            "results": _read_json(folder / "outer_results.json"),
            "winner": winner,
            "model_selection": configs,
            "final_runs": final_runs,
        }

    def _run_node(self, number: int, folder: Path, run_type: str) -> dict[str, Any]:
        """Build a lightweight sidebar node for one training run."""
        metrics = folder / "metrics_data.torch"
        return {
            "number": number,
            "path": self._relative(folder),
            "type": run_type,
            "has_metrics": metrics.is_file(),
            "modified_at": metrics.stat().st_mtime if metrics.is_file() else None,
        }

    def details(
        self, relative_path: str, include_final_siblings: bool = False
    ) -> dict[str, Any]:
        """Load metric histories and relevant JSON metadata for a selection."""
        target = self.resolve(relative_path)
        if not target.is_dir():
            raise ValueError("Select a run or configuration directory.")

        selection_kind = self._selection_kind(target)
        metrics_root = target
        metrics_files: Iterable[Path]
        direct_metrics = target / "metrics_data.torch"
        if selection_kind == "Final run" and include_final_siblings:
            metrics_root = target.parent
            metrics_files = [
                folder / "metrics_data.torch"
                for _, folder in _numbered_directories(
                    target.parent, _FINAL_RUN_PATTERN
                )
                if (folder / "metrics_data.torch").is_file()
            ][:_MAX_METRICS_FILES_PER_SELECTION]
        elif direct_metrics.is_file():
            metrics_files = [direct_metrics]
        else:
            metrics_files = sorted(target.rglob("metrics_data.torch"))[
                :_MAX_METRICS_FILES_PER_SELECTION
            ]

        series = []
        errors = []
        modified_at = None
        file_count = 0
        for metrics_file in metrics_files:
            file_count += 1
            try:
                file_series, file_stat = self._metrics_file_series(metrics_file)
                source = metrics_file.parent.relative_to(metrics_root).as_posix()
                source = source if source != "." else target.name
                series.extend({**item, "source": source} for item in file_series)
                timestamp = file_stat.st_mtime
                modified_at = max(modified_at or timestamp, timestamp)
            except (
                OSError,
                RuntimeError,
                ValueError,
                TypeError,
                EOFError,
                pickle.UnpicklingError,
            ) as error:
                errors.append(
                    {
                        "file": self._relative(metrics_file),
                        "message": str(error),
                    }
                )

        if selection_kind == "Model-selection configuration":
            plot_scope = "model_selection_configuration"
        elif selection_kind == "Final run":
            plot_scope = "final_runs"
        else:
            plot_scope = "single_run"

        return {
            "selection": {
                "name": target.name,
                "path": self._relative(target),
                "kind": selection_kind,
                "plot_scope": plot_scope,
                "selected_source": target.name,
                "final_runs_included": (
                    selection_kind == "Final run" and include_final_siblings
                ),
            },
            "series": series,
            "metrics_file_count": file_count,
            "modified_at": modified_at,
            "metadata": self._metadata_for(target),
            "overview": self._experiment_overview(target),
            "errors": errors,
            "cache": self.cache_status(),
        }

    def _experiment_overview(self, target: Path) -> Optional[dict[str, Any]]:
        """Summarize progress and recorded run times for the selected experiment."""
        assessment = next(
            (
                ancestor
                for ancestor in self._ancestors_within_root(target)
                if ancestor.name == MODEL_ASSESSMENT
            ),
            None,
        )
        if assessment is None:
            return None
        experiment = assessment.parent
        run_folders = []
        config_folders = []
        for _, outer_folder in _numbered_directories(assessment, _OUTER_FOLD_PATTERN):
            selection = outer_folder / "MODEL_SELECTION"
            for _, config_folder in _numbered_directories(selection, _CONFIG_PATTERN):
                config_folders.append(config_folder)
                for _, inner_folder in _numbered_directories(
                    config_folder, _INNER_FOLD_PATTERN
                ):
                    run_folders.extend(
                        folder
                        for _, folder in _numbered_directories(
                            inner_folder, _RUN_PATTERN
                        )
                    )
            run_folders.extend(
                folder
                for _, folder in _numbered_directories(outer_folder, _FINAL_RUN_PATTERN)
            )

        statuses = [self._run_status(folder) for folder in run_folders]
        durations = [
            status["duration_seconds"]
            for status in statuses
            if status["duration_seconds"] is not None
        ]
        completed = sum(status["state"] == "completed" for status in statuses)
        running = sum(status["state"] == "running" for status in statuses)
        failed = sum(status["state"] == "failed" for status in statuses)
        queued = sum(status["state"] == "queued" for status in statuses)
        average = statistics.fmean(durations) if durations else None
        incomplete = max(len(statuses) - completed, 0)
        return {
            "experiment": self._relative(experiment),
            "name": experiment.name,
            "state": (
                "completed"
                if (assessment / "assessment_results.json").is_file()
                else "running"
            ),
            "runs": {
                "total": len(statuses),
                "completed": completed,
                "running": running,
                "queued": queued,
                "failed": failed,
            },
            "configurations": {
                "total": len(config_folders),
                "completed": sum(
                    (folder / "config_results.json").is_file()
                    for folder in config_folders
                ),
            },
            "timing": {
                "recorded_total_seconds": sum(durations) if durations else None,
                "average_run_seconds": average,
                "median_run_seconds": statistics.median(durations)
                if durations
                else None,
                "fastest_run_seconds": min(durations) if durations else None,
                "slowest_run_seconds": max(durations) if durations else None,
                "estimated_remaining_compute_seconds": average * incomplete
                if average is not None
                else None,
                "timed_runs": len(durations),
            },
        }

    def _run_status(self, run_folder: Path) -> dict[str, Any]:
        """Classify one run and extract completed profiler time markers."""
        log_path = run_folder / "experiment.log"
        duration = None
        if log_path.is_file():
            elapsed = 0.0
            try:
                with log_path.open("r", encoding="utf-8", errors="replace") as log:
                    for line in log:
                        match = _ELAPSED_PATTERN.search(line)
                        if match:
                            elapsed += float(match.group(1))
            except OSError:
                elapsed = 0.0
            if elapsed > 0:
                duration = elapsed

        result_files = list(run_folder.glob("run_*_results.dill"))
        error_path = run_folder / "experiment.err"
        if duration is not None or result_files:
            state = "completed"
        elif error_path.is_file() and error_path.stat().st_size > 0:
            state = "failed"
        elif any(
            (run_folder / filename).exists()
            for filename in (
                "metrics_data.torch",
                "experiment.log",
                "last_checkpoint.pth",
                "best_checkpoint.pth",
            )
        ):
            state = "running"
        else:
            state = "queued"
        return {"state": state, "duration_seconds": duration}

    def _metrics_file_series(
        self, metrics_file: Path
    ) -> tuple[list[dict[str, Any]], Any]:
        """Return normalized series through the live LRU cache."""
        file_stat = metrics_file.stat()
        signature = (file_stat.st_mtime_ns, file_stat.st_size)
        file_series = self.metrics_cache.get(metrics_file, signature)
        if file_series is None:
            file_series = self._load_metrics_file(metrics_file)
            final_stat = metrics_file.stat()
            final_signature = (final_stat.st_mtime_ns, final_stat.st_size)
            if final_signature == signature:
                self.metrics_cache.put(metrics_file, signature, file_series)
        return file_series, file_stat

    def _load_metrics_file(self, metrics_file: Path) -> list[dict[str, Any]]:
        """Load and normalize one metrics artifact on demand."""
        stored = torch.load(metrics_file, map_location="cpu", weights_only=True)
        if not isinstance(stored, dict):
            raise ValueError("Expected a dictionary at the file root.")
        series = []
        for group in ("losses", "scores"):
            metrics = stored.get(group, {})
            if not isinstance(metrics, dict):
                continue
            for name, values in metrics.items():
                normalized = _numeric_series(values)
                if normalized is not None:
                    series.append(
                        {
                            "group": group,
                            "name": str(name),
                            "values": normalized,
                        }
                    )
        return series

    def _selection_kind(self, target: Path) -> str:
        """Return a human-readable type for a selected result directory."""
        if _FINAL_RUN_PATTERN.fullmatch(target.name):
            return "Final run"
        if _RUN_PATTERN.fullmatch(target.name):
            return "Model-selection run"
        if _CONFIG_PATTERN.fullmatch(target.name):
            return "Model-selection configuration"
        return "Experiment selection"

    def _ancestors_within_root(self, target: Path) -> Iterable[Path]:
        """Yield ``target`` and its ancestors without leaving the result root."""
        current = target
        while current.is_relative_to(self.root):
            yield current
            if current == self.root:
                break
            current = current.parent

    def _metadata_for(self, target: Path) -> list[dict[str, Any]]:
        """Collect configuration, fold, and assessment JSON near a selection."""
        metadata = []
        seen = set()

        def add(label: str, path: Path) -> None:
            """Append a readable JSON artifact once when it exists."""
            if path in seen:
                return
            payload = _read_json(path)
            if payload is not None:
                seen.add(path)
                metadata.append(
                    {
                        "label": label,
                        "path": self._relative(path),
                        "data": _json_safe(payload),
                    }
                )

        for ancestor in self._ancestors_within_root(target):
            if _CONFIG_PATTERN.fullmatch(ancestor.name):
                add("Configuration results", ancestor / "config_results.json")
                break

        outer_folder = next(
            (
                ancestor
                for ancestor in self._ancestors_within_root(target)
                if _OUTER_FOLD_PATTERN.fullmatch(ancestor.name)
            ),
            None,
        )
        if outer_folder is not None:
            add(
                "Selected configuration",
                outer_folder / "MODEL_SELECTION" / "winner_config.json",
            )
            add("Outer-fold results", outer_folder / "outer_results.json")
            assessment = outer_folder.parent
            add("Assessment results", assessment / "assessment_results.json")
        return metadata


class DashboardServer(ThreadingHTTPServer):
    """Threaded HTTP server carrying a :class:`ResultsRepository`."""

    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        repository: ResultsRepository,
    ):
        """Bind the server address and attach its result repository."""
        super().__init__(server_address, DashboardRequestHandler)
        self.repository = repository


class DashboardRequestHandler(BaseHTTPRequestHandler):
    """Serve the dashboard assets and read-only JSON API."""

    server: DashboardServer

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        """Handle dashboard, asset, and API requests."""
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/tree":
                self._send_json(self.server.repository.tree())
                return
            if parsed.path == "/api/details":
                query = parse_qs(parsed.query)
                relative_path = query.get("path", [None])[0]
                if not relative_path:
                    self._send_error(
                        HTTPStatus.BAD_REQUEST, "Missing the 'path' parameter."
                    )
                    return
                include_final_siblings = query.get("aggregate_final_runs", ["0"])[
                    0
                ] in {"1", "true", "yes"}
                self._send_json(
                    self.server.repository.details(
                        relative_path,
                        include_final_siblings=include_final_siblings,
                    )
                )
                return
            if parsed.path == "/api/cache":
                self._send_json(self.server.repository.cache_status())
                return
            if parsed.path == "/api/model-graph":
                query = parse_qs(parsed.query)
                relative_path = query.get("path", [None])[0]
                if not relative_path:
                    self._send_error(
                        HTTPStatus.BAD_REQUEST, "Missing the 'path' parameter."
                    )
                    return
                checkpoint_kind = query.get("checkpoint", ["auto"])[0]
                graph_mode = query.get("mode", ["architecture"])[0]
                self._send_json(
                    self.server.repository.model_graph(
                        relative_path,
                        checkpoint_kind=checkpoint_kind,
                        graph_mode=graph_mode,
                    )
                )
                return
            if parsed.path == "/api/model-graph-info":
                query = parse_qs(parsed.query)
                relative_path = query.get("path", [None])[0]
                if not relative_path:
                    self._send_error(
                        HTTPStatus.BAD_REQUEST, "Missing the 'path' parameter."
                    )
                    return
                self._send_json(
                    self.server.repository.model_graph_info(relative_path)
                )
                return
            if parsed.path == "/api/experiment-filter":
                query = parse_qs(parsed.query)
                relative_path = query.get("path", [None])[0]
                if not relative_path:
                    self._send_error(
                        HTTPStatus.BAD_REQUEST, "Missing the 'path' parameter."
                    )
                    return
                self._send_json(
                    self.server.repository.experiment_filter_data(relative_path)
                )
                return
            if parsed.path in ("/", "/index.html"):
                self._send_file(_ASSET_DIRECTORY / "index.html")
                return
            if parsed.path.startswith("/assets/"):
                asset_name = Path(parsed.path).name
                asset = (_ASSET_DIRECTORY / asset_name).resolve()
                if asset.parent != _ASSET_DIRECTORY.resolve():
                    self._send_error(HTTPStatus.NOT_FOUND, "Asset not found.")
                    return
                self._send_file(asset)
                return
            self._send_error(HTTPStatus.NOT_FOUND, "Page not found.")
        except FileNotFoundError:
            self._send_error(HTTPStatus.NOT_FOUND, "Selection not found.")
        except ValueError as error:
            self._send_error(HTTPStatus.BAD_REQUEST, str(error))
        except OSError as error:
            self._send_error(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                f"Could not read the result directory: {error}",
            )

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        """Handle dashboard configuration requests."""
        parsed = urlparse(self.path)
        if parsed.path == "/api/cache/reset":
            self._send_json(self.server.repository.reset_cache())
            return
        if parsed.path != "/api/cache":
            self._send_error(HTTPStatus.NOT_FOUND, "Page not found.")
            return
        try:
            payload = self._read_json_body()
            if "max_mb" not in payload:
                raise ValueError("Missing the 'max_mb' field.")
            self._send_json(self.server.repository.configure_cache(payload["max_mb"]))
        except (ValueError, json.JSONDecodeError) as error:
            self._send_error(HTTPStatus.BAD_REQUEST, str(error))

    def _read_json_body(self) -> dict[str, Any]:
        """Read a small JSON request body."""
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError as error:
            raise ValueError("Invalid Content-Length header.") from error
        if not 0 < content_length <= 65536:
            raise ValueError("Request body must be between 1 byte and 64 KiB.")
        payload = json.loads(self.rfile.read(content_length))
        if not isinstance(payload, dict):
            raise ValueError("Expected a JSON object.")
        return payload

    def _send_json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        """Stream a no-cache JSON response with bounded temporary memory."""
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        encoder = json.JSONEncoder(
            separators=(",", ":"), ensure_ascii=False, allow_nan=False
        )
        chunks = []
        chunk_bytes = 0
        try:
            for chunk in encoder.iterencode(payload):
                encoded = chunk.encode("utf-8")
                chunks.append(encoded)
                chunk_bytes += len(encoded)
                if chunk_bytes >= 64 * 1024:
                    self.wfile.write(b"".join(chunks))
                    chunks = []
                    chunk_bytes = 0
            if chunks:
                self.wfile.write(b"".join(chunks))
        except (BrokenPipeError, ConnectionResetError):
            return

    def _send_error(self, status: HTTPStatus, message: str) -> None:
        """Send a structured JSON error response."""
        self._send_json({"error": message}, status=status)

    def _send_file(self, path: Path) -> None:
        """Serve one bundled frontend asset with its inferred content type."""
        try:
            body = path.read_bytes()
        except FileNotFoundError:
            self._send_error(HTTPStatus.NOT_FOUND, "Asset not found.")
            return
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        """Use a concise dashboard access log."""
        print(f"[mlwiz-dashboard] {self.address_string()} {format % args}")


def create_server(
    logdir: str | Path, host: str = "127.0.0.1", port: int = 6006
) -> DashboardServer:
    """Create, but do not start, a dashboard HTTP server."""
    return DashboardServer((host, port), ResultsRepository(logdir))


def get_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse dashboard command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Inspect MLWiz model-selection and final-run metrics."
    )
    parser.add_argument(
        "--logdir",
        default="RESULTS",
        help="Experiment directory or parent results directory (default: RESULTS).",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=6006, type=int)
    parser.add_argument(
        "--open",
        action="store_true",
        dest="open_browser",
        help="Open the dashboard in the default browser after startup.",
    )
    args = parser.parse_args(argv)
    if not Path(args.logdir).expanduser().is_dir():
        parser.error(f"--logdir is not a directory: {args.logdir}")
    if not 0 <= args.port <= 65535:
        parser.error("--port must be between 0 and 65535")
    return args


def main() -> None:
    """Start the MLWiz dashboard until interrupted."""
    args = get_args()
    server = create_server(args.logdir, args.host, args.port)
    actual_port = server.server_address[1]
    display_host = "127.0.0.1" if args.host in ("0.0.0.0", "::") else args.host
    url = f"http://{display_host}:{actual_port}/"
    print(f"MLWiz Dashboard is watching {Path(args.logdir).resolve()}")
    print(f"Open {url} (press Ctrl-C to stop)")
    if args.open_browser:
        threading.Timer(0.15, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping MLWiz Dashboard.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
