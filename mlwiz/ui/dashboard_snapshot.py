"""Portable, project-independent snapshots for MLWiz Dashboard."""

from __future__ import annotations

import argparse
import copy
import io
import json
import threading
import webbrowser
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from mlwiz.ui.dashboard import DashboardServer, ResultsRepository

SNAPSHOT_FORMAT = "mlwiz-dashboard-snapshot"
SNAPSHOT_VERSION = 1
SNAPSHOT_MEMBER = "snapshot.json"
_MAX_SNAPSHOT_BYTES = 1024**3


def _selection_paths(experiment: dict[str, Any]) -> list[str]:
    """Return every configuration and run path exposed by an experiment tree."""
    paths: list[str] = []
    for outer_fold in experiment.get("outer_folds", []):
        for config in outer_fold.get("model_selection", []):
            paths.append(config["path"])
            for inner_fold in config.get("inner_folds", []):
                paths.extend(run["path"] for run in inner_fold.get("runs", []))
        paths.extend(run["path"] for run in outer_fold.get("final_runs", []))
    return paths


def _selected_experiment(
    tree: dict[str, Any], selected_path: str
) -> dict[str, Any]:
    """Find the experiment containing a dashboard-relative selection path."""
    matches = [
        experiment
        for experiment in tree.get("experiments", [])
        if selected_path == experiment["path"]
        or selected_path.startswith(f"{experiment['path']}/")
    ]
    if not matches:
        raise ValueError(f"Selection is not present in the dashboard: {selected_path}")
    return max(matches, key=lambda item: len(item["path"]))


def build_snapshot(
    repository: ResultsRepository,
    state: dict[str, Any],
    selected_only: bool = False,
) -> dict[str, Any]:
    """Materialize all experiments, or only the selected experiment, as JSON."""
    selected_path = state.get("selectedPath")
    tree = repository.tree()
    experiments = tree.get("experiments", [])
    if selected_only:
        if not isinstance(selected_path, str) or not selected_path:
            raise ValueError("A non-empty dashboard path is required for --path.")
        experiment = _selected_experiment(tree, selected_path)
        paths = _selection_paths(experiment)
        if selected_path not in paths:
            raise ValueError("Select a configuration or run before exporting the view.")
        experiments = [experiment]
    elif isinstance(selected_path, str) and selected_path:
        # Validate restored selections while still exporting the complete root.
        _selected_experiment(tree, selected_path)

    details: dict[str, Any] = {}
    filters: dict[str, Any] = {}
    analyses: dict[str, Any] = {}
    for experiment in experiments:
        for path in _selection_paths(experiment):
            details[path] = repository.details(path)
            selection = details[path].get("selection", {})
            if selection.get("plot_scope") == "final_runs":
                details[f"{path}?aggregate_final_runs=1"] = repository.details(
                    path, include_final_siblings=True
                )
        experiment_path = experiment["path"]
        filters[experiment_path] = repository.experiment_filter_data(experiment_path)
        for outer_fold in experiment.get("outer_folds", []):
            inner_numbers = {
                inner_fold["number"]
                for config in outer_fold.get("model_selection", [])
                for inner_fold in config.get("inner_folds", [])
            }
            for inner_number in sorted(inner_numbers):
                key = (
                    f"{experiment_path}?outer={outer_fold['number']}"
                    f"&inner={inner_number}"
                )
                analyses[key] = repository.model_selection_analysis(
                    experiment_path, outer_fold["number"], inner_number
                )

    snapshot_tree = {
        "root": "Portable dashboard snapshot",
        "experiments": copy.deepcopy(experiments),
        "experiment_count": len(experiments),
    }
    return {
        "format": SNAPSHOT_FORMAT,
        "version": SNAPSHOT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "state": copy.deepcopy(state),
        "tree": snapshot_tree,
        "details": details,
        "filters": filters,
        "analyses": analyses,
    }


def _validate_snapshot(snapshot: Any) -> dict[str, Any]:
    """Validate the small structural contract of an imported snapshot."""
    if not isinstance(snapshot, dict):
        raise ValueError("Snapshot payload must be a JSON object.")
    if snapshot.get("format") != SNAPSHOT_FORMAT:
        raise ValueError("File is not an MLWiz Dashboard snapshot.")
    if snapshot.get("version") != SNAPSHOT_VERSION:
        raise ValueError(
            f"Unsupported snapshot version: {snapshot.get('version')!r}."
        )
    for key in ("state", "tree", "details", "filters"):
        if not isinstance(snapshot.get(key), dict):
            raise ValueError(f"Snapshot field '{key}' must be an object.")
    return snapshot


def write_snapshot(snapshot: dict[str, Any], output: str | Path) -> Path:
    """Write a validated snapshot as a compressed ``.mlwiz`` archive."""
    _validate_snapshot(snapshot)
    destination = Path(output).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        archive.writestr(
            SNAPSHOT_MEMBER,
            json.dumps(snapshot, ensure_ascii=False, allow_nan=False),
        )
    return destination.resolve()


def snapshot_bytes(snapshot: dict[str, Any]) -> bytes:
    """Serialize a snapshot for a browser download without touching results."""
    stream = io.BytesIO()
    with zipfile.ZipFile(
        stream, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        archive.writestr(
            SNAPSHOT_MEMBER,
            json.dumps(_validate_snapshot(snapshot), ensure_ascii=False, allow_nan=False),
        )
    return stream.getvalue()


def read_snapshot(source: str | Path) -> dict[str, Any]:
    """Read and validate a snapshot without extracting archive members."""
    path = Path(source).expanduser()
    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            if names != [SNAPSHOT_MEMBER]:
                raise ValueError("Snapshot archive has unexpected contents.")
            if archive.getinfo(SNAPSHOT_MEMBER).file_size > _MAX_SNAPSHOT_BYTES:
                raise ValueError("Snapshot payload exceeds the 1 GiB safety limit.")
            snapshot = json.loads(archive.read(SNAPSHOT_MEMBER))
    except (OSError, zipfile.BadZipFile, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read dashboard snapshot: {error}") from error
    return _validate_snapshot(snapshot)


class SnapshotRepository:
    """Read-only dashboard repository backed solely by normalized JSON."""

    def __init__(self, snapshot: dict[str, Any]):
        """Retain a validated snapshot for subsequent read-only API calls."""
        self.snapshot = _validate_snapshot(snapshot)

    @staticmethod
    def _cache() -> dict[str, Any]:
        """Return the inert cache status used by snapshot servers."""
        return {
            "entries": 0,
            "max_bytes": 0,
            "used_mb": 0,
            "max_mb": 0,
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "invalidations": 0,
            "skipped": 0,
        }

    def tree(self) -> dict[str, Any]:
        """Return the captured experiment hierarchy."""
        return copy.deepcopy(self.snapshot["tree"])

    def details(
        self, relative_path: str, include_final_siblings: bool = False
    ) -> dict[str, Any]:
        """Return captured normalized metrics for a selection."""
        key = (
            f"{relative_path}?aggregate_final_runs=1"
            if include_final_siblings
            else relative_path
        )
        payload = self.snapshot["details"].get(key)
        if payload is None:
            raise FileNotFoundError(relative_path)
        return copy.deepcopy(payload)

    def experiment_filter_data(self, relative_path: str) -> dict[str, Any]:
        """Return captured configuration-filter values for an experiment."""
        payload = self.snapshot["filters"].get(relative_path)
        if payload is None:
            raise FileNotFoundError(relative_path)
        return copy.deepcopy(payload)

    def model_selection_analysis(
        self, relative_path: str, outer_fold: int, inner_fold: int
    ) -> dict[str, Any]:
        """Return a captured fold comparison when the snapshot includes it."""
        key = f"{relative_path}?outer={outer_fold}&inner={inner_fold}"
        payload = self.snapshot.get("analyses", {}).get(key)
        if payload is None:
            raise FileNotFoundError(
                "Model selection analysis was not captured in this snapshot."
            )
        return copy.deepcopy(payload)

    def snapshot_state(self) -> dict[str, Any]:
        """Return the browser state that should be restored on page load."""
        return copy.deepcopy(self.snapshot["state"])

    def cache_status(self) -> dict[str, Any]:
        """Report that imported normalized data requires no metric cache."""
        return self._cache()

    def configure_cache(self, _max_mb: float) -> dict[str, Any]:
        """Keep cache configuration inert for immutable snapshots."""
        return self._cache()

    def reset_cache(self) -> dict[str, Any]:
        """Keep cache reset inert for immutable snapshots."""
        return self._cache()

    def model_graph_info(self, _relative_path: str) -> dict[str, Any]:
        """Explain that checkpoints and model graphs were deliberately omitted."""
        raise ValueError("Model graphs are not included in portable snapshots.")

    def model_graph(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
        """Reject model-graph loading because snapshots contain no checkpoints."""
        raise ValueError("Model graphs are not included in portable snapshots.")


def export_get_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse portable-dashboard export arguments."""
    parser = argparse.ArgumentParser(
        description="Export a project-independent MLWiz Dashboard snapshot."
    )
    parser.add_argument("--logdir", default="RESULTS")
    parser.add_argument("--path", help="Dashboard-relative configuration or run path.")
    parser.add_argument("--state", help="Optional dashboard-state JSON file.")
    parser.add_argument("--output", "-o", required=True)
    args = parser.parse_args(argv)
    if not Path(args.logdir).expanduser().is_dir():
        parser.error(f"--logdir is not a directory: {args.logdir}")
    return args


def export_main() -> None:
    """Build a portable dashboard archive from a live result directory."""
    args = export_get_args()
    state: dict[str, Any] = {}
    if args.state:
        try:
            state = json.loads(Path(args.state).expanduser().read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise SystemExit(f"Could not read --state: {error}") from error
        if not isinstance(state, dict):
            raise SystemExit("--state must contain a JSON object")
    if args.path:
        state["selectedPath"] = args.path
    try:
        snapshot = build_snapshot(
            ResultsRepository(args.logdir), state, selected_only=bool(args.path)
        )
        destination = write_snapshot(snapshot, args.output)
    except (OSError, ValueError) as error:
        raise SystemExit(f"Could not export dashboard snapshot: {error}") from error
    print(f"Exported MLWiz Dashboard snapshot to {destination}")


def import_get_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse portable-dashboard import server arguments."""
    parser = argparse.ArgumentParser(
        description="Serve a portable MLWiz Dashboard snapshot."
    )
    parser.add_argument("snapshot")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=6006, type=int)
    parser.add_argument("--open", action="store_true", dest="open_browser")
    parser.add_argument(
        "--show-logs",
        action="store_true",
        help="Show dashboard HTTP request logs in the terminal.",
    )
    args = parser.parse_args(argv)
    if not Path(args.snapshot).expanduser().is_file():
        parser.error(f"snapshot is not a file: {args.snapshot}")
    if not 0 <= args.port <= 65535:
        parser.error("--port must be between 0 and 65535")
    return args


def import_main() -> None:
    """Start an ad-hoc read-only server over one portable snapshot."""
    args = import_get_args()
    try:
        snapshot = read_snapshot(args.snapshot)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    server = DashboardServer(
        (args.host, args.port),
        SnapshotRepository(snapshot),
        show_logs=args.show_logs,
    )
    actual_port = server.server_address[1]
    display_host = "127.0.0.1" if args.host in ("0.0.0.0", "::") else args.host
    url = f"http://{display_host}:{actual_port}/"
    print(f"MLWiz Dashboard is serving {Path(args.snapshot).resolve()}")
    print(f"Open {url} (press Ctrl-C to stop)")
    if args.open_browser:
        threading.Timer(0.15, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping MLWiz Dashboard.")
    finally:
        server.server_close()
