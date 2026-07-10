"""Browser dashboard for inspecting MLWiz experiment metrics.

The dashboard mirrors MLWiz's result hierarchy and reads the
``metrics_data.torch`` artifacts produced by :class:`~mlwiz.training.callback.plotter.Plotter`.
It intentionally uses Python's standard HTTP server so it can ship with MLWiz
without adding a web-framework dependency.
"""

from __future__ import annotations

import argparse
import json
import math
import mimetypes
import pickle
import re
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterable, Optional
from urllib.parse import parse_qs, unquote, urlparse

import torch

from mlwiz.static import MODEL_ASSESSMENT


_OUTER_FOLD_PATTERN = re.compile(r"OUTER_FOLD_(\d+)$")
_CONFIG_PATTERN = re.compile(r"config_(\d+)$")
_INNER_FOLD_PATTERN = re.compile(r"INNER_FOLD_(\d+)$")
_RUN_PATTERN = re.compile(r"run_?(\d+)$")
_FINAL_RUN_PATTERN = re.compile(r"final_run_?(\d+)$")
_ASSET_DIRECTORY = Path(__file__).with_name("web_assets")
_MAX_METRICS_FILES_PER_SELECTION = 256


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
            return json.load(stream)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


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


class ResultsRepository:
    """Read-only view over one experiment or a directory of experiments."""

    def __init__(self, logdir: str | Path):
        """Create a repository rooted at ``logdir``.

        Passing a ``MODEL_ASSESSMENT`` directory is treated as passing its
        parent experiment directory, which keeps all API paths inside the root.
        """
        root = Path(logdir).expanduser().resolve()
        self.root = root.parent if root.name == MODEL_ASSESSMENT else root

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

    def details(self, relative_path: str) -> dict[str, Any]:
        """Load metric histories and relevant JSON metadata for a selection."""
        target = self.resolve(relative_path)
        if not target.is_dir():
            raise ValueError("Select a run or configuration directory.")

        direct_metrics = target / "metrics_data.torch"
        if direct_metrics.is_file():
            metrics_files: Iterable[Path] = [direct_metrics]
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
                stored = torch.load(metrics_file, map_location="cpu", weights_only=True)
                if not isinstance(stored, dict):
                    raise ValueError("Expected a dictionary at the file root.")
                source = metrics_file.parent.relative_to(target).as_posix()
                source = source if source != "." else target.name
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
                                    "source": source,
                                    "values": normalized,
                                }
                            )
                timestamp = metrics_file.stat().st_mtime
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

        return {
            "selection": {
                "name": target.name,
                "path": self._relative(target),
                "kind": self._selection_kind(target),
            },
            "series": series,
            "metrics_file_count": file_count,
            "modified_at": modified_at,
            "metadata": self._metadata_for(target),
            "errors": errors,
        }

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
                self._send_json(self.server.repository.details(relative_path))
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

    def _send_json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        """Send a no-cache JSON response."""
        body = json.dumps(_json_safe(payload), separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

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
