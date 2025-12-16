import datetime
import os
import select
import shutil
import sys
import signal
import termios
import threading
import time
import traceback
import tty
from copy import deepcopy
from dataclasses import dataclass
from pprint import pformat
from typing import Callable, Optional, Sequence, Tuple, Union
import tqdm
import ray

from mlwiz.static import *

SelectionIdentifier = Tuple[str, int, int, int, int]
FinalIdentifier = Tuple[str, int, int]
_ViewIdentifier = Union[SelectionIdentifier, FinalIdentifier]


def clear_screen():
    """
    Clears the CLI interface.
    """
    # Shared helper called before drawing headers, bars, or switching views.
    # Mimic Ctrl+L: clear visible screen and move cursor to top without
    # wiping scrollback/history.
    print("\033[2J\033[H", end="", flush=True)


@dataclass
class _RenderState:
    header: str = ""
    last_progress: str = ""
    failure: str = ""
    moves: str = ""
    rendered_lines: int = 0
    origin_row: int = 1
    global_header: str = ""

    def reset_area(self):
        """
        Reset the tracked render area to the default origin.

        Side effects:
            Mutates ``rendered_lines`` and ``origin_row`` so the next render
            pass re-anchors from the top of the screen.
        """
        self.rendered_lines = 0
        self.origin_row = 1

    def append_moves(self, text: str):
        """
        Append terminal cursor/erase escape sequences to apply after rendering.

        Args:
            text (str): Raw escape sequence string.

        Side effects:
            Mutates ``moves`` by appending ``text``.
        """
        self.moves += text

    def clear_moves(self):
        """
        Clear buffered terminal escape sequences.

        Side effects:
            Resets ``moves`` to an empty string.
        """
        self.moves = ""


@ray.remote(num_cpus=0)
class ProgressManagerActor:
    """
    Ray actor used to aggregate progress updates from workers.
    The driver periodically pulls pending messages instead of relying on a queue.
    """

    def __init__(self):
        """
        Initialize the progress actor.

        Side effects:
            Initializes the internal message buffer and lifecycle flags used by
            workers and the driver process.
        """
        # Buffer messages coming from workers while the driver polls via drain().
        self._pending_messages = []
        self._closed = False
        self._terminated = False

    def push(self, payload: dict):
        """
        Enqueue a progress update from a worker.

        Args:
            payload (dict): Progress update payload (will be deep-copied).

        Side effects:
            Appends the payload to the internal buffer unless the actor has
            been closed.
        """
        # Enqueue updates from workers; ignored when actor is marked closed.
        if self._closed:
            return
        self._pending_messages.append(deepcopy(payload))

    def terminate(self):
        """
        Requests all workers to terminate gracefully and stops further updates.
        """
        # Exposed so evaluator can broadcast shutdown and stop buffering.
        self._terminated = True
        self._closed = True

    def is_terminated(self) -> bool:
        """
        Return whether termination has been requested.

        Returns:
            bool: ``True`` if :meth:`terminate` was called.
        """
        # Queried by workers to decide whether to exit early.
        return self._terminated

    def drain(self):
        """
        Returns buffered messages and clears the buffer.
        """
        # Called by ProgressManager.update_state to fetch and clear messages.
        pending = self._pending_messages
        self._pending_messages = []
        return pending, self._closed

    def close(self):
        """
        Stop accepting further progress messages.

        Side effects:
            Marks the actor as closed; future :meth:`push` calls are ignored.
        """
        # Signals no further messages should be accepted.
        self._closed = True


class ProgressManager:
    r"""
    Class that is responsible for drawing progress bars.

    Args:
        outer_folds (int): number of external folds for model assessment
        inner_folds (int): number of internal folds for model selection
        no_configs (int): number of possible configurations in model selection
        final_runs (int): number of final runs per outer fold once the
            best model has been selected
        debug (bool): whether to run in debug mode (no tqdm, simple prints)
        progress_actor (:class:`~ray.actor.ActorHandle`): handle used to pull
            aggregated progress updates from different workers
    """

    # Possible vars of ``bar_format``:
    #       * ``l_bar, bar, r_bar``,
    #       * ``n, n_fmt, total, total_fmt``,
    #       * ``percentage, elapsed, elapsed_s``,
    #       * ``ncols, nrows, desc, unit``,
    #       * ``rate, rate_fmt, rate_noinv``,
    #       * ``rate_noinv_fmt, rate_inv, rate_inv_fmt``,
    #       * ``postfix, unit_divisor, remaining, remaining_s``

    def __init__(
        self,
        outer_folds,
        inner_folds,
        no_configs,
        config_runs,
        final_runs,
        debug=True,
        progress_actor=None,
        poll_interval: float = 0.2,
    ):
        """
        Initialize the progress manager UI.

        Args:
            outer_folds (int): Number of outer folds.
            inner_folds (int): Number of inner folds.
            no_configs (int): Number of hyper-parameter configurations.
            config_runs (int): Number of runs per configuration/inner fold.
            final_runs (int): Number of final runs per outer fold.
            debug (bool): If ``True``, disable interactive rendering and rely on
                simple prints.
            progress_actor (ray.actor.ActorHandle | None): Actor handle used to
                pull aggregated progress updates from worker processes.
            poll_interval (float): Minimum poll interval (seconds) used when
                draining the actor.

        Side effects:
            Clears the screen, initializes progress bars when not in debug mode,
            starts the input listener thread, and registers a resize handler.
        """
        # Wire progress data structures and start UI scaffolding used by render/refresh.
        self.ncols = 100
        self.outer_folds = outer_folds
        self.inner_folds = inner_folds
        self.no_configs = no_configs
        self.config_runs = config_runs
        self.final_runs = final_runs
        self.pbars = []
        self.debug = debug
        self.progress_actor = progress_actor
        self._poll_interval = poll_interval

        # Keep track of the latest message per individual run
        # Structured as [outer][inner][config][run] -> msg
        self._last_run_messages = {}
        # Keep track of the latest message for final runs
        # Structured as [outer][run] -> msg
        self._final_run_messages = {}
        # Keep track of the latest progress text per individual run
        # Structured as [outer][inner][config][run] -> str
        self._last_progress_messages = {}
        # Keep track of the latest progress text for final runs
        # Structured as [outer][run] -> str
        self._final_progress_messages = {}
        # Keep identifiers of the most recent model selection and final runs
        self._last_seen_model_selection_identifier = None
        self._last_seen_final_run_identifier = None

        # Track what is currently visible on screen.
        self._render_state = _RenderState()
        self._render_state.global_header = self._build_global_header()

        # when NOT in debug mode, this is None
        # when the global view is active, otherwise
        # we will visualize the progress of a single
        # configuration
        self._to_visualize = "g"
        self._input_buffer = ""  # currently typed command (prefixed with ':')
        self._input_active = False  # whether command mode is active
        self._input_render_len = 0  # last rendered width to properly erase
        self._input_lock = threading.Lock()  # guards input state
        self._stop_input_event = threading.Event()  # stops listener on exit
        self._input_thread = None  # background input listener
        self._model_configs = None
        clear_screen()

        if not self.debug:
            self.show_header()
            for i in range(self.outer_folds):
                for j in range(self.inner_folds):
                    self.pbars.append(self._init_selection_pbar(i, j))

            for i in range(self.outer_folds):
                self.pbars.append(self._init_assessment_pbar(i))

            self.show_footer()

            self.times = [{} for _ in range(len(self.pbars))]
            self._start_input_listener()
        self._register_resize_handler()

    def set_model_configs(self, model_configs):
        """
        Set model configurations so copies don't have to be stored
        on the progress actor.

        :param model_configs: list of hparams
        """
        # Called from evaluator to provide config data used by _format_run_message.
        self._model_configs = model_configs

    def _change_view_mode(
        self, identifier: str = None, force_refresh: bool = False
    ) -> None:
        """
        Changes the view mode of the progress manager.
        If config_id is None, the global view is activated.
        Otherwise, the progress of a single configuration is visualized.

        Args:
            identifier (str): the reference to the configuration
            to visualize in the form of "outer_id_inner_id_config_id_run_id"

        """
        # Switch between global view and focused run view; drives rendering pipeline.
        if self.debug:
            raise RuntimeError("Cannot change view mode in debug mode")
        identifier = (identifier or "g").strip()

        if identifier == "" or identifier in {"g", "global"}:
            identifier = "g"

        if force_refresh or self._to_visualize != identifier:
            self._to_visualize = identifier
            clear_screen()
            self._render_state.reset_area()

        if self._to_visualize in {"g", "global"}:
            self.refresh_global_view()
            self._render_user_input()
            return

        invalid_id_msg = 'ProgressManager: invalid identifier format, use "outer inner config run" format or "outer run" format...'

        parsed_identifier = self._parse_view_identifier(self._to_visualize)
        if parsed_identifier is None or any(
            value <= 0 for value in parsed_identifier[1:]
        ):
            self._render_invalid_identifier(invalid_id_msg)
            return

        kind, *raw_values = parsed_identifier
        zero_based = [value - 1 for value in raw_values]

        self._render_state.last_progress = ""
        try:
            if kind == "final":
                msg, progress_msg = self._final_view_state(zero_based)
            else:
                msg, progress_msg = self._selection_view_state(zero_based)

            self._render_state.last_progress = progress_msg or ""
            if msg is not None:
                self._handle_message(msg, store=False)

        except KeyError as e:
            self._print_missing_update(kind, raw_values, e)

        except Exception:
            self._render_invalid_identifier(invalid_id_msg)
            return

        self._render_user_input()

    def _render_invalid_identifier(self, invalid_id_msg: str) -> None:
        """Clear the screen and display guidance when the view ID is invalid."""
        clear_screen()
        self._render_state.reset_area()
        print(invalid_id_msg, end="", flush=True)

    def _final_view_state(self, values: Sequence[int]) -> Tuple[dict, str]:
        """Prepare header/message pair for a specific final run view."""
        outer, run = values
        self._last_seen_final_run_identifier = f"{outer + 1} {run + 1}"
        self._render_state.header = (
            f"Risk assessment run {run + 1} for outer fold {outer + 1}..."
        )
        msg = self._final_run_messages[outer][run]
        progress_msg = self._get_progress_text(
            self._final_progress_messages, outer, run
        )
        return msg, progress_msg

    def _selection_view_state(
        self, values: Sequence[int]
    ) -> Tuple[dict, str]:
        """Prepare header/message pair for a specific model selection run view."""
        outer, inner, config, run = values
        self._last_seen_model_selection_identifier = (
            f"{outer + 1} {inner + 1} {config + 1} {run + 1}"
        )
        self._render_state.header = (
            f"Model selection run {run + 1} for config {config + 1} "
            f"for outer fold {outer + 1}, inner fold {inner + 1}..."
        )
        msg = self._last_run_messages[outer][inner][config][run]
        progress_msg = self._get_progress_text(
            self._last_progress_messages, outer, run, inner, config
        )
        return msg, progress_msg

    def _print_missing_update(
        self, kind: str, values: Sequence[int], missing_key
    ) -> None:
        """Emit a placeholder line while waiting for a missing cached update."""
        if kind == "final":
            outer, run = values
            msg = (
                f"ProgressManager: waiting for updates for final run {run} "
                f"of outer fold {outer} (missing key {missing_key})..."
            )
        else:
            outer, inner, config, run = values
            msg = (
                "ProgressManager: waiting for updates for model selection "
                f"run {run} of config {config} (outer {outer}, inner {inner}) "
                f"(missing key {missing_key})..."
            )
        print(msg)

    def _get_progress_text(
        self, container: dict, outer: int, run: int, inner: int = None, config: int = None
    ) -> str:
        """Fetch the last stored progress text for the given run coordinates."""
        current = container.get(outer, {})
        if inner is not None:
            current = current.get(inner, {}).get(config, {})
        return current.get(run, "")

    def _start_input_listener(self) -> None:
        """Start non-blocking stdin listener thread for navigation commands."""
        if self.debug or self.progress_actor is None:
            return
        if self._input_thread is not None:
            return
        if not sys.stdin.isatty():
            return

        self._input_thread = threading.Thread(
            target=self._listen_for_user_input, daemon=True
        )
        self._input_thread.start()

    def _register_resize_handler(self) -> None:
        """Repaint the current view when the terminal is resized."""
        if self.debug or not sys.stdout.isatty():
            return
        try:
            signal.signal(signal.SIGWINCH, self._handle_resize)
        except Exception:
            pass

    def _handle_resize(self, signum, frame) -> None:
        """Signal handler that refreshes the UI on terminal resize."""
        if self.debug or not sys.stdout.isatty():
            return
        cols, _ = shutil.get_terminal_size((self.ncols, 24))
        self.ncols = cols
        self._render_state.global_header = self._build_global_header()
        for pbar in self.pbars:
            pbar.ncols = self.ncols
        self._refresh_view()

    def _listen_for_user_input(self) -> None:
        """Poll stdin in cbreak mode to capture keypresses without blocking tqdm."""
        fd = sys.stdin.fileno()
        try:
            old_settings = termios.tcgetattr(fd)
        except Exception:
            return

        try:
            tty.setcbreak(fd)
            while not self._stop_input_event.is_set():
                readable, _, _ = select.select([fd], [], [], 0.1)
                if not readable:
                    continue
                try:
                    char_bytes = os.read(fd, 1)
                    if not char_bytes:
                        continue
                    char = char_bytes.decode(errors="ignore")
                    # Best-effort assembly of ANSI escape sequences (e.g., arrows).
                    if char == "\x1b":
                        seq = [char]
                        while True:
                            ready, _, _ = select.select([fd], [], [], 0)
                            if not ready or len(seq) >= 3:
                                break
                            next_bytes = os.read(fd, 1)
                            if not next_bytes:
                                break
                            seq.append(next_bytes.decode(errors="ignore"))
                        char = "".join(seq)
                except Exception:
                    break
                if char:
                    self._handle_keypress(char)
        finally:
            try:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            except Exception:
                pass
            self._clear_input_overlay()

    def _handle_keypress(self, char: str) -> None:
        """Route keyboard input into navigation or command handling."""
        arrow_handlers = {
            "\x1b[A": self._handle_arrow_up,
            "\x1b[B": self._handle_arrow_down,
            "\x1b[C": self._handle_arrow_right,
            "\x1b[D": self._handle_arrow_left,
        }
        handler = arrow_handlers.get(char)
        if handler is not None:
            handler()
            return

        if char == ":" and not self._input_active:
            with self._input_lock:
                self._input_active = True
                self._input_buffer = ":"
            self._render_user_input()
            return

        if not self._input_active:
            return

        if char in {"\r", "\n"}:
            with self._input_lock:
                command = self._input_buffer.lstrip(":")
                self._input_buffer = ""
                self._input_active = False
            self._render_user_input()
            cmd = (command or "").strip().lower()
            if cmd in {"r", "refresh"}:
                self._refresh_view()
            else:
                self._change_view_mode(command if command else None)
            return

        if char in {"\x7f", "\b"}:
            with self._input_lock:
                self._input_buffer = self._input_buffer[:-1]
                if self._input_buffer == "":
                    self._input_active = False
            self._render_user_input()
            return

        with self._input_lock:
            self._input_buffer += char
        self._render_user_input()

    def _handle_arrow_up(self) -> None:
        """Toggle between the last seen selection and final views."""
        self._toggle_last_views()

    def _handle_arrow_down(self) -> None:
        """Toggle between the last seen selection and final views."""
        self._toggle_last_views()

    def _handle_arrow_right(self) -> None:
        """Move forward through run identifiers."""
        self._navigate_identifier(direction=1)

    def _handle_arrow_left(self) -> None:
        """Move backward through run identifiers."""
        self._navigate_identifier(direction=-1)

    def _toggle_last_views(self) -> None:
        """Switch focus between the most recent selection and final run views."""
        selection_id = self._last_seen_model_selection_identifier
        final_id = self._last_seen_final_run_identifier
        kind, *_ = self._parse_view_identifier(self._to_visualize) or (None,)

        if kind == "selection" and final_id:
            self._change_view_mode(final_id)
        elif kind == "final" and selection_id:
            self._change_view_mode(selection_id)
        elif kind is None:
            pass  # global view, don't do anything

    def _navigate_identifier(self, direction: int) -> None:
        """Step to the next or previous identifier based on current focus."""
        parsed = self._current_navigation_identifier()
        if parsed is None:
            return

        kind, *values = parsed
        if kind == "selection":
            new_values = self._step_selection(values, direction)
        else:
            new_values = self._step_final(values, direction)

        if new_values is None:
            return
        new_identifier = self._format_view_identifier(kind, new_values)
        self._change_view_mode(new_identifier)

    def _parse_view_identifier(
        self, identifier: Optional[str]
    ) -> Optional[_ViewIdentifier]:
        """Parse user-entered identifier into a typed tuple for navigation."""
        if not identifier or identifier in {"g", "global"}:
            return None
        try:
            values = [int(v) for v in identifier.split(" ")]
        except Exception:
            return None

        if len(values) == 2:
            return ("final", *values)
        if len(values) == 4:
            return ("selection", *values)
        return None

    def _current_navigation_identifier(self) -> Optional[_ViewIdentifier]:
        """
        Resolve the current identifier used for navigation, falling back to the
        last visited run if no explicit selection is active.
        """
        parsed = self._parse_view_identifier(self._to_visualize)
        if parsed is not None:
            return parsed

        fallback = (
            self._last_seen_model_selection_identifier
            or self._last_seen_final_run_identifier
        )
        if fallback is None:
            return None
        return self._parse_view_identifier(fallback)

    def _format_view_identifier(self, kind: str, values: Sequence[int]) -> str:
        """Rebuild an identifier string after a navigation step."""
        if kind == "final":
            outer, run = values
            return f"{outer} {run}"
        outer, inner, config, run = values
        return f"{outer} {inner} {config} {run}"

    def _step_selection(
        self, values: Sequence[int], direction: int
    ) -> Tuple[int, int, int, int]:
        """Increment/decrement selection run coordinates with wraparound semantics."""
        outer, inner, config, run = values

        if direction > 0:
            if run < self.config_runs:
                run += 1
            elif config < self.no_configs:
                config += 1
                run = 1
            elif inner < self.inner_folds:
                inner += 1
                config = 1
                run = 1
            elif outer < self.outer_folds:
                outer += 1
                inner = 1
                config = 1
                run = 1
        else:
            if run > 1:
                run -= 1
            elif config > 1:
                config -= 1
                run = self.config_runs
            elif inner > 1:
                inner -= 1
                config = self.no_configs
                run = self.config_runs
            elif outer > 1:
                outer -= 1
                inner = self.inner_folds
                config = self.no_configs
                run = self.config_runs

        return outer, inner, config, run

    def _step_final(
        self, values: Sequence[int], direction: int
    ) -> Tuple[int, int]:
        """Increment/decrement final run coordinates with wraparound semantics."""
        outer, run = values

        if direction > 0:
            if run < self.final_runs:
                run += 1
            elif outer < self.outer_folds:
                outer += 1
                run = 1
        else:
            if run > 1:
                run -= 1
            elif outer > 1:
                outer -= 1
                run = self.final_runs

        return outer, run

    def _clear_input_overlay(self) -> None:
        """Reset command overlay state and trigger redraw."""
        with self._input_lock:
            self._input_buffer = ""
            self._input_active = False
        self._render_user_input()

    def _refresh_view(self) -> None:
        """
        Force a redraw of the current view (global or focused).
        """
        # Forces a full render pass using current selection/global mode.
        if self.debug:
            return
        identifier = self._to_visualize or "g"
        self._change_view_mode(identifier, force_refresh=True)

    def _render_user_input(self) -> None:
        """Paint the command-line overlay while preserving current bars."""
        if self.debug or not sys.stdout.isatty():
            self._render_failure_message()
            return

        cols, rows = shutil.get_terminal_size((self.ncols, 24))
        with self._input_lock:
            text = self._input_buffer if self._input_active else ""
            prev_width = self._input_render_len
            width = min(max(prev_width, len(text)), cols)
            if len(text) > width and width > 0:
                text = text[-width:]
            self._input_render_len = width if self._input_active else 0

        if width == 0:
            self._render_failure_message()
            return

        # Save/restore cursor (\0337/\0338) so overlay does not affect tqdm cursor.
        col = max(1, cols - width + 1)
        line = text.ljust(width)
        print(f"\0337\033[{rows};{col}H{line}\0338", end="", flush=True)
        self._render_failure_message()

    def _render_failure_message(self) -> None:
        """
        Render a persistent failure message at the bottom-left corner.
        """
        # Keeps latest failure visible alongside overlay without disrupting bars.
        failure = self._render_state.failure
        if failure == "":
            return
        if not sys.stdout.isatty():
            return

        cols, rows = shutil.get_terminal_size((self.ncols, 24))
        with self._input_lock:
            overlay_width = self._input_render_len if self._input_active else 0

        # Reserve space for the input overlay if active.
        overlay_start = (
            max(1, cols - overlay_width + 1) if overlay_width else cols + 1
        )
        max_width = overlay_start - 1 if overlay_start > 1 else cols
        max_width = max(1, max_width)

        text = failure[:max_width].ljust(max_width)
        # Save/restore cursor so we don't move the tqdm cursor.
        print(f"\0337\033[{rows};1H{text}\0338", end="", flush=True)

    def _is_active_view(self, msg: dict) -> bool:
        """Check whether a message targets the currently selected view."""
        if self.debug:
            return True

        target = self._to_visualize
        if target in {None, "g", "global"}:
            return False

        try:
            values = [int(v) - 1 for v in target.split(" ")]
        except Exception:
            return False

        if len(values) == 2:
            outer, run = values
            return (
                msg.get(IS_FINAL)
                and msg.get(OUTER_FOLD) == outer
                and msg.get(RUN_ID) == run
            )

        if len(values) == 4:
            outer, inner, config, run = values
            return (
                not msg.get(IS_FINAL)
                and msg.get(OUTER_FOLD) == outer
                and msg.get(INNER_FOLD) == inner
                and msg.get(CONFIG_ID) == config
                and msg.get(RUN_ID) == run
            )

        return False

    def _render_progress(self, printer: Callable[[], int]) -> None:
        """
        Clear previous progress lines and invoke the provided printer.
        """
        # Centralized rendering hook used by message handlers to draw bars/text.
        # Render from a fixed top-left anchor so scrolling does not confuse the cursor.
        render_state = self._render_state
        if not sys.stdout.isatty():
            rendered_lines = printer()
            render_state.rendered_lines = (
                rendered_lines if isinstance(rendered_lines, int) else 0
            )
            return

        lines_to_clear = max(1, render_state.rendered_lines)
        # Jump to the anchor row before clearing anything.
        render_state.append_moves(f"\033[{render_state.origin_row};1H")
        for idx in range(lines_to_clear):
            self._clear_line()
            if idx < lines_to_clear - 1:
                self._cursor_down()

        # Reset to the anchor and render the new content.
        render_state.append_moves(f"\033[{render_state.origin_row};1H")
        rendered_lines = printer()
        render_state.rendered_lines = (
            rendered_lines if isinstance(rendered_lines, int) else 0
        )

    def _make_config_readable(self, obj):
        """
        Prepare configuration dictionaries for display.
        Ensures keys are strings and class specs show up as class_name(args).
        """
        # Used by _format_run_message to pretty-print configs.

        class _ClassSpec:
            def __init__(self, class_name, args):
                """
                Store a class spec for pretty-printing.

                Args:
                    class_name (str): Dotted class path or short class name.
                    args (dict | object | None): Constructor arguments.

                Side effects:
                    Normalizes ``args`` to an empty dict when ``None``.
                """
                self.class_name = class_name
                self.args = args if args is not None else {}

            def __repr__(self):
                """
                Return a readable ``ClassName(arg=value, ...)`` representation.

                Returns:
                    str: Human-friendly representation for terminal display.
                """
                if not self.args:
                    return f"{self.class_name}()"
                if not isinstance(self.args, dict):
                    return f"{self.class_name}({repr(self.args)})"

                parts = []
                for key, value in self.args.items():
                    value_repr = pformat(value, sort_dicts=False)
                    if "\n" in value_repr:
                        value_repr = value_repr.replace(
                            "\n", "\n" + " " * (len(key) + 1)
                        )
                    parts.append(f"{key}={value_repr}")
                return f"{self.class_name}({', '.join(parts)})"

        if isinstance(obj, dict):
            if "class_name" in obj:
                return _ClassSpec(
                    obj.get("class_name"),
                    self._make_config_readable(obj.get("args", {})),
                )
            return {
                str(k): self._make_config_readable(v) for k, v in obj.items()
            }
        if isinstance(obj, list):
            return [self._make_config_readable(v) for v in obj]
        if isinstance(obj, tuple):
            return tuple(self._make_config_readable(v) for v in obj)
        return obj

    def _stringify_config(self, readable_config) -> str:
        """
        Render the readable config without dict braces, as key: value lines.
        """
        # Converts processed config into multiline string for focused view.
        if not isinstance(readable_config, dict):
            return pformat(readable_config, sort_dicts=False)

        lines = []
        for key, value in readable_config.items():
            value_str = pformat(value, sort_dicts=False)
            if "\n" in value_str:
                value_str = value_str.replace(
                    "\n", "\n" + " " * (len(str(key)) + 2)
                )
            lines.append(f"{key}: {value_str}")
        return "\n".join(lines)

    def _format_run_message(self, msg: dict) -> str:
        """
        Format the progress message, appending the config (pretty printed)
        when available.
        """
        # Builds user-facing message combining payload and config details.
        base_message = msg.get("message") or ""
        config_id = msg.get(CONFIG_ID)
        config = self._model_configs[config_id]
        if config is None:
            return base_message

        readable_config = self._make_config_readable(config)
        config_str = self._stringify_config(readable_config)
        if base_message:
            return f"{base_message}\nConfig:\n{config_str}"
        return f"Config:\n{config_str}"

    def _handle_message(self, msg: dict, store: bool = True) -> None:
        """
        Handle a single progress message (shared between the polling loop
        and view replays).
        """
        # Core dispatcher that updates cached messages and drives rendering.
        msg_type = msg.get("type")
        render_state = self._render_state

        outer_fold = msg.get(OUTER_FOLD)
        inner_fold = msg.get(INNER_FOLD)
        config_id = msg.get(CONFIG_ID)
        run_id = msg.get(RUN_ID)

        if msg_type == START_CONFIG:
            # Avoid changing the header while a specific configuration view is active.
            if inner_fold is None and self._is_active_view(msg):
                render_state.header = (
                    f"Risk assessment run {run_id + 1} for outer fold {outer_fold + 1}..."
                )
            elif self._is_active_view(msg):
                render_state.header = (
                    f"Model selection run {run_id + 1} for config {config_id + 1} "
                    f"for outer fold {outer_fold + 1}, inner fold {inner_fold + 1}..."
                )

        elif msg_type == BATCH_PROGRESS:
            if self._is_active_view(msg):
                batch_id = msg.get(BATCH)
                total_batches = msg.get(TOTAL_BATCHES)
                epoch = msg.get(EPOCH)
                desc = msg.get("message")

                self._render_progress(
                    lambda: self._print_train_progress_bar(
                        batch_id, total_batches, epoch, desc
                    )
                )
            if store:
                self._store_last_run_message(msg)

        elif msg_type == RUN_PROGRESS:
            progress_msg = self._format_run_message(msg)
            if store:
                self._store_last_run_message(msg)
                self._store_last_progress_message(msg, progress_msg)

            if self._is_active_view(msg):
                batch_id = msg.get(BATCH)
                total_batches = msg.get(TOTAL_BATCHES)
                epoch = msg.get(EPOCH)
                desc = msg.get(MODE)
                render_state.last_progress = progress_msg
                self._render_progress(
                    lambda: self._print_train_progress_bar(
                        batch_id, total_batches, epoch, desc
                    )
                )

        elif msg_type == RUN_COMPLETED:
            pass  # do not store this message, not useful for now

        elif msg_type in {RUN_FAILED}:
            if store:
                self._store_last_run_message(msg)

            outer_fold = msg.get(OUTER_FOLD)
            inner_fold = msg.get(INNER_FOLD)
            config_id = msg.get(CONFIG_ID)
            run_id = msg.get(RUN_ID)

            if self._is_active_view(msg):
                clear_screen()
                render_state.reset_area()

                print(
                    f"Run failed: run {run_id + 1} for config {config_id + 1} for outer fold {outer_fold + 1}, inner fold {inner_fold + 1}... \nMessage: {msg.get('message')}"
                )
            else:
                failure_desc = (
                    f"Run failed: run {run_id + 1} for config {config_id + 1} "
                    f"for outer fold {outer_fold + 1}"
                )
                if inner_fold is not None:
                    failure_desc += f", inner fold {inner_fold + 1}"
                failure_desc += f"... Message: {msg.get('message')}"
                render_state.failure = failure_desc
                self._render_failure_message()

        elif msg_type == END_CONFIG:
            position = outer_fold * self.inner_folds + inner_fold
            elapsed = msg.get(ELAPSED)
            configs_times = self.times[position]
            # Compute delta t for a specific config
            configs_times[config_id] = (
                elapsed,
                True,
            )  # (time.time() - configs_times[config_id][0], True)

            # Update progress bar only when global view is visible to avoid redraws.
            if self._to_visualize in {"g", "global", None}:
                self.pbars[position].update()
                self.refresh_global_view()
            else:
                # manually modify the state only, otherwise tqdm will redraw
                pbar = self.pbars[position]
                pbar.n += 1
                pbar.last_print_n = pbar.n

        elif msg_type == END_FINAL_RUN:
            position = self.outer_folds * self.inner_folds + outer_fold
            elapsed = msg.get(ELAPSED)
            configs_times = self.times[position]
            # Compute delta t for a specific config
            configs_times[run_id] = (
                elapsed,
                True,
            )  # (time.time() - configs_times[run_id][0], True)
            # Update progress bar only when global view is visible to avoid redraws.
            if self._to_visualize in {"g", "global", None}:
                self.pbars[position].update()
                self.refresh_global_view()  # does not do anything in debug mode
            else:
                # manually modify the state only, otherwise tqdm will redraw
                pbar = self.pbars[position]
                pbar.n += 1
                pbar.last_print_n = pbar.n
        else:
            print(f"Cannot parse type of message {msg_type}, fix this.")

    def _print_train_progress_bar(
        self, batch_id: int, total_batches: int, epoch: int, desc: str
    ) -> int:
        """
        Simple progress bar printer for debug mode, avoids tqdm dependency.
        """
        # Used when debug mode disables tqdm; renders in-place via buffer helpers.
        total = max(total_batches, 1)
        progress = min(batch_id, total)
        filled = int(30 * progress / total)
        bar = "#" * filled + "-" * (30 - filled)
        percentage = int(progress / total * 100)
        msg = f"{desc} Epoch {epoch}: [{bar}] {percentage:3d}% ({progress}/{total})"

        render_state = self._render_state
        if render_state.header:
            msg = render_state.header + "\n" + msg
        if render_state.last_progress:
            msg = msg + "\n" + render_state.last_progress

        self._append_to_buffer(msg)
        self._flush_buffer()
        return msg.count("\n") + 1

    def update_state(self) -> None:
        """
        Updates the state of the progress bar (different from showing it
        on screen, see :func:`refresh`) by pulling batched updates from
        the progress actor.
        """
        # Polls actor for new messages and routes them through _handle_message.
        if self.progress_actor is None:
            print(
                "ProgressManager: cannot update the UI, no progress actor provided..."
            )
            return

        try:
            while True:
                messages, closed = ray.get(self.progress_actor.drain.remote())
                for msg in messages:
                    self._handle_message(msg)
                if closed and not messages:
                    break
                time.sleep(self._poll_interval)

        except Exception as e:
            print(f"{e}\n{traceback.format_exc()}")
            return
        finally:
            self._stop_input_event.set()

    def _cursor_up(self) -> None:
        """
        Moves cursor one line up without clearing it.
        """
        # Helper for manual rendering path in debug mode.
        # ANSI: move cursor up one line.
        self._render_state.append_moves("\033[F")

    def _cursor_down(self) -> None:
        """
        Moves cursor one line down without adding a newline.
        """
        # Helper for manual rendering path in debug mode.
        # ANSI: move cursor down one line.
        self._render_state.append_moves("\033[E")

    def _clear_line(self) -> None:
        """
        Clears the current line in the terminal.
        """
        # Helper for manual rendering path in debug mode.
        # ANSI: return carriage then clear to end of line.
        self._render_state.append_moves("\r\033[K")

    def _append_to_buffer(self, msg: str) -> None:
        """
        Appends text or ANSI moves to the pending buffer.
        """
        # Shared buffer to batch cursor moves + messages.
        self._render_state.append_moves(msg)

    def _clear_moves_buffer(self) -> None:
        """
        Clears the moves buffer.
        """
        # Reset buffered cursor/text before reuse.
        self._render_state.clear_moves()

    def _flush_buffer(self, render_overlay: bool = True) -> None:
        """Emit buffered cursor moves/text and optionally redraw overlays."""
        print(self._render_state.moves, end="", flush=True)
        self._clear_moves_buffer()
        if render_overlay:
            self._render_user_input()

    def _init_selection_pbar(self, i: int, j: int) -> tqdm.tqdm:
        """
        Initializes the progress bar for model selection

        Args:
            i (int): the id of the outer fold (from 0 to outer folds - 1)
            j (int): the id of the inner fold (from 0 to inner folds - 1)
        """
        # Creates tqdm bar representing one inner fold/config grid slot.
        position = i * self.inner_folds + j
        pbar = tqdm.tqdm(
            total=self.no_configs,
            ncols=self.ncols,
            ascii=True,
            position=position,
            unit="config",
            bar_format=" {desc} {percentage:3.0f}%|{bar}|{n_fmt}/{total_fmt}{postfix}",
            leave=False,
        )
        pbar.set_description(f"Out_{i + 1}/Inn_{j + 1}")
        mean = str(datetime.timedelta(seconds=0))
        pbar.set_postfix_str(f"(1 cfg every {mean})")
        return pbar

    def _init_assessment_pbar(self, i: int) -> tqdm.tqdm:
        """
        Initializes the progress bar for risk assessment

        Args:
            i (int): the id of the outer fold (from 0 to outer folds - 1)
        """
        # Creates tqdm bar tracking final runs per outer fold.
        position = self.outer_folds * self.inner_folds + i
        pbar = tqdm.tqdm(
            total=self.final_runs,
            ncols=self.ncols,
            ascii=True,
            position=position,
            unit="config",
            bar_format=" {desc} {percentage:3.0f}%|{bar}|{n_fmt}/{total_fmt}{postfix}",
            leave=False,
        )
        pbar.set_description(f"Final run {i + 1}")
        mean = str(datetime.timedelta(seconds=0))
        pbar.set_postfix_str(f"(1 run every {mean})")
        return pbar

    def _build_global_header(self) -> str:
        """Construct the banner shown in the global (aggregated) view."""
        left = "*" * ((self.ncols - 21) // 2 + 1)
        right = "*" * ((self.ncols - 21) // 2)
        return f"\033[F\033[A{left} Experiment Progress {right}\n\n"

    def _render_global_header(self) -> None:
        """Render the global header using the stored render state."""
        if self.debug:
            return
        header = self._render_state.global_header or self._build_global_header()
        self._render_state.global_header = header
        self._append_to_buffer(header)
        self._flush_buffer(render_overlay=False)

    def _render_global_view(self) -> None:
        """Render the header and all progress bars for the global view."""
        self._render_global_header()
        for i, pbar in enumerate(self.pbars):
            pbar.ncols = self.ncols
            # When resuming, do not consider completed exp. (delta approx. < 1)
            completion_times = [
                delta
                for k, (delta, completed) in self.times[i].items()
                if completed and delta > 1
            ]

            if len(completion_times) > 0:
                min_seconds = min(completion_times)
                max_seconds = max(completion_times)
                mean_seconds = sum(completion_times) / len(completion_times)
            else:
                min_seconds = 0
                max_seconds = 0
                mean_seconds = 0

            mean_time = str(datetime.timedelta(seconds=mean_seconds)).split(
                "."
            )[0]
            min_time = str(datetime.timedelta(seconds=min_seconds)).split(
                "."
            )[0]
            max_time = str(datetime.timedelta(seconds=max_seconds)).split(
                "."
            )[0]

            pbar.set_postfix_str(
                f"min:{min_time}|avg:{mean_time}|max:{max_time}"
            )
            pbar.refresh()
        self.show_footer()

    def show_header(self):
        """
        Prints the header of the progress bar
        """
        """
        \033[F --> move cursor to the beginning of the previous line
        \033[A --> move cursor up one line
        \033[<N>A --> move cursor up N lines
        """
        # Draws banner; called before rendering bars.
        self._render_global_header()

    def show_footer(self):
        """
        Prints the footer of the progress bar
        """
        # Placeholder for future footer output; kept for symmetry.
        pass  # need to work how how to print after tqdm

    def refresh_global_view(self):
        """
        Refreshes the progress bar
        """
        # Recomputes tqdm postfixes and redraws global view.
        if self.debug:
            return
        if self._to_visualize in {"g", "global", None}:
            self._render_global_view()
        self._render_user_input()

    def _selection_slot(
        self, container: dict, outer: int, inner: int, config: int
    ) -> dict:
        """Helper to build nested dict structure for selection runs."""
        return (
            container.setdefault(outer, {})
            .setdefault(inner, {})
            .setdefault(config, {})
        )

    def _final_slot(self, container: dict, outer: int) -> dict:
        """Helper to build nested dict structure for final runs."""
        return container.setdefault(outer, {})

    def _store_last_run_message(self, msg: dict) -> None:
        """
        Stores the latest progress message for a specific run.
        """
        # Keeps latest raw payloads for replay when user focuses a run.
        outer = msg.get(OUTER_FOLD)
        run = msg.get(RUN_ID)

        if msg.get(IS_FINAL):
            self._final_slot(self._final_run_messages, outer)[run] = msg
            return

        inner = msg.get(INNER_FOLD)
        config = msg.get(CONFIG_ID)

        selection_runs = self._selection_slot(
            self._last_run_messages, outer, inner, config
        )
        selection_runs[run] = msg

    def _store_last_progress_message(
        self, msg: dict, progress_msg: str
    ) -> None:
        """
        Stores the latest formatted progress text for a specific run.
        """
        # Persists rendered text per run for immediate replay in focused view.
        outer = msg.get(OUTER_FOLD)
        run = msg.get(RUN_ID)

        if msg.get(IS_FINAL):
            self._final_slot(self._final_progress_messages, outer)[
                run
            ] = progress_msg
            return

        inner = msg.get(INNER_FOLD)
        config = msg.get(CONFIG_ID)

        selection_runs = self._selection_slot(
            self._last_progress_messages, outer, inner, config
        )
        selection_runs[run] = progress_msg or ""
