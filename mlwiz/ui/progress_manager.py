import datetime
import os
import select
import shutil
import sys
import termios
import threading
import time
import tty
from copy import deepcopy
from pprint import pformat
from typing import Callable
import tqdm
import ray

from mlwiz.static import *


def clear_screen():
    """
    Clears the CLI interface.
    """
    # Mimic Ctrl+L: clear visible screen and move cursor to top without
    # wiping scrollback/history.
    print("\033[2J\033[H", end="", flush=True)


@ray.remote(num_cpus=0)
class ProgressManagerActor:
    """
    Ray actor used to aggregate progress updates from workers.
    The driver periodically pulls pending messages instead of relying on a queue.
    """

    def __init__(self):
        self._pending_messages = []
        self._closed = False

    def push(self, payload: dict):
        if self._closed:
            return
        self._pending_messages.append(deepcopy(payload))

    def drain(self):
        """
        Returns buffered messages and clears the buffer.
        """
        pending = self._pending_messages
        self._pending_messages = []
        return pending, self._closed

    def close(self):
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
        final_runs,
        debug=True,
        progress_actor=None,
        poll_interval: float = 0.2,
    ):
        self.ncols = 100
        self.outer_folds = outer_folds
        self.inner_folds = inner_folds
        self.no_configs = no_configs
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

        # used to combine printing of multiple information in debug mode
        self._header_run_message = ""
        self._last_progress_msg = ""
        self._moves_buffer = ""
        self._failure_message = ""
        self._rendered_lines = 0
        # Anchor row (1-based) where custom rendering starts; keeps drawing absolute.
        self._render_origin_row = 1

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

    def set_model_configs(self, model_configs):
        """
        Set model configurations so copies don't have to be stored
        on the progress actor.

        :param model_configs: list of hparams
        """
        self._model_configs = model_configs

    def _change_view_mode(
        self, identifier: str = None, force_refresh: bool = False
    ):
        """
        Changes the view mode of the progress manager.
        If config_id is None, the global view is activated.
        Otherwise, the progress of a single configuration is visualized.

        Args:
            identifier (str): the reference to the configuration
            to visualize in the form of "outer_id_inner_id_config_id_run_id"

        """
        assert not self.debug, "Cannot change view mode in debug mode"
        identifier = (identifier or "g").strip()

        if identifier == "" or identifier in {"g", "global"}:
            identifier = "g"

        if force_refresh or self._to_visualize != identifier:
            self._to_visualize = identifier
            clear_screen()
            self._rendered_lines = 0
            self._render_origin_row = 1

        if self._to_visualize in {"g", "global"}:
            self.refresh()
            self._render_user_input()
            return

        invalid_id_msg = "ProgressManager: invalid identifier format, use outer_inner_config_run format or outer_run..."

        # put last stored message on screen to display it
        try:
            values = self._to_visualize.split("_")
            if (
                (len(values) != 2 and len(values) != 4)
                or "0" in values
                or 0 in values
            ):
                raise Exception(invalid_id_msg)
        except Exception:
            clear_screen()
            self._render_origin_row = 1
            print(invalid_id_msg, end="", flush=True)
            return

        try:
            msg = None
            if len(values) == 2:
                outer, run = int(values[0]) - 1, int(values[1]) - 1
                msg = self._final_run_messages[int(outer)][int(run)]
                self._header_run_message = f"Risk assessment run {run + 1} for outer fold {outer + 1}..."

            elif len(values) == 4:
                outer, inner, config, run = (
                    int(values[0]) - 1,
                    int(values[1]) - 1,
                    int(values[2]) - 1,
                    int(values[3]) - 1,
                )
                msg = self._last_run_messages[int(outer)][int(inner)][
                    int(config)
                ][int(run)]
                self._header_run_message = f"Model selection run {run + 1} for config {config + 1} for outer fold {outer + 1}, inner fold {inner + 1}..."

            if msg is not None:
                self._handle_message(
                    msg, store=False
                )  # do not store message already stored
        except KeyError:
            print("ProgressManager: waiting for the next update...")
        except Exception:
            clear_screen()
            print(invalid_id_msg, end="", flush=True)
            return

        self._render_user_input()

    def _start_input_listener(self):
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

    def _listen_for_user_input(self):
        # Put stdin in cbreak mode and poll for keypresses without blocking tqdm.
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
                    char = os.read(fd, 1).decode(errors="ignore")
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

    def _handle_keypress(self, char: str):
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

    def _clear_input_overlay(self):
        with self._input_lock:
            self._input_buffer = ""
            self._input_active = False
        self._render_user_input()

    def _refresh_view(self):
        """
        Force a redraw of the current view (global or focused).
        """
        if self.debug:
            return
        identifier = self._to_visualize or "g"
        self._change_view_mode(identifier, force_refresh=True)

    def _render_user_input(self):
        if self.debug or not sys.stdout.isatty():
            self._render_failure_message()
            return

        cols, rows = shutil.get_terminal_size((self.ncols, 24))
        with self._input_lock:
            text = self._input_buffer if self._input_active else ""
            width = max(self._input_render_len, len(text))
            self._input_render_len = len(text)

        if width == 0:
            self._render_failure_message()
            return

        # Save/restore cursor (\0337/\0338) so overlay does not affect tqdm cursor.
        col = max(1, cols - width + 1)
        line = text.ljust(width)
        print(f"\0337\033[{rows};{col}H{line}\0338", end="", flush=True)
        self._render_failure_message()

    def _render_failure_message(self):
        """
        Render a persistent failure message at the bottom-left corner.
        """
        if self._failure_message == "":
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

        text = self._failure_message[:max_width].ljust(max_width)
        # Save/restore cursor so we don't move the tqdm cursor.
        print(f"\0337\033[{rows};1H{text}\0338", end="", flush=True)

    def _is_active_view(self, msg: dict) -> bool:
        # Decide whether a progress message belongs to the currently selected view.
        if self.debug:
            return True

        target = self._to_visualize
        if target in {None, "g", "global"}:
            return False

        try:
            values = [int(v) - 1 for v in target.split("_")]
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

    def _render_progress(self, printer: Callable[[], int]):
        """
        Clear previous progress lines and invoke the provided printer.
        """
        # Render from a fixed top-left anchor so scrolling does not confuse the cursor.
        if not sys.stdout.isatty():
            rendered_lines = printer()
            self._rendered_lines = (
                rendered_lines if isinstance(rendered_lines, int) else 0
            )
            return

        lines_to_clear = max(1, self._rendered_lines)
        # Jump to the anchor row before clearing anything.
        self._moves_buffer += f"\033[{self._render_origin_row};1H"
        for idx in range(lines_to_clear):
            self._clear_line()
            if idx < lines_to_clear - 1:
                self._cursor_down()

        # Reset to the anchor and render the new content.
        self._moves_buffer += f"\033[{self._render_origin_row};1H"
        rendered_lines = printer()
        self._rendered_lines = (
            rendered_lines if isinstance(rendered_lines, int) else 0
        )

    def _print_run_progress(self, msg: str):
        """
        Print a generic progress message respecting header ordering.
        """
        text = msg or ""
        parts = []
        if self._header_run_message:
            parts.append(self._header_run_message)
        if text:
            parts.append(text)
        rendered = "\n".join(parts)
        self._append_to_buffer(rendered)
        self._flush_buffer()
        return rendered.count("\n") + 1 if rendered else 0

    def _make_config_readable(self, obj):
        """
        Prepare configuration dictionaries for display.
        Ensures keys are strings and class specs show up as class_name(args).
        """

        class _ClassSpec:
            def __init__(self, class_name, args):
                self.class_name = class_name
                self.args = args if args is not None else {}

            def __repr__(self):
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

    def _handle_message(self, msg: dict, store: bool = True):
        """
        Handle a single progress message (shared between the polling loop
        and view replays).
        """
        type = msg.get("type")

        outer_fold = msg.get(OUTER_FOLD)
        inner_fold = msg.get(INNER_FOLD)
        config_id = msg.get(CONFIG_ID)
        run_id = msg.get(RUN_ID)

        if type == START_CONFIG:
            # Avoid changing the header while a specific configuration view is active.
            if inner_fold is None and self._is_active_view(msg):
                self._header_run_message = f"Risk assessment run {run_id + 1} for outer fold {outer_fold + 1}..."
            elif self._is_active_view(msg):
                self._header_run_message = f"Model selection run {run_id + 1} for config {config_id + 1} for outer fold {outer_fold + 1}, inner fold {inner_fold + 1}..."

        elif type == BATCH_PROGRESS:
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

        elif type == RUN_PROGRESS:
            if store:
                self._store_last_run_message(msg)
            if self._is_active_view(msg):
                self._last_progress_msg = self._format_run_message(msg)
                self._render_progress(
                    lambda: self._print_run_progress(self._last_progress_msg)
                )

        elif type == RUN_COMPLETED:
            pass  # do not store this message, not useful for now

        elif type in {RUN_FAILED}:
            if store:
                self._store_last_run_message(msg)

            outer_fold = msg.get(OUTER_FOLD)
            inner_fold = msg.get(INNER_FOLD)
            config_id = msg.get(CONFIG_ID)
            run_id = msg.get(RUN_ID)

            if self._is_active_view(msg):
                clear_screen()
                self._render_origin_row = 1

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
                self._failure_message = failure_desc
                self._render_failure_message()

        elif type == END_CONFIG:
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
                self.refresh()  # does not do anything in debug mode
            else:
                # manually modify the state only, otherwise tqdm will redraw
                pbar = self.pbars[position]
                pbar.n += 1
                pbar.last_print_n = pbar.n

        elif type == END_FINAL_RUN:
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
                self.refresh()  # does not do anything in debug mode
            else:
                # manually modify the state only, otherwise tqdm will redraw
                pbar = self.pbars[position]
                pbar.n += 1
                pbar.last_print_n = pbar.n
        else:
            print(f"Cannot parse type of message {type}, fix this.")

    def _print_train_progress_bar(
        self, batch_id: int, total_batches: int, epoch: int, desc: str
    ):
        """
        Simple progress bar printer for debug mode, avoids tqdm dependency.
        """
        total = max(total_batches, 1)
        progress = min(batch_id, total)
        filled = int(30 * progress / total)
        bar = "#" * filled + "-" * (30 - filled)
        percentage = int(progress / total * 100)
        msg = f"{desc} Epoch {epoch}: [{bar}] {percentage:3d}% ({progress}/{total})"

        if self._header_run_message != "":
            msg = self._header_run_message + "\n" + msg
        if self._last_progress_msg != "":
            msg = msg + "\n" + self._last_progress_msg

        self._append_to_buffer(msg)
        self._flush_buffer()
        return msg.count("\n") + 1

    def update_state(self):
        """
        Updates the state of the progress bar (different from showing it
        on screen, see :func:`refresh`) by pulling batched updates from
        the progress actor.
        """
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
            print(e)
            return
        finally:
            self._stop_input_event.set()

    def _cursor_up(self):
        """
        Moves cursor one line up without clearing it.
        """
        # ANSI: move cursor up one line.
        self._moves_buffer += "\033[F"

    def _cursor_down(self):
        """
        Moves cursor one line down without adding a newline.
        """
        # ANSI: move cursor down one line.
        self._moves_buffer += "\033[E"

    def _clear_line(self):
        """
        Clears the current line in the terminal.
        """
        # ANSI: return carriage then clear to end of line.
        self._moves_buffer += "\r\033[K"

    def _append_to_buffer(self, msg: str):
        """
        Clears the moves buffer.
        """
        self._moves_buffer += msg

    def _clear_moves_buffer(self):
        """
        Clears the moves buffer.
        """
        self._moves_buffer = ""

    def _flush_buffer(self):
        # Dump buffered cursor moves + text, then redraw the input overlay.
        print(self._moves_buffer, end="", flush=True)
        self._clear_moves_buffer()
        self._render_user_input()

    def _init_selection_pbar(self, i: int, j: int):
        """
        Initializes the progress bar for model selection

        Args:
            i (int): the id of the outer fold (from 0 to outer folds - 1)
            j (int): the id of the inner fold (from 0 to inner folds - 1)
        """
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

    def _init_assessment_pbar(self, i: int):
        """
        Initializes the progress bar for risk assessment

        Args:
            i (int): the id of the outer fold (from 0 to outer folds - 1)
        """
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

    def show_header(self):
        """
        Prints the header of the progress bar
        """
        """
        \033[F --> move cursor to the beginning of the previous line
        \033[A --> move cursor up one line
        \033[<N>A --> move cursor up N lines
        """
        print(
            f"\033[F\033[A{'*' * ((self.ncols - 21) // 2 + 1)} "
            f"Experiment Progress {'*' * ((self.ncols - 21) // 2)}\n",
            end="\n",
            flush=True,
        )

    def show_footer(self):
        """
        Prints the footer of the progress bar
        """
        pass  # need to work how how to print after tqdm

    def refresh(self):
        """
        Refreshes the progress bar
        """
        if self.debug:
            return
        if self._to_visualize in {"g", "global", None}:
            self.show_header()
            for i, pbar in enumerate(self.pbars):
                # When resuming, do not consider completed exp. (delta approx. < 1)
                completion_times = [
                    delta
                    for k, (delta, completed) in self.times[i].items()
                    if completed and delta > 1
                ]

                if len(completion_times) > 0:
                    min_seconds = min(completion_times)
                    max_seconds = max(completion_times)
                    mean_seconds = sum(completion_times) / len(
                        completion_times
                    )
                else:
                    min_seconds = 0
                    max_seconds = 0
                    mean_seconds = 0

                mean_time = str(
                    datetime.timedelta(seconds=mean_seconds)
                ).split(".")[0]
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
        self._render_user_input()

    def _store_last_run_message(self, msg: dict):
        """
        Stores the latest progress message for a specific run.
        """
        outer = msg.get(OUTER_FOLD)
        run = msg.get(RUN_ID)

        if msg.get(IS_FINAL, False):
            if outer not in self._final_run_messages:
                self._final_run_messages[outer] = {}
            self._final_run_messages[outer][run] = msg
            return

        inner = msg.get(INNER_FOLD)
        config = msg.get(CONFIG_ID)

        if outer not in self._last_run_messages:
            self._last_run_messages[outer] = {}
        if inner not in self._last_run_messages[outer]:
            self._last_run_messages[outer][inner] = {}
        if config not in self._last_run_messages[outer][inner]:
            self._last_run_messages[outer][inner][config] = {}

        self._last_run_messages[outer][inner][config][run] = msg
