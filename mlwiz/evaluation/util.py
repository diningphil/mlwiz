import builtins
import datetime
import json
import math
import os
import random
import select
import shutil
import sys
import termios
import threading
import tty
from typing import List, Tuple, Callable

import numpy as np
import pandas as pd
import torch
import tqdm
from scipy import stats

from mlwiz.data.dataset import DatasetInterface
from mlwiz.data.provider import DataProvider
from mlwiz.model.interface import ModelInterface
from mlwiz.static import *
from mlwiz.util import dill_load, return_class_and_args, s2c


def clear_screen():
    """
    Clears the CLI interface.
    """
    # Mimic Ctrl+L: clear visible screen and move cursor to top without
    # wiping scrollback/history.
    print("\033[2J\033[H", end="", flush=True)


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
        progress_queue (ray.util.Queue): the queue used to receive progress
            messages from different workers
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
        progress_queue=None,
    ):
        self.ncols = 100
        self.outer_folds = outer_folds
        self.inner_folds = inner_folds
        self.no_configs = no_configs
        self.final_runs = final_runs
        self.pbars = []
        self.debug = debug
        self.progress_queue = progress_queue

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

    def _change_view_mode(self, identifier: str = None):
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

        if self._to_visualize != identifier:
            self._to_visualize = identifier
            clear_screen()

        if self._to_visualize in {"g", "global"}:
            self.refresh()
            self._render_user_input()
            return

        invalid_id_msg = "ProgressManager: invalid identifier format, use outer_inner_config_run format or outer_run..."

        # put last stored message in queue to display it
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
        if self.debug or self.progress_queue is None:
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
        overlay_start = max(1, cols - overlay_width + 1) if overlay_width else cols + 1
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

    def _render_progress(self, printer: Callable[[], None]):
        """
        Clear previous progress lines and invoke the provided printer.
        """
        if self._header_run_message != "":
            self._clear_line()
            self._cursor_up()
        if self._last_progress_msg != "":
            self._clear_line()
            self._cursor_up()

        self._clear_line()
        self._cursor_up()
        printer()

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
        self._append_to_buffer("\n".join(parts))
        self._flush_buffer()

    def _handle_message(self, msg: dict, store: bool = True):
        """
        Handle a single progress message (shared between queue consumer
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

        elif type == RUN_PROGRESS:
            if store:
                self._store_last_run_message(msg)

            if self._is_active_view(msg):
                self._last_progress_msg = msg.get("message")
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

    def update_state(self):
        """
        Updates the state of the progress bar (different from showing it
        on screen, see :func:`refresh`) once a message is sent to
        the progress queue
        """
        if self.progress_queue is None:
            print(
                "ProgressManager: cannot update the UI, no progress queue provided..."
            )
            return

        try:
            while True:
                msg = self.progress_queue.get()

                if msg is None:
                    break

                self._handle_message(msg)

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


"""
Various options for random search model selection
"""


def choice(*args):
    """
    Implements a random choice between a list of values
    """
    return random.choice(args)


def uniform(*args):
    """
    Implements a uniform sampling given an interval
    """
    return random.uniform(*args)


def normal(*args):
    """
    Implements a univariate normal sampling given its parameters
    """
    return random.normalvariate(*args)


def randint(*args):
    """
    Implements a random integer sampling in an interval
    """
    return random.randint(*args)


def loguniform(*args):
    r"""
    Performs a log-uniform random selection.

    Args:
        *args: a tuple of (log min, log max, [base]) to use. Base 10 is used
            if the third argument is not available.

    Returns:
        a randomly chosen value
    """
    log_min, log_max, *base = args
    base = base[0] if len(base) > 0 else 10

    log_min = math.log(log_min) / math.log(base)
    log_max = math.log(log_max) / math.log(base)

    return base ** (random.uniform(log_min, log_max))


def retrieve_experiments(
    model_selection_folder, skip_results_not_found: bool = False
) -> List[dict]:
    """
    Once the experiments are done, retrieves the config_results.json files of
    all configurations in a specific model selection folder, and returns them
    as a list of dictionaries

    :param model_selection_folder: path to the folder of a model selection,
        that is, your_results_path/..../MODEL_SELECTION/
    :param skip_results_not_found: whether to skip an experiment if a
        `config_results.json` file has not been produced yet. Useful when
        analyzing experiments while others still run.
    :return: a list of dictionaries, one per configuration, each with an extra
        key "exp_folder" which identifies the config folder.
    """
    config_directory = os.path.join(model_selection_folder)

    if not os.path.exists(config_directory):
        raise FileNotFoundError(f"Directory not found: {config_directory}")

    folder_names = []
    for _, dirs, _ in os.walk(config_directory):
        for d in dirs:
            if "config" in d:
                folder_names.append(os.path.join(config_directory, d))
        break  # do not recursively explore subfolders

    configs = []
    for cf in folder_names:
        config_results_path = os.path.join(cf, "config_results.json")
        if not os.path.exists(config_results_path) and skip_results_not_found:
            continue

        exp_info = json.load(open(config_results_path, "rb"))
        exp_config = exp_info

        exp_config["exp_folder"] = cf
        configs.append(exp_config)

    return configs


def create_dataframe(
    config_list: List[dict], key_mappings: List[Tuple[str, Callable]]
):
    """
    Creates a pandas DataFrame from a list of configuration dictionaries and key mappings.

    Args:
        config_list : List[dict]
            A list of dictionaries, where each dictionary represents a configuration. Each configuration
            must contain an `exp_folder` key and may include nested keys corresponding to hyperparameter names.

        key_mappings : List[Tuple[str, Callable]]
            A list of tuples where:
            - The first element (`str`) is the hyperparameter name to extract from the configurations.
            - The second element (`Callable`) is a transformation function to apply to the extracted value.

    Returns:
        df : pandas.DataFrame
            A DataFrame containing rows generated from `config_list` with columns for `exp_folder`
            and the specified key_mappings. If a mapping value is missing, the corresponding
            DataFrame cell will contain `None`.
    """

    def _finditem(obj, key):
        if key in obj:
            return obj[key]

        for k, v in obj.items():
            if isinstance(v, dict):
                item = _finditem(v, key)
                if item is not None:
                    return item

        return None

    df_rows = []

    for config in config_list:
        new_row = {"exp_folder": config["exp_folder"]}
        for hp_name, t_caster in key_mappings:
            cf_v = _finditem(config, hp_name)
            new_row[hp_name] = t_caster(cf_v) if cf_v is not None else None

        # Append the new row to the DataFrame
        df_rows.append(new_row)

    df = pd.DataFrame.from_records(
        df_rows, columns=[h[0] for h in key_mappings] + ["exp_folder"]
    )

    return df


def filter_experiments(
    config_list: List[dict], logic: bool = "AND", parameters: dict = {}
):
    """
    Filters the list of configurations returned by the method ``retrieve_experiments`` according to a dictionary.
    The dictionary contains the keys and values of the configuration files you are looking for.

    If you specify more then one key/value pair to look for, then the `logic` parameter specifies whether you want to filter
    using the AND/OR rule.

    For a key, you can specify more than one possible value you are interested in by passing a list as the value, for instance
    {'device': 'cpu', 'lr': [0.1, 0.01]}

    Args:
        config_list: The list of configuration files
        logic: if ``AND``, a configuration is selected iff all conditions are satisfied. If ``OR``, a config is selected when at least
            one of the criteria is met.
        parameters: dictionary with parameters used to filter the configurations

    Returns:
        a list of filtered configurations like the one in input
    """

    def _finditem(obj, key):
        if key in obj:
            return obj[key]

        for k, v in obj.items():
            if isinstance(v, dict):
                item = _finditem(v, key)
                if item is not None:
                    return item

        return None

    assert logic in ["AND", "OR"], "logic can only be AND/OR case sensitive"

    filtered_config_list = []

    for config in config_list:
        keep = True if logic == "AND" else False

        for k, v in parameters.items():
            cf_v = _finditem(config, k)
            assert (
                cf_v is not None
            ), f"Key {k} not found in the configuration, check your input"

            if type(v) == list:
                assert len(v) > 0, (
                    f'the list of values for key "{k}" cannot be'
                    f" empty, consider removing this key"
                )

                # the user specified a list of acceptable values
                # it is sufficient that one of them is present to return True
                if cf_v in v and logic == "OR":
                    keep = True
                    break

                if cf_v not in v and logic == "AND":
                    keep = False
                    break

            else:
                if v == cf_v and logic == "OR":
                    keep = True
                    break

                if v != cf_v and logic == "AND":
                    keep = False
                    break

        if keep:
            filtered_config_list.append(config)

    return filtered_config_list


def retrieve_best_configuration(model_selection_folder) -> dict:
    """
    Once the experiments are done, retrieves the winning configuration from
    a specific model selection folder, and returns it as a dictionaries

    :param model_selection_folder: path to the folder of a model selection,
        that is, your_results_path/..../MODEL_SELECTION/
    :return: a dictionary with info about the best configuration
    """
    config_directory = os.path.join(model_selection_folder)

    if not os.path.exists(config_directory):
        raise FileNotFoundError(f"Directory not found: {config_directory}")

    best_config = json.load(
        open(os.path.join(config_directory, "winner_config.json"), "rb")
    )
    return best_config


def instantiate_dataset_from_config(config: dict) -> DatasetInterface:
    """
    Instantiate a dataset from a configuration file.

    :param config (dict): the configuration file
    :return: an instance of DatasetInterface, i.e., the dataset
    """
    storage_folder = config[CONFIG][STORAGE_FOLDER]
    dataset_class = s2c(config[CONFIG][DATASET_CLASS])
    return dataset_class(storage_folder)


def instantiate_data_provider_from_config(
    config: dict, splits_filepath: str, n_outer_folds: int, n_inner_folds: int
) -> DataProvider:
    """
    Instantiate a data provider from a configuration file.
    :param config (dict): the configuration file
    :param splits_filepath (str): the path to data splits file
    :param n_outer_folds (int): the number of outer folds
    :param n_inner_folds (int): the number of inner folds
    :return: an instance of DataProvider, i.e., the data provider
    """
    storage_folder = config[CONFIG][STORAGE_FOLDER]
    dataset_class = s2c(config[CONFIG][DATASET_CLASS])
    dataset_getter = s2c(config[CONFIG][DATASET_GETTER])
    dl_class, dl_args = return_class_and_args(config[CONFIG], DATA_LOADER)

    return dataset_getter(
        storage_folder=storage_folder,
        splits_filepath=splits_filepath,
        dataset_class=dataset_class,
        data_loader_class=dl_class,
        data_loader_args=dl_args,
        outer_folds=n_outer_folds,
        inner_folds=n_inner_folds,
    )


def instantiate_model_from_config(
    config: dict,
    dataset: DatasetInterface,
) -> ModelInterface:
    """
    Instantiate a model from a configuration file.
    :param config (dict): the configuration file
    :param dataset (DatasetInterface): the dataset used in the experiment
    :return: an instance of ModelInterface, i.e., the model
    """
    config_ = config[CONFIG]
    model_class = s2c(config_[MODEL])
    model = model_class(
        dataset.dim_input_features,
        dataset.dim_target,
        config=config_,
    )

    return model


def load_checkpoint(
    checkpoint_path: str, model: ModelInterface, device: torch.device
):
    """
    Load a checkpoint from a checkpoint file into a model.
    :param checkpoint_path: the checkpoint file path
    :param model (ModelInterface): the model
    :param device (torch.device): the device, e.g, "cpu" or "cuda"
    """
    ckpt_dict = torch.load(
        checkpoint_path,
        map_location="cpu" if device == "cpu" else None,
        weights_only=True,
    )
    model_state = ckpt_dict[MODEL_STATE]

    # Needed only when moving from cpu to cuda (due to changes in config
    # file). Move all parameters to cuda.
    for param in model_state.keys():
        model_state[param] = model_state[param].to(device)

    model.load_state_dict(model_state)


def get_scores_from_outer_results(
    exp_folder, outer_fold_id, metric_key="main_score"
) -> dict:
    """
    Extracts scores from the configuration dictionary.
    Args:
        exp_folder (str): The path to the experiment folder.
        outer_fold_id (int): The ID of the outer fold, from 1 on.
        metric_key (str): The key for the metric to extract. Default is 'main_score'.
    """
    config_dict = json.load(
        open(
            os.path.join(
                exp_folder,
                f"MODEL_ASSESSMENT/OUTER_FOLD_{outer_fold_id}/outer_results.json",
            ),
            "rb",
        )
    )

    # Extract scores for the specified metric from the config dictionary
    scores = {
        "training": config_dict["outer_train"][metric_key],
        "validation": config_dict["outer_validation"][metric_key],
        "test": config_dict["outer_test"][metric_key],
        "training_std": config_dict["outer_train"][metric_key + "_std"],
        "validation_std": config_dict["outer_validation"][metric_key + "_std"],
        "test_std": config_dict["outer_test"][metric_key + "_std"],
    }

    return scores


def get_scores_from_assessment_results(
    exp_folder, metric_key="main_score"
) -> dict:
    """
    Extracts scores from the configuration dictionary.
    Args:
        exp_folder (str): The path to the experiment folder.
        metric_key (str): The key for the metric to extract. Default is 'main_score'.
    """
    config_dict = json.load(
        open(
            os.path.join(
                exp_folder, "MODEL_ASSESSMENT/assessment_results.json"
            ),
            "rb",
        )
    )

    # Extract scores for the specified metric from the config dictionary
    scores = {
        "training": config_dict["avg_training_" + metric_key],
        "validation": config_dict["avg_validation_" + metric_key],
        "test": config_dict["avg_test_" + metric_key],
        "training_std": config_dict["std_training_" + metric_key],
        "validation_std": config_dict["std_validation_" + metric_key],
        "test_std": config_dict["std_test_" + metric_key],
    }

    return scores


def _df_to_latex_table(df, no_decimals=2, model_as_row=True):
    # Pivot the table: index=model, columns=dataset, values=training (training_std)
    float_format = f".{no_decimals}f"

    def format_entry(x, mode="test"):
        return f"{round(x[f'{mode}'], no_decimals):{float_format}} ({round(x[f'{mode}_std'], no_decimals):{float_format}})"

    # Apply the formatting row-wise
    df["formatted"] = df.apply(format_entry, axis=1)

    # Pivot to desired shape
    if model_as_row:
        pivot_df = df.pivot(
            index="model", columns="dataset", values="formatted"
        )
    else:
        pivot_df = df.pivot(
            index="dataset", columns="model", values="formatted"
        )

    # Reset index to have 'model' or 'dataset' as a column
    pivot_df = pivot_df.reset_index()

    # Generate LaTeX
    latex = pivot_df.to_latex(index=False, escape=False, na_rep="--")
    return latex


def create_latex_table_from_assessment_results(
    exp_metadata,
    metric_key="main_score",
    no_decimals="2",
    model_as_row=True,
    use_single_outer_fold=False,
) -> str:
    """
    Creates a LaTeX table from a list of experiment folders, each containing assessment results.
    Args:
        exp_metadata (list[tuple(str,str,str)]): A list of (paths to the experiment folder, model name, dataset name).
        metric_key (str): The key for the metric to extract. Default is 'main_score'.
        no_decimals (int): The number of rounded decimal places to display in the LaTeX table.
        model_as_row (bool): If True, models are rows and datasets are columns. If False, the opposite.
        use_single_outer_fold (bool): If True, only the first outer fold is used. This is useful
            because when the number of outer folds is 1, the std in the assessment file is 0,
            therefore we want to recover the std across the final runs of the unique outer fold.
    """
    # Initialize a list to store the data frames
    dataframes = []

    # Loop through each experiment folder
    for exp_folder, model, dataset in exp_metadata:
        # Load the assessment results from the JSON file
        if not use_single_outer_fold:
            assessment_results = get_scores_from_assessment_results(
                exp_folder, metric_key
            )
        else:
            assessment_results = get_scores_from_outer_results(
                exp_folder, 1, metric_key
            )

        assessment_results["model"] = model
        assessment_results["dataset"] = dataset

        # Convert the dictionary to a DataFrame with a single row
        df = pd.DataFrame(assessment_results, index=[0])

        # Append the DataFrame to the list
        dataframes.append(df)

    # Concatenate all the DataFrames into a single DataFrame
    combined_df = pd.concat(dataframes, ignore_index=True)

    # Create a LaTeX table from the DataFrame
    latex_table = _df_to_latex_table(
        combined_df, no_decimals=no_decimals, model_as_row=model_as_row
    )

    return latex_table


def _list_outer_fold_ids(exp_folder: str) -> List[int]:
    assessment_folder = os.path.join(exp_folder, MODEL_ASSESSMENT)
    if not os.path.isdir(assessment_folder):
        raise FileNotFoundError(
            f"Missing MODEL_ASSESSMENT folder in experiment {exp_folder}"
        )

    outer_folds = []
    for entry in os.listdir(assessment_folder):
        if entry.startswith("OUTER_FOLD_"):
            try:
                outer_folds.append(int(entry.split("_")[-1]))
            except ValueError:
                continue

    return sorted(outer_folds)


def _load_final_run_metric_samples(
    exp_folder: str, outer_fold_id: int, set_key: str, metric_key: str
) -> np.ndarray:
    samples = []
    run_id = 1
    set_key = set_key.lower()

    while True:
        run_results_path = os.path.join(
            exp_folder,
            MODEL_ASSESSMENT,
            f"OUTER_FOLD_{outer_fold_id}",
            f"final_run{run_id}",
            f"run_{run_id}_results.dill",
        )

        if not os.path.exists(run_results_path):
            break

        training_res, validation_res, test_res, _ = dill_load(run_results_path)

        set_results = {
            TRAINING: training_res,
            VALIDATION: validation_res,
            TEST: test_res,
        }[set_key]

        score_dict = (
            set_results[SCORE] if SCORE in set_results else set_results
        )

        if metric_key not in score_dict:
            raise KeyError(
                f"Metric '{metric_key}' not found in final run results at {run_results_path}"
            )

        samples.append(float(score_dict[metric_key]))
        run_id += 1

    if len(samples) == 0:
        raise ValueError(
            f"No final run results found for outer fold {outer_fold_id} in {exp_folder}"
        )

    return np.array(samples, dtype=float)


def _collect_metric_samples(
    exp_folder: str, metric_key: str, set_key: str
) -> Tuple[np.ndarray, str]:
    set_key = set_key.lower()
    if set_key not in [TRAINING, VALIDATION, TEST]:
        raise ValueError(
            f"set_key must be one of {[TRAINING, VALIDATION, TEST]}, received '{set_key}'"
        )

    outer_fold_ids = _list_outer_fold_ids(exp_folder)

    if len(outer_fold_ids) == 0:
        raise ValueError(f"No outer folds found in experiment {exp_folder}")

    if len(outer_fold_ids) == 1:
        return (
            _load_final_run_metric_samples(
                exp_folder, outer_fold_ids[0], set_key, metric_key
            ),
            "final_runs",
        )

    samples = np.array(
        [
            get_scores_from_outer_results(
                exp_folder, outer_fold_id, metric_key
            )[set_key]
            for outer_fold_id in outer_fold_ids
        ],
        dtype=float,
    )

    return samples, "outer_fold_means"


def _summarize_samples(samples: np.ndarray, confidence_level: float):
    mean = float(samples.mean())
    std = float(samples.std())
    z_value = stats.norm.ppf(0.5 + confidence_level / 2.0)
    ci_half_width = float(z_value * std / math.sqrt(len(samples)))
    return mean, std, ci_half_width


def statistical_significance(
    highlighted_exp_metadata: Tuple[str, str, str],
    other_exp_metadata: List[Tuple[str, str, str]],
    metric_key: str = "main_score",
    set_key: str = TEST,
    confidence_level: float = 0.95,
) -> pd.DataFrame:
    """
    Compares the statistical significance of a highlighted model against
    a list of other experiments using a Welch's t-test.

    Args:
        highlighted_exp_metadata (tuple[str, str, str]):
            (experiment_folder, model_name, dataset_name) for the reference model.
        other_exp_metadata (list[tuple[str, str, str]]):
            List of (experiment_folder, model_name, dataset_name) for the
            models to compare against the highlighted one.
        metric_key (str): The metric to compare. Default is "main_score".
        set_key (str): Which dataset split to consider: "training",
            "validation", or "test". Default is "test".
        confidence_level (float): Confidence level for CI computation and
            significance test. Default is 0.95.

    Returns:
        pandas.DataFrame: Each row contains the mean/std/CI for the highlighted
        and compared models plus the p-value (two-sided) and a boolean flag
        indicating whether the difference is statistically significant at the
        provided confidence level.

    Notes:
        - If multiple outer folds are present, their averaged scores are used
          as samples.
        - If only one outer fold exists, the scores of the final runs are
          used as samples.
    """

    set_key = set_key.lower()
    alpha = 1 - confidence_level

    ref_folder, ref_model, ref_dataset = highlighted_exp_metadata
    ref_samples, ref_source = _collect_metric_samples(
        ref_folder, metric_key, set_key
    )
    ref_mean, ref_std, ref_ci = _summarize_samples(
        ref_samples, confidence_level
    )

    results = []
    for exp_folder, model, dataset in other_exp_metadata:
        comp_samples, comp_source = _collect_metric_samples(
            exp_folder, metric_key, set_key
        )
        comp_mean, comp_std, comp_ci = _summarize_samples(
            comp_samples, confidence_level
        )

        ttest_res = stats.ttest_ind(ref_samples, comp_samples, equal_var=False)
        p_value = (
            float(ttest_res.pvalue)
            if not math.isnan(ttest_res.pvalue)
            else 1.0
        )

        results.append(
            {
                "metric": metric_key,
                "set": set_key,
                "reference_model": ref_model,
                "reference_dataset": ref_dataset,
                "reference_sample_source": ref_source,
                "reference_samples": len(ref_samples),
                "reference_mean": ref_mean,
                "reference_std": ref_std,
                "reference_ci": ref_ci,
                "model": model,
                "dataset": dataset,
                "sample_source": comp_source,
                "samples": len(comp_samples),
                "mean": comp_mean,
                "std": comp_std,
                "ci": comp_ci,
                "p_value": p_value,
                "statistically_significant": bool(p_value < alpha),
            }
        )

    return pd.DataFrame(results)
    # Keep a reference to the original print so Ray monkeypatching does not affect us
    _print = staticmethod(builtins.print)
