import traceback
import json
import operator
import os
import os.path as osp
import random
import re
import threading
import time
from copy import deepcopy
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import ray
import requests
import torch

from mlwiz.data.provider import DataProvider
from mlwiz.evaluation.config import Config
from mlwiz.evaluation.grid import Grid
from mlwiz.evaluation.random_search import RandomSearch
from mlwiz.ui.progress_manager import (
    ProgressManager,
    ProgressManagerActor,
    clear_screen,
)
from mlwiz.experiment.experiment import Experiment
from mlwiz.exceptions import ExperimentTerminated
from mlwiz.log.logger import Logger
from mlwiz.static import (
    AVG,
    BATCH,
    BEST_CONFIG,
    CI,
    CONFIG,
    CONFIG_ID,
    END_CONFIG,
    END_FINAL_RUN,
    EPOCH,
    EXPERIMENT_LOGFILE,
    FOLDS,
    INNER_FOLD,
    IS_FINAL,
    LOG_FINAL_RUNS,
    LOG_MODEL_SELECTION,
    LOSS,
    MAIN_LOSS,
    MAIN_SCORE,
    MLWIZ_RAY_NUM_GPUS_PER_TASK,
    MODEL_ASSESSMENT,
    MODEL_SELECTION_TRAINING_RUNS,
    OUTER_FOLD,
    OUTER_TEST,
    OUTER_TRAIN,
    OUTER_VALIDATION,
    RUN_FAILED,
    RUN_ID,
    RUN_PROGRESS,
    SCORE,
    STD,
    TELEGRAM_BOT_CHAT_ID,
    TELEGRAM_BOT_TOKEN,
    TEST,
    TOTAL_BATCHES,
    TOTAL_EPOCHS,
    TRAINING,
    TR_LOSS,
    TR_SCORE,
    VALIDATION,
    VL_LOSS,
    VL_SCORE,
)
from mlwiz.util import dill_load, atomic_dill_save, s2c


def send_telegram_update(bot_token: str, bot_chat_ID: str, bot_message: str):
    """
    Sends a message using Telegram APIs. Markdown can be used.

    Args:
        bot_token (str): token of the user's bot
        bot_chat_ID (str): identifier of the chat where to write the message
        bot_message (str): the message to be sent
    """
    send_text = (
        "https://api.telegram.org/bot"
        + str(bot_token)
        + "/sendMessage?chat_id="
        + str(bot_chat_ID)
        + "&parse_mode=Markdown&text="
        + str(bot_message)
    )
    response = requests.get(send_text)
    return response.json()


def extract_and_sum_elapsed_seconds(file_path):
    """
    Sum per-run elapsed time entries from an experiment log file.

    The evaluator writes elapsed-time markers to the experiment log in the
    form:

        ``Total time of the experiment in seconds: <SECONDS>``

    This helper scans the file for all such entries and returns their sum.

    Args:
        file_path (str | os.PathLike): Path to the experiment log file.

    Returns:
        float: Sum of all matched elapsed seconds.

    Side effects:
        Reads the file from disk.
    """
    # Open the file and read its contents
    with open(file_path, "r") as f:
        content = f.read()

    # Find all instances of the pattern "Total time of the experiment in seconds: [SECONDS] \n"
    matches = re.findall(
        r"Total time of the experiment in seconds: (\d+(?:\.\d+)?) \n", content
    )
    # Convert the matches to floats and sum them together
    total_seconds = sum(map(float, matches))

    return total_seconds


def _mean_std_ci(values: np.ndarray) -> Tuple[float, float, float]:
    """
    Computes mean, std, and 95% confidence interval for the provided values.
    """
    mean = float(values.mean())
    std = float(values.std())
    se = std / np.sqrt(len(values))
    half_width = 1.96 * se
    return mean, std, half_width


def _push_progress_update(progress_actor, payload: dict):
    """
    Safely forwards progress updates to the shared actor.
    """
    if progress_actor is None:
        return
    try:
        progress_actor.push.remote(deepcopy(payload))
    except Exception:
        pass


def _set_cuda_memory_limit_from_env():
    """
    Best-effort limit of per-process GPU memory based on the configured Ray
    GPU fraction. No-op if CUDA is unavailable or the value is invalid.
    """
    gpus_per_task = _get_ray_num_gpus_per_task()

    if not (0.0 < gpus_per_task <= 1.0):
        return

    if torch.cuda.is_available():
        torch.cuda.init()
        visible_gpus = torch.cuda.device_count()
        for gpu_idx in range(visible_gpus):
            torch.cuda.memory.set_per_process_memory_fraction(
                gpus_per_task, device=torch.device(f"cuda:{gpu_idx}")
            )
            # print(f"Setting max GPU {gpu_idx} memory of process to: {gpus_per_task}")


def _make_termination_checker(
    progress_actor, min_interval: float = 0.2
) -> Callable[[], bool]:
    """
    Creates a closure that checks for termination requests without hammering the actor.
    """
    termination_state = {"stop": False, "last_check": 0.0}

    def _should_terminate() -> bool:
        """
        Check whether the run should terminate.

        This is a lightweight wrapper around the progress actor termination
        flag with a minimum polling interval. It errs on the safe side: if the
        actor cannot be queried, it returns ``True``.

        Returns:
            bool: ``True`` if termination was requested or cannot be checked;
            ``False`` otherwise.
        """
        if termination_state["stop"]:
            return True
        if progress_actor is None:
            return False
        now = time.time()
        if now - termination_state["last_check"] < min_interval:
            return False
        termination_state["last_check"] = now
        try:
            termination_state["stop"] = ray.get(
                progress_actor.is_terminated.remote()
            )
        except Exception:
            return True  # errs on the safe side: stop if we cannot check
        return termination_state["stop"]

    return _should_terminate


def _get_ray_num_gpus_per_task(default: float = 0.0) -> float:
    """
    Return the Ray GPU request per task from the environment.

    This exists primarily to keep module import side-effect free (e.g. during
    Sphinx autodoc) when the variable is unset or malformed.
    """
    raw_value = os.environ.get(MLWIZ_RAY_NUM_GPUS_PER_TASK)
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        return default
    return value if value >= 0.0 else default


@ray.remote(
    num_cpus=1,
    num_gpus=_get_ray_num_gpus_per_task(),
    max_calls=1,
    # max_calls=1 --> the worker automatically exits after executing the task
    # (thereby releasing the GPU resources).
)
def run_valid(
    experiment_class: Callable[..., Experiment],
    dataset_getter: Callable[..., DataProvider],
    config: dict,
    config_id: int,
    run_id: int,
    fold_run_exp_folder: str,
    fold_run_results_torch_path: str,
    exp_seed: int,
    training_timeout_seconds: int,
    logger: Logger,
    progress_actor=None,
) -> Tuple[int, int, int, int, float]:
    r"""
    Ray job that performs a model selection run and returns bookkeeping
    information for the progress manager.

    Args:
        experiment_class
            (Callable[..., :class:`~mlwiz.experiment.experiment.Experiment`]):
            the class of the experiment to instantiate
        dataset_getter
            (Callable[..., :class:`~mlwiz.data.provider.DataProvider`]):
            the class of the data provider to instantiate
        config (dict): the configuration of this specific experiment
        config_id (int): the id of the configuration (for bookkeeping reasons)
        run_id (int): the id of the training run (for bookkeeping reasons)
        fold_run_exp_folder (str): path of the experiment root folder
        fold_run_results_torch_path (str): path where to store the
            results of the experiment
        exp_seed (int): seed of the experiment
        training_timeout_seconds (int): timeout for the experiment in seconds
        logger (:class:`~mlwiz.log.logger.Logger`): a logger to log
            information in the appropriate file

    Returns:
        a tuple with outer fold id, inner fold id, config id, run id,
            and time elapsed
    """
    _should_terminate = _make_termination_checker(progress_actor)

    if _should_terminate():
        return None

    # if not osp.exists(fold_run_results_torch_path):
    try:
        _set_cuda_memory_limit_from_env()
        experiment = experiment_class(config, fold_run_exp_folder, exp_seed)

        # This is used to comunicate with the progress manager
        # to display the UI
        def _report_progress(payload: dict):
            """
            Forward per-epoch/batch progress updates to the shared progress UI.

            Args:
                payload (dict): Progress fields produced by the experiment.

            Side effects:
                Sends the update to the Ray actor backing the terminal UI.
            """
            payload = deepcopy(payload)
            payload.update(
                {
                    OUTER_FOLD: dataset_getter.outer_k,
                    INNER_FOLD: dataset_getter.inner_k,
                    CONFIG_ID: config_id,
                    RUN_ID: run_id,
                    IS_FINAL: False,
                }
            )
            _push_progress_update(progress_actor, payload)

        train_res, val_res = experiment.run_valid(
            dataset_getter,
            training_timeout_seconds,
            logger,
            progress_callback=_report_progress,
            should_terminate=_should_terminate,
        )
        elapsed = extract_and_sum_elapsed_seconds(
            osp.join(fold_run_exp_folder, EXPERIMENT_LOGFILE)
        )
        atomic_dill_save(
            (train_res, val_res, elapsed), fold_run_results_torch_path
        )
    except ExperimentTerminated:
        return None
    except Exception as e:

        _push_progress_update(
            progress_actor,
            {
                "type": RUN_FAILED,
                str(OUTER_FOLD): dataset_getter.outer_k,
                str(INNER_FOLD): dataset_getter.inner_k,
                str(CONFIG_ID): config_id,
                str(RUN_ID): run_id,
                str(IS_FINAL): False,
                str(EPOCH): 0,
                str(TOTAL_EPOCHS): 0,
                "message": f"{e}\n{traceback.format_exc()}",
            },
        )

        elapsed = -1
        return None

    return (
        dataset_getter.outer_k,
        dataset_getter.inner_k,
        config_id,
        run_id,
        elapsed,
    )


@ray.remote(
    num_cpus=1,
    num_gpus=_get_ray_num_gpus_per_task(),
    max_calls=1,
    # max_calls=1 --> the worker automatically exits after executing the task
    # (thereby releasing the GPU resources).
)
def run_test(
    experiment_class: Callable[..., Experiment],
    dataset_getter: Callable[..., DataProvider],
    best_config: dict,
    outer_k: int,
    run_id: int,
    final_run_exp_path: str,
    final_run_torch_path: str,
    exp_seed: int,
    training_timeout_seconds: int,
    logger: Logger,
    progress_actor=None,
) -> Tuple[int, int, float]:
    r"""
    Ray job that performs a risk assessment run and returns bookkeeping
    information for the progress manager.

    Args:
        experiment_class
            (Callable[..., :class:`~mlwiz.experiment.experiment.Experiment`]):
            the class of the experiment to instantiate
        dataset_getter
            (Callable[..., :class:`~mlwiz.data.provider.DataProvider`]):
            the class of the data provider to instantiate
        best_config (dict): the best configuration to use for this
            specific outer fold
        run_id (int): the id of the final run (for bookkeeping reasons)
        final_run_exp_path (str): path of the experiment root folder
        final_run_torch_path (str): path where to store the results
            of the experiment
        exp_seed (int): seed of the experiment
        training_timeout_seconds (int): timeout for the experiment in seconds
        logger (:class:`~mlwiz.log.logger.Logger`): a logger to log
            information in the appropriate file

    Returns:
        a tuple with outer fold id, final run id, and time elapsed
    """
    _should_terminate = _make_termination_checker(progress_actor)

    if _should_terminate():
        return None

    # if not osp.exists(final_run_torch_path):
    try:
        _set_cuda_memory_limit_from_env()
        experiment = experiment_class(
            best_config[CONFIG], final_run_exp_path, exp_seed
        )

        # This is used to comunicate with the progress manager
        # to display the UI
        def _report_progress(payload: dict):
            """
            Forward per-epoch/batch progress updates to the shared progress UI.

            Args:
                payload (dict): Progress fields produced by the experiment.

            Side effects:
                Sends the update to the Ray actor backing the terminal UI.
            """
            payload = deepcopy(payload)
            payload.update(
                {
                    OUTER_FOLD: dataset_getter.outer_k,
                    INNER_FOLD: None,
                    CONFIG_ID: best_config["best_config_id"] - 1,
                    RUN_ID: run_id,
                    IS_FINAL: True,
                }
            )
            _push_progress_update(progress_actor, payload)

        res = experiment.run_test(
            dataset_getter,
            training_timeout_seconds,
            logger,
            progress_callback=_report_progress,
            should_terminate=_should_terminate,
        )
        elapsed = extract_and_sum_elapsed_seconds(
            osp.join(final_run_exp_path, EXPERIMENT_LOGFILE)
        )
        train_res, val_res, test_res = res
        atomic_dill_save(
            (train_res, val_res, test_res, elapsed), final_run_torch_path
        )
    except ExperimentTerminated:
        return None
    except Exception as e:

        _push_progress_update(
            progress_actor,
            {
                "type": RUN_FAILED,
                str(OUTER_FOLD): dataset_getter.outer_k,
                str(INNER_FOLD): None,
                str(CONFIG_ID): best_config["best_config_id"] - 1,
                str(RUN_ID): run_id,
                str(IS_FINAL): True,
                str(EPOCH): 0,
                str(TOTAL_EPOCHS): 0,
                "message": f"{e}\n{traceback.format_exc()}",
            },
        )
        elapsed = -1
        return None

    return outer_k, run_id, elapsed


class RiskAssesser:
    r"""
    Class implementing a K-Fold technique to do Risk Assessment
    (estimate of the true generalization performances)
    and K-Fold Model Selection (select the best hyper-parameters
    for **each** external fold

    Args:
        outer_folds (int): The number K of outer TEST folds.
            You should have generated the splits accordingly
        inner_folds (int): The number K of inner VALIDATION folds.
            You should have generated the splits accordingly
        experiment_class
            (Callable[..., :class:`~mlwiz.experiment.experiment.Experiment`]):
            the experiment class to be instantiated
        exp_path (str): The folder in which to store **all** results
        splits_filepath (str): The splits filepath with additional
            meta information
        model_configs
            (Union[:class:`~mlwiz.evaluation.grid.Grid`,
            :class:`~mlwiz.evaluation.random_search.RandomSearch`]):
            an object storing all possible model configurations,
            e.g., config.base.Grid
        risk_assessment_training_runs (int): no of final training runs to
            mitigate bad initializations
        model_selection_training_runs (int): no of training runs to mitigate
            bad initializations at model selection time
        higher_is_better (bool): whether the best model
            for each external fold should be selected by higher
            or lower score values
        gpus_per_task (float): Number of gpus to assign to each
            experiment. Can be < ``1``.
        base_seed (int): Seed used to generate experiments seeds.
            Used to replicate results. Default is ``42``
        training_timeout_seconds (int): optional timeout limit per
            experiment in seconds
    """

    def __init__(
        self,
        outer_folds: int,
        inner_folds: int,
        experiment_class: Callable[..., Experiment],
        exp_path: str,
        splits_filepath: str,
        model_configs: Union[Grid, RandomSearch],
        risk_assessment_training_runs: int,
        model_selection_training_runs: int,
        higher_is_better: bool,
        gpus_per_task: float,
        base_seed: int = 42,
        training_timeout_seconds: int = -1,
    ):
        r"""
        Initialize the risk assessment evaluator.

        Args:
            outer_folds (int): Number of outer folds (risk assessment).
            inner_folds (int): Number of inner folds (model selection).
            experiment_class (Callable[..., Experiment]): Experiment class used
                to run training/evaluation.
            exp_path (str): Root folder where all results will be written.
            splits_filepath (str): Path to the serialized splits file.
            model_configs (Grid | RandomSearch): Search object yielding model
                configurations to evaluate.
            risk_assessment_training_runs (int): Number of repeated "final"
                runs per outer fold to reduce variance from random
                initialization.
            model_selection_training_runs (int): Number of repeated runs per
                configuration during model selection.
            higher_is_better (bool): Whether a larger score indicates a better
                configuration.
            gpus_per_task (float): GPUs assigned per Ray task (may be < 1.0).
            base_seed (int): Base seed used to derive per-run seeds.
            training_timeout_seconds (int): Optional per-run timeout in seconds.
                Use ``-1`` to disable.

        Side effects:
            Seeds NumPy/PyTorch/Python RNGs for reproducibility and initializes
            internal bookkeeping/state used by Ray jobs and the progress UI.
        """
        # REPRODUCIBILITY:
        # https://pytorch.org/docs/stable/notes/randomness.html
        self.base_seed = base_seed
        # Impost the manual seed from the start
        np.random.seed(self.base_seed)
        torch.manual_seed(self.base_seed)
        torch.cuda.manual_seed(self.base_seed)
        random.seed(self.base_seed)

        self.outer_folds = outer_folds
        self.inner_folds = inner_folds
        self.experiment_class = experiment_class

        # Iterator producing the list of all possible configs
        self.model_configs = model_configs
        self.risk_assessment_training_runs = risk_assessment_training_runs
        self.model_selection_training_runs = model_selection_training_runs
        self.training_timeout_seconds = training_timeout_seconds
        self.higher_is_better = higher_is_better
        if higher_is_better:
            self.operator = operator.gt
        else:
            self.operator = operator.lt
        self.gpus_per_task = gpus_per_task

        # Main folders
        self.exp_path = exp_path

        # Splits filepath
        self.splits_filepath = splits_filepath

        # Model assessment filenames
        self._ASSESSMENT_FOLDER = osp.join(exp_path, MODEL_ASSESSMENT)
        self._OUTER_FOLD_BASE = "OUTER_FOLD_"
        self._OUTER_RESULTS_FILENAME = "outer_results.json"
        self._ASSESSMENT_FILENAME = "assessment_results.json"

        # Model selection filenames
        self._SELECTION_FOLDER = "MODEL_SELECTION"
        self._INNER_FOLD_BASE = "INNER_FOLD_"
        self._CONFIG_BASE = "config_"
        self._CONFIG_RESULTS = "config_results.json"
        self._WINNER_CONFIG = "winner_config.json"

        # Used to keep track of the scheduled jobs
        self.model_selection_job_list = []
        self.final_runs_job_list = []
        # Runs that were already completed on disk (restart flow)
        self.completed_model_selection_runs = []
        self.completed_final_runs = []
        self.progress_actor = None

        # telegram config
        tc = model_configs.telegram_config
        self.telegram_bot_token = (
            tc[TELEGRAM_BOT_TOKEN] if tc is not None else None
        )
        self.telegram_bot_chat_ID = (
            tc[TELEGRAM_BOT_CHAT_ID] if tc is not None else None
        )
        self.log_model_selection = (
            tc[LOG_MODEL_SELECTION] if tc is not None else None
        )
        self.log_final_runs = tc[LOG_FINAL_RUNS] if tc is not None else None
        self.failure_message = None

    def _create_dataset_getter(
        self, outer_k: int, inner_k: Optional[int]
    ) -> DataProvider:
        """
        Instantiates and configures a dataset provider for the requested folds.
        """
        dataset_getter_class = s2c(self.model_configs.dataset_getter)
        dataset_getter = dataset_getter_class(
            self.model_configs.storage_folder,
            self.splits_filepath,
            s2c(self.model_configs.dataset_class),
            s2c(self.model_configs.data_loader_class),
            self.model_configs.data_loader_args,
            self.outer_folds,
            self.inner_folds,
        )
        dataset_getter.set_outer_k(outer_k)
        dataset_getter.set_inner_k(inner_k)
        return dataset_getter

    def _request_termination(self):
        """
        Signals all workers and the UI to terminate gracefully.
        """
        clear_screen()
        print("Termination requested. Stopping scheduled jobs...")
        self.failure_message = "Execution interrupted by user."
        if self.progress_actor is not None:
            try:
                self.progress_actor.terminate.remote()
            except Exception:
                pass
        else:
            for job in (
                self.model_selection_job_list + self.final_runs_job_list
            ):
                try:
                    ray.cancel(job, force=True)
                except Exception:
                    pass
        if getattr(self, "progress_manager", None) is not None:
            try:
                self.progress_manager._stop_input_event.set()
            except Exception:
                pass

    def risk_assessment(
        self,
        debug: bool,
        execute_config_id: int = None,
        skip_config_ids: List[int] = None,
    ):
        r"""
        Performs risk assessment to evaluate the performances of a model.

        Args:
            debug: if ``True``, sequential execution is performed and logs are
                printed to screen
            execute_config_id: if debug mode is enabled, it will prioritize the
                execution of this configuration for each model selection
                procedure. It assumes indices start from 1.
                Use this to debug specific configurations.
            skip_config_ids: if provided, the provided list of configurations
                will not be considered for model selection. Use it,
                for instance, when a run is taking too long to execute and you
                decide it is not worth to wait for it.
        """
        if not osp.exists(self._ASSESSMENT_FOLDER):
            os.makedirs(self._ASSESSMENT_FOLDER)

        # internally skip-config ids start from 0
        if skip_config_ids is None:
            skip_config_ids = []
        else:
            skip_config_ids = [s - 1 for s in skip_config_ids]

        # Reset failure state at the beginning of each evaluation
        self.failure_message = None

        self.model_selection_job_list = []
        self.final_runs_job_list = []
        self.completed_model_selection_runs = []
        self.completed_final_runs = []

        self.progress_actor = None
        try:
            self.progress_actor = ProgressManagerActor.remote()
        except Exception:
            # If the actor cannot be created, fall back to global-only view
            print("Cannot create Ray progress actor, progress UI disabled.")

        # Show progress
        progress_manager = ProgressManager(
            self.outer_folds,
            self.inner_folds,
            len(self.model_configs),
            self.model_selection_training_runs,
            self.risk_assessment_training_runs,
            debug=debug,
            progress_actor=self.progress_actor,
            poll_interval=0.1,
        )
        self.progress_manager = progress_manager

        # Start listening to the progress actor
        progress_thread = threading.Thread(
            target=progress_manager.update_state, daemon=True
        )
        progress_thread.start()

        # NOTE: Pre-computing seeds in advance simplifies the code
        # Pre-compute in advance the seeds for model selection to aid
        # replicability
        self.model_selection_seeds = [
            [
                [
                    [
                        random.randrange(2**32 - 1)
                        for _ in range(self.model_selection_training_runs)
                    ]
                    for _ in range(self.inner_folds)
                ]
                for _ in range(len(self.model_configs))
            ]
            for _ in range(self.outer_folds)
        ]
        # Pre-compute in advance the seeds for the final runs to aid
        # replicability
        self.final_runs_seeds = [
            [
                random.randrange(2**32 - 1)
                for _ in range(self.risk_assessment_training_runs)
            ]
            for _ in range(self.outer_folds)
        ]

        try:
            success = True
            for outer_k in range(self.outer_folds):
                # Create a separate folder for each experiment
                kfold_folder = osp.join(
                    self._ASSESSMENT_FOLDER,
                    self._OUTER_FOLD_BASE + str(outer_k + 1),
                )
                if not osp.exists(kfold_folder):
                    os.makedirs(kfold_folder)

                # Perform model selection. This determines a best config
                # FOR EACH of the k outer folds
                self.model_selection(
                    kfold_folder,
                    outer_k,
                    debug,
                    execute_config_id,
                    skip_config_ids,
                )

                if self.failure_message is not None:
                    success = False
                    break

                # Must stay separate from Ray distributed computing logic
                if debug:
                    self.run_final_model(outer_k, True)
                    if self.failure_message is not None:
                        success = False
                        break

            # We launched all model selection jobs, now it is time to wait
            if not debug and success:
                # This will also launch the final runs jobs once the model
                # selection for a specific outer folds is completed.
                # It returns when everything has completed
                success = self.wait_configs(skip_config_ids)

            if not success or self.failure_message is not None:
                if self.failure_message is not None:
                    print(self.failure_message)
                if self.progress_actor is not None:
                    try:
                        self.progress_actor.close.remote()
                    except Exception as e:
                        print(f"{e}\n{traceback.format_exc()}")

                progress_thread.join()
                return

            # Produces the self._ASSESSMENT_FILENAME file
            self.compute_risk_assessment_result()

            if self.progress_actor is not None:
                try:
                    self.progress_actor.close.remote()
                except Exception:
                    print

        except KeyboardInterrupt:
            self._request_termination()
            if self.progress_actor is not None:
                try:
                    self.progress_actor.close.remote()
                except Exception:
                    pass
            progress_thread.join()
            return

        progress_thread.join()

    def wait_configs(self, skip_config_ids: List[int]) -> bool:
        r"""
        Waits for configurations to terminate and updates the state of the
        progress manager

        Returns:
            bool: ``True`` if all runs completed successfully, ``False`` otherwise.
        """
        no_model_configs = len(self.model_configs)
        skip_model_selection = no_model_configs == 1

        # Copy the list of jobs (only for model selection atm)
        waiting = [el for el in self.model_selection_job_list]

        # Counters to keep track of what has completed
        inner_runs_completed = np.zeros(
            (self.outer_folds, no_model_configs, self.inner_folds), dtype=int
        )
        # keeps track of the final run jobs completed for each outer fold
        # the max value for each position is self.risk_assessment_training_runs
        final_runs_completed = [0 for _ in range(self.outer_folds)]

        # Cached (already completed) runs that must be replayed

        def handle_model_selection_result(result):
            """
            Process a completed model-selection run and update bookkeeping/UI.

            Args:
                result (tuple): Tuple returned by :func:`run_valid` containing
                    ``(outer_k, inner_k, config_id, run_id, elapsed)``.

            Side effects:
                Updates internal completion counters, pushes UI events, and may
                schedule final runs once an outer fold finishes model selection.
            """
            outer_k, inner_k, config_id, run_id, elapsed = result

            ms_exp_path = osp.join(
                self._ASSESSMENT_FOLDER,
                self._OUTER_FOLD_BASE + str(outer_k + 1),
                self._SELECTION_FOLDER,
            )
            config_folder = osp.join(
                ms_exp_path, self._CONFIG_BASE + str(config_id + 1)
            )

            config_inner_fold_folder = osp.join(
                config_folder, self._INNER_FOLD_BASE + str(inner_k + 1)
            )

            # if all runs for a certain config and inner fold are
            # completed, then update the progress manager
            inner_runs_completed[outer_k][config_id][inner_k] += 1
            if (
                inner_runs_completed[outer_k][config_id][inner_k]
                == self.model_selection_training_runs
            ):
                _push_progress_update(
                    self.progress_actor,
                    dict(
                        type=END_CONFIG,
                        outer_fold=outer_k,
                        inner_fold=inner_k,
                        config_id=config_id,
                        elapsed=elapsed,
                        message=[
                            f"Outer fold {outer_k + 1}, "
                            f"inner fold {inner_k + 1}, "
                            f"configuration {config_id + 1} completed."
                        ],
                    ),
                )

                self.process_model_selection_runs(
                    config_inner_fold_folder, inner_k
                )

            # if all inner folds completed (including their runs),
            # process that configuration and compute its average scores
            if (
                inner_runs_completed[outer_k, config_id, :].sum()
                == self.inner_folds * self.model_selection_training_runs
            ):
                self.process_config_results_across_inner_folds(
                    config_folder,
                    deepcopy(self.model_configs[config_id]),
                )

            # if model selection is complete, launch final runs
            if (
                inner_runs_completed[outer_k, :, :].sum()
                == self.inner_folds
                * no_model_configs
                * self.model_selection_training_runs
            ):  # outer fold completed - schedule final runs
                self.compute_best_hyperparameters(
                    ms_exp_path,
                    outer_k,
                    len(self.model_configs),
                    skip_config_ids,
                )
                # Only enqueue newly scheduled final runs to avoid duplicates
                prev_len = len(self.final_runs_job_list)
                self.run_final_model(outer_k, False)
                waiting.extend(self.final_runs_job_list[prev_len:])

        def handle_final_run_result(result):
            """
            Process a completed final run and update bookkeeping/UI.

            Args:
                result (tuple): Tuple returned by :func:`run_test` containing
                    ``(outer_k, run_id, elapsed)``.

            Side effects:
                Updates internal completion counters, pushes UI events, and may
                compute outer-fold results once all final runs complete.
            """
            outer_k, run_id, elapsed = result
            _push_progress_update(
                self.progress_actor,
                dict(
                    type=END_FINAL_RUN,
                    outer_fold=outer_k,
                    run_id=run_id,
                    elapsed=elapsed,
                    message=[
                        f"Outer fold {outer_k + 1}, "
                        f"final run {run_id + 1} completed."
                    ],
                ),
            )

            final_runs_completed[outer_k] += 1
            if (
                final_runs_completed[outer_k]
                == self.risk_assessment_training_runs
            ):
                # Time to produce self._OUTER_RESULTS_FILENAME
                self.compute_final_runs_score_per_fold(outer_k)

        def process_cached_results():
            """
            Replay previously completed runs to keep UI and counters consistent.

            Side effects:
                Invokes the local handlers for all cached model-selection and
                final-run results, then clears the caches.
            """
            for cached in self.completed_model_selection_runs:
                handle_model_selection_result(cached)
            self.completed_model_selection_runs = []
            for cached in self.completed_final_runs:
                handle_final_run_result(cached)
            self.completed_final_runs = []

        if skip_model_selection:
            for outer_k in range(self.outer_folds):
                # no need to call process_inner_results() here
                prev_len = len(self.final_runs_job_list)
                self.run_final_model(outer_k, False)
                waiting.extend(self.final_runs_job_list[prev_len:])
            process_cached_results()

        # Ad-hoc code to skip configs in the evaluator
        for skipped_config_id in skip_config_ids:
            inner_runs_completed[:, skipped_config_id, :] = (
                self.model_selection_training_runs
            )

            for outer_k in range(self.outer_folds):
                for inner_k in range(self.inner_folds):
                    _push_progress_update(
                        self.progress_actor,
                        dict(
                            type=END_CONFIG,
                            outer_fold=outer_k,
                            inner_fold=inner_k,
                            config_id=skipped_config_id,
                            elapsed=0.0,
                            message=[
                                f"Outer fold {outer_k + 1}, "
                                f"inner fold {inner_k + 1}, "
                                f"configuration {skipped_config_id + 1} skipped."
                            ],
                        ),
                    )

        # This is necessary in case all model selection runs
        # have been completed and we only need to launch
        # the final runs. If this is not called and waiting is empty,
        # handle_model_selection_result will not launch the final
        # run jobs
        process_cached_results()

        success = True
        while waiting:
            completed, waiting = ray.wait(waiting)
            for future in completed:
                is_model_selection_run = (
                    future in self.model_selection_job_list
                )
                is_final_run = future in self.final_runs_job_list
                result = ray.get(future)

                if result is None:
                    if self.failure_message is None:
                        self.failure_message = (
                            "A model selection run failed; stopping before computing the best configuration. "
                            "Check the run logs for details."
                            if is_model_selection_run
                            else "A final run failed; skipping outer fold scoring and final assessment. "
                            "Check the run logs for details."
                        )
                    success = False

                elif is_model_selection_run:  # Model selection
                    handle_model_selection_result(result)
                elif is_final_run:  # Risk ass. final runs
                    handle_final_run_result(result)

        return success

    def model_selection(
        self,
        kfold_folder: str,
        outer_k: int,
        debug: bool,
        execute_config_id: Optional[int],
        skip_config_ids: List[int],
    ):
        r"""
        Performs model selection.

        Args:
            kfold_folder: The root folder for model selection
            outer_k: the current outer fold to consider
            debug: if ``True``, sequential execution is performed and logs are
                printed to screen
            execute_config_id: if debug mode is enabled, it will prioritize the
                execution of this configuration. It assumes indices start
                from 1. Use this to debug specific configurations.
            skip_config_ids: if provided, the provided list of configurations
                will not be considered for model selection. Use it,
                for instance, when a run is taking too long to execute and you
        decide it is not worth to wait for it.
        """
        if len(skip_config_ids) > 0 and execute_config_id is not None:
            raise ValueError(
                "Cannot specify both skip_config_id and execute_config_id"
            )

        model_selection_folder = osp.join(kfold_folder, self._SELECTION_FOLDER)

        # Create the dataset provider
        dataset_getter = self._create_dataset_getter(outer_k, None)

        if not osp.exists(model_selection_folder):
            os.makedirs(model_selection_folder)

        # Inform progress manager about the configs even when model selection is skipped.
        self.progress_manager.set_model_configs(
            deepcopy(self.model_configs.hparams)
        )

        # if the # of configs to try is 1, simply skip model selection
        if len(self.model_configs) > 1:
            _model_configs = [
                (config_id, config)
                for config_id, config in enumerate(self.model_configs)
                if config_id not in skip_config_ids
            ]

            # Prioritizing executions in debug mode for debugging purposes
            if debug and execute_config_id is not None:
                element = _model_configs.pop(execute_config_id - 1)
                _model_configs.insert(0, element)
                print(
                    f"Prioritizing execution of configuration"
                    f" {_model_configs[0][0] + 1} as requested..."
                )
                print(element)
            # Launch one job for each inner_fold for each configuration
            for config_id, config in _model_configs:
                # Create a separate folder for each configuration
                config_folder = osp.join(
                    model_selection_folder,
                    self._CONFIG_BASE + str(config_id + 1),
                )
                if not osp.exists(config_folder):
                    os.makedirs(config_folder)

                for k in range(self.inner_folds):
                    # Create a separate folder for each fold for each config.
                    config_inner_fold_folder = osp.join(
                        config_folder, self._INNER_FOLD_BASE + str(k + 1)
                    )

                    # Tell the data provider to take data relative
                    # to a specific INNER split
                    dataset_getter.set_inner_k(k)

                    for run_id in range(self.model_selection_training_runs):
                        fold_run_exp_folder = osp.join(
                            config_inner_fold_folder, f"run_{run_id + 1}"
                        )
                        fold_run_results_torch_path = osp.join(
                            fold_run_exp_folder,
                            f"run_{run_id + 1}_results.dill",
                        )

                        # Use pre-computed random seed for the experiment
                        exp_seed = self.model_selection_seeds[outer_k][
                            config_id
                        ][k][run_id]
                        dataset_getter.set_exp_seed(exp_seed)

                        logger = Logger(
                            osp.join(fold_run_exp_folder, EXPERIMENT_LOGFILE),
                            mode="a",
                            debug=debug,
                        )
                        logger.log(
                            json.dumps(
                                dict(
                                    outer_k=dataset_getter.outer_k,
                                    inner_k=dataset_getter.inner_k,
                                    run_id=run_id,
                                    exp_seed=exp_seed,
                                    **config,
                                ),
                                sort_keys=False,
                                indent=4,
                            )
                        )
                        if not debug:
                            if osp.exists(fold_run_results_torch_path):
                                train_res, val_res, cached_elapsed = dill_load(
                                    fold_run_results_torch_path
                                )
                                run_log_path = osp.join(
                                    fold_run_exp_folder, EXPERIMENT_LOGFILE
                                )
                                elapsed = (
                                    extract_and_sum_elapsed_seconds(
                                        run_log_path
                                    )
                                    if osp.exists(run_log_path)
                                    else cached_elapsed
                                )
                                self.completed_model_selection_runs.append(
                                    (
                                        dataset_getter.outer_k,
                                        dataset_getter.inner_k,
                                        config_id,
                                        run_id,
                                        elapsed,
                                    )
                                )

                                # When reusing cached results, still surface a progress message so the UI
                                # can show the latest information for this run.
                                cached_msg = "Recovered cached result."
                                try:
                                    summary_parts = []
                                    tr_loss = float(train_res[LOSS][MAIN_LOSS])
                                    val_loss = float(val_res[LOSS][MAIN_LOSS])
                                    tr_score = float(
                                        train_res[SCORE][MAIN_SCORE]
                                    )
                                    val_score = float(
                                        val_res[SCORE][MAIN_SCORE]
                                    )
                                    summary_parts.append(
                                        f"TR/VL/TE loss: {tr_loss:.2f}/{val_loss:.2f}/N/A TR/VL/TE score: {tr_score:.2f}/{val_score:.2f}/N/A"
                                    )
                                    if summary_parts:
                                        cached_msg = " | ".join(summary_parts)
                                except Exception as e:
                                    cached_msg += (
                                        f" {e}\n{traceback.format_exc()}",
                                    )

                                _push_progress_update(
                                    self.progress_actor,
                                    {
                                        "type": RUN_PROGRESS,
                                        OUTER_FOLD: dataset_getter.outer_k,
                                        INNER_FOLD: dataset_getter.inner_k,
                                        CONFIG_ID: config_id,
                                        RUN_ID: run_id,
                                        IS_FINAL: False,
                                        EPOCH: 0,
                                        TOTAL_EPOCHS: 0,
                                        BATCH: 1,
                                        TOTAL_BATCHES: 1,
                                        "message": cached_msg,
                                    },
                                )
                            else:
                                # Launch the job and append to list of outer jobs
                                future = run_valid.remote(
                                    self.experiment_class,
                                    dataset_getter,
                                    config,
                                    config_id,
                                    run_id,
                                    fold_run_exp_folder,
                                    fold_run_results_torch_path,
                                    exp_seed,
                                    self.training_timeout_seconds,
                                    logger,
                                    self.progress_actor,
                                )
                                self.model_selection_job_list.append(future)
                        else:  # debug mode
                            if not osp.exists(fold_run_results_torch_path):
                                experiment = self.experiment_class(
                                    config, fold_run_exp_folder, exp_seed
                                )

                                # This is used to comunicate with the progress manager
                                # to display the UI
                                def _report_progress(payload: dict):
                                    """
                                    Forward progress updates to the shared UI (debug mode).

                                    Args:
                                        payload (dict): Progress fields produced by the experiment.

                                    Side effects:
                                        Sends the update to the Ray actor backing the terminal UI.
                                    """
                                    payload = deepcopy(payload)
                                    payload.update(
                                        {
                                            OUTER_FOLD: dataset_getter.outer_k,
                                            INNER_FOLD: dataset_getter.inner_k,
                                            CONFIG_ID: config_id,
                                            RUN_ID: run_id,
                                            IS_FINAL: False,
                                        }
                                    )
                                    _push_progress_update(
                                        self.progress_actor, payload
                                    )

                                try:
                                    (
                                        training_score,
                                        validation_score,
                                    ) = experiment.run_valid(
                                        dataset_getter,
                                        self.training_timeout_seconds,
                                        logger,
                                        progress_callback=_report_progress,
                                    )
                                    elapsed = extract_and_sum_elapsed_seconds(
                                        osp.join(
                                            fold_run_exp_folder,
                                            EXPERIMENT_LOGFILE,
                                        )
                                    )
                                    atomic_dill_save(
                                        (
                                            training_score,
                                            validation_score,
                                            elapsed,
                                        ),
                                        fold_run_results_torch_path,
                                    )
                                except Exception as e:
                                    _push_progress_update(
                                        self.progress_actor,
                                        {
                                            "type": RUN_FAILED,
                                            str(
                                                OUTER_FOLD
                                            ): dataset_getter.outer_k,
                                            str(
                                                INNER_FOLD
                                            ): dataset_getter.inner_k,
                                            str(CONFIG_ID): config_id,
                                            str(RUN_ID): run_id,
                                            str(IS_FINAL): False,
                                            str(EPOCH): 0,
                                            str(TOTAL_EPOCHS): 0,
                                            "message": f"{e}\n{traceback.format_exc()}",
                                        },
                                    )

                                    elapsed = -1
                                    if self.failure_message is None:
                                        self.failure_message = (
                                            "A model selection run failed; "
                                            "stopping before computing the best configuration. "
                                            "Check the run logs for details."
                                        )
                                    return None

                    if debug:
                        self.process_model_selection_runs(
                            config_inner_fold_folder, k
                        )
                if debug:
                    self.process_config_results_across_inner_folds(
                        config_folder, deepcopy(config)
                    )
            if debug:
                self.compute_best_hyperparameters(
                    model_selection_folder,
                    outer_k,
                    len(self.model_configs),
                    skip_config_ids,
                )
        else:
            # Performing model selection for a single configuration is useless
            with open(
                osp.join(model_selection_folder, self._WINNER_CONFIG), "w"
            ) as fp:
                json.dump(
                    dict(best_config_id=1, config=self.model_configs[0]),
                    fp,
                    sort_keys=False,
                    indent=4,
                )

    def run_final_model(self, outer_k: int, debug: bool):
        r"""
        Performs the final runs once the best model for outer fold ``outer_k``
        has been chosen.

        Args:
            outer_k (int): the current outer fold to consider
            debug (bool): if ``True``, sequential execution is performed and
                logs are printed to screen
        """
        outer_folder = osp.join(
            self._ASSESSMENT_FOLDER, self._OUTER_FOLD_BASE + str(outer_k + 1)
        )
        config_fname = osp.join(
            outer_folder, self._SELECTION_FOLDER, self._WINNER_CONFIG
        )

        with open(config_fname, "r") as f:
            best_config = json.load(f)

        dataset_getter = self._create_dataset_getter(outer_k, None)

        # Mitigate bad random initializations with more runs
        for i in range(self.risk_assessment_training_runs):
            final_run_exp_path = osp.join(outer_folder, f"final_run{i + 1}")
            final_run_torch_path = osp.join(
                final_run_exp_path, f"run_{i + 1}_results.dill"
            )

            # Use pre-computed random seed for the experiment
            exp_seed = self.final_runs_seeds[outer_k][i]
            dataset_getter.set_exp_seed(exp_seed)

            # Retrain with the best configuration and test
            # Set up a log file for this experiment (run in a separate process)
            logger = Logger(
                osp.join(final_run_exp_path, EXPERIMENT_LOGFILE),
                mode="a",
                debug=debug,
            )
            logger.log(
                json.dumps(
                    dict(
                        outer_k=dataset_getter.outer_k,
                        inner_k=dataset_getter.inner_k,
                        exp_seed=exp_seed,
                        **best_config,
                    ),
                    sort_keys=False,
                    indent=4,
                )
            )

            if not debug:
                if osp.exists(final_run_torch_path):
                    final_run_log_path = osp.join(
                        final_run_exp_path, EXPERIMENT_LOGFILE
                    )
                    cached_elapsed = None
                    try:
                        res = dill_load(final_run_torch_path)
                        cached_elapsed = res[-1]
                        train_res, val_res, test_res = res[:3]

                        elapsed = (
                            extract_and_sum_elapsed_seconds(final_run_log_path)
                            if osp.exists(final_run_log_path)
                            else cached_elapsed
                        )
                    except Exception:
                        train_res, val_res, test_res = None, None, None
                        elapsed = 0.0

                    self.completed_final_runs.append(
                        (dataset_getter.outer_k, i, elapsed)
                    )

                    cached_msg = "Recovered cached result."
                    try:
                        summary_parts = []
                        tr_loss = float(train_res[LOSS][MAIN_LOSS])
                        val_loss = float(val_res[LOSS][MAIN_LOSS])
                        test_loss = float(test_res[LOSS][MAIN_LOSS])
                        tr_score = float(train_res[SCORE][MAIN_SCORE])
                        val_score = float(val_res[SCORE][MAIN_SCORE])
                        test_score = float(test_res[SCORE][MAIN_SCORE])
                        summary_parts.append(
                            f"TR/VL/TE loss: {tr_loss:.2f}/{val_loss:.2f}/{test_loss:.2f} TR/VL/TE score: {tr_score:.2f}/{val_score:.2f}/{test_score:.2f}"
                        )
                        cached_msg = " | ".join(summary_parts)
                    except Exception as e:
                        cached_msg += (f" {e}\n{traceback.format_exc()}",)

                    _push_progress_update(
                        self.progress_actor,
                        {
                            "type": RUN_PROGRESS,
                            OUTER_FOLD: dataset_getter.outer_k,
                            INNER_FOLD: None,
                            CONFIG_ID: best_config["best_config_id"] - 1,
                            RUN_ID: i,
                            IS_FINAL: True,
                            EPOCH: 0,
                            TOTAL_EPOCHS: 0,
                            BATCH: 1,
                            TOTAL_BATCHES: 1,
                            "message": cached_msg,
                        },
                    )
                else:
                    # Launch the job and append to list of final runs jobs
                    future = run_test.remote(
                        self.experiment_class,
                        dataset_getter,
                        best_config,
                        outer_k,
                        i,
                        final_run_exp_path,
                        final_run_torch_path,
                        exp_seed,
                        self.training_timeout_seconds,
                        logger,
                        self.progress_actor,
                    )
                    self.final_runs_job_list.append(future)
            else:
                if not osp.exists(final_run_torch_path):
                    experiment = self.experiment_class(
                        best_config[CONFIG], final_run_exp_path, exp_seed
                    )

                    # This is used to comunicate with the progress manager
                    # to display the UI
                    def _report_progress(payload: dict):
                        """
                        Forward progress updates to the shared UI (debug mode).

                        Args:
                            payload (dict): Progress fields produced by the experiment.

                        Side effects:
                            Sends the update to the Ray actor backing the terminal UI.
                        """
                        payload = deepcopy(payload)
                        payload.update(
                            {
                                OUTER_FOLD: dataset_getter.outer_k,
                                INNER_FOLD: None,
                                CONFIG_ID: best_config["best_config_id"] - 1,
                                RUN_ID: i,
                                IS_FINAL: True,
                            }
                        )
                        _push_progress_update(self.progress_actor, payload)

                    try:
                        res = experiment.run_test(
                            dataset_getter,
                            self.training_timeout_seconds,
                            logger,
                            progress_callback=_report_progress,
                        )
                        elapsed = extract_and_sum_elapsed_seconds(
                            osp.join(final_run_exp_path, EXPERIMENT_LOGFILE)
                        )

                        training_res, val_res, test_res = res
                        atomic_dill_save(
                            (training_res, val_res, test_res, elapsed),
                            final_run_torch_path,
                        )
                    except Exception as e:

                        _push_progress_update(
                            self.progress_actor,
                            {
                                "type": RUN_FAILED,
                                str(OUTER_FOLD): dataset_getter.outer_k,
                                str(INNER_FOLD): None,
                                str(CONFIG_ID): best_config["best_config_id"]
                                - 1,
                                str(RUN_ID): i,
                                str(IS_FINAL): True,
                                str(EPOCH): 0,
                                str(TOTAL_EPOCHS): 0,
                                "message": f"{e}\n{traceback.format_exc()}",
                            },
                        )

                        elapsed = -1
                        if self.failure_message is None:
                            self.failure_message = (
                                f"Final run {i + 1} for outer fold {outer_k + 1} failed; "
                                "skipping outer fold scoring and final assessment. "
                                "Check the run logs for details."
                            )
                        return None

        if debug:
            self.compute_final_runs_score_per_fold(outer_k)

    def process_model_selection_runs(
        self, inner_fold_config_folder: str, inner_k: int
    ):
        r"""
        Computes the average performances for the training runs about
            a specific configuration and a specific inner_fold split

        Args:
            inner_fold_config_folder (str): an inner fold experiment folder
                of a specific configuration
            inner_k (int): the inner fold id
        """
        fold_results_filename = osp.join(
            inner_fold_config_folder, f"fold_{str(inner_k + 1)}_results.dill"
        )
        fold_info_filename = osp.join(
            inner_fold_config_folder, f"fold_{str(inner_k + 1)}_results.info"
        )
        run_dict = [{} for _ in range(self.model_selection_training_runs)]
        results_dict = {}

        if self.model_selection_training_runs <= 0:
            raise ValueError(
                "model_selection_training_runs must be > 0, "
                f"got {self.model_selection_training_runs}."
            )
        for run_id in range(self.model_selection_training_runs):
            fold_run_exp_folder = osp.join(
                inner_fold_config_folder, f"run_{run_id + 1}"
            )
            fold_run_results_torch_path = osp.join(
                fold_run_exp_folder, f"run_{run_id + 1}_results.dill"
            )

            training_res, validation_res, _ = dill_load(
                fold_run_results_torch_path
            )

            training_loss, validation_loss = (
                training_res[LOSS],
                validation_res[LOSS],
            )
            training_score, validation_score = (
                training_res[SCORE],
                validation_res[SCORE],
            )

            for res_type, mode, res, main_res_type in [
                (LOSS, TRAINING, training_loss, MAIN_LOSS),
                (LOSS, VALIDATION, validation_loss, MAIN_LOSS),
                (SCORE, TRAINING, training_score, MAIN_SCORE),
                (SCORE, VALIDATION, validation_score, MAIN_SCORE),
            ]:
                for key in res.keys():
                    if main_res_type in key:
                        continue
                    run_dict[run_id][f"{mode}_{key}_{res_type}"] = float(
                        res[key]
                    )

            # Rename main loss key for aesthetic
            run_dict[run_id][TR_LOSS] = float(training_loss[MAIN_LOSS])
            run_dict[run_id][VL_LOSS] = float(validation_loss[MAIN_LOSS])
            run_dict[run_id][TR_SCORE] = float(training_score[MAIN_SCORE])
            run_dict[run_id][VL_SCORE] = float(validation_score[MAIN_SCORE])
            del training_loss[MAIN_LOSS]
            del validation_loss[MAIN_LOSS]
            del training_score[MAIN_SCORE]
            del validation_score[MAIN_SCORE]

        # Note that training/validation loss/score will be used only to extract
        # the proper keys
        for key_dict, set_type, res_type in [
            (training_loss, TRAINING, LOSS),
            (validation_loss, VALIDATION, LOSS),
            (training_score, TRAINING, SCORE),
            (validation_score, VALIDATION, SCORE),
        ]:
            for key in list(key_dict.keys()) + [res_type]:
                suffix = f"_{res_type}" if key != res_type else ""
                set_res = np.array(
                    [
                        run_dict[i][f"{set_type}_{key}{suffix}"]
                        for i in range(self.model_selection_training_runs)
                    ]
                )
                mean, std, ci = _mean_std_ci(set_res)
                results_dict[f"{set_type}_{key}{suffix}"] = mean
                results_dict[f"{STD}_{set_type}_{key}{suffix}"] = std
                results_dict[f"{CI}_{set_type}_{key}{suffix}"] = ci

        results_dict.update({MODEL_SELECTION_TRAINING_RUNS: run_dict})

        with open(fold_info_filename, "w") as fp:
            json.dump(results_dict, fp, sort_keys=False, indent=4)

        atomic_dill_save(
            (
                {
                    LOSS: {MAIN_LOSS: results_dict[f"{TRAINING}_{LOSS}"]},
                    SCORE: {MAIN_SCORE: results_dict[f"{TRAINING}_{SCORE}"]},
                },
                {
                    LOSS: {MAIN_LOSS: results_dict[f"{VALIDATION}_{LOSS}"]},
                    SCORE: {MAIN_SCORE: results_dict[f"{VALIDATION}_{SCORE}"]},
                },
                None,
            ),
            fold_results_filename,
        )

    def process_config_results_across_inner_folds(
        self, config_folder: str, config: Config
    ):
        r"""
        Averages the results for each configuration across inner folds and
        stores it into a file.

        Args:
            config_folder (str):
            config (:class:`~mlwiz.evaluation.config.Config`): the
                configuration object
        """
        config_filename = osp.join(config_folder, self._CONFIG_RESULTS)
        k_fold_dict = {
            CONFIG: config,
            FOLDS: [{} for _ in range(self.inner_folds)],
        }

        if self.inner_folds <= 0:
            raise ValueError(
                f"inner_folds must be > 0, got {self.inner_folds}."
            )
        for k in range(self.inner_folds):
            # Set up a log file for this experiment (run in a separate process)
            config_inner_fold_folder = osp.join(
                config_folder, self._INNER_FOLD_BASE + str(k + 1)
            )
            fold_results_torch_path = osp.join(
                config_inner_fold_folder, f"fold_{str(k + 1)}_results.dill"
            )

            training_res, validation_res, _ = dill_load(
                fold_results_torch_path
            )

            training_loss, validation_loss = (
                training_res[LOSS],
                validation_res[LOSS],
            )
            training_score, validation_score = (
                training_res[SCORE],
                validation_res[SCORE],
            )

            for res_type, mode, res, main_res_type in [
                (LOSS, TRAINING, training_loss, MAIN_LOSS),
                (LOSS, VALIDATION, validation_loss, MAIN_LOSS),
                (SCORE, TRAINING, training_score, MAIN_SCORE),
                (SCORE, VALIDATION, validation_score, MAIN_SCORE),
            ]:
                for key in res.keys():
                    if main_res_type in key:
                        continue
                    k_fold_dict[FOLDS][k][f"{mode}_{key}_{res_type}"] = float(
                        res[key]
                    )

            # Rename main loss key for aesthetic
            k_fold_dict[FOLDS][k][TR_LOSS] = float(training_loss[MAIN_LOSS])
            k_fold_dict[FOLDS][k][VL_LOSS] = float(validation_loss[MAIN_LOSS])
            k_fold_dict[FOLDS][k][TR_SCORE] = float(training_score[MAIN_SCORE])
            k_fold_dict[FOLDS][k][VL_SCORE] = float(
                validation_score[MAIN_SCORE]
            )
            del training_loss[MAIN_LOSS]
            del validation_loss[MAIN_LOSS]
            del training_score[MAIN_SCORE]
            del validation_score[MAIN_SCORE]

        # Note that training/validation loss/score will be used only to extract
        # the proper keys
        for key_dict, set_type, res_type in [
            (training_loss, TRAINING, LOSS),
            (validation_loss, VALIDATION, LOSS),
            (training_score, TRAINING, SCORE),
            (validation_score, VALIDATION, SCORE),
        ]:
            for key in list(key_dict.keys()) + [res_type]:
                suffix = f"_{res_type}" if key != res_type else ""
                set_res = np.array(
                    [
                        k_fold_dict[FOLDS][i][f"{set_type}_{key}{suffix}"]
                        for i in range(self.inner_folds)
                    ]
                )
                mean, std, ci = _mean_std_ci(set_res)
                k_fold_dict[f"{AVG}_{set_type}_{key}{suffix}"] = mean
                k_fold_dict[f"{STD}_{set_type}_{key}{suffix}"] = std
                k_fold_dict[f"{CI}_{set_type}_{key}{suffix}"] = ci

        with open(config_filename, "w") as fp:
            json.dump(k_fold_dict, fp, sort_keys=False, indent=4)

    def compute_best_hyperparameters(
        self,
        folder: str,
        outer_k: int,
        no_configurations: int,
        skip_config_ids: List[int],
    ):
        r"""
        Chooses the best hyper-parameters configuration using the proper
        validation mean score.

        Args:
            folder (str): the model selection folder associated with
                outer fold k
            outer_k (int): the current outer fold to consider. Used for
                telegram updates
            no_configurations (int): number of possible configurations
            skip_config_ids: list of configuration ids to skip
        """
        best_avg_vl = -float("inf") if self.higher_is_better else float("inf")
        best_std_vl = float("inf")

        for i in range(1, no_configurations + 1):
            if (i - 1) in skip_config_ids:
                continue
            config_filename = osp.join(
                folder, self._CONFIG_BASE + str(i), self._CONFIG_RESULTS
            )

            with open(config_filename, "r") as fp:
                config_dict = json.load(fp)

                avg_vl = config_dict[f"{AVG}_{VALIDATION}_{SCORE}"]
                std_vl = config_dict[f"{STD}_{VALIDATION}_{SCORE}"]

                if self.operator(avg_vl, best_avg_vl) or (
                    best_avg_vl == avg_vl and best_std_vl > std_vl
                ):
                    best_i = i
                    best_avg_vl = avg_vl
                    best_std_vl = std_vl
                    best_config = config_dict

        # Send telegram update
        if (
            self.model_configs.telegram_config is not None
            and self.log_model_selection
        ):
            exp_name = os.path.basename(self.exp_path)
            telegram_msg = (
                f"Exp *{exp_name}* \n"
                f"Model Sel. ended for outer fold *{outer_k + 1}* \n"
                f"Best config id: *{best_i}* \n"
                f"Main score: avg *{best_avg_vl:.4f}* "
                f"/ std *{best_std_vl:.4f}*"
            )
            send_telegram_update(
                self.telegram_bot_token,
                self.telegram_bot_chat_ID,
                telegram_msg,
            )

        with open(osp.join(folder, self._WINNER_CONFIG), "w") as fp:
            json.dump(
                dict(best_config_id=best_i, **best_config),
                fp,
                sort_keys=False,
                indent=4,
            )

    def compute_final_runs_score_per_fold(self, outer_k: int):
        r"""
        Computes the average scores for the final runs of a specific outer fold

        Args:
            outer_k (int): id of the outer fold from 0 to K-1
        """
        outer_folder = osp.join(
            self._ASSESSMENT_FOLDER, self._OUTER_FOLD_BASE + str(outer_k + 1)
        )
        config_fname = osp.join(
            outer_folder, self._SELECTION_FOLDER, self._WINNER_CONFIG
        )

        with open(config_fname, "r") as f:
            best_config = json.load(f)

            training_losses, validation_losses, test_losses = [], [], []
            training_scores, validation_scores, test_scores = [], [], []
            for i in range(self.risk_assessment_training_runs):
                final_run_exp_path = osp.join(
                    outer_folder, f"final_run{i + 1}"
                )
                final_run_torch_path = osp.join(
                    final_run_exp_path, f"run_{i + 1}_results.dill"
                )
                res = dill_load(final_run_torch_path)

                tr_res, vl_res, te_res = {}, {}, {}

                training_res, validation_res, test_res, _ = res
                training_loss, validation_loss, test_loss = (
                    training_res[LOSS],
                    validation_res[LOSS],
                    test_res[LOSS],
                )
                training_score, validation_score, test_score = (
                    training_res[SCORE],
                    validation_res[SCORE],
                    test_res[SCORE],
                )

                training_losses.append(training_loss)
                validation_losses.append(validation_loss)
                test_losses.append(test_loss)
                training_scores.append(training_score)
                validation_scores.append(validation_score)
                test_scores.append(test_score)

                # this block may be unindented, *_score used only to retrieve
                # keys
                scores = [
                    (training_score, tr_res, training_scores),
                    (validation_score, vl_res, validation_scores),
                    (test_score, te_res, test_scores),
                ]
                losses = [
                    (training_loss, tr_res, training_losses),
                    (validation_loss, vl_res, validation_losses),
                    (test_loss, te_res, test_losses),
                ]

                # this block may be unindented, set_score used only to retrieve
                # keys
                for res_type, res in [(LOSS, losses), (SCORE, scores)]:
                    for set_res_type, set_dict, set_results in res:
                        for key in set_res_type.keys():
                            suffix = (
                                f"_{res_type}"
                                if (key != MAIN_LOSS and key != MAIN_SCORE)
                                else ""
                            )
                            values = np.array(
                                [
                                    float(set_run[key])
                                    for set_run in set_results
                                ]
                            )
                            mean, std, ci = _mean_std_ci(values)
                            set_dict[key + suffix] = mean
                            set_dict[key + f"{suffix}_{STD}"] = std
                            set_dict[key + f"{suffix}_{CI}"] = ci

        # Send telegram update
        if (
            self.model_configs.telegram_config is not None
            and self.log_final_runs
        ):
            exp_name = os.path.basename(self.exp_path)
            telegram_msg = (
                f"Exp *{exp_name}* \n"
                f"Final runs ended for outer fold *{outer_k + 1}* \n"
                f"Main test score: avg *{scores[2][1][MAIN_SCORE]:.4f}* "
                f"/ std *{scores[2][1][f'{MAIN_SCORE}_{STD}']:.4f}*"
            )
            send_telegram_update(
                self.telegram_bot_token,
                self.telegram_bot_chat_ID,
                telegram_msg,
            )

        with open(
            osp.join(outer_folder, self._OUTER_RESULTS_FILENAME), "w"
        ) as fp:
            json.dump(
                {
                    BEST_CONFIG: best_config,
                    OUTER_TRAIN: tr_res,
                    OUTER_VALIDATION: vl_res,
                    OUTER_TEST: te_res,
                },
                fp,
                sort_keys=False,
                indent=4,
            )

    def compute_risk_assessment_result(self):
        r"""
        Aggregates Outer Folds results and compute Training and Test mean/std
        """
        outer_tr_results = []
        outer_vl_results = []
        outer_ts_results = []
        assessment_results = {}

        for i in range(1, self.outer_folds + 1):
            config_filename = osp.join(
                self._ASSESSMENT_FOLDER,
                self._OUTER_FOLD_BASE + str(i),
                self._OUTER_RESULTS_FILENAME,
            )

            with open(config_filename, "r") as fp:
                outer_fold_results = json.load(fp)
                outer_tr_results.append(outer_fold_results[OUTER_TRAIN])
                outer_vl_results.append(outer_fold_results[OUTER_VALIDATION])
                outer_ts_results.append(outer_fold_results[OUTER_TEST])

                for k in outer_fold_results[
                    OUTER_TRAIN
                ].keys():  # train keys are the same as valid and test keys
                    # Do not want to average std of different final runs in
                    # different outer folds
                    if k.endswith(STD) or k.endswith(CI):
                        continue

                    # there may be different optimal losses for each outer
                    # fold, so we cannot always compute the average over
                    # K outer folds of the same loss this is not so
                    # problematic as one can always recover the average and
                    # standard loss values across outer folds when we
                    # have the same loss for all outer folds using
                    # a jupyter notebook
                    if "_loss" in k:
                        continue

                    outer_results = [
                        (outer_tr_results, TRAINING),
                        (outer_vl_results, VALIDATION),
                        (outer_ts_results, TEST),
                    ]

                    for results, set in outer_results:
                        set_results = np.array([res[k] for res in results])
                        mean, std, ci = _mean_std_ci(set_results)
                        assessment_results[f"{AVG}_{set}_{k}"] = mean
                        assessment_results[f"{STD}_{set}_{k}"] = std
                        assessment_results[f"{CI}_{set}_{k}"] = ci

        # Send telegram update
        if self.model_configs.telegram_config is not None:
            exp_name = os.path.basename(self.exp_path)
            telegram_msg = (
                f"Exp *{exp_name}* \n"
                f"Experiment has finished \n"
                f"Test score: avg "
                f"*{assessment_results[f'{AVG}_{TEST}_{MAIN_SCORE}']:.4f}* "
                f"/ std"
                f" *{assessment_results[f'{STD}_{TEST}_{MAIN_SCORE}']:.4f}*"
            )
            send_telegram_update(
                self.telegram_bot_token,
                self.telegram_bot_chat_ID,
                telegram_msg,
            )

        with open(
            osp.join(self._ASSESSMENT_FOLDER, self._ASSESSMENT_FILENAME), "w"
        ) as fp:
            json.dump(assessment_results, fp, sort_keys=False, indent=4)
