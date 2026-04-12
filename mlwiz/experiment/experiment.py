"""Experiment wrapper for building and running training jobs.

Defines :class:`~mlwiz.experiment.experiment.Experiment`, which instantiates models/engines from configs and runs validation/test loops.
"""

import random
import os
import socket
import threading
import time
import traceback
import faulthandler
from queue import Empty
from typing import Callable, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from mlwiz.exceptions import (
    ExperimentTerminated,
    TerminationRequested,
)
from mlwiz.evaluation.config import Config
from mlwiz.util import return_class_and_args, s2c
from mlwiz.model.interface import ModelInterface
from mlwiz.static import DEFAULT_ENGINE_CALLBACK
from mlwiz.static import LOSS, SCORE
from mlwiz.static import MLWIZ_RAY_NUM_GPUS_PER_TASK
from mlwiz.log.logger import Logger
from mlwiz.training.distributed import dist_is_initialized
from mlwiz.training.engine import TrainingEngine


def _find_free_port() -> int:
    """
    Return a free localhost TCP port.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _to_dotted_path(obj) -> str:
    """
    Return the dotted path for a class/function object.
    """
    return f"{obj.__module__}.{obj.__qualname__}"


def _to_queue_safe(obj):
    """
    Convert tensors/numpy values to plain Python before queue transfer.
    """
    if torch.is_tensor(obj):
        obj = obj.detach().cpu()
        return obj.item() if obj.numel() == 1 else obj.tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _to_queue_safe(v) for k, v in obj.items()}
    if isinstance(obj, tuple):
        return tuple(_to_queue_safe(v) for v in obj)
    if isinstance(obj, list):
        return [_to_queue_safe(v) for v in obj]
    return obj


def _ddp_worker(
    rank: int,
    world_size: int,
    mode: str,
    experiment_spec,
    dataset_getter_spec,
    training_timeout_seconds,
    logger_spec,
    master_port: int,
    result_queue,
    progress_queue,
    stop_flag,
):
    """
    Worker used by DDP spawn.
    """
    experiment = None
    rank_log = None
    res = None
    try:
        os.makedirs(experiment_spec["exp_path"], exist_ok=True)
        rank_log = open(
            os.path.join(
                experiment_spec["exp_path"], f"ddp_rank_{rank}.log"
            ),
            "a",
        )
        faulthandler.enable(rank_log)
        rank_log.write("DDP worker started.\n")
        rank_log.flush()

        experiment_class = s2c(experiment_spec["class_name"])
        experiment = experiment_class(
            experiment_spec["model_configuration"],
            experiment_spec["exp_path"],
            experiment_spec["exp_seed"],
        )

        dataset_getter_class = s2c(dataset_getter_spec["class_name"])
        dataset_class = s2c(dataset_getter_spec["dataset_class"])
        data_loader_class = s2c(dataset_getter_spec["data_loader_class"])
        dataset_getter = dataset_getter_class(
            dataset_getter_spec["storage_folder"],
            dataset_getter_spec["splits_filepath"],
            dataset_class,
            data_loader_class,
            dataset_getter_spec["data_loader_args"],
            dataset_getter_spec["outer_folds"],
            dataset_getter_spec["inner_folds"],
        )
        dataset_getter.set_outer_k(dataset_getter_spec["outer_k"])
        dataset_getter.set_inner_k(dataset_getter_spec["inner_k"])
        dataset_getter.set_exp_seed(dataset_getter_spec["exp_seed"])

        logger = None
        if logger_spec is not None:
            logger = Logger(
                logger_spec["filepath"],
                logger_spec["mode"],
                logger_spec["debug"],
            )

        np.random.seed(experiment_spec["exp_seed"])
        torch.manual_seed(experiment_spec["exp_seed"])
        torch.cuda.manual_seed(experiment_spec["exp_seed"])
        random.seed(experiment_spec["exp_seed"])

        experiment._setup_ddp(rank, world_size, master_port)
        progress_cb = None
        if rank == 0 and progress_queue is not None:
            # Rank 0 forwards progress to parent process.
            progress_cb = lambda payload: progress_queue.put(payload)
        should_terminate_cb = lambda: bool(stop_flag.value)

        if mode == "valid":
            res = experiment._run_valid_impl(
                dataset_getter,
                training_timeout_seconds,
                logger,
                progress_callback=progress_cb,
                should_terminate=should_terminate_cb,
                ddp_rank=rank,
                ddp_world_size=world_size,
            )
        elif mode == "test":
            res = experiment._run_test_impl(
                dataset_getter,
                training_timeout_seconds,
                logger,
                progress_callback=progress_cb,
                should_terminate=should_terminate_cb,
                ddp_rank=rank,
                ddp_world_size=world_size,
            )
        else:
            raise ValueError(f"Unsupported DDP mode: {mode}")
    except ExperimentTerminated:
        res = None
    except Exception:
        stop_flag.value = True
        tb = traceback.format_exc()
        if rank_log is not None:
            rank_log.write(tb + "\n")
            rank_log.flush()
        res = {"__ddp_error__": tb}
    finally:
        if rank == 0:
            # Keep queue payload plain Python to avoid shared-memory FD issues.
            result_queue.put(_to_queue_safe(res))
        try:
            if experiment is not None:
                experiment._cleanup_ddp()
        except Exception:
            pass
        if rank_log is not None:
            rank_log.close()


class Experiment:
    r"""
    Class that handles a single standard experiment.

    Args:
        model_configuration (dict): the dictionary holding the
            experiment-specific configuration
        exp_path (str): path to the experiment folder
        exp_seed (int): the experiment's seed to use
    """

    def __init__(
        self, model_configuration: dict, exp_path: str, exp_seed: int
    ):
        r"""
        Initialize an experiment and set deterministic seeds/flags.

        Args:
            model_configuration (dict): Experiment configuration dictionary
                (model, training engine, optimizer, metrics, etc.).
            exp_path (str): Folder where this experiment run will write its
                artifacts (logs, checkpoints, results).
            exp_seed (int): Seed used to initialize RNGs for reproducibility.

        Side effects:
            - Wraps the configuration with :class:`~mlwiz.evaluation.config.Config`.
            - Sets NumPy, Python, and PyTorch RNG seeds.
            - Enables deterministic CuDNN behavior and disables benchmarking.
        """
        self.model_config = Config(model_configuration)
        self.exp_path = exp_path
        self.exp_seed = exp_seed
        # Set seed here to aid reproducibility
        np.random.seed(self.exp_seed)
        torch.manual_seed(self.exp_seed)
        torch.cuda.manual_seed(self.exp_seed)
        random.seed(self.exp_seed)

        # torch.use_deterministic_algorithms(True) for future versions of
        # Pytorch
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _should_use_ddp(self) -> bool:
        """
        Enable DDP when multiple CUDA devices are visible in this process.
        """
        raw_gpus_per_task = os.environ.get(MLWIZ_RAY_NUM_GPUS_PER_TASK, "1")
        try:
            gpus_per_task = float(raw_gpus_per_task)
        except ValueError:
            gpus_per_task = 1.0

        device = str(self.model_config.get("device", "cpu"))
        return (
            "cuda" in device
            and torch.cuda.is_available()
            and gpus_per_task > 1
            and float(gpus_per_task).is_integer()
            and torch.cuda.device_count() >= gpus_per_task
        )

    def _set_worker_device(self, ddp_rank: Optional[int]):
        """
        Set per-rank device in config.
        """
        if ddp_rank is not None:
            self.model_config.config_dict["device"] = f"cuda:{ddp_rank}"

    def _setup_ddp(self, rank: int, world_size: int, master_port: int):
        """
        Initialize the process group for this rank.
        """
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(master_port)
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(rank)
        torch.cuda.set_device(rank)
        dist.init_process_group(
            backend="nccl", rank=rank, world_size=world_size
        )
        self._set_worker_device(rank)

    def _cleanup_ddp(self):
        """
        Tear down process group.
        """
        if dist_is_initialized():
            dist.destroy_process_group()

    def _wrap_ddp_model(self, model, ddp_rank: Optional[int]):
        """
        Wrap model in DDP (single-device and model-parallel cases).
        """
        if ddp_rank is None:
            return model

        # Check whether model params live on a single CUDA device or not.
        param_devices = {p.device for p in model.parameters() if p.is_cuda}
        if len(param_devices) <= 1:
            # Standard case: one GPU per rank, pin this process to local GPU.
            return DDP(
                model, device_ids=[ddp_rank], output_device=ddp_rank
            )
        # Model-parallel case: params span multiple GPUs, so do not pin.
        return DDP(model)

    def _run_ddp(
        self,
        mode: str,
        dataset_getter,
        training_timeout_seconds,
        logger,
        progress_callback: Callable[[dict], None] = None,
        should_terminate: Optional[Callable[[], bool]] = None,
    ):
        """
        Spawn one local process per visible GPU and return rank-0 result.
        """
        requested = int(float(os.environ[MLWIZ_RAY_NUM_GPUS_PER_TASK]))
        world_size = min(requested, torch.cuda.device_count())
        # Use spawn context for all IPC objects to match mp.spawn().
        ctx = mp.get_context("spawn")
        result_queue = ctx.Queue()
        progress_queue = ctx.Queue()
        stop_flag = ctx.Value("b", False)
        master_port = _find_free_port()
        stop_event = threading.Event()
        experiment_spec = {
            "class_name": _to_dotted_path(self.__class__),
            "model_configuration": self.model_config.config_dict,
            "exp_path": self.exp_path,
            "exp_seed": self.exp_seed,
        }
        dataset_getter_spec = {
            "class_name": _to_dotted_path(dataset_getter.__class__),
            "storage_folder": dataset_getter.storage_folder,
            "splits_filepath": dataset_getter.splits_filepath,
            "dataset_class": (
                dataset_getter.dataset_class
                if isinstance(dataset_getter.dataset_class, str)
                else _to_dotted_path(dataset_getter.dataset_class)
            ),
            "data_loader_class": (
                dataset_getter.data_loader_class
                if isinstance(dataset_getter.data_loader_class, str)
                else _to_dotted_path(dataset_getter.data_loader_class)
            ),
            "data_loader_args": dataset_getter.data_loader_args,
            "outer_folds": dataset_getter.outer_folds,
            "inner_folds": dataset_getter.inner_folds,
            "outer_k": dataset_getter.outer_k,
            "inner_k": dataset_getter.inner_k,
            "exp_seed": dataset_getter.exp_seed,
        }
        logger_spec = (
            {
                "filepath": str(logger.filepath),
                "mode": logger.mode,
                "debug": logger.debug,
            }
            if logger is not None
            else None
        )

        def _progress_loop():
            while not stop_event.is_set():
                try:
                    payload = progress_queue.get(timeout=0.1)
                except Empty:
                    continue
                if payload is None:
                    break
                if progress_callback is not None:
                    progress_callback(payload)

        def _termination_loop():
            while not stop_event.is_set():
                if should_terminate is not None:
                    try:
                        if should_terminate():
                            stop_flag.value = True
                            break
                    except Exception:
                        stop_flag.value = True
                        break
                time.sleep(0.2)

        progress_thread = threading.Thread(
            target=_progress_loop, daemon=True
        )
        termination_thread = threading.Thread(
            target=_termination_loop, daemon=True
        )
        progress_thread.start()
        termination_thread.start()

        try:
            mp.spawn(
                _ddp_worker,
                nprocs=world_size,
                join=True,
                args=(
                    world_size,
                    mode,
                    experiment_spec,
                    dataset_getter_spec,
                    training_timeout_seconds,
                    logger_spec,
                    master_port,
                    result_queue,
                    progress_queue,
                    stop_flag,
                ),
            )
            res = result_queue.get()
            if isinstance(res, dict) and "__ddp_error__" in res:
                raise RuntimeError(res["__ddp_error__"])
            return res
        except Exception as e:
            try:
                res = result_queue.get_nowait()
                if isinstance(res, dict) and "__ddp_error__" in res:
                    raise RuntimeError(res["__ddp_error__"]) from e
            except Exception:
                pass
            raise RuntimeError(
                f"DDP worker exited early. Check rank logs under: {self.exp_path}"
            ) from e
        finally:
            stop_event.set()
            progress_queue.put(None)
            progress_thread.join()
            termination_thread.join()

    def _return_class_and_args(
        config: Config, key: str
    ) -> Tuple[Callable[..., object], dict]:
        r"""
        Returns the class and arguments associated to a specific key in the
        configuration file.

        Args:
            config: the configuration dictionary
            key: a string representing a particular class in the
                configuration dictionary

        Returns:
            a tuple (class, dict of arguments), or (None, None) if the key
            is not present in the config dictionary
        """
        if key not in config or config[key] is None:
            return None, None
        elif isinstance(config[key], str):
            return s2c(config[key]), {}
        elif isinstance(config[key], dict):
            return (
                s2c(config[key]["class_name"]),
                config[key]["args"] if "args" in config[key] else {},
            )
        else:
            raise NotImplementedError(
                "Parameter has not been formatted properly"
            )

    def create_model(
        self,
        dim_input_features: Union[int, Tuple[int]],
        dim_target: int,
        config: Config,
    ) -> ModelInterface:
        r"""
        Instantiates a model that implements the
        :class:`~mlwiz.model.model.ModelInterface` interface

        Args:
            dim_input_features (Union[int, Tuple[int]]): number of node features
            dim_target (int): target dimension
            config (:class:`~mlwiz.evaluation.config.Config`):
                the configuration dictionary

        Returns:
            a model that implements the
            :class:`~mlwiz.model.model.ModelInterface` interface
        """
        model = s2c(config["model"])(
            dim_input_features=dim_input_features,
            dim_target=dim_target,
            config=config,
        )

        # move to device
        # model .to() may not return anything
        model.to(self.model_config.device)
        return model

    def create_engine(
        self,
        config: Config,
        model: ModelInterface,
    ) -> TrainingEngine:
        r"""
        Utility that instantiates the training engine. It looks for
        pre-defined fields in the configuration file, i.e. ``loss``,
        ``scorer``, ``optimizer``, ``scheduler``, ``gradient_clipper``,
        ``early_stopper`` and ``plotter``, all of which should be classes
        implementing the :class:`~mlwiz.training.event.handler.EventHandler`
        interface

        Args:
            config (:class:`~mlwiz.evaluation.config.Config`):
                the configuration dictionary
            model: the  model that needs be trained

        Returns:
            a :class:`~mlwiz.training.engine.TrainingEngine` object
        """
        device = config["device"]
        evaluate_every = config["evaluate_every"]

        loss_class, loss_args = return_class_and_args(config, "loss")
        loss_args.update(device=device)
        loss = (
            loss_class(use_as_loss=True, **loss_args)
            if loss_class is not None
            else None
        )

        scorer_class, scorer_args = return_class_and_args(config, "scorer")
        scorer_args.update(device=device)
        scorer = (
            scorer_class(use_as_loss=False, **scorer_args)
            if scorer_class is not None
            else None
        )

        optim_class, optim_args = return_class_and_args(config, "optimizer")
        optimizer = (
            optim_class(model=model, **optim_args)
            if optim_class is not None
            else None
        )

        sched_class, sched_args = return_class_and_args(config, "scheduler")
        if sched_args is not None:
            sched_args["optimizer"] = optimizer.optimizer
        scheduler = (
            sched_class(**sched_args) if sched_class is not None else None
        )
        # Remove the optimizer obj ow troubles when dumping the config file
        if sched_args is not None:
            sched_args.pop("optimizer", None)

        grad_clip_class, grad_clip_args = return_class_and_args(
            config, "gradient_clipper"
        )
        grad_clipper = (
            grad_clip_class(**grad_clip_args)
            if grad_clip_class is not None
            else None
        )

        early_stop_class, early_stop_args = return_class_and_args(
            config, "early_stopper"
        )
        early_stopper = (
            early_stop_class(**early_stop_args)
            if early_stop_class is not None
            else None
        )

        plot_class, plot_args = return_class_and_args(config, "plotter")
        plotter = (
            plot_class(exp_path=self.exp_path, **plot_args)
            if plot_class is not None
            else None
        )

        store_last_checkpoint = config.get("checkpoint", False)
        engine_class, engine_args = return_class_and_args(config, "engine")
        engine_callback = s2c(
            engine_args.get("engine_callback", DEFAULT_ENGINE_CALLBACK)
        )
        eval_training = engine_args.get("eval_training", False)
        mixed_precision = engine_args.get("mixed_precision", False)
        mixed_precision_dtype = engine_args.get(
            "mixed_precision_dtype", "torch.float16"
        )

        engine = engine_class(
            engine_callback=engine_callback,
            model=model,
            loss=loss,
            optimizer=optimizer,
            scorer=scorer,
            scheduler=scheduler,
            early_stopper=early_stopper,
            gradient_clipper=grad_clipper,
            device=device,
            plotter=plotter,
            exp_path=self.exp_path,
            evaluate_every=evaluate_every,
            eval_training=eval_training,
            store_last_checkpoint=store_last_checkpoint,
            mixed_precision=mixed_precision,
            mixed_precision_dtype=mixed_precision_dtype,
        )
        return engine

    def run_valid(
        self,
        dataset_getter,
        training_timeout_seconds,
        logger,
        progress_callback: Callable[[dict], None] = None,
        should_terminate: Optional[Callable[[], bool]] = None,
    ):
        r"""
        This function returns the training and validation results
        for a `model selection run`.
        **Do not attempt to load the test set inside this method!**
        **If possible, rely on already available subclasses of this class**.

        It implements a simple training scheme.

        Args:
            dataset_getter (:class:`~mlwiz.data.provider.DataProvider`):
                a data provider
            training_timeout_seconds (int): timeout for the experiment in seconds
            logger (:class:`~mlwiz.log.logger.Logger`): the logger

        Returns:
            a tuple of training and test dictionaries.
            Each dictionary has two keys:

            * ``LOSS`` (as defined in ``mlwiz.static``)
            * ``SCORE`` (as defined in ``mlwiz.static``)

            For instance, training_results[SCORE] is a dictionary itself
            with other fields to be used by the evaluator.
        """
        if self._should_use_ddp():
            return self._run_ddp(
                "valid",
                dataset_getter,
                training_timeout_seconds,
                logger,
                progress_callback=progress_callback,
                should_terminate=should_terminate,
            )
        return self._run_valid_impl(
            dataset_getter,
            training_timeout_seconds,
            logger,
            progress_callback=progress_callback,
            should_terminate=should_terminate,
        )

    def _run_valid_impl(
        self,
        dataset_getter,
        training_timeout_seconds,
        logger,
        progress_callback: Callable[[dict], None] = None,
        should_terminate: Optional[Callable[[], bool]] = None,
        ddp_rank: Optional[int] = None,
        ddp_world_size: int = 1,
    ):
        """
        Internal validation run used by both single-process and DDP paths.
        """
        self._set_worker_device(ddp_rank)
        batch_size = self.model_config["batch_size"]
        shuffle = (
            self.model_config["shuffle"]
            if "shuffle" in self.model_config
            else True
        )

        # Instantiate the Dataset
        train_loader = dataset_getter.get_inner_train(
            batch_size=batch_size,
            shuffle=shuffle,
            ddp_rank=ddp_rank,
            ddp_world_size=ddp_world_size,
        )
        val_loader = dataset_getter.get_inner_val(
            batch_size=batch_size, shuffle=shuffle
        )

        dim_input_features = dataset_getter.get_dim_input_features()
        dim_target = dataset_getter.get_dim_target()

        # Instantiate the Model
        model = self.create_model(
            dim_input_features, dim_target, self.model_config
        )
        model = self._wrap_ddp_model(model, ddp_rank)

        # Instantiate the engine (it handles the training loop and the
        # inference phase by abstracting the specifics)
        training_engine = self.create_engine(self.model_config, model)

        try:
            (
                train_loss,
                train_score,
                _,  # check the ordering is correct
                val_loss,
                val_score,
                _,
                _,
                _,
                _,
            ) = training_engine.train(
                train_loader=train_loader,
                validation_loader=val_loader,
                test_loader=None,
                max_epochs=self.model_config["epochs"],
                logger=logger,
                training_timeout_seconds=training_timeout_seconds,
                progress_callback=progress_callback,
                should_terminate=should_terminate,
            )
        except TerminationRequested as exc:
            raise ExperimentTerminated("Validation run terminated.") from exc

        train_res = {LOSS: train_loss, SCORE: train_score}
        val_res = {LOSS: val_loss, SCORE: val_score}
        return train_res, val_res

    def run_test(
        self,
        dataset_getter,
        training_timeout_seconds,
        logger,
        progress_callback: Callable[[dict], None] = None,
        should_terminate: Optional[Callable[[], bool]] = None,
    ):
        """
        This function returns the training, validation and test results
        for a `final run`.
        **Do not use the test to train the model
        nor for early stopping reasons!**
        **If possible, rely on already available subclasses of this class**.

        It implements a simple training scheme.

        Args:
            dataset_getter (:class:`~mlwiz.data.provider.DataProvider`):
                a data provider
            training_timeout_seconds (int): timeout for the experiment in seconds
            logger (:class:`~mlwiz.log.logger.Logger`): the logger

        Returns:
            a tuple of training,validation,test dictionaries.
            Each dictionary has two keys:

            * ``LOSS`` (as defined in ``mlwiz.static``)
            * ``SCORE`` (as defined in ``mlwiz.static``)

            For instance, training_results[SCORE] is a dictionary itself with
            other fields to be used by the evaluator.
        """
        if self._should_use_ddp():
            return self._run_ddp(
                "test",
                dataset_getter,
                training_timeout_seconds,
                logger,
                progress_callback=progress_callback,
                should_terminate=should_terminate,
            )
        return self._run_test_impl(
            dataset_getter,
            training_timeout_seconds,
            logger,
            progress_callback=progress_callback,
            should_terminate=should_terminate,
        )

    def _run_test_impl(
        self,
        dataset_getter,
        training_timeout_seconds,
        logger,
        progress_callback: Callable[[dict], None] = None,
        should_terminate: Optional[Callable[[], bool]] = None,
        ddp_rank: Optional[int] = None,
        ddp_world_size: int = 1,
    ):
        """
        Internal final run used by both single-process and DDP paths.
        """
        self._set_worker_device(ddp_rank)
        batch_size = self.model_config["batch_size"]
        shuffle = (
            self.model_config["shuffle"]
            if "shuffle" in self.model_config
            else True
        )

        # Instantiate the Dataset
        train_loader = dataset_getter.get_outer_train(
            batch_size=batch_size,
            shuffle=shuffle,
            ddp_rank=ddp_rank,
            ddp_world_size=ddp_world_size,
        )
        val_loader = dataset_getter.get_outer_val(
            batch_size=batch_size, shuffle=shuffle
        )
        test_loader = dataset_getter.get_outer_test(
            batch_size=batch_size, shuffle=shuffle
        )

        # Call this after the loaders: the datasets may need to be instantiated
        # with additional parameters
        dim_input_features = dataset_getter.get_dim_input_features()
        dim_target = dataset_getter.get_dim_target()

        # Instantiate the Model
        model = self.create_model(
            dim_input_features, dim_target, self.model_config
        )
        model = self._wrap_ddp_model(model, ddp_rank)

        # Instantiate the engine (it handles the training loop and the
        # inference phase by abstracting the specifics)
        training_engine = self.create_engine(self.model_config, model)

        try:
            (
                train_loss,
                train_score,
                _,
                val_loss,
                val_score,
                _,
                test_loss,
                test_score,
                _,
            ) = training_engine.train(
                train_loader=train_loader,
                validation_loader=val_loader,
                test_loader=test_loader,
                max_epochs=self.model_config["epochs"],
                logger=logger,
                training_timeout_seconds=training_timeout_seconds,
                progress_callback=progress_callback,
                should_terminate=should_terminate,
            )
        except TerminationRequested as exc:
            raise ExperimentTerminated("Final run terminated.") from exc

        train_res = {LOSS: train_loss, SCORE: train_score}
        val_res = {LOSS: val_loss, SCORE: val_score}
        test_res = {LOSS: test_loss, SCORE: test_score}
        return train_res, val_res, test_res
