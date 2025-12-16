import os
from pathlib import Path
from typing import Callable, List, Union, Tuple, Optional

import time
import torch
import torch_geometric
from torch.utils.data import SequentialSampler
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from mlwiz.exceptions import TerminationRequested
from mlwiz.log.logger import Logger
from mlwiz.model.interface import ModelInterface
from mlwiz.static import (
    BATCH,
    BATCH_PROGRESS,
    BEST_CHECKPOINT_FILENAME,
    BEST_EPOCH,
    CUMULATIVE_UNSENT_DELTA,
    EMB_TUPLE_SUBSTR,
    EPOCH,
    LAST_CHECKPOINT_FILENAME,
    LAST_RUN_ELAPSED_TIME,
    LOSSES,
    MAIN_LOSS,
    MAIN_SCORE,
    MODE,
    MODEL_STATE,
    OPTIMIZER_STATE,
    RUN_COMPLETED,
    RUN_PROGRESS,
    SCHEDULER_STATE,
    SCORES,
    START_CONFIG,
    STOP_TRAINING,
    TEST,
    TIME_DELTA,
    TOTAL_BATCHES,
    TOTAL_EPOCHS,
    TRAINING,
    VALIDATION,
)
from mlwiz.training.callback.early_stopping import EarlyStopper
from mlwiz.training.callback.engine_callback import EngineCallback
from mlwiz.training.callback.gradient_clipping import GradientClipper
from mlwiz.training.callback.metric import Metric
from mlwiz.training.callback.optimizer import Optimizer
from mlwiz.training.callback.plotter import Plotter
from mlwiz.training.callback.scheduler import Scheduler
from mlwiz.training.event.dispatcher import EventDispatcher
from mlwiz.training.event.handler import EventHandler
from mlwiz.training.event.state import State
from mlwiz.training.profiler import Profiler
from copy import deepcopy


def log(msg, logger: Logger):
    """
    Logs a message using a logger
    """
    if logger is not None:
        logger.log(msg)


def fmt(x, decimals=2, sci_decimals=2):
    """Format number with fixed-point unless it's small, then scientific."""
    thresh = 10 ** (-decimals - 1)
    x = float(x)
    if x == 0.0:
        return f"{0:.{decimals}f}"
    if abs(x) < thresh:
        return f"{x:.{sci_decimals}e}"
    return f"{x:.{decimals}f}"


def reorder(obj: List[object], perm: List[int]):
    """
    Reorders a list of objects in ascending order according to the indices
    defined in permutation argument.

    Args:
        obj (List[object]): the list of objects
        perm (List[int]): the permutation

    Returns:
        The reordered list of objects
    """
    if len(obj) != len(perm) or len(obj) == 0:
        raise ValueError(
            "Expected `obj` and `perm` to have the same non-zero length, "
            f"got len(obj)={len(obj)} and len(perm)={len(perm)}."
        )
    return [
        y for (x, y) in sorted(zip(perm, obj))
    ]  # sort items by original dataset index


class TrainingEngine(EventDispatcher):
    """
    This is the most important class when it comes to training a model.
    It implements the :class:`~mlwiz.training.event.dispatcher.EventDispatcher`
    interface, which means that after registering some callbacks in a
    given order, it will proceed to trigger specific events that
    will result in the shared :class:`~mlwiz.training.event.state.State`
    object being updated by the callbacks. Callbacks implement the EventHandler
    interface, and they receive the shared State object when any event is
    triggered. Knowing the order in which
    callbacks are called is important.
    The order is:

    * loss function
    * score function
    * gradient clipper
    * optimizer
    * early stopper
    * scheduler
    * plotter

    Args:
        engine_callback
            (Callable[...,
            :class:`~mlwiz.training.callback.engine_callback.EngineCallback`]):
            the engine callback object to be used for data fetching and
            checkpoints (or even other purposes if necessary)
        model (:class:`~mlwiz.model.interface.ModelInterface`):
            the model to be trained
        loss (:class:`~mlwiz.training.callback.metric.Metric`):
            the loss to be used
        optimizer (:class:`~mlwiz.training.callback.optimizer.Optimizer`):
            the optimizer to be used
        scorer (:class:`~mlwiz.training.callback.metric.Metric`):
            the score to be used
        scheduler (:class:`~mlwiz.training.callback.scheduler.Scheduler`):
            the scheduler to be used Default is ``None``.
        early_stopper
            (:class:`~mlwiz.training.callback.early_stopping.EarlyStopper`):
                the early stopper to be used. Default is ``None``.
        gradient_clipper
            (:class:`~mlwiz.training.callback.gradient_clipper.GradientClipper`
            ): the gradient clipper to be used. Default is ``None``.
        device (str): the device on which to train. Default is ``cpu``.
        plotter (:class:`~mlwiz.training.callback.plotter.Plotter`):
            the plotter to be used. Default is ``None``.
        exp_path (str): the path of the experiment folder.
            Default is ``None`` but it is always instantiated.
        evaluate_every (int): the frequency of logging epoch results.
            Default is ``1``.
        eval_training (bool): whether to re-evaluate loss and scores on the
            training set after a training epoch. Defaults to False.
        eval_test_every_epoch (bool): whether to evaluate loss and scores on
            the test set (if available) after each training epoch. Defaults to
            False because one should not care about test performance until the
            very end of risk assessment. However, set this to True if you want
            to log test metrics during training.
        store_last_checkpoint (bool): whether to store a checkpoint at
            the end of each epoch. Allows to resume training from last epoch.
            Default is ``False``.
    """

    cumulative_batch_unsent_time: float = 0.0
    cumulative_epoch_unsent_time: float = 0.0

    def __init__(
        self,
        engine_callback: Callable[..., EngineCallback],
        model: ModelInterface,
        loss: Metric,
        optimizer: Optimizer,
        scorer: Metric,
        scheduler: Scheduler = None,
        early_stopper: EarlyStopper = None,
        gradient_clipper: GradientClipper = None,
        device: str = "cpu",
        plotter: Plotter = None,
        exp_path: str = None,
        evaluate_every: int = 1,
        eval_training: bool = False,
        eval_test_every_epoch: bool = False,
        store_last_checkpoint: bool = False,
    ):
        """
        Initialize the training engine and register event callbacks.

        Args:
            engine_callback (Callable[..., EngineCallback]): Callback class used
                for data fetching, forward pass, and checkpointing.
            model (ModelInterface): Model to train.
            loss (Metric): Metric used as loss (``use_as_loss=True``).
            optimizer (Optimizer): Optimizer callback.
            scorer (Metric): Metric used as score (``use_as_loss=False``).
            scheduler (Scheduler, optional): Learning-rate scheduler callback.
            early_stopper (EarlyStopper, optional): Early stopping callback.
            gradient_clipper (GradientClipper, optional): Gradient clipping
                callback.
            device (str): Device identifier (e.g., ``'cpu'`` or ``'cuda:0'``).
            plotter (Plotter, optional): Plotting callback.
            exp_path (str, optional): Experiment output folder used by some
                callbacks for logging and checkpoints.
            evaluate_every (int): Frequency (in epochs) at which epoch-level
                results are reported/logged.
            eval_training (bool): Whether to re-evaluate loss/score on the
                training set after each epoch.
            eval_test_every_epoch (bool): Whether to evaluate on the test set
                after each epoch (if available).
            store_last_checkpoint (bool): Whether to store a checkpoint at the
                end of each epoch.

        Side effects:
            Registers all non-``None`` callbacks (loss, scorer, optimizer, etc.)
            plus an engine callback, and initializes the shared
            :class:`~mlwiz.training.event.state.State`.
        """
        super().__init__()

        self.engine_callback = engine_callback
        self.model = model
        self.loss_fun = loss
        self.optimizer = optimizer
        self.score_fun = scorer
        self.scheduler = scheduler
        self.early_stopper = early_stopper
        self.gradient_clipper = gradient_clipper
        self.device = device
        self.plotter = plotter
        self.exp_path = exp_path
        self.evaluate_every = evaluate_every
        self.eval_training = eval_training
        self.eval_test_every_epoch = eval_test_every_epoch
        self.store_last_checkpoint = store_last_checkpoint
        self.training = False
        self.logger = None

        self._total_batches = None
        self._should_terminate = None

        self.profiler = Profiler(threshold=1e-5)

        # Now register the callbacks (IN THIS ORDER, KNOWN TO THE USER)
        # Decorate with a profiler
        self.callbacks = [
            self.profiler(c)
            for c in [
                self.loss_fun,
                self.score_fun,
                self.gradient_clipper,
                self.optimizer,
                self.early_stopper,
                self.scheduler,
                self.plotter,
            ]
            if c is not None
        ]  # filter None callbacks

        # Add an Engine specific callback to profile different passages of
        # _loop
        self.callbacks.append(
            self.profiler(
                self.engine_callback(
                    store_last_checkpoint=self.store_last_checkpoint
                )
            )
        )

        for c in self.callbacks:
            self.register(c)

        self.state = State(
            self.model, self.optimizer, self.device
        )  # Initialize the state
        self.state.update(exp_path=self.exp_path)

    def _check_termination(self):
        """
        Raises if an external termination has been requested.
        """
        if self._should_terminate is None:
            return
        try:
            should_stop = self._should_terminate()
        except TerminationRequested:
            raise
        except Exception:
            return
        if should_stop:
            raise TerminationRequested("Termination requested.")

    def _to_data_list(
        self, x: torch.Tensor, batch: torch.Tensor, y: Optional[torch.Tensor]
    ) -> List[Data]:
        """
        Converts model outputs back to a list of Data elements. Used for graph data.

        Args:
            x (:class:`torch.Tensor`): tensor holding embedding information of different
                nodes/graphs embeddings
            batch (:class:`torch.Tensor`): the usual PyG batch tensor.
                Used to split node/graph embeddings graph-wise.
            y (:class:`torch.Tensor`): target labels, used to determine
                whether the task is graph prediction or node prediction.
                Can be ``None``.

        Returns:
            a list of PyG Data objects (with only ``x`` and ``y`` attributes)
        """
        data_list = []

        _, counts = torch.unique_consecutive(batch, return_counts=True)
        cumulative = torch.cumsum(counts, dim=0)

        if y is not None:
            is_graph_prediction = y.shape[0] == len(cumulative)
            y = y.unsqueeze(1) if y.dim() == 1 else y

            tmp_y = (
                y[0].unsqueeze(0)
                if is_graph_prediction
                else y[: cumulative[0]]
            )
            data_list.append((Data(x=x[: cumulative[0]], y=tmp_y), tmp_y))

            for i in range(1, len(cumulative)):
                tmp_y = (
                    y[i].unsqueeze(0)
                    if is_graph_prediction
                    else y[cumulative[i - 1] : cumulative[i]]
                )
                g = Data(x=x[cumulative[i - 1] : cumulative[i]], y=tmp_y)
                data_list.append((g, tmp_y))
        else:
            data_list.append(Data(x=x[: cumulative[0]]))
            for i in range(1, len(cumulative)):
                g = Data(x=x[cumulative[i - 1] : cumulative[i]])
                data_list.append((g, None))

        return data_list

    def _to_list(
        self,
        data_list: List[Data],
        embeddings: Union[Tuple[torch.Tensor], torch.Tensor],
        batch: torch.Tensor,
        y: Optional[torch.Tensor],
    ) -> List[Data]:
        """
        Extends the ``data_list`` list of PyG Data objects with new samples.

        Args:
            data_list: a list of PyG Data objects (with only ``x`` and ``y``
                attributes)
            embeddings (:class:`torch.Tensor`): tensor holding information
                of different nodes/graphs embeddings
            batch (:class:`torch.Tensor`): the usual PyG batch tensor.
                Used to split node/graph embeddings graph-wise.
            y (:class:`torch.Tensor`): target labels, used to determine
                whether the task is graph prediction or node prediction.
                Can be ``None``.

        Returns:
            a list of PyG Data objects (with only ``x`` and ``y`` attributes)
        """
        # Crucial: Detach the embeddings to free the computation graph!!
        if isinstance(embeddings, tuple):
            embeddings = tuple(
                [
                    e.detach().cpu() if e is not None else None
                    for e in embeddings
                ]
            )
        elif isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.detach().cpu()
        else:
            raise NotImplementedError(
                "Embeddings not understood, should be torch.Tensor or "
                "Tuple of torch.Tensor"
            )

        # Convert embeddings back to a list of torch_geometric Data objects
        # Needs also y information to (possibly) use them as a tensor dataset
        # CRUCIAL: remember, the loader could have shuffled the data, so we
        # need to carry y around as well
        if data_list is None:
            data_list = []
        data_list.extend(self._to_data_list(embeddings, batch, y))
        return data_list

    def set_device(self):
        """
        Moves the model and the loss metric to the proper device.
        """
        self.model.to(self.device)
        self.loss_fun.to(self.device)

    def set_training_mode(self):
        """
        Sets the model and the internal state in ``TRAINING`` mode
        """
        self.model.train()
        self.training = True

    def set_eval_mode(self):
        """
        Sets the model and the internal state in ``EVALUATION`` mode
        """
        self.model.eval()
        self.training = False

    def _loop_helper(self):
        """
        Helper function that loops over the data.
        """
        # Move data to device
        data = self.state.batch_input
        x = data[0].to(self.device)
        targets = data[1].to(self.device)

        # Helpful when you need to access again the input batch, e.g for some
        # continual learning strategy
        self.state.update(batch_input=x)
        self.state.update(batch_targets=targets)

        # Reset batch_loss and batch_score states
        self.state.update(batch_loss=None)
        self.state.update(batch_score=None)

        if self.training:
            self._dispatch(EventHandler.ON_TRAINING_BATCH_START, self.state)
        else:
            self._dispatch(EventHandler.ON_EVAL_BATCH_START, self.state)

        self._dispatch(EventHandler.ON_FORWARD, self.state)

        # EngineCallback will store the outputs in state.batch_outputs
        output = self.state.batch_outputs

        if len(output) > 1 and self.state.return_embeddings:
            # Embeddings should be in position 2 of the output
            embeddings = output[1]

            epoch_data_list = self.state.epoch_data_list

            if isinstance(x, torch.Tensor):  # classical ML
                if epoch_data_list is None:
                    epoch_data_list = []
                epoch_data_list.extend(
                    [
                        (embeddings[i], targets[i])
                        for i in range(embeddings.shape[0])
                    ]
                )
            elif isinstance(x, torch_geometric.data.Batch):  # graph dataset
                # graphs need special handling
                batch_idx = x.batch
                epoch_data_list = self._to_list(
                    self.state.epoch_data_list,
                    embeddings,
                    batch_idx,
                    targets,
                )
            elif isinstance(x, torch_geometric.data.Data):  # single graph
                # graphs need special handling
                batch_idx = torch.zeros(x.x.shape[0])
                epoch_data_list = self._to_list(
                    self.state.epoch_data_list,
                    embeddings,
                    batch_idx,
                    targets,
                )
            else:
                raise NotImplementedError(
                    f"Input type not understood. Allowed data types are torch "
                    f"Tensors and torch_geometric Batches, got {type(x)}."
                )

            # I am extending the data list, not replacing! Hence the name
            # "epoch" data list
            self.state.update(epoch_data_list=epoch_data_list)

        self._dispatch(EventHandler.ON_COMPUTE_METRICS, self.state)

        if self.training:
            self._dispatch(EventHandler.ON_BACKWARD, self.state)
            self._dispatch(EventHandler.ON_TRAINING_BATCH_END, self.state)
        else:
            self._dispatch(EventHandler.ON_EVAL_BATCH_END, self.state)

    # loop over all data (i.e. computes an epoch)
    def _loop(
        self, loader: DataLoader, _notify_progress: Callable[[str, dict], None]
    ):
        """
        Main method that computes a pass over the dataset using the data
        loader provided.

        Args:
            loader (:class:`torch_geometric.loader.DataLoader`):
                the loader to be used
        """
        # Reset epoch state (DO NOT REMOVE)
        self.state.update(epoch_data_list=None)
        self.state.update(epoch_loss=None)
        self.state.update(epoch_score=None)
        self.state.update(loader_iterable=iter(loader))

        # Loop over data
        total_batches = len(loader)
        self._total_batches = total_batches
        batch_progress_time = time.time()
        for id_batch in range(total_batches):
            self._check_termination()
            self.state.update(id_batch=id_batch)
            # EngineCallback will store fetched data in state.batch_input
            self._dispatch(EventHandler.ON_FETCH_DATA, self.state)

            self._loop_helper()

            batch_progress_time_delta = time.time() - batch_progress_time

            if (
                batch_progress_time_delta >= TIME_DELTA
                or self.cumulative_batch_unsent_time > CUMULATIVE_UNSENT_DELTA
            ):
                # Batch has completed
                _notify_progress(
                    BATCH_PROGRESS,
                    {
                        EPOCH: self.state.epoch + 1,
                        TOTAL_EPOCHS: self.state.total_epochs,
                        BATCH: id_batch + 1,
                        TOTAL_BATCHES: total_batches,
                        "message": f"{deepcopy(self.state.set.capitalize())} Progress",
                    },
                )
                # reset
                self.cumulative_batch_unsent_time = 0.0
            else:
                self.cumulative_batch_unsent_time += batch_progress_time_delta

            batch_progress_time = time.time()

    def _train(self, loader, _notify_progress: Callable[[str, dict], None]):
        """
        Implements a loop over the data in training mode
        """
        self.set_training_mode()

        self._dispatch(EventHandler.ON_TRAINING_EPOCH_START, self.state)

        self._loop(loader, _notify_progress)

        self._dispatch(EventHandler.ON_TRAINING_EPOCH_END, self.state)

        if self.state.epoch_loss is None:
            raise RuntimeError(
                "TrainingEngine epoch_loss was not computed. "
                "Ensure the loss Metric callback is registered and the loader yields at least one batch."
            )
        loss, score = self.state.epoch_loss, self.state.epoch_score
        return loss, score, None

    def infer(
        self,
        loader: DataLoader,
        set: str,
        _notify_progress: Callable[[str, dict], None],
    ) -> Tuple[dict, dict, List[Union[torch.Tensor, Data]]]:
        """
        Performs an evaluation step on the data.

        Args:
            loader (:class:`torch_geometric.loader.DataLoader`):
                the loader to be used
            set (str): the type of dataset being used, can be ``TRAINING``,
                ``VALIDATION`` or ``TEST`` (as defined in ``mlwiz.static``)

        Returns:
             a tuple (loss dict, score dict, list of
             :class:`torch_geometric.data.Data` objects with ``x`` and
             ``y`` attributes only).
             The data list can be used, for instance, in
             semi-supervised experiments or in incremental architectures
        """
        self.set_eval_mode()  # inference runs with dropout/bn disabled, no gradients
        self.state.update(
            set=set
        )  # track which split (TRAINING/VALIDATION/TEST) is being evaluated

        self._dispatch(
            EventHandler.ON_EVAL_EPOCH_START, self.state
        )  # let callbacks reset epoch accumulators

        with torch.no_grad():
            self._loop(loader, _notify_progress)  # state has been updated

        self._dispatch(
            EventHandler.ON_EVAL_EPOCH_END, self.state
        )  # let callbacks finalize epoch metrics

        if self.state.epoch_loss is None:
            raise RuntimeError(
                "TrainingEngine epoch_loss was not computed during inference. "
                "Ensure the loss Metric callback is registered and the loader yields at least one batch."
            )
        loss, score, data_list = (
            self.state.epoch_loss,
            self.state.epoch_score,
            self.state.epoch_data_list,  # per-sample (embedding, target) tuples when return_embeddings=True
        )

        # Add the main loss we want to return as a special key
        main_loss_name = self.loss_fun.get_main_metric_name()
        loss[MAIN_LOSS] = loss[
            main_loss_name
        ]  # normalized key for downstream evaluation

        # Add the main score we want to return as a special key
        # Needed by the experimental evaluation framework
        main_score_name = self.score_fun.get_main_metric_name()
        score[MAIN_SCORE] = score[
            main_score_name
        ]  # normalized key for downstream evaluation

        # If samples been shuffled by the data loader, i.e., in the
        # last inference phase of the engine, require a sampler-provided
        # permutation to restore the embedding/data list in the original
        # (unshuffled) order.
        if data_list is not None and loader.sampler is not None:
            # if SequentialSampler then shuffle was false
            if not isinstance(loader.sampler, SequentialSampler):
                permutation = getattr(
                    loader.sampler, "permutation", None
                )  # dataset indices in the order sampled
                if permutation is None:
                    raise ValueError(
                        "TrainingEngine requires the DataLoader sampler to expose a non-None "
                        "`permutation` so embeddings can be returned in the original sample order."
                    )

                if isinstance(permutation, torch.Tensor):
                    permutation = permutation.tolist()
                else:
                    permutation = list(permutation)

                data_list = reorder(
                    data_list, permutation
                )  # back to original dataset-index order

        return loss, score, data_list

    def train(
        self,
        train_loader: DataLoader,
        validation_loader: DataLoader = None,
        test_loader: DataLoader = None,
        max_epochs: int = 100,
        zero_epoch: bool = False,
        logger: Logger = None,
        training_timeout_seconds: int = -1,
        progress_callback: Callable[[dict], None] = None,
        should_terminate: Optional[Callable[[], bool]] = None,
    ) -> Tuple[
        dict,
        dict,
        List[Union[torch.Tensor, Data]],
        dict,
        dict,
        List[Union[torch.Tensor, Data]],
        dict,
        dict,
        List[Union[torch.Tensor, Data]],
    ]:
        """
        Trains the model and regularly evaluates on validation and test data
        (if given). May perform early stopping and checkpointing.

        Args:
            train_loader (:class:`torch_geometric.loader.DataLoader`):
                the DataLoader associated with training data
            validation_loader (:class:`torch_geometric.loader.DataLoader`):
                the DataLoader associated with validation data, if any
            test_loader (:class:`torch_geometric.loader.DataLoader`):
                the DataLoader associated with test data, if any
            max_epochs (int): maximum number of training epochs.
                Default is ``100``
            zero_epoch: if ``True``, starts again from epoch 0 and resets
                optimizer and scheduler states. Default is ``False``
            logger: the logger
            progress_callback: optional callable that receives dictionaries
                with progress information for external consumers
            should_terminate: optional callable returning ``True`` when a
                graceful termination has been requested

        Returns:
             a tuple (train_loss, train_score, train_embeddings,
             validation_loss, validation_score, validation_embeddings,
             test_loss, test_score, test_embeddings)
        """
        self.logger = logger
        self._should_terminate = should_terminate

        def _notify_progress(event_type: str, payload: dict):
            """
            Sends progress updates to an external callback, if provided.
            """
            if progress_callback is None:
                return

            data = {"type": event_type}
            data.update(payload)

            try:
                progress_callback(data)
            except Exception:
                # Do not break training if the progress callback fails
                pass

        final_epoch = None

        try:
            self._check_termination()
            # Initialize variables
            val_loss, val_score, val_embeddings_tuple = None, None, None
            test_loss, test_score, test_embeddings_tuple = None, None, None

            self.set_device()

            # Restore training from last checkpoint if possible!
            ckpt_filename = Path(self.exp_path, LAST_CHECKPOINT_FILENAME)
            best_ckpt_filename = Path(self.exp_path, BEST_CHECKPOINT_FILENAME)
            is_early_stopper_ckpt = (
                self.early_stopper.checkpoint
                if self.early_stopper is not None
                else False
            )
            # If one changes the options in the config file, the existence of
            # a checkpoint is not enough to
            # decide whether to resume training or not!
            if os.path.exists(ckpt_filename) and (
                self.store_last_checkpoint or is_early_stopper_ckpt
            ):
                self._restore_checkpoint_and_best_results(
                    ckpt_filename, best_ckpt_filename, zero_epoch
                )
                log(
                    f"START AGAIN FROM EPOCH {self.state.initial_epoch}",
                    logger,
                )

            self._dispatch(EventHandler.ON_FIT_START, self.state)

            # In case we already have a trained model
            epoch = self.state.initial_epoch

            self.state.update(TOTAL_EPOCHS=max_epochs)
            _notify_progress(START_CONFIG, {TOTAL_EPOCHS: max_epochs})

            last_run_elapsed_time = self.state.current_elapsed_time

            epoch_progress_time = time.time()
            # Loop over the entire dataset dataset
            for epoch in range(self.state.initial_epoch, max_epochs):
                self._check_termination()
                if training_timeout_seconds > 0:
                    # update the current time including the time of the last run
                    self.state.update(
                        current_elapsed_time=self.profiler.total_elapsed_time.seconds
                        + last_run_elapsed_time
                    )

                    if (
                        self.state.current_elapsed_time
                        >= training_timeout_seconds
                    ):
                        msg = f"Skipping training of new epoch {epoch + 1} because time limit of {training_timeout_seconds} has been reached. Current time elapsed: {self.state.current_elapsed_time}"
                        log(msg, logger)
                        self.state.update(stop_training=True)
                        break

                self.state.update(epoch=epoch)
                self.state.update(return_embeddings=False)

                self._dispatch(EventHandler.ON_EPOCH_START, self.state)

                self.state.update(set=TRAINING)
                train_loss, train_score, _ = self._train(
                    train_loader, _notify_progress
                )

                # Update state with epoch results
                epoch_results = {LOSSES: {}, SCORES: {}}

                if (
                    (epoch + 1) >= self.evaluate_every
                    and (epoch + 1) % self.evaluate_every == 0
                ) or epoch == 0:
                    if self.eval_training:
                        # Compute training output (necessary because on_backward
                        # has been called)
                        train_loss, train_score, _ = self.infer(
                            train_loader, TRAINING, _notify_progress
                        )
                    else:
                        # Add the main loss we want to return as a special key
                        main_loss_name = self.loss_fun.get_main_metric_name()
                        train_loss[MAIN_LOSS] = train_loss[main_loss_name]

                        # Add the main score we want to return as a special key
                        # Needed by the experimental evaluation framework
                        main_score_name = self.score_fun.get_main_metric_name()
                        train_score[MAIN_SCORE] = train_score[main_score_name]

                    # Compute validation output
                    if validation_loader is not None:
                        val_loss, val_score, _ = self.infer(
                            validation_loader, VALIDATION, _notify_progress
                        )

                    # Compute test output for visualization purposes only (e.g.
                    # to debug an incorrect data split for link prediction)
                    if test_loader is not None and self.eval_test_every_epoch:
                        test_loss, test_score, _ = self.infer(
                            test_loader, TEST, _notify_progress
                        )

                    epoch_results[LOSSES].update(
                        {f"{TRAINING}_{k}": v for k, v in train_loss.items()}
                    )
                    epoch_results[SCORES].update(
                        {f"{TRAINING}_{k}": v for k, v in train_score.items()}
                    )

                    if validation_loader is not None:
                        epoch_results[LOSSES].update(
                            {
                                f"{VALIDATION}_{k}": v
                                for k, v in val_loss.items()
                            }
                        )
                        epoch_results[SCORES].update(
                            {
                                f"{VALIDATION}_{k}": v
                                for k, v in val_score.items()
                            }
                        )
                        val_msg_str = (
                            f", VL loss: {val_loss} VL score: {val_score}"
                        )
                    else:
                        val_msg_str = ""

                    if test_loader is not None and self.eval_test_every_epoch:
                        epoch_results[LOSSES].update(
                            {f"{TEST}_{k}": v for k, v in test_loss.items()}
                        )
                        epoch_results[SCORES].update(
                            {f"{TEST}_{k}": v for k, v in test_score.items()}
                        )
                        test_msg_str = (
                            f", TE loss: {test_loss} TE score: {test_score}"
                        )
                    else:
                        test_msg_str = ""

                    # Message for Progress Manager
                    tr_loss = fmt(train_loss[MAIN_LOSS].item())
                    tr_score = fmt(train_score[MAIN_SCORE].item())
                    if validation_loader is not None:
                        val_loss = fmt(val_loss[MAIN_LOSS].item())
                        val_score = fmt(val_score[MAIN_SCORE].item())
                    else:
                        val_loss = "N/A"
                        val_score = "N/A"
                    if test_loader is not None and self.eval_test_every_epoch:
                        test_loss = fmt(test_loss[MAIN_LOSS].item())
                        test_score = fmt(test_score[MAIN_SCORE].item())
                    else:
                        test_loss = "N/A"
                        test_score = "N/A"
                    progress_manager_msg = (
                        f"Epoch: {epoch + 1} "
                        + f"TR/VL/TE loss: {tr_loss}/{val_loss}/{test_loss} "
                        + f"TR/VL/TE score: {tr_score}/{val_score}/{test_score}"
                    )

                    epoch_progress_time_delta = (
                        time.time() - epoch_progress_time
                    )
                    if (
                        epoch == 0
                        and epoch_progress_time_delta >= TIME_DELTA
                        or self.cumulative_epoch_unsent_time
                        > CUMULATIVE_UNSENT_DELTA
                    ):
                        _notify_progress(
                            RUN_PROGRESS,
                            {
                                EPOCH: epoch + 1,
                                TOTAL_EPOCHS: max_epochs,
                                BATCH: self._total_batches + 1,
                                TOTAL_BATCHES: self._total_batches + 1,
                                MODE: f"{deepcopy(self.state.set.capitalize())} Progress",
                                "message": deepcopy(progress_manager_msg),
                            },
                        )
                        # reset
                        self.cumulative_epoch_unsent_time = 0.0
                    else:
                        self.cumulative_epoch_unsent_time += (
                            epoch_progress_time_delta
                        )

                    epoch_progress_time = time.time()

                    # Log performances
                    logger_msg = (
                        f"Epoch: {epoch + 1}, TR loss: {train_loss} "
                        f"TR score: {train_score}" + val_msg_str + test_msg_str
                    )
                    log(logger_msg, logger)

                # Update state with the result of this epoch
                self.state.update(epoch_results=epoch_results)

                # We can apply early stopping here
                self._dispatch(EventHandler.ON_EPOCH_END, self.state)

                if self.state.stop_training:
                    log(f"Stopping at epoch {self.state.epoch}.", logger)
                    break

            # Needed to indicate that training has ended
            self.state.update(stop_training=True)

            # We reached the max # of epochs, get best scores from the early
            # stopper (if any) and restore best model
            if self.early_stopper is not None:
                ber = self.state.best_epoch_results
                # Restore the model according to the best validation score!
                self.model.load_state_dict(ber[MODEL_STATE])
            else:
                self.state.update(best_epoch_results={BEST_EPOCH: epoch})
                ber = self.state.best_epoch_results

            self.state.update(return_embeddings=True)

            final_epoch = self.state.epoch

            # Compute training output
            train_loss, train_score, train_embeddings_tuple = self.infer(
                train_loader, TRAINING, _notify_progress
            )
            # ber[f'{TRAINING}_loss'] = train_loss
            ber[f"{TRAINING}{EMB_TUPLE_SUBSTR}"] = train_embeddings_tuple
            ber.update({f"{TRAINING}_{k}": v for k, v in train_loss.items()})
            ber.update({f"{TRAINING}_{k}": v for k, v in train_score.items()})

            # Compute validation output
            if validation_loader is not None:
                val_loss, val_score, val_embeddings_tuple = self.infer(
                    validation_loader, VALIDATION, _notify_progress
                )
                # ber[f'{VALIDATION}_loss'] = val_loss
                ber[f"{VALIDATION}{EMB_TUPLE_SUBSTR}"] = val_embeddings_tuple
                ber.update({f"{TRAINING}_{k}": v for k, v in val_loss.items()})
                ber.update(
                    {f"{TRAINING}_{k}": v for k, v in val_score.items()}
                )

            # Compute test output
            if test_loader is not None:
                test_loss, test_score, test_embeddings_tuple = self.infer(
                    test_loader, TEST, _notify_progress
                )
                # ber[f'{TEST}_loss'] = test_loss
                ber[f"{TEST}{EMB_TUPLE_SUBSTR}"] = test_embeddings_tuple
                ber.update({f"{TEST}_{k}": v for k, v in test_loss.items()})
                ber.update({f"{TEST}_{k}": v for k, v in test_score.items()})

            self._dispatch(EventHandler.ON_FIT_END, self.state)

            self.state.update(return_embeddings=False)

            log(
                f"Chosen is Epoch {ber[BEST_EPOCH] + 1} "
                f"TR loss: {train_loss} TR score: {train_score}, "
                f"VL loss: {val_loss} VL score: {val_score} "
                f"TE loss: {test_loss} TE score: {test_score}",
                logger,
            )

            self.state.update(set=None)

        except TerminationRequested:
            raise
        except (KeyboardInterrupt, RuntimeError, FileNotFoundError) as e:
            report = self.profiler.report()
            log(str(e), logger)
            log(report, logger)
            raise e
            exit(0)

        # Log profile results
        report = self.profiler.report()
        log(report, logger)

        _notify_progress(
            RUN_COMPLETED,
            {
                EPOCH: (
                    (final_epoch + 1)
                    if final_epoch is not None
                    else max_epochs
                ),
                TOTAL_EPOCHS: max_epochs,
            },
        )

        return (
            train_loss,
            train_score,
            train_embeddings_tuple,
            val_loss,
            val_score,
            val_embeddings_tuple,
            test_loss,
            test_score,
            test_embeddings_tuple,
        )

    def _restore_checkpoint_and_best_results(
        self, ckpt_filename, best_ckpt_filename, zero_epoch
    ):
        """
        Restores the (best or last) checkpoint from a given file, and loads
        the best results so far into the state if any.
        """
        # When changing exp config from cuda to cpu, cuda will not be available
        # to pytorch (due to Ray management of resources). Hence, we need to
        # specify explicitly the map location as cpu. The other way around
        # (cpu to cuda) poses no problem since GPUs are visible.
        ckpt_dict = torch.load(
            ckpt_filename,
            map_location="cpu" if self.device == "cpu" else None,
            weights_only=True,
        )

        self.state.update(
            current_elapsed_time=ckpt_dict[LAST_RUN_ELAPSED_TIME]
        )

        self.state.update(
            initial_epoch=int(ckpt_dict[EPOCH]) + 1 if not zero_epoch else 0
        )
        self.state.update(stop_training=ckpt_dict[STOP_TRAINING])

        model_state = ckpt_dict[MODEL_STATE]

        # Needed only when moving from cpu to cuda (due to changes in config
        # file). Move all parameters to cuda.
        for param in model_state.keys():
            model_state[param] = model_state[param].to(self.device)

        self.model.load_state_dict(model_state)

        if not zero_epoch:
            optimizer_state = ckpt_dict[OPTIMIZER_STATE]
            self.state.update(optimizer_state=optimizer_state)

        if os.path.exists(best_ckpt_filename):
            best_ckpt_dict = torch.load(
                best_ckpt_filename,
                map_location="cpu" if self.device == "cpu" else None,
                weights_only=True,
            )
            self.state.update(best_epoch_results=best_ckpt_dict)

        if self.scheduler is not None and not zero_epoch:
            scheduler_state = ckpt_dict[SCHEDULER_STATE]
            if scheduler_state is None:
                raise RuntimeError(
                    "Checkpoint is missing scheduler state, but a scheduler is configured."
                )
            self.state.update(scheduler_state=scheduler_state)


class DataStreamTrainingEngine(TrainingEngine):
    """
    Class that handles a stream of samples that could end at any moment.
    """

    # loop over all data (i.e. computes an epoch)
    def _loop(
        self, loader: DataLoader, _notify_progress: Callable[[str, dict], None]
    ):
        """
        Compared to superclass version, handles the issue of a stream of data
        that could end at any moment. This is done using an additional
        boolean state variable.
        """
        # Reset epoch state (DO NOT REMOVE)
        self.state.update(epoch_data_list=None)
        self.state.update(epoch_loss=None)
        self.state.update(epoch_score=None)
        self.state.update(loader_iterable=iter(loader))

        # Loop over data
        id_batch = 0
        self.state.update(stop_fetching=False)
        batch_progress_time = time.time()

        while not self.state.stop_fetching:
            self.state.update(id_batch=id_batch)
            id_batch += 1

            # EngineCallback will store fetched data in state.batch_input
            self._dispatch(EventHandler.ON_FETCH_DATA, self.state)

            if self.state.stop_fetching:
                return

            self._loop_helper()

            batch_progress_time_delta = time.time() - batch_progress_time

            if (
                batch_progress_time_delta >= TIME_DELTA
                or self.cumulative_batch_unsent_time > CUMULATIVE_UNSENT_DELTA
            ):
                # Batch has completed
                _notify_progress(
                    BATCH_PROGRESS,
                    {
                        EPOCH: self.state.epoch + 1,
                        TOTAL_EPOCHS: self.state.total_epochs,
                        BATCH: id_batch + 1,
                        TOTAL_BATCHES: id_batch + 1,  # patch
                        "message": f"{deepcopy(self.state.set.capitalize())} Progress",
                    },
                )
                # reset
                self.cumulative_batch_unsent_time = 0.0
            else:
                self.cumulative_batch_unsent_time += batch_progress_time_delta

            batch_progress_time = time.time()
