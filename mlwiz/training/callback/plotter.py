"""TensorBoard logging callback for training runs.

The :class:`~mlwiz.training.callback.plotter.Plotter` writes per-epoch metrics and optional on-disk histories.
"""

from pathlib import Path
from typing import Optional

import torch
from torch.utils.tensorboard import SummaryWriter

from mlwiz.static import (
    LOSSES,
    SCORES,
    TENSORBOARD,
    TEST,
    TRAINING,
    VALIDATION,
)
from mlwiz.training.event.handler import EventHandler
from mlwiz.training.event.state import State
from mlwiz.training.distributed import is_main_process
from mlwiz.training.util import atomic_torch_save


class _NullSummaryWriter:
    """Drop-in writer used when TensorBoard logging is disabled."""

    def add_scalars(self, *args, **kwargs):
        """Ignore scalar logging calls."""

    def close(self):
        """No-op close for API compatibility."""


class Plotter(EventHandler):
    r"""
    Plotter is the main event handler for plotting at training time.

    Args:
        exp_path (str): path where to store the Tensorboard logs
        store_on_disk (bool): whether to store all metrics on disk.
            Defaults to False
        kwargs (dict): additional arguments that may depend on the plotter
    """

    def __init__(
        self,
        exp_path: str,
        store_on_disk: bool = False,
        store_every_N_epochs: Optional[int] = None,
        enable_tensorboard: bool = True,
        **kwargs: dict,
    ):
        r"""
        Initialize the plotter and tensorboard writer.

        Args:
            exp_path (str): Experiment folder where tensorboard logs are stored.
            store_on_disk (bool): If ``True``, persist raw metric histories to
                ``metrics_data.torch`` in ``exp_path`` in addition to
                tensorboard summaries.
            store_every_N_epochs (int, optional): If set, flushes metrics to
                disk every ``N`` epochs instead of every epoch. Metrics are
                always flushed on termination.
            enable_tensorboard (bool): If ``False``, skip TensorBoard event
                file creation and only keep optional ``metrics_data.torch``
                persistence.
            **kwargs: Unused extra arguments (kept for configuration
                compatibility).

        Side effects:
            When ``enable_tensorboard`` is ``True``, creates the tensorboard
            folder if missing and instantiates a
            :class:`torch.utils.tensorboard.SummaryWriter`. Always loads
            previously stored metrics if present.
        """
        super().__init__()
        self.exp_path = exp_path
        self.store_on_disk = store_on_disk
        self.store_every_N_epochs = store_every_N_epochs
        self.enable_tensorboard = enable_tensorboard
        self.main_process = is_main_process()
        if self.store_every_N_epochs is not None and (
            not isinstance(self.store_every_N_epochs, int)
            or self.store_every_N_epochs <= 0
        ):
            raise ValueError(
                "`store_every_N_epochs` must be a positive integer."
            )

        if self.enable_tensorboard and self.main_process:
            tensorboard_dir = Path(self.exp_path, TENSORBOARD)
            tensorboard_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir=tensorboard_dir)
        else:
            # Keep the same writer API while preventing any TensorBoard files.
            self.writer = _NullSummaryWriter()

        if not self.main_process:
            self.store_on_disk = False

        self.stored_metrics = {"losses": {}, "scores": {}}
        self.stored_metrics_path = Path(self.exp_path, "metrics_data.torch")
        if self.stored_metrics_path.exists():
            self.stored_metrics = torch.load(
                self.stored_metrics_path, weights_only=True
            )

    def _update_stored_metrics(self, metric_type: str, key: str, value):
        """
        Append one scalar metric value to in-memory metric history.
        """
        if key not in self.stored_metrics[metric_type]:
            self.stored_metrics[metric_type][key] = [value.item()]
        else:
            self.stored_metrics[metric_type][key].append(value.item())

    def _store_metrics(self):
        """
        Persist in-memory metrics to disk.
        """
        try:
            atomic_torch_save(self.stored_metrics, self.stored_metrics_path)
        except RuntimeError as e:
            print(e)

    def on_epoch_end(self, state: State):
        """
        Writes Training, Validation and (if any) Test metrics to Tensorboard

        Args:
            state (:class:`~training.event.state.State`):
                object holding training information
        """
        if not self.main_process:
            return

        for k, v in state.epoch_results[LOSSES].items():
            loss_scalars = {}
            # Remove training/validation/test prefix (coupling with Engine)
            loss_name = " ".join(k.split("_")[1:])
            if TRAINING in k:
                loss_scalars[f"{TRAINING}"] = v
            elif VALIDATION in k:
                loss_scalars[f"{VALIDATION}"] = v
            elif TEST in k:
                loss_scalars[f"{TEST}"] = v

            self.writer.add_scalars(loss_name, loss_scalars, state.epoch)

            if self.store_on_disk:
                self._update_stored_metrics("losses", k, v)

        for k, v in state.epoch_results[SCORES].items():
            score_scalars = {}
            # Remove training/validation/test prefix (coupling with Engine)
            score_name = " ".join(k.split("_")[1:])
            if TRAINING in k:
                score_scalars[f"{TRAINING}"] = v
            elif VALIDATION in k:
                score_scalars[f"{VALIDATION}"] = v
            elif TEST in k:
                score_scalars[f"{TEST}"] = v

            self.writer.add_scalars(score_name, score_scalars, state.epoch)

            if self.store_on_disk:
                self._update_stored_metrics("scores", k, v)

        if (
            self.store_on_disk
            and self.store_every_N_epochs is not None
            and (state.epoch + 1) % self.store_every_N_epochs == 0
        ):
            self._store_metrics()

    def on_fit_end(self, state: State):
        """
        Frees resources by closing the Tensorboard writer

        Args:
            state (:class:`~training.event.state.State`):
                object holding training information
        """
        self.writer.close()

    def on_termination(self, state: State):
        """
        Persist metrics at termination and free TensorBoard resources.

        Args:
            state (:class:`~training.event.state.State`):
                object holding training information
        """
        if self.main_process and self.store_on_disk:
            self._store_metrics()
        self.writer.close()
