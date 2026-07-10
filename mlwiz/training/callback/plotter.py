"""Metric-history callback for MLWiz Dashboard.

The :class:`~mlwiz.training.callback.plotter.Plotter` persists per-epoch losses
and scores to ``metrics_data.torch`` for live and post-run inspection.
"""

from pathlib import Path
from typing import Optional

import torch

from mlwiz.static import LOSSES, SCORES
from mlwiz.training.event.handler import EventHandler
from mlwiz.training.event.state import State
from mlwiz.training.distributed import is_main_process
from mlwiz.training.util import atomic_torch_save


class Plotter(EventHandler):
    r"""
    Plotter is the main event handler for plotting at training time.

    Args:
        exp_path (str): path where to store ``metrics_data.torch``
        store_on_disk (bool): whether to store all metrics on disk.
            Defaults to ``True``
        kwargs (dict): additional arguments that may depend on the plotter
    """

    def __init__(
        self,
        exp_path: str,
        store_on_disk: bool = True,
        store_every_N_epochs: Optional[int] = None,
        **kwargs: dict,
    ):
        r"""
        Initialize dashboard metric persistence.

        Args:
            exp_path (str): Experiment folder where metrics are stored.
            store_on_disk (bool): If ``True``, persist raw metric histories to
                ``metrics_data.torch`` in ``exp_path``. Defaults to ``True``.
            store_every_N_epochs (int, optional): If set, flushes metrics to
                disk every ``N`` epochs instead of every epoch. Metrics are
                always flushed on termination.
            **kwargs: Unused extra arguments (kept for configuration
                compatibility).

        Side effects:
            Loads previously stored metrics when ``metrics_data.torch`` exists.
        """
        super().__init__()
        self.exp_path = exp_path
        self.store_on_disk = store_on_disk
        self.store_every_N_epochs = store_every_N_epochs
        self.main_process = is_main_process()
        if self.store_every_N_epochs is not None and (
            not isinstance(self.store_every_N_epochs, int)
            or self.store_every_N_epochs <= 0
        ):
            raise ValueError("`store_every_N_epochs` must be a positive integer.")

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
        Append the epoch's available loss and score metrics.

        Args:
            state (:class:`~training.event.state.State`):
                object holding training information
        """
        if not self.main_process or not self.store_on_disk:
            return

        for k, v in state.epoch_results[LOSSES].items():
            self._update_stored_metrics("losses", k, v)

        for k, v in state.epoch_results[SCORES].items():
            self._update_stored_metrics("scores", k, v)

        if (
            self.store_on_disk
            and self.store_every_N_epochs is not None
            and (state.epoch + 1) % self.store_every_N_epochs == 0
        ):
            self._store_metrics()

    def on_termination(self, state: State):
        """
        Persist metrics at termination.

        Args:
            state (:class:`~training.event.state.State`):
                object holding training information
        """
        if self.main_process and self.store_on_disk:
            self._store_metrics()
