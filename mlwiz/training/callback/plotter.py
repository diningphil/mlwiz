"""Metric-history callback for MLWiz Dashboard.

The :class:`~mlwiz.training.callback.plotter.Plotter` persists epoch losses
and scores, plus optional training-step histories, to ``metrics_data.torch``
for live and post-run inspection.
"""

from pathlib import Path
from typing import Optional

import torch

from mlwiz.static import LOSSES, SCORES, TRAINING
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
        store_every_N_epochs (int, optional): epoch metric flush interval.
            Defaults to ``1``.
        store_every_N_steps (int, optional): training-step sampling and flush
            interval. Defaults to ``None`` (disabled).
        kwargs (dict): additional arguments that may depend on the plotter
    """

    def __init__(
        self,
        exp_path: str,
        store_on_disk: bool = True,
        store_every_N_epochs: Optional[int] = 1,
        store_every_N_steps: Optional[int] = None,
        **kwargs: dict,
    ):
        r"""
        Initialize dashboard metric persistence.

        Args:
            exp_path (str): Experiment folder where metrics are stored.
            store_on_disk (bool): If ``True``, persist raw metric histories to
                ``metrics_data.torch`` in ``exp_path``. Defaults to ``True``.
            store_every_N_epochs (int, optional): Flush metrics to disk every
                ``N`` epochs. Defaults to ``1``. Set to ``None`` to flush only
                when training terminates.
            store_every_N_steps (int, optional): Record and flush training
                batch metrics every ``N`` optimizer steps. Defaults to
                ``None``, which disables step histories.
            **kwargs: Unused extra arguments (kept for configuration
                compatibility).

        Side effects:
            Loads previously stored metrics when ``metrics_data.torch`` exists.
        """
        super().__init__()
        self.exp_path = exp_path
        self.store_on_disk = store_on_disk
        self.store_every_N_epochs = store_every_N_epochs
        self.store_every_N_steps = store_every_N_steps
        self.main_process = is_main_process()
        self._validate_interval(
            "store_every_N_epochs", self.store_every_N_epochs
        )
        self._validate_interval("store_every_N_steps", self.store_every_N_steps)

        if not self.main_process:
            self.store_on_disk = False

        self.stored_metrics = {"losses": {}, "scores": {}}
        self.stored_metrics_path = Path(self.exp_path, "metrics_data.torch")
        if self.stored_metrics_path.exists():
            self.stored_metrics = torch.load(
                self.stored_metrics_path, weights_only=True
            )
        step_metrics = self.stored_metrics.get("step", {})
        self._training_step = (
            int(step_metrics.get("last_step", 0))
            if isinstance(step_metrics, dict)
            else 0
        )

    @staticmethod
    def _validate_interval(name: str, value: Optional[int]):
        """Validate one optional positive persistence interval."""
        if value is not None and (
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
        ):
            raise ValueError(f"`{name}` must be a positive integer or None.")

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

    def on_epoch_start(self, state: State):
        """Reset the opt-in batch-score request before this epoch starts."""
        state.update(log_step_metrics=False)

    def on_training_batch_start(self, state: State):
        """Request a batch score only when the next step will be sampled."""
        state.update(
            log_step_metrics=(
                self.main_process
                and self.store_on_disk
                and self.store_every_N_steps is not None
                and (self._training_step + 1) % self.store_every_N_steps == 0
            )
        )

    def on_eval_batch_start(self, state: State):
        """Keep validation/test batches out of training-step histories."""
        state.update(log_step_metrics=False)

    def on_training_batch_end(self, state: State):
        """Append and persist metrics at the configured global step interval."""
        if (
            not self.main_process
            or not self.store_on_disk
            or self.store_every_N_steps is None
        ):
            return

        self._training_step += 1
        step_metrics = self.stored_metrics.setdefault(
            "step", {"steps": [], "losses": {}, "scores": {}}
        )
        step_metrics["last_step"] = self._training_step
        if self._training_step % self.store_every_N_steps != 0:
            return

        step_metrics.setdefault("steps", []).append(self._training_step)
        for metric_type, values in (
            ("losses", state.batch_loss),
            ("scores", state.batch_score),
        ):
            if not isinstance(values, dict):
                continue
            histories = step_metrics.setdefault(metric_type, {})
            for key, value in values.items():
                name = f"{TRAINING}_{key}"
                histories.setdefault(name, []).append(value.item())
        self._store_metrics()

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


class WidthPlotter(Plotter):
    """Example plotter that records the output width of every model layer.

    The resulting ``model_widths`` entry is an ``epochs × layers`` matrix.
    It is intentionally simple: fixed-width models such as :class:`MLP`
    produce flat trends, while models that replace or resize layers during
    training produce changing curves.
    """

    @staticmethod
    def _model_widths(model: torch.nn.Module) -> list[int]:
        """Return output dimensions for common learnable layer types."""
        model = getattr(model, "module", model)
        widths = []
        for layer in model.modules():
            width = getattr(layer, "out_features", None)
            if width is None:
                width = getattr(layer, "out_channels", None)
            if isinstance(width, int):
                widths.append(width)
        return widths

    def on_epoch_end(self, state: State):
        """Store regular metrics followed by one width vector for this epoch."""
        super().on_epoch_end(state)
        if not self.main_process or not self.store_on_disk:
            return
        widths = self._model_widths(state.model)
        if not widths:
            return
        self.stored_metrics.setdefault("model_widths", []).append(widths)
        if self.store_every_N_epochs is not None and (
            (state.epoch + 1) % self.store_every_N_epochs == 0
        ):
            # ``Plotter.on_epoch_end`` flushes regular metrics first. Re-flush
            # so this epoch's width vector is included in the same artifact.
            self._store_metrics()
