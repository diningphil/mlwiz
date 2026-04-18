"""Optimizer callback wrapper for the training engine.

Instantiates a PyTorch optimizer from a dotted path and exposes lifecycle hooks.
"""

import torch

from mlwiz.util import s2c
from mlwiz.model.interface import ModelInterface
from mlwiz.training.event.handler import EventHandler
from mlwiz.training.util import clone_to_cpu


class Optimizer(EventHandler):
    """
    Optimizer is the main event handler for optimizers.
    Just pass a PyTorch optimizer together with its arguments in the
    configuration file.

    Args:
        model (:class:`~mlwiz.model.interface.ModelInterface`):
            the model that has to be trained
        optimizer_class_name (str): dotted path to the optimizer class to use
        accumulate_gradients (bool): if ``True``, accumulate mini-batch
            gradients to perform a batch gradient update without
            loading the entire batch in memory
        kwargs (dict): additional parameters for the specific optimizer
    """

    def __init__(
        self,
        model: ModelInterface,
        optimizer_class_name: str,
        accumulate_gradients: bool = False,
        **kwargs: dict,
    ):
        """
        Instantiate the underlying PyTorch optimizer.

        Args:
            model (ModelInterface): Model whose parameters will be optimized.
            optimizer_class_name (str): Dotted path to the optimizer class.
            accumulate_gradients (bool): If ``True``, gradients are accumulated
                across batches and an optimizer step is performed at the end of
                the epoch. If ``False``, step/zero_grad happen per batch.
            **kwargs: Additional keyword arguments forwarded to the optimizer
                constructor.

        Side effects:
            Stores the instantiated optimizer on ``self.optimizer``.
        """
        super().__init__()
        self.optimizer = s2c(optimizer_class_name)(
            model.parameters(), **kwargs
        )
        self._param_names = (
            {
                id(param): name
                for name, param in model.named_parameters()
            }
            if hasattr(model, "named_parameters")
            else {}
        )
        self.accumulate_gradients = accumulate_gradients

    def _collect_state_shape_mismatches(self):
        """
        Collects optimizer state tensors whose shapes differ from their
        corresponding model parameter shapes.
        """
        mismatches = []
        for group_idx, group in enumerate(self.optimizer.param_groups):
            for param_idx, param in enumerate(group["params"]):
                state = self.optimizer.state.get(param, {})
                if not state:
                    continue

                param_shape = tuple(param.shape)
                grad_shape = (
                    tuple(param.grad.shape)
                    if getattr(param, "grad", None) is not None
                    else None
                )
                param_name = self._param_names.get(
                    id(param), f"group[{group_idx}].param[{param_idx}]"
                )

                for state_key, state_value in state.items():
                    if not torch.is_tensor(state_value):
                        continue
                    # Optimizers can keep scalar counters (e.g. Adam's step).
                    if state_value.ndim == 0:
                        continue

                    state_shape = tuple(state_value.shape)
                    if state_shape != param_shape:
                        mismatches.append(
                            {
                                "group_idx": group_idx,
                                "param_idx": param_idx,
                                "param_name": param_name,
                                "param_shape": param_shape,
                                "grad_shape": grad_shape,
                                "state_key": state_key,
                                "state_shape": state_shape,
                            }
                        )

        return mismatches

    def _step_or_raise_with_state_mismatches(self):
        """
        Executes an optimizer step and, on RuntimeError, reports any optimizer
        state/parameter shape mismatches before re-raising.
        """
        try:
            self.optimizer.step()
        except RuntimeError as e:
            mismatches = self._collect_state_shape_mismatches()
            if not mismatches:
                raise

            lines = [
                f"{str(e)}",
                "Optimizer state shape mismatches (state tensor != parameter tensor):",
            ]
            for mismatch in mismatches:
                lines.append(
                    (
                        f"- {mismatch['param_name']} "
                        f"(group={mismatch['group_idx']}, index={mismatch['param_idx']}): "
                        f"param={mismatch['param_shape']}, "
                        f"grad={mismatch['grad_shape']}, "
                        f"state[{mismatch['state_key']}]={mismatch['state_shape']}"
                    )
                )

            raise RuntimeError("\n".join(lines)) from e

    def load_state_dict(self, state_dict):
        """
        Loads the state_dict of the optimizer from a checkpoint

        Args:
            state (:class:`~training.event.state.State`):
                object holding training information
        """
        self.optimizer.load_state_dict(state_dict)

    def on_fit_start(self, state):
        """
        If a checkpoint is present, load the state of the optimizer

        Args:
            state (:class:`~training.event.state.State`):
                object holding training information
        """
        if state.optimizer_state is not None:
            self.optimizer.load_state_dict(state.optimizer_state)

    def on_training_epoch_start(self, state):
        """
        At the start of epoch, and if the gradient has been accumulated
        across the entire epoch, zeroes the gradient of the optimizer.

        Args:
            state (:class:`~training.event.state.State`):
                object holding training information
        """
        if self.accumulate_gradients:
            self.optimizer.zero_grad()

    def on_training_batch_start(self, state):
        """
        At the start of a batch, if batch updates are in order,
        zeroes the gradient of the optimizer

        Args:
            state (:class:`~training.event.state.State`):
                object holding training information
        """
        if not self.accumulate_gradients:
            self.optimizer.zero_grad()

    def on_training_batch_end(self, state):
        """
        At the end of a batch, if batch updates are in order,
        performs a weight update

        Args:
            state (:class:`~training.event.state.State`):
                object holding training information
        """
        if not self.accumulate_gradients:
            grad_scaler = getattr(state, "grad_scaler", None)
            if grad_scaler is not None:
                grad_scaler.step(self.optimizer)
                grad_scaler.update()
            else:
                self._step_or_raise_with_state_mismatches()

    def on_training_epoch_end(self, state):
        """
        At the end of a batch, and if the gradient has been
        accumulated across the entire epoch, performs a weight update

        Args:
            state (:class:`~training.event.state.State`):
                object holding training information
        """
        if self.accumulate_gradients:
            grad_scaler = getattr(state, "grad_scaler", None)
            if grad_scaler is not None:
                grad_scaler.step(self.optimizer)
                grad_scaler.update()
            else:
                self._step_or_raise_with_state_mismatches()

    def on_epoch_end(self, state):
        """
        Updates the state of the optimizer into the state
        at the end of the epoch

        Args:
            state (:class:`~training.event.state.State`):
                object holding training information
        """
        grad_scaler = getattr(state, "grad_scaler", None)
        scaler_state = (
            clone_to_cpu(grad_scaler.state_dict())
            if grad_scaler is not None
            else None
        )
        state.update(
            # Keep optimizer/scaler checkpoints on CPU to reduce GPU memory spikes.
            optimizer_state=clone_to_cpu(self.optimizer.state_dict()),
            scaler_state=scaler_state,
        )
