"""Optimizer callback wrapper for the training engine.

Instantiates a PyTorch optimizer from a dotted path and exposes lifecycle hooks.
"""

from copy import deepcopy

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
        if hasattr(model, "named_parameters"):
            named_parameters = list(model.named_parameters())
            optimizer_parameters = named_parameters
            self._param_names = {
                id(param): name
                for name, param in named_parameters
            }
        else:
            optimizer_parameters = model.parameters()
            self._param_names = {}
        self.optimizer = s2c(optimizer_class_name)(
            optimizer_parameters, **kwargs
        )
        self.accumulate_gradients = accumulate_gradients

    @staticmethod
    def _clone_optimizer_state_value(value):
        """
        Clone optimizer state values to avoid in-place aliasing.
        """
        if torch.is_tensor(value):
            return value.clone()
        if isinstance(value, dict):
            return {
                k: Optimizer._clone_optimizer_state_value(v)
                for k, v in value.items()
            }
        if isinstance(value, list):
            return [Optimizer._clone_optimizer_state_value(v) for v in value]
        if isinstance(value, tuple):
            return tuple(
                Optimizer._clone_optimizer_state_value(v) for v in value
            )
        return deepcopy(value)

    def _adapt_state_dict_by_param_names(self, loaded_state_dict: dict):
        """
        Remap optimizer state to current parameters by ``param_names``.

        If either optimizer lacks ``param_names`` metadata, returns the input
        state dictionary unchanged so PyTorch keeps the order-based
        loading behavior.
        """
        current_state_dict = self.optimizer.state_dict()
        loaded_groups = loaded_state_dict.get("param_groups", [])
        current_groups = current_state_dict.get("param_groups", [])

        # Fall back to PyTorch's default order-based loading if named metadata
        # is missing or incompatible.
        if not loaded_groups or len(loaded_groups) != len(current_groups):
            return loaded_state_dict
        if not all("param_names" in g for g in loaded_groups):
            return loaded_state_dict
        if not all("param_names" in g for g in current_groups):
            return loaded_state_dict

        # Start from the current optimizer layout so parameter IDs always match
        # the live optimizer instance.
        adapted_state_dict = deepcopy(current_state_dict)
        loaded_state = loaded_state_dict.get("state", {})

        for group_idx, (loaded_group, current_group) in enumerate(
            zip(loaded_groups, current_groups)
        ):
            loaded_names = loaded_group.get("param_names", [])
            loaded_ids = loaded_group.get("params", [])
            current_names = current_group.get("param_names", [])
            current_ids = current_group.get("params", [])

            if len(loaded_names) != len(loaded_ids):
                return loaded_state_dict
            if len(current_names) != len(current_ids):
                return loaded_state_dict

            # Copy non-parameter group settings (e.g. lr/betas/weight_decay)
            # from the checkpointed optimizer.
            for key, value in loaded_group.items():
                if key not in ("params", "param_names"):
                    adapted_state_dict["param_groups"][group_idx][key] = value

            # Reattach state by parameter name so reordering does not break the
            # mapping between model parameters and optimizer slots.
            loaded_id_by_name = dict(zip(loaded_names, loaded_ids))
            for current_id, current_name in zip(current_ids, current_names):
                loaded_id = loaded_id_by_name.get(current_name)
                if loaded_id is None or loaded_id not in loaded_state:
                    continue
                adapted_state_dict["state"][
                    current_id
                ] = self._clone_optimizer_state_value(loaded_state[loaded_id])

        return adapted_state_dict

    def load_state_dict(self, state_dict):
        """
        Loads the state_dict of the optimizer from a checkpoint.
        MLWiz stores and loads named metadata for optimizer state
        to be sure parameters are matched correctly to their optimizer state.

        Args:
            state (:class:`~training.event.state.State`):
                object holding training information
        """
        adapted_state_dict = self._adapt_state_dict_by_param_names(state_dict)
        self.optimizer.load_state_dict(adapted_state_dict)

    def on_fit_start(self, state):
        """
        If a checkpoint is present, load the state of the optimizer

        Args:
            state (:class:`~training.event.state.State`):
                object holding training information
        """
        if state.optimizer_state is not None:
            self.load_state_dict(state.optimizer_state)

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
                self.optimizer.step()

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
                self.optimizer.step()

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
