from typing import List, Optional, Tuple, Union

import torch
from torch.nn import Linear

from mlwiz.model.interface import ModelInterface


class GRU(ModelInterface):
    """
    An RNN model used to test the library.
    """

    def __init__(
        self,
        dim_input_features: Union[int, Tuple[int]],
        dim_target: int,
        config: dict,
    ):
        """
        Initialize a GRU-based model for sequence/time-series inputs.

        Args:
            dim_input_features (Union[int, Tuple[int]]): Per-timestep input
                feature dimension. Must be an ``int`` for this model.
            dim_target (int): Output dimension (e.g., number of classes).
            config (dict): Model configuration. Expected keys:
                - ``dim_embedding`` (int): Hidden state size of the GRU.

        Raises:
            TypeError: If ``dim_input_features`` is not an ``int``.
            KeyError: If ``config`` does not contain ``dim_embedding``.

        Side effects:
            Initializes internal Torch modules (GRU + output projection).
        """
        super().__init__(
            dim_input_features,
            dim_target,
            config,
        )
        if not isinstance(dim_input_features, int):
            raise TypeError(
                "dim_input_features must be an int for time-series models, "
                f"got {dim_input_features!r} ({type(dim_input_features).__name__})."
            )

        dim_embedding = config["dim_embedding"]

        self.rnn = torch.nn.GRU(
            input_size=dim_input_features,
            hidden_size=dim_embedding,
            num_layers=1,
            batch_first=True,
        )
        self.out_layer = Linear(dim_embedding, dim_target)

    def forward(self, data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform a forward pass of the GRU model.

        Args:
            data (torch.Tensor): Batched sequence tensor of shape
                ``(batch, timesteps, dim_input_features)``.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Model outputs of shape ``(batch, dim_target)`` produced from
                  the last GRU timestep.
                - Sequence embeddings of shape ``(batch, timesteps, dim_embedding)``
                  (the per-timestep GRU outputs).

            Note:
                Some MLWiz models return an additional third element with
                auxiliary outputs. This model returns only ``(output, embeddings)``.
        """
        h, hidden_state = self.rnn(data)
        # get last time step
        h_last = h[:, -1, :]  # batch size x dim_embedding
        o = self.out_layer(h_last)
        return o, h
