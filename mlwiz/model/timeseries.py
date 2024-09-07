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
        super().__init__(
            dim_input_features,
            dim_target,
            config,
        )
        assert type(dim_input_features) == int

        dim_embedding = config["dim_embedding"]

        self.rnn = torch.nn.GRU(
            input_size=dim_input_features,
            hidden_size=dim_embedding,
            num_layers=1,
            batch_first=True,
        )
        self.out_layer = Linear(dim_embedding, dim_target)

    def forward(
        self, data: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[object]]]:
        """
        Implements an MLP forward pass

        Args:
            data (torch.Tensor): a batched tensor

        Returns:
            a tuple (output, node_embedddings)
        """
        h, hidden_state = self.rnn(data)
        # get last time step
        h_last = h[:, -1, :]  # batch size x dim_embedding
        o = self.out_layer(h_last)
        return o, h
