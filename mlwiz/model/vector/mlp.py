from typing import List, Optional, Tuple

import torch
from torch import nn
from torch_geometric.data import Batch
from torch_geometric.nn import global_add_pool

from mlwiz.model.interface import ModelInterface


class MLP(ModelInterface):
    """
    An MLP model used to test the library.
    """

    def __init__(
        self,
        dim_input_features,
        dim_target,
        config,
    ):
        super().__init__(
            dim_input_features,
            dim_target,
            config,
        )

        # TODO IMPLEMENT A REAL MLP
        dim_embedding = config["dim_embedding"]
        self.W = nn.Linear(dim_input_features, dim_target, bias=True)

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
        # TODO IMPLEMENT A REAL MLP
        hg = global_add_pool(x, batch)
        return self.W(hg), x
