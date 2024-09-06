from typing import List, Optional, Tuple, Union

import torch
from torch import nn
from torch_geometric.data import Batch, Data
from torch_geometric.nn import SAGEConv

# from torch_geometric_temporal import DCRNN

from mlwiz.model.interface import ModelInterface


class ToyDGN(ModelInterface):
    """
    A Deep Graph Network used to test the library
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

        num_layers = config["num_layers"]
        dim_embedding = config["dim_embedding"]
        self.aggregation = config["aggregation"]  # can be mean or max

        if self.aggregation == "max":
            self.fc_max = nn.Linear(dim_embedding, dim_embedding)

        self.layers = nn.ModuleList([])
        for i in range(num_layers):
            dim_input = dim_input_features if i == 0 else dim_embedding

            conv = SAGEConv(dim_input, dim_embedding)
            # Overwrite aggregation method (default is set to mean
            conv.aggr = self.aggregation

            self.layers.append(conv)

    def forward(
        self, data: Batch
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[object]]]:
        """
        Implements a DGN with simple graph convolutional layers.

        Args:
            data (torch_geometric.data.Batch): a batch of graphs

        Returns:
            the output depends on the readout passed to the model as argument.
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x_all = []

        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if self.aggregation == "max":
                x = torch.relu(self.fc_max(x))
            x_all.append(x)

        node_embs = torch.cat(x_all, dim=1)

        return self.readout(node_embs, batch, **dict(edge_index=edge_index))
