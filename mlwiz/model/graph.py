from typing import List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import Linear
from torch_geometric.data import Batch
from torch_geometric.nn import SAGEConv, global_mean_pool

from mlwiz.model.interface import ModelInterface


# from torch_geometric_temporal import DCRNN


class DGN(ModelInterface):
    """
    A Deep Graph Network used to test the library.
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
        if type(dim_input_features) == tuple:
            dim_feats = dim_input_features[0]
        else:
            dim_feats = dim_input_features

        self.is_graph_classification = config["is_graph_classification"]
        self.is_single_graph = config["is_single_graph"]

        num_layers = config["num_layers"]
        dim_embedding = config["dim_embedding"]

        self.layers = nn.ModuleList([])
        for i in range(num_layers):
            dim_input = dim_feats if i == 0 else dim_embedding
            conv = SAGEConv(dim_input, dim_embedding)
            self.layers.append(conv)

        self.out_layer = Linear(dim_embedding * num_layers, dim_target)

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
            x_all.append(x)

        h = torch.cat(x_all, dim=1)

        idxs = None
        if self.is_single_graph:
            if self.training:
                idxs = data.training_indices
            else:
                idxs = data.eval_indices

        if self.is_graph_classification:
            h = global_mean_pool(h, batch)
            o = self.out_layer(h)
        else:
            o = self.out_layer(h)

        if self.is_single_graph:
            o = o[idxs]
            h = h[idxs]

        return o, h, idxs
