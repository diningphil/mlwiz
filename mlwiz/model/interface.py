from typing import Tuple, Optional, List, Union

import torch
import torch_geometric


class ModelInterface(torch.nn.Module):
    r"""
    Provides the signature for any main model to be trained under MLWiz

    Args:
        dim_input_features (Union[int, Tuple[int]]): dimension of node features
            (according to the :class:`~mlwiz.data.dataset.DatasetInterface`
            property)
        dim_target (int): dimension of the target
            (according to the :class:`~mlwiz.data.dataset.DatasetInterface`
            property)
        config (dict): config dictionary containing all the necessary
            hyper-parameters plus additional information (if needed)
    """

    def __init__(
        self,
        dim_input_features: Union[int, Tuple[int]],
        dim_target: int,
        config: dict,
    ):
        super().__init__()
        self.dim_input_features = dim_input_features
        self.dim_target = dim_target
        self.config = config

    def forward(
        self, data: Union[torch.Tensor, torch_geometric.data.Batch]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[object]]]:
        r"""
        Performs a forward pass over a batch of graphs

        Args:
            data: a batch of samples

        Returns:
            a tuple (model's output, [optional] node embeddings,
            [optional] additional outputs
        """
        raise NotImplementedError("You need to implement this method!")
