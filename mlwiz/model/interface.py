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
        r"""
        Initialize the model interface.

        Args:
            dim_input_features (Union[int, Tuple[int]]): Input feature dimension.
                For graph datasets this may be a tuple to represent multiple
                feature blocks (e.g., node features and edge features).
            dim_target (int): Target dimension (e.g., number of classes or
                regression outputs).
            config (dict): Free-form configuration dictionary used by concrete
                models (hyper-parameters and optional flags).

        Side effects:
            Stores the arguments as instance attributes.
        """
        super().__init__()
        self.dim_input_features = dim_input_features
        self.dim_target = dim_target
        self.config = config

    def forward(
        self, data: Union[torch.Tensor, torch_geometric.data.Batch]
    ) -> Union[
        Tuple[torch.Tensor, Optional[torch.Tensor]],
        Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[object]]],
    ]:
        r"""
        Perform a forward pass over a batch of samples.

        Args:
            data: a batch of samples

        Returns:
            Either:

            - ``(output, embeddings)``
            - ``(output, embeddings, additional_outputs)``

            where ``embeddings`` and ``additional_outputs`` are optional and may
            be omitted by models that do not produce them.
        """
        raise NotImplementedError("You need to implement this method!")
