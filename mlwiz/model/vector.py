from typing import List, Optional, Tuple, Union

import torch
import torchvision
from torch import relu
from torch.nn import Linear, Conv2d, MaxPool2d

from mlwiz.model.interface import ModelInterface


class MLP(ModelInterface):
    """
    An MLP model used to test the library.
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
        self.mlp = torchvision.ops.MLP(
            in_channels=dim_input_features,
            hidden_channels=[dim_embedding, dim_embedding],
        )
        self.out_layer = Linear(dim_embedding, dim_target)

        self._testing = config.get("mlwiz_tests", False)

    def forward(
        self, data: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[object]]]:
        """
        Implements an MLP forward pass

        Args:
            data (torch.Tensor): a batched tensor

        Returns:
            a tuple (output, embedddings)
        """
        # for testing
        if self._testing:
            if data.shape[1] == 1:
                # MNIST dataset, let's remove channel dim and flatten
                assert len(data.shape) > 2
                data = data.squeeze(1)
                data = torch.reshape(data, (-1, self.dim_input_features))
            elif data.shape == (1, 28, 28):
                data = torch.reshape(data, (-1, self.dim_input_features))
        # --

        h = self.mlp(data)
        o = self.out_layer(h)
        return o, h


class CNN(ModelInterface):
    """
    A CNN model used to test the library.
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

        # taken from https://medium.com/@bpoyeka1/building-simple-neural-networks-nn-cnn-using-pytorch-for-mnist-dataset-31e459d17788
        self.conv1 = Conv2d(
            in_channels=1,
            out_channels=8,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )

        self.pool = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv2 = Conv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )
        # last conv out was 8 so this conv input is 8.

        self.fc1 = Linear(16 * 7 * 7, dim_target)

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
        h = relu(self.conv1(data))
        h = self.pool(h)
        h = relu(self.conv2(h))
        h = self.pool(h)
        h = h.reshape(h.shape[0], -1)
        o = self.fc1(h)
        return o, h
