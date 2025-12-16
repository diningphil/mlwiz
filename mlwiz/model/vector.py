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
        """
        Initialize a simple multi-layer perceptron for vector inputs.

        Args:
            dim_input_features (Union[int, Tuple[int]]): Input feature dimension.
                Must be an ``int`` for this model (flat vector features).
            dim_target (int): Output dimension (e.g., number of classes).
            config (dict): Model configuration. Expected keys:
                - ``dim_embedding`` (int): Hidden embedding size.
                Optional keys:
                - ``mlwiz_tests`` (bool): Enable test-only input reshaping
                  for MNIST-like tensors.

        Raises:
            TypeError: If ``dim_input_features`` is not an ``int``.
            KeyError: If ``config`` does not contain ``dim_embedding``.

        Side effects:
            Initializes internal Torch modules and stores a testing flag in
            ``self._testing``.
        """
        super().__init__(
            dim_input_features,
            dim_target,
            config,
        )
        if not isinstance(dim_input_features, int):
            raise TypeError(
                "dim_input_features must be an int for vector models, "
                f"got {dim_input_features!r} ({type(dim_input_features).__name__})."
            )

        dim_embedding = config["dim_embedding"]
        self.mlp = torchvision.ops.MLP(
            in_channels=dim_input_features,
            hidden_channels=[dim_embedding, dim_embedding],
        )
        self.out_layer = Linear(dim_embedding, dim_target)

        self._testing = config.get("mlwiz_tests", False)

    def forward(self, data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform a forward pass of the MLP.

        When running MLWiz integration tests (``config['mlwiz_tests']``),
        this method supports MNIST-like inputs by flattening images to a
        vector before applying the MLP.

        Args:
            data (torch.Tensor): Input batch tensor. Expected shapes:
                - ``(batch, dim_input_features)`` for standard vector inputs.
                - ``(batch, 1, H, W)`` or ``(1, 28, 28)`` in test mode, which
                  will be reshaped to ``(batch, dim_input_features)``.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Model outputs of shape ``(batch, dim_target)``.
                - Embeddings of shape ``(batch, dim_embedding)``.

            Note:
                Some MLWiz models return an additional third element with
                auxiliary outputs. This model returns only ``(output, embeddings)``.

        Raises:
            ValueError: If test-mode MNIST reshaping encounters an
                unexpected input shape.
        """
        # time.sleep(0.2)  # simulate some delay
        # for testing
        if self._testing:
            if data.shape[1] == 1:
                # MNIST dataset, let's remove channel dim and flatten
                if data.dim() <= 2:
                    raise ValueError(
                        "In test mode, expected an MNIST-like tensor with at least 3 dimensions "
                        f"when data.shape[1] == 1, got shape {tuple(data.shape)}."
                    )
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
        """
        Initialize a small convolutional network for image-like inputs.

        Args:
            dim_input_features (Union[int, Tuple[int]]): Input feature dimension.
                Must be an ``int`` for this model; the input is expected to be
                an image tensor with a single channel.
            dim_target (int): Output dimension (e.g., number of classes).
            config (dict): Model configuration dictionary (currently unused).

        Raises:
            TypeError: If ``dim_input_features`` is not an ``int``.

        Side effects:
            Initializes internal convolution/pooling/linear Torch modules.
        """
        super().__init__(
            dim_input_features,
            dim_target,
            config,
        )
        if not isinstance(dim_input_features, int):
            raise TypeError(
                "dim_input_features must be an int for vector models, "
                f"got {dim_input_features!r} ({type(dim_input_features).__name__})."
            )

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

    def forward(self, data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform a forward pass of the CNN.

        Args:
            data (torch.Tensor): Input image batch tensor of shape
                ``(batch, 1, H, W)``.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Model outputs of shape ``(batch, dim_target)``.
                - Embeddings/features of shape ``(batch, 16 * 7 * 7)`` for the
                  default architecture (post-conv flattened representation).

            Note:
                Some MLWiz models return an additional third element with
                auxiliary outputs. This model returns only ``(output, embeddings)``.
        """
        h = relu(self.conv1(data))
        h = self.pool(h)
        h = relu(self.conv2(h))
        h = self.pool(h)
        h = h.reshape(h.shape[0], -1)
        o = self.fc1(h)
        return o, h
