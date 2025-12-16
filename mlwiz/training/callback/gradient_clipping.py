from torch.nn.utils import clip_grad_value_

from mlwiz.training.event.handler import EventHandler
from mlwiz.training.event.state import State


class GradientClipper(EventHandler):
    r"""
    GradientClipper is the main event handler for gradient clippers.
    Configure it in the experiment configuration to enable gradient clipping.

    Args:
        clip_value (float): the gradient will be clipped in
            [-clip_value, clip_value]
        kwargs (dict): additional arguments
    """

    def __init__(self, clip_value: float, **kwargs: dict):
        """
        Initialize the gradient clipper.

        Args:
            clip_value (float): Clip value used by
                :func:`torch.nn.utils.clip_grad_value_`.
            **kwargs: Unused extra arguments (kept for configuration
                compatibility).

        Side effects:
            Stores ``clip_value`` on the instance.
        """
        self.clip_value = clip_value

    def on_backward(self, state: State):
        """
        Clips the gradients of the model before the weights are updated.

        Args:
            state (:class:`~training.event.state.State`):
                object holding training information
        """
        clip_grad_value_(state.model.parameters(), clip_value=self.clip_value)
