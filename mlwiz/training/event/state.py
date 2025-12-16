class State:
    """
    Any object of this class contains training information that is handled and
    modified by a :class:`TrainingEngine` as well as by the
    `EventHandler` objects implementing callbacks

    Args:
        model (torch.nn.Module): the model
        optimizer (training.callback.optimizer.Optimizer): the optimizer
        device (str): the device on which to run computations
    """

    def __init__(self, model, optimizer, device):
        """
        Initialize the training state container.

        Args:
            model (torch.nn.Module): Model being trained/evaluated.
            optimizer (mlwiz.training.callback.optimizer.Optimizer): Optimizer
                callback wrapper.
            device (str): Device identifier used for computation
                (e.g., ``'cpu'`` or ``'cuda:0'``).

        Side effects:
            Initializes common bookkeeping fields (epoch counters, flags, and
            callback state placeholders) used by the training engine and event
            handlers.
        """
        self.initial_epoch = 0
        self.epoch = self.initial_epoch
        self.total_epochs = -1
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.optimizer_state = None
        self.scheduler = None
        self.scheduler_state = None
        self.stop_training = False
        self.return_embeddings = False
        self.set = None
        self.current_elapsed_time = 0

    def __getitem__(self, name):
        """
        Returns the value associated with argument `name`, otherwise returns
        :obj:`None`
        """
        return getattr(self, name, None)

    def __contains__(self, name):
        """
        Returns true if state contains the field `name`, and False otherwise
        """
        return name in self.__dict__

    def update(self, **values: dict):
        """
        The method sets new attributes or updates existing ones using the
        key,value pairs in ``values``

        Args:
            values: a dictionary of key,value pairs to store in
                the global state
        """
        for name, value in values.items():
            setattr(self, name, value)
