import random
from typing import Callable, Tuple, Union

import numpy as np
import torch

from mlwiz.evaluation.config import Config
from mlwiz.util import return_class_and_args, s2c
from mlwiz.model.interface import ModelInterface
from mlwiz.static import DEFAULT_ENGINE_CALLBACK
from mlwiz.static import LOSS, SCORE
from mlwiz.training.engine import TrainingEngine


class Experiment:
    r"""
    Class that handles a single standard experiment.

    Args:
        model_configuration (dict): the dictionary holding the
            experiment-specific configuration
        exp_path (str): path to the experiment folder
        exp_seed (int): the experiment's seed to use
    """

    def __init__(
        self, model_configuration: dict, exp_path: str, exp_seed: int
    ):
        self.model_config = Config(model_configuration)
        self.exp_path = exp_path
        self.exp_seed = exp_seed
        # Set seed here to aid reproducibility
        np.random.seed(self.exp_seed)
        torch.manual_seed(self.exp_seed)
        torch.cuda.manual_seed(self.exp_seed)
        random.seed(self.exp_seed)

        # torch.use_deterministic_algorithms(True) for future versions of
        # Pytorch
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _return_class_and_args(
        config: Config, key: str
    ) -> Tuple[Callable[..., object], dict]:
        r"""
        Returns the class and arguments associated to a specific key in the
        configuration file.

        Args:
            config: the configuration dictionary
            key: a string representing a particular class in the
                configuration dictionary

        Returns:
            a tuple (class, dict of arguments), or (None, None) if the key
            is not present in the config dictionary
        """
        if key not in config or config[key] is None:
            return None, None
        elif isinstance(config[key], str):
            return s2c(config[key]), {}
        elif isinstance(config[key], dict):
            return (
                s2c(config[key]["class_name"]),
                config[key]["args"] if "args" in config[key] else {},
            )
        else:
            raise NotImplementedError(
                "Parameter has not been formatted properly"
            )

    def create_model(
        self,
        dim_input_features: Union[int, Tuple[int]],
        dim_target: int,
        config: Config,
    ) -> ModelInterface:
        r"""
        Instantiates a model that implements the
        :class:`~mlwiz.model.model.ModelInterface` interface

        Args:
            dim_input_features (Union[int, Tuple[int]]): number of node features
            dim_target (int): target dimension
            config (:class:`~mlwiz.evaluation.config.Config`):
                the configuration dictionary

        Returns:
            a model that implements the
            :class:`~mlwiz.model.model.ModelInterface` interface
        """
        model = s2c(config["model"])(
            dim_input_features=dim_input_features,
            dim_target=dim_target,
            config=config,
        )

        # move to device
        # model .to() may not return anything
        model.to(self.model_config.device)
        return model

    def create_engine(
        self,
        config: Config,
        model: ModelInterface,
    ) -> TrainingEngine:
        r"""
        Utility that instantiates the training engine. It looks for
        pre-defined fields in the configuration file, i.e. ``loss``,
        ``scorer``, ``optimizer``, ``scheduler``, ``gradient_clipper``,
        ``early_stopper`` and ``plotter``, all of which should be classes
        implementing the :class:`~mlwiz.training.event.handler.EventHandler`
        interface

        Args:
            config (:class:`~mlwiz.evaluation.config.Config`):
                the configuration dictionary
            model: the  model that needs be trained

        Returns:
            a :class:`~mlwiz.training.engine.TrainingEngine` object
        """
        device = config["device"]
        evaluate_every = config["evaluate_every"]

        loss_class, loss_args = return_class_and_args(config, "loss")
        loss_args.update(device=device)
        loss = (
            loss_class(use_as_loss=True, **loss_args)
            if loss_class is not None
            else None
        )

        scorer_class, scorer_args = return_class_and_args(config, "scorer")
        scorer_args.update(device=device)
        scorer = (
            scorer_class(use_as_loss=False, **scorer_args)
            if scorer_class is not None
            else None
        )

        optim_class, optim_args = return_class_and_args(config, "optimizer")
        optimizer = (
            optim_class(model=model, **optim_args)
            if optim_class is not None
            else None
        )

        sched_class, sched_args = return_class_and_args(config, "scheduler")
        if sched_args is not None:
            sched_args["optimizer"] = optimizer.optimizer
        scheduler = (
            sched_class(**sched_args) if sched_class is not None else None
        )
        # Remove the optimizer obj ow troubles when dumping the config file
        if sched_args is not None:
            sched_args.pop("optimizer", None)

        grad_clip_class, grad_clip_args = return_class_and_args(
            config, "gradient_clipper"
        )
        grad_clipper = (
            grad_clip_class(**grad_clip_args)
            if grad_clip_class is not None
            else None
        )

        early_stop_class, early_stop_args = return_class_and_args(
            config, "early_stopper"
        )
        early_stopper = (
            early_stop_class(**early_stop_args)
            if early_stop_class is not None
            else None
        )

        plot_class, plot_args = return_class_and_args(config, "plotter")
        plotter = (
            plot_class(exp_path=self.exp_path, **plot_args)
            if plot_class is not None
            else None
        )

        store_last_checkpoint = config.get("checkpoint", False)
        engine_class, engine_args = return_class_and_args(config, "engine")
        engine_callback = s2c(
            engine_args.get("engine_callback", DEFAULT_ENGINE_CALLBACK)
        )
        eval_training = engine_args.get("eval_training", False)

        engine = engine_class(
            engine_callback=engine_callback,
            model=model,
            loss=loss,
            optimizer=optimizer,
            scorer=scorer,
            scheduler=scheduler,
            early_stopper=early_stopper,
            gradient_clipper=grad_clipper,
            device=device,
            plotter=plotter,
            exp_path=self.exp_path,
            evaluate_every=evaluate_every,
            eval_training=eval_training,
            store_last_checkpoint=store_last_checkpoint,
        )
        return engine

    def run_valid(self, dataset_getter, logger):
        r"""
        This function returns the training and validation results
        for a `model selection run`.
        **Do not attempt to load the test set inside this method!**
        **If possible, rely on already available subclasses of this class**.

        It implements a simple training scheme.

        Args:
            dataset_getter (:class:`~mlwiz.data.provider.DataProvider`):
                a data provider
            logger (:class:`~mlwiz.log.logger.Logger`): the logger

        Returns:
            a tuple of training and test dictionaries.
            Each dictionary has two keys:

            * ``LOSS`` (as defined in ``mlwiz.static``)
            * ``SCORE`` (as defined in ``mlwiz.static``)

            For instance, training_results[SCORE] is a dictionary itself
            with other fields to be used by the evaluator.
        """
        batch_size = self.model_config["batch_size"]
        shuffle = (
            self.model_config["shuffle"]
            if "shuffle" in self.model_config
            else True
        )

        # Instantiate the Dataset
        train_loader = dataset_getter.get_inner_train(
            batch_size=batch_size, shuffle=shuffle
        )
        val_loader = dataset_getter.get_inner_val(
            batch_size=batch_size, shuffle=shuffle
        )

        dim_input_features = dataset_getter.get_dim_input_features()
        dim_target = dataset_getter.get_dim_target()

        # Instantiate the Model
        model = self.create_model(
            dim_input_features, dim_target, self.model_config
        )

        # Instantiate the engine (it handles the training loop and the
        # inference phase by abstracting the specifics)
        training_engine = self.create_engine(self.model_config, model)

        (
            train_loss,
            train_score,
            _,  # check the ordering is correct
            val_loss,
            val_score,
            _,
            _,
            _,
            _,
        ) = training_engine.train(
            train_loader=train_loader,
            validation_loader=val_loader,
            test_loader=None,
            max_epochs=self.model_config["epochs"],
            logger=logger,
        )

        train_res = {LOSS: train_loss, SCORE: train_score}
        val_res = {LOSS: val_loss, SCORE: val_score}
        return train_res, val_res

    def run_test(self, dataset_getter, logger):
        """
        This function returns the training, validation and test results
        for a `final run`.
        **Do not use the test to train the model
        nor for early stopping reasons!**
        **If possible, rely on already available subclasses of this class**.

        It implements a simple training scheme.

        Args:
            dataset_getter (:class:`~mlwiz.data.provider.DataProvider`):
                a data provider
            logger (:class:`~mlwiz.log.logger.Logger`): the logger

        Returns:
            a tuple of training,validation,test dictionaries.
            Each dictionary has two keys:

            * ``LOSS`` (as defined in ``mlwiz.static``)
            * ``SCORE`` (as defined in ``mlwiz.static``)

            For instance, training_results[SCORE] is a dictionary itself with
            other fields to be used by the evaluator.
        """
        batch_size = self.model_config["batch_size"]
        shuffle = (
            self.model_config["shuffle"]
            if "shuffle" in self.model_config
            else True
        )

        # Instantiate the Dataset
        train_loader = dataset_getter.get_outer_train(
            batch_size=batch_size, shuffle=shuffle
        )
        val_loader = dataset_getter.get_outer_val(
            batch_size=batch_size, shuffle=shuffle
        )
        test_loader = dataset_getter.get_outer_test(
            batch_size=batch_size, shuffle=shuffle
        )

        # Call this after the loaders: the datasets may need to be instantiated
        # with additional parameters
        dim_input_features = dataset_getter.get_dim_input_features()
        dim_target = dataset_getter.get_dim_target()

        # Instantiate the Model
        model = self.create_model(
            dim_input_features, dim_target, self.model_config
        )

        # Instantiate the engine (it handles the training loop and the
        # inference phase by abstracting the specifics)
        training_engine = self.create_engine(self.model_config, model)

        (
            train_loss,
            train_score,
            _,
            val_loss,
            val_score,
            _,
            test_loss,
            test_score,
            _,
        ) = training_engine.train(
            train_loader=train_loader,
            validation_loader=val_loader,
            test_loader=test_loader,
            max_epochs=self.model_config["epochs"],
            logger=logger,
        )

        train_res = {LOSS: train_loss, SCORE: train_score}
        val_res = {LOSS: val_loss, SCORE: val_score}
        test_res = {LOSS: test_loss, SCORE: test_score}
        return train_res, val_res, test_res
