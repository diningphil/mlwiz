from mlwiz.experiment.experiment import Experiment
from mlwiz.static import LOSS, SCORE


class SupervisedTask(Experiment):
    """
    Class that implements a standard supervised experiment.
    """

    def run_valid(self, dataset_getter, logger):
        r"""
        This function returns the training and validation results
        for a `model selection run`.
        **Do not attempt to load the test set inside this method!**
        **If possible, rely on already available subclasses of this class**.

        It implements a simple supervised training scheme.

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
        batch_size = self.model_config.supervised_config["batch_size"]
        shuffle = (
            self.model_config.supervised_config["shuffle"]
            if "shuffle" in self.model_config.supervised_config
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
        dim_edge_features = dataset_getter.get_dim_edge_features()
        dim_target = dataset_getter.get_dim_target()

        # Instantiate the Model
        model = self.create_supervised_model(
            dim_input_features, dim_edge_features, dim_target
        )

        # Instantiate the engine (it handles the training loop and the
        # inference phase by abstracting the specifics)
        supervised_training_engine = self.create_supervised_engine(model)

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
        ) = supervised_training_engine.train(
            train_loader=train_loader,
            validation_loader=val_loader,
            test_loader=None,
            max_epochs=self.model_config.supervised_config["epochs"],
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

        It implements a simple supervised training scheme.

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
        batch_size = self.model_config.supervised_config["batch_size"]
        shuffle = (
            self.model_config.supervised_config["shuffle"]
            if "shuffle" in self.model_config.supervised_config
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
        dim_edge_features = dataset_getter.get_dim_edge_features()
        dim_target = dataset_getter.get_dim_target()

        # Instantiate the Model
        model = self.create_supervised_model(
            dim_input_features, dim_edge_features, dim_target
        )

        # Instantiate the engine (it handles the training loop and the
        # inference phase by abstracting the specifics)
        supervised_training_engine = self.create_supervised_engine(model)

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
        ) = supervised_training_engine.train(
            train_loader=train_loader,
            validation_loader=val_loader,
            test_loader=test_loader,
            max_epochs=self.model_config.supervised_config["epochs"],
            logger=logger,
        )

        train_res = {LOSS: train_loss, SCORE: train_score}
        val_res = {LOSS: val_loss, SCORE: val_score}
        test_res = {LOSS: test_loss, SCORE: test_score}
        return train_res, val_res, test_res
