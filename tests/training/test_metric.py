from typing import Tuple, List

import pytest
import torch

from mlwiz.training.callback.metric import Metric, AdditiveLoss, MultiScore
from mlwiz.training.event.state import State


class FakeMetric(Metric):
    r"""
    This is a fake metric class for testing purposes. It is designed to be used in machine learning workflows as a placeholder or a reference for implementing new metrics.

    Parameters:
        use_as_loss (bool, optional): Whether this metric should be used as a loss function. Defaults to False.
        reduction (str, optional): How the metric should be reduced. Supported values are 'mean' and 'sum'. Defaults to 'mean'.
        accumulate_over_epoch (bool, optional): Whether the metric should be computed over the whole dataset rather than averaged across minibatches. Defaults to True.
        force_cpu (bool, optional): Whether the metric should force execution on the CPU. Defaults to True.`
        device (str, optional): The device on which the metric should be executed. Defaults to 'cpu'.
        **kwargs: Additional keyword arguments to pass to the parent class.
    """

    def __init__(
        self,
        use_as_loss: bool = False,
        reduction: str = "mean",
        accumulate_over_epoch: bool = True,
        force_cpu: bool = True,
        device: str = "cpu",
        **kwargs: dict,
    ):
        super().__init__(
            use_as_loss=use_as_loss,
            reduction=reduction,
            accumulate_over_epoch=accumulate_over_epoch,
            force_cpu=force_cpu,
            device=device,
            **kwargs,
        )
        self.called = 0
        self.num_nodes = 20

    @property
    def name(self) -> str:
        return "Fake Metric"

    def get_predictions_and_targets(
        self, targets: torch.Tensor, *outputs: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get predictions and targets that will be passed to compute_metric.

        Parameters:
            targets (torch.Tensor): The target values.
            *outputs: Output of the model.

        Returns:
            tuple: A tuple containing the predictions and targets.
        """
        pred = torch.arange(self.num_nodes).float() + float(self.called)
        return pred, torch.zeros(1)

    def compute_metric(
        self, targets: torch.Tensor, predictions: torch.Tensor
    ) -> torch.tensor:
        """
        Compute the metric value based on the targets and predictions.

        Parameters:
            targets (torch.Tensor): The target values.
            predictions (torch.Tensor): The predicted values.

        Returns:
            torch.tensor: The computed metric value.
        """
        return predictions.mean()


def test_metric(fake_metric):
    """
    Check that batch/epoch loss/scores are correctly computed in terms of
    averaging over batches or over the entire loss
    """
    # repetition serves to deal with additive loss, where the same mocked
    # loss will be summed over many times

    reduction = "mean"  # not necessary in this test
    for use_as_loss in [False, True]:
        for accumulate_over_epoch in [False, True]:
            for force_cpu in [False, True]:
                metric = fake_metric(
                    use_as_loss, reduction, accumulate_over_epoch, force_cpu
                )
                for num_batch_calls in [1, 10]:
                    for ep_start, ba_start, forw, comp_met, ba_end, ep_end in [
                        (
                            metric.on_training_epoch_start,
                            metric.on_training_batch_start,
                            metric.on_forward,
                            metric.on_compute_metrics,
                            metric.on_training_batch_end,
                            metric.on_training_epoch_end,
                        ),
                        (
                            metric.on_eval_epoch_start,
                            metric.on_eval_batch_start,
                            metric.on_forward,
                            metric.on_compute_metrics,
                            metric.on_eval_batch_end,
                            metric.on_eval_epoch_end,
                        ),
                    ]:
                        # counter used to make the score change a bit
                        metric.called = 0.0

                        num_nodes = metric.num_nodes
                        state = State(model=None, optimizer=None, device="cpu")
                        state.batch_outputs = (torch.ones(1), torch.ones(1))
                        state.batch_targets = (torch.ones(1), torch.ones(1))

                        # Simulate training epoch
                        ep_start(state)

                        for batch in range(num_batch_calls):
                            state.update(batch_loss=None)
                            state.update(batch_score=None)

                            ba_start(state)

                            forw(state)

                            comp_met(state)

                            ba_end(state)

                            # change a bit the next score
                            metric.called += 1

                        ep_end(state)

                        expected_results = [
                            torch.arange(num_nodes) + float(i)
                            for i in range(num_batch_calls)
                        ]
                        if accumulate_over_epoch:
                            expected_results = torch.cat(expected_results)
                        else:
                            # for each batch compute the average score and then
                            # average again
                            expected_results = torch.stack(
                                expected_results,
                                dim=0,
                            )
                            # average resuls for each individual batch
                            expected_results = expected_results.mean(dim=1)

                        assert (
                            expected_results.mean()
                            == state.epoch_loss[metric.name]
                            if use_as_loss
                            else state.epoch_score[metric.name]
                        )


class FakeAdditiveLoss(AdditiveLoss):
    """
    A fake Additive Loss class for testing purposes.

    Args:
        use_as_loss (bool, optional): Whether to use this loss as a loss function. Defaults to False.
        reduction (str, optional): How to reduce the loss. Options are "mean", "sum", or "none". Defaults to "mean".
        accumulate_over_epoch (bool, optional): Whether to accumulate the loss over an epoch. Defaults to True.
        force_cpu (bool, optional): Whether to force the loss to be calculated on the CPU. Defaults to True.
        device (str, optional): The device to use for the loss calculation. Defaults to "cpu".
        **losses: Additional keyword arguments to pass to the parent class.
    """

    def __init__(
        self,
        use_as_loss: bool = False,
        reduction: str = "mean",
        accumulate_over_epoch: bool = True,
        force_cpu: bool = True,
        device: str = "cpu",
        **losses,
    ):
        super().__init__(
            use_as_loss=use_as_loss,
            reduction=reduction,
            accumulate_over_epoch=accumulate_over_epoch,
            force_cpu=force_cpu,
            device=device,
            **losses,
        )
        self.called = 0
        self.num_nodes = 20

    @property
    def name(self) -> str:
        """
        The name of the loss.

        Returns:
            str: The name of the loss.
        """
        return "Fake Additive Loss"


@pytest.fixture
def fake_additive_loss():
    def metric_init_fun(
        use_as_loss, reduction, accumulate_over_epoch, force_cpu
    ):
        return FakeAdditiveLoss(
            use_as_loss,
            reduction,
            accumulate_over_epoch,
            force_cpu,
            loss1="tests.training.test_metric.FakeMetric",
            loss2="tests.training.test_metric.FakeMetric",
            loss3="tests.training.test_metric.FakeMetric",
        )

    # Return how many times the fake metric will be summed over
    return metric_init_fun


def test_additive_loss(fake_additive_loss):
    """
    Check that batch/epoch loss/scores are correctly computed in terms of
    averaging over batches or over the entire loss
    """
    # repetition serves to deal with additive loss, where the same mocked
    # loss will be summed over many times

    reduction = "mean"  # not necessary in this test
    for use_as_loss in [True]:
        for accumulate_over_epoch in [False, True]:
            for force_cpu in [False, True]:
                metric = fake_additive_loss(
                    use_as_loss, reduction, accumulate_over_epoch, force_cpu
                )
                for num_batch_calls in [1, 10]:
                    for ep_start, ba_start, forw, comp_met, ba_end, ep_end in [
                        (
                            metric.on_training_epoch_start,
                            metric.on_training_batch_start,
                            metric.on_forward,
                            metric.on_compute_metrics,
                            metric.on_training_batch_end,
                            metric.on_training_epoch_end,
                        ),
                        (
                            metric.on_eval_epoch_start,
                            metric.on_eval_batch_start,
                            metric.on_forward,
                            metric.on_compute_metrics,
                            metric.on_eval_batch_end,
                            metric.on_eval_epoch_end,
                        ),
                    ]:
                        # counter used to make the score change a bit
                        for m in metric.losses:
                            m.called = 0.0

                        num_nodes = metric.num_nodes
                        state = State(model=None, optimizer=None, device="cpu")
                        state.batch_outputs = (torch.ones(1), torch.ones(1))
                        state.batch_targets = (torch.ones(1), torch.ones(1))

                        # Simulate training epoch
                        ep_start(state)

                        for batch in range(num_batch_calls):
                            state.update(batch_loss=None)
                            state.update(batch_score=None)

                            ba_start(state)

                            forw(state)

                            comp_met(state)

                            ba_end(state)

                            # change a bit the next score
                            for m in metric.losses:
                                m.called += 1.0

                        ep_end(state)

                        expected_results = [
                            (torch.arange(num_nodes) + float(i)) * 3.0
                            for i in range(num_batch_calls)
                        ]
                        if accumulate_over_epoch:
                            expected_results = torch.cat(expected_results)
                        else:
                            # for each batch compute the average score and then
                            # average again
                            expected_results = torch.stack(
                                expected_results,
                                dim=0,
                            )
                            # average resuls for each individual batch
                            expected_results = expected_results.mean(dim=1)

                        assert (
                            expected_results.mean()
                            == state.epoch_loss[metric.name]
                            if use_as_loss
                            else state.epoch_score[metric.name]
                        )


class FakeMultiScore(MultiScore):
    """
    A Fake Multi Score class that simulates the MultiScore class with additional properties and methods.

    Parameters:
        use_as_loss (bool): If True, the score is used as a loss.
        reduction (str): The reduction method for the scores.
        accumulate_over_epoch (bool): If True, the scores are accumulated over an epoch.
        force_cpu (bool): If True, the scores are forced to be on the CPU.
        device (str): The device to use for the scores.
        main_scorer (Metric): The main scorer for the scores.
        **extra_scorers (dict): Additional scorers to be used.
    """

    def __init__(
        self,
        use_as_loss: bool = False,
        reduction: str = "mean",
        accumulate_over_epoch: bool = True,
        force_cpu: bool = True,
        device: str = "cpu",
        main_scorer: Metric = None,
        **extra_scorers,
    ):
        super().__init__(
            use_as_loss=use_as_loss,
            reduction=reduction,
            accumulate_over_epoch=accumulate_over_epoch,
            force_cpu=force_cpu,
            device=device,
            main_scorer=main_scorer,
            **extra_scorers,
        )
        self.called = 0
        self.num_nodes = 20

    @property
    def name(self) -> str:
        """
        Returns the name of the score.

        Returns:
            str: The name of the score.
        """
        return "Fake Multi Score"


@pytest.fixture
def fake_multi_score():
    def metric_init_fun(
        use_as_loss, reduction, accumulate_over_epoch, force_cpu
    ):
        return FakeMultiScore(
            use_as_loss,
            reduction,
            accumulate_over_epoch,
            force_cpu,
            main_scorer="tests.training.test_metric.FakeMetric",
            score2="tests.training.test_metric.FakeMetric",
            score3="tests.training.test_metric.FakeMetric",
        )

    # Return how many times the fake metric will be summed over
    return metric_init_fun


def test_multi_score(fake_multi_score):
    """
    Check that batch/epoch loss/scores are correctly computed in terms of
    averaging over batches or over the entire loss
    """
    # repetition serves to deal with additive loss, where the same mocked
    # loss will be summed over many times

    reduction = "mean"  # not necessary in this test
    for use_as_loss in [False]:
        for accumulate_over_epoch in [False, True]:
            for force_cpu in [False, True]:
                metric = fake_multi_score(
                    use_as_loss, reduction, accumulate_over_epoch, force_cpu
                )
                for num_batch_calls in [1, 10]:
                    for ep_start, ba_start, forw, comp_met, ba_end, ep_end in [
                        (
                            metric.on_training_epoch_start,
                            metric.on_training_batch_start,
                            metric.on_forward,
                            metric.on_compute_metrics,
                            metric.on_training_batch_end,
                            metric.on_training_epoch_end,
                        ),
                        (
                            metric.on_eval_epoch_start,
                            metric.on_eval_batch_start,
                            metric.on_forward,
                            metric.on_compute_metrics,
                            metric.on_eval_batch_end,
                            metric.on_eval_epoch_end,
                        ),
                    ]:
                        # counter used to make the score change a bit
                        for s in metric.scores:
                            s.called = 0.0

                        num_nodes = metric.num_nodes
                        state = State(model=None, optimizer=None, device="cpu")
                        state.batch_outputs = (torch.ones(1), torch.ones(1))
                        state.batch_targets = (torch.ones(1), torch.ones(1))

                        # Simulate training epoch
                        ep_start(state)

                        for batch in range(num_batch_calls):
                            state.update(batch_loss=None)
                            state.update(batch_score=None)

                            ba_start(state)

                            forw(state)

                            comp_met(state)

                            ba_end(state)

                            # change a bit the next score
                            for s in metric.scores:
                                s.called += 1.0

                        ep_end(state)

                        expected_results = [
                            torch.arange(num_nodes) + float(i)
                            for i in range(num_batch_calls)
                        ]
                        if accumulate_over_epoch:
                            expected_results = torch.cat(expected_results)
                        else:
                            # for each batch compute the average score and then
                            # average again
                            expected_results = torch.stack(
                                expected_results,
                                dim=0,
                            )
                            # average resuls for each individual batch
                            expected_results = expected_results.mean(dim=1)

                        assert (
                            expected_results.mean()
                            == state.epoch_loss[metric.get_main_metric_name()]
                            if use_as_loss
                            else state.epoch_score[
                                metric.get_main_metric_name()
                            ]
                        )
