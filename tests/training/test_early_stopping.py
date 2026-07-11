"""Unit tests for early stopping callbacks.

Validates patience-based stopping behavior over synthetic epoch results.
"""

import torch
from torch.nn import Linear

from mlwiz.model.interface import ModelInterface
from mlwiz.static import (
    BEST_CHECKPOINT_FILENAME,
    BEST_EPOCH,
    BEST_OPTIMIZER_CHECKPOINT_FILENAME,
    LOSSES,
    MAX,
    MIN,
    MODEL_STATE,
    OPTIMIZER_STATE,
    SCORES,
)
from mlwiz.training.callback.early_stopping import PatienceEarlyStopper
from mlwiz.training.event.state import State


class FakeModel(ModelInterface):
    """
    A fake model class that implements the `ModelInterface` interface.
    """

    def __init__(self):
        """
        Initializes a new instance of the `FakeModel` class.
        """
        super().__init__(0, 0, None)
        self.lin = Linear(10, 10)


def test_early_stopping_patience():
    """
    Tests the `PatienceEarlyStopper` class with different parameters.

    This function iterates over `use_as_loss` and `patience` values to test the behavior of the `PatienceEarlyStopper` class.
    For each combination, it creates an instance of the `PatienceEarlyStopper`, initializes a `State` object with a `FakeModel`, updates the state with epoch results,
    and checks the behavior of the `PatienceEarlyStopper` during the training loop.
    """
    for use_as_loss in [False, True]:
        for patience in [2, 10]:
            early_stopper = PatienceEarlyStopper(
                (
                    "validation_main_loss"
                    if use_as_loss
                    else "validation_main_score"
                ),
                mode=MIN if use_as_loss else MAX,
                patience=patience,
                checkpoint=False,
            )

            state = State(model=FakeModel(), optimizer=None, device="cpu")

            # Update state with epoch results
            epoch_results = {LOSSES: {}, SCORES: {}}
            state.update(epoch_results=epoch_results)

            num_epochs = 30
            for epoch in range(1, num_epochs + 1):
                state.update(epoch=epoch)

                state.epoch_results[LOSSES].update(
                    {"validation_main_loss": epoch}
                )
                state.epoch_results[SCORES].update(
                    {"validation_main_score": epoch}
                )

                early_stopper.on_epoch_end(state)

                if state.stop_training:
                    break

            if use_as_loss:
                # implies MIN is used. For this test we should stop and exit
                # the loop after patience epochs (the best epoch will always
                # be the first)
                assert state.best_epoch_results[BEST_EPOCH] == epoch - patience
            else:
                assert state.best_epoch_results[BEST_EPOCH] == num_epochs


def test_early_stopper_writes_split_best_checkpoint(tmp_path):
    """Best model and optimizer snapshots are persisted to distinct files."""
    early_stopper = PatienceEarlyStopper(
        "validation_main_score",
        mode=MAX,
        patience=2,
        checkpoint=True,
    )
    state = State(model=FakeModel(), optimizer=None, device="cpu")
    state.update(
        exp_path=str(tmp_path),
        epoch=0,
        epoch_results={
            LOSSES: {},
            SCORES: {"validation_main_score": 0.5},
        },
        optimizer_state={"state": {}, "param_groups": []},
        scheduler_state=None,
        scaler_state=None,
    )

    early_stopper.on_epoch_end(state)

    model_payload = torch.load(
        tmp_path / BEST_CHECKPOINT_FILENAME, weights_only=True
    )
    optimizer_payload = torch.load(
        tmp_path / BEST_OPTIMIZER_CHECKPOINT_FILENAME, weights_only=True
    )
    assert MODEL_STATE in model_payload
    assert OPTIMIZER_STATE not in model_payload
    assert optimizer_payload[OPTIMIZER_STATE] == {
        "state": {},
        "param_groups": [],
    }
