import random
from typing import Tuple

import numpy as np
import torch
from sklearn.model_selection import (
    StratifiedKFold,
    StratifiedShuffleSplit,
    ShuffleSplit,
    KFold,
)

import mlwiz
from mlwiz.data.dataset import IterableDatasetInterface
from mlwiz.util import s2c, dill_load, dill_save


class Fold:
    r"""
    Simple class that stores training, validation, and test indices.

    Args:
        train_idxs (Union[list, tuple]): training indices
        val_idxs (Union[list, tuple]): validation indices. Default is ``None``
        test_idxs (Union[list, tuple]): test indices. Default is ``None``
    """

    def __init__(self, train_idxs, val_idxs=None, test_idxs=None):
        self.train_idxs = train_idxs
        self.val_idxs = val_idxs
        self.test_idxs = test_idxs


class InnerFold(Fold):
    r"""
    Simple extension of the Fold class that returns a dictionary with
    training and validation indices (model selection).
    """

    def todict(self) -> dict:
        r"""
        Creates a dictionary with the training/validation indices.

        Returns:
            a dict with keys ``['train', 'val']`` associated with the
            respective indices
        """
        return {"train": self.train_idxs, "val": self.val_idxs}


class OuterFold(Fold):
    r"""
    Simple extension of the Fold class that returns a dictionary with training
    and test indices (risk assessment)
    """

    def todict(self) -> dict:
        r"""
        Creates a dictionary with the training/validation/test indices.

        Returns:
            a dict with keys ``['train', 'val', 'test']``
            associated with the respective indices
        """
        return {
            "train": self.train_idxs,
            "val": self.val_idxs,
            "test": self.test_idxs,
        }


class _NoShuffleTrainTestSplit:
    r"""
    Class that implements a very simple training/test split. Can be used to
    further split training data into training and validation.

    Args:
        test_ratio: percentage of data to use for evaluation.
    """

    def __init__(self, test_ratio):
        self.test_ratio = test_ratio

    # Leave the arguments as they are. The parameter `y` is needed to
    # implement the same interface as sklearn
    def split(self, idxs, y=None):
        """
        Splits the data.

        Args:
            idxs: the indices to split according to the `test_ratio` parameter
            y: Unused argument

        Returns:
            a list of a single tuple (train indices, test/eval indices)
        """
        n_samples = len(idxs)
        n_test = int(n_samples * self.test_ratio)
        n_train = n_samples - n_test
        train_idxs = np.arange(n_train)
        test_idxs = np.arange(n_train, n_train + n_test)
        return [(train_idxs, test_idxs)]


class Splitter:
    r"""
    Class that generates and stores the data splits at dataset creation time.

    Args:
        n_outer_folds (int): number of outer folds (risk assessment).
            1 means hold-out, >1 means k-fold
        n_inner_folds (int): number of inner folds (model selection).
            1 means hold-out, >1 means k-fold
        seed (int): random seed for reproducibility (on the same machine)
        stratify (bool): whether to apply stratification or not
            (should be true for classification tasks)
        shuffle (bool): whether to apply shuffle or not
        inner_val_ratio  (float): percentage of validation set for
            hold_out model selection. Default is ``0.1``
        outer_val_ratio  (float): percentage of validation set for
            hold_out model assessment (final training runs).
            Default is ``0.1``
        test_ratio  (float): percentage of test set for
            hold_out model assessment. Default is ``0.1``
    """

    def __init__(
        self,
        n_outer_folds: int,
        n_inner_folds: int,
        seed: int,
        stratify: bool,
        shuffle: bool,
        inner_val_ratio: float = 0.1,
        outer_val_ratio: float = 0.1,
        test_ratio: float = 0.1,
    ):
        self.outer_folds = []
        self.inner_folds = []
        self._stratify = stratify
        self.shuffle = shuffle

        self.n_outer_folds = n_outer_folds
        self.n_inner_folds = n_inner_folds

        self.seed = seed
        if not shuffle and seed is not None:
            print(f"Shuffle is {shuffle}, seed set to None.")
            self.seed = None

        self.inner_val_ratio = inner_val_ratio
        self.outer_val_ratio = outer_val_ratio
        self.test_ratio = test_ratio

    @property
    def stratify(self) -> bool:
        """
        Whether the splitter has to apply target stratification or not
        """
        return self._stratify

    def get_targets(
        self, dataset: mlwiz.data.dataset.DatasetInterface
    ) -> Tuple[bool, np.ndarray]:
        r"""
        Reads the entire dataset and returns the targets.

        Args:
            dataset (:class:`~mlwiz.data.dataset.DatasetInterface`):
                the dataset

        Returns:
            a tuple of two elements. The first element is a boolean, which
            is ``True`` if target values exist or an exception has not
            been thrown. The second value holds the actual targets or ``None``,
            depending on the first boolean value.
        """
        print("Extracting targets from dataset...")
        targets = []
        for sample in dataset:
            x, y = sample

            if y is None:
                print("No target found, skipping.")
                return False, None

            if not isinstance(y, torch.Tensor):
                y = torch.Tensor([y])

            targets.append(y)

        targets = torch.cat(targets, dim=0).numpy()
        print("Done.")
        return True, targets

    @classmethod
    def load(cls, path: str):
        r"""
        Loads the data splits from disk.

        Args:
            :param path: the path of the yaml file with the splits

        Returns:
            a :class:`~mlwiz.data.splitter.Splitter` object
        """
        splits = dill_load(path)

        splitter_classname = splits.get("splitter_class", "Splitter")
        splitter_class = s2c(splitter_classname)

        splitter_args = splits.get("splitter_args")
        splitter = splitter_class(**splitter_args)

        assert splitter.n_outer_folds == len(splits["outer_folds"])
        assert splitter.n_inner_folds == len(splits["inner_folds"][0])

        for fold_data in splits["outer_folds"]:
            splitter.outer_folds.append(
                OuterFold(
                    fold_data["train"],
                    val_idxs=fold_data["val"],
                    test_idxs=fold_data["test"],
                )
            )

        for inner_split in splits["inner_folds"]:
            inner_split_data = []
            for fold_data in inner_split:
                inner_split_data.append(
                    InnerFold(fold_data["train"], val_idxs=fold_data["val"])
                )
            splitter.inner_folds.append(inner_split_data)

        return splitter

    def _get_splitter(
        self, n_splits: int, stratified: bool, eval_ratio: float
    ):
        r"""
        Instantiates the appropriate splitter to use depending on the situation

        Args:
            n_splits (int): the number of different splits to create
            stratified (bool): whether to perform stratification.
                **Works with classification tasks only!**
            eval_ratio (float): the amount of evaluation (validation/test)
                data to use in case ``n_splits==1`` (i.e., hold-out data split)

        Returns:
            a :class:`~mlwiz.data.splitter.Splitter` object
        """
        if n_splits == 1:
            if not self.shuffle:
                assert (
                    stratified is False
                ), "Stratified not implemented when shuffle is False"
                splitter = _NoShuffleTrainTestSplit(test_ratio=eval_ratio)
            else:
                if stratified:
                    splitter = StratifiedShuffleSplit(
                        n_splits, test_size=eval_ratio, random_state=self.seed
                    )
                else:
                    splitter = ShuffleSplit(
                        n_splits, test_size=eval_ratio, random_state=self.seed
                    )
        elif n_splits > 1:
            if stratified:
                splitter = StratifiedKFold(
                    n_splits, shuffle=self.shuffle, random_state=self.seed
                )
            else:
                splitter = KFold(
                    n_splits, shuffle=self.shuffle, random_state=self.seed
                )
        else:
            raise ValueError(f"'n_splits' must be >=1, got {n_splits}")

        return splitter

    def split(
        self,
        dataset: mlwiz.data.dataset.DatasetInterface,
        targets: np.ndarray = None,
    ):
        r"""
        Computes the splits and stores them in the list fields
        ``self.outer_folds`` and ``self.inner_folds``.
        IMPORTANT: calling split() sets the seed of numpy, torch, and
        random for reproducibility.

        Args:
            dataset (:class:`~mlwiz.data.dataset.DatasetInterface`):
                the Dataset object
            targets (np.ndarray]): targets used for stratification.
                Default is ``None``
        """
        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed(self.seed)
            random.seed(self.seed)

        idxs = range(len(dataset))

        stratified = self.stratify
        outer_idxs = np.array(idxs)

        outer_splitter = self._get_splitter(
            n_splits=self.n_outer_folds,
            stratified=stratified,
            eval_ratio=self.test_ratio,
        )  # This is the true test (outer test)

        for train_idxs, test_idxs in outer_splitter.split(
            outer_idxs, y=targets if stratified else None
        ):
            assert set(train_idxs) == set(outer_idxs[train_idxs])
            assert set(test_idxs) == set(outer_idxs[test_idxs])

            inner_fold_splits = []
            inner_idxs = outer_idxs[
                train_idxs
            ]  # equals train_idxs because outer_idxs was ordered
            inner_targets = (
                targets[train_idxs] if targets is not None else None
            )

            inner_splitter = self._get_splitter(
                n_splits=self.n_inner_folds,
                stratified=stratified,
                eval_ratio=self.inner_val_ratio,
            )  # The inner "test" is, instead, the validation set

            for inner_train_idxs, inner_val_idxs in inner_splitter.split(
                inner_idxs, y=inner_targets if stratified else None
            ):
                inner_fold = InnerFold(
                    train_idxs=inner_idxs[inner_train_idxs].tolist(),
                    val_idxs=inner_idxs[inner_val_idxs].tolist(),
                )
                inner_fold_splits.append(inner_fold)

                # False if empty
                assert not bool(
                    set(inner_train_idxs)
                    & set(inner_val_idxs)
                    & set(test_idxs)
                )
                assert not bool(
                    set(inner_idxs[inner_train_idxs])
                    & set(inner_idxs[inner_val_idxs])
                    & set(test_idxs)
                )

            self.inner_folds.append(inner_fold_splits)

            # Obtain outer val from outer train in an holdout fashion
            outer_val_splitter = self._get_splitter(
                n_splits=1,
                stratified=stratified,
                eval_ratio=self.outer_val_ratio,
            )
            outer_train_idxs, outer_val_idxs = list(
                outer_val_splitter.split(inner_idxs, y=inner_targets)
            )[0]

            # False if empty
            assert not bool(
                set(outer_train_idxs) & set(outer_val_idxs) & set(test_idxs)
            )
            assert not bool(
                set(outer_train_idxs) & set(outer_val_idxs) & set(test_idxs)
            )
            assert not bool(
                set(inner_idxs[outer_train_idxs])
                & set(inner_idxs[outer_val_idxs])
                & set(test_idxs)
            )

            np.random.shuffle(outer_train_idxs)
            np.random.shuffle(outer_val_idxs)
            np.random.shuffle(test_idxs)
            outer_fold = OuterFold(
                train_idxs=inner_idxs[outer_train_idxs].tolist(),
                val_idxs=inner_idxs[outer_val_idxs].tolist(),
                test_idxs=outer_idxs[test_idxs].tolist(),
            )
            self.outer_folds.append(outer_fold)

    def check_splits_overlap(self, skip_check: bool = False):
        """
        Checks if the splits created are non-overlapping or overlapping.
        If overlapping, an error message is returned.
        :param skip_check: whether to skip this check
        """
        if skip_check:
            print("User asked to skip data checking data splits for overlaps.")
        else:
            err_msg = "Data splits overlap! Please check your splitter code for errors."

            for outer_fold in self.outer_folds:
                assert set(outer_fold.train_idxs).isdisjoint(
                    set(outer_fold.test_idxs)
                ), err_msg
                assert set(outer_fold.val_idxs).isdisjoint(
                    set(outer_fold.test_idxs)
                ), err_msg
                assert set(outer_fold.train_idxs).isdisjoint(
                    set(outer_fold.val_idxs)
                ), err_msg

            for inner_fold_list in self.inner_folds:
                for inner_fold in inner_fold_list:
                    assert set(inner_fold.train_idxs).isdisjoint(
                        set(inner_fold.val_idxs)
                    ), err_msg
                    assert (
                        inner_fold.test_idxs is None
                    ), "Test indices should not be present in the inner folds."
            print("Check data splits not overlapping: passed.")

    def _splitter_args(self) -> dict:
        r"""
        Returns a dict with all the splitter's arguments for subsequent
        re-loading at experiment time.

        Returns:
            a dict containing all splitter's arguments.
        """
        return {
            "n_outer_folds": self.n_outer_folds,
            "n_inner_folds": self.n_inner_folds,
            "seed": self.seed,
            "stratify": self.stratify,
            "shuffle": self.shuffle,
            "inner_val_ratio": self.inner_val_ratio,
            "outer_val_ratio": self.outer_val_ratio,
            "test_ratio": self.test_ratio,
        }

    def save(self, path: str):
        r"""
        Saves the split as a dictionary into a ``torch`` file. The arguments
        of the dictionary are
        * seed (int)
        * splitter_class (str)
        * splitter_args (dict)
        * outer_folds (list of dicts)
        * inner_folds (list of lists of dicts)

        Args:
            path (str): filepath where to save the object
        """
        print("Saving splits on disk...")

        # save split class name
        module = self.__class__.__module__
        splitter_class = self.__class__.__qualname__
        if module is not None and module != "__builtin__":
            splitter_class = module + "." + splitter_class

        savedict = {
            "seed": self.seed,
            "splitter_class": splitter_class,
            "splitter_args": self._splitter_args(),
            "outer_folds": [o.todict() for o in self.outer_folds],
            "inner_folds": [],
        }

        for (
            inner_split
        ) in self.inner_folds:  # len(self.inner_folds) == # of **outer** folds
            savedict["inner_folds"].append([i.todict() for i in inner_split])
        dill_save(savedict, path)
        print("Done.")


class SingleGraphSplitter(Splitter):
    r"""
    A splitter for a single graph dataset that randomly splits nodes into
    training/validation/test

    Args:
        n_outer_folds (int): number of outer folds (risk assessment).
            1 means hold-out, >1 means k-fold
        n_inner_folds (int): number of inner folds (model selection).
            1 means hold-out, >1 means k-fold
        seed (int): random seed for reproducibility (on the same machine)
        stratify (bool): whether to apply stratification or not
            (should be true for classification tasks)
        shuffle (bool): whether to apply shuffle or not
        inner_val_ratio  (float): percentage of validation set for
            hold_out model selection. Default is ``0.1``
        outer_val_ratio  (float): percentage of validation set for
            hold_out model assessment (final training runs).
            Default is ``0.1``
        test_ratio  (float): percentage of test set for
            hold_out model assessment. Default is ``0.1``
    """

    def split(
        self,
        dataset: mlwiz.data.dataset.DatasetInterface,
        targets: np.ndarray = None,
    ):
        r"""
        Compared with the superclass version, the only difference is that the
        range of indices spans across the number of nodes of the single
        graph taken into consideration.

        Args:
            dataset (:class:`~mlwiz.data.dataset.DatasetInterface`):
                the Dataset object
            targets (np.ndarray]): targets used for stratification.
                Default is ``None``
        """
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        random.seed(self.seed)

        assert (
            len(dataset) == 1
        ), "This class works only with single graph datasets"
        idxs = range(dataset[0][0].x.shape[0])
        stratified = self.stratify
        outer_idxs = np.array(idxs)

        # all this code below could be reused from the original splitter
        outer_splitter = self._get_splitter(
            n_splits=self.n_outer_folds,
            stratified=stratified,
            eval_ratio=self.test_ratio,
        )  # This is the true test (outer test)

        for train_idxs, test_idxs in outer_splitter.split(
            outer_idxs, y=targets
        ):
            assert set(train_idxs) == set(outer_idxs[train_idxs])
            assert set(test_idxs) == set(outer_idxs[test_idxs])

            inner_fold_splits = []
            inner_idxs = outer_idxs[
                train_idxs
            ]  # equals train_idxs because outer_idxs was ordered
            inner_targets = (
                targets[train_idxs] if targets is not None else None
            )

            inner_splitter = self._get_splitter(
                n_splits=self.n_inner_folds,
                stratified=stratified,
                eval_ratio=self.inner_val_ratio,
            )  # The inner "test" is, instead, the validation set

            for inner_train_idxs, inner_val_idxs in inner_splitter.split(
                inner_idxs, y=inner_targets
            ):
                inner_fold = InnerFold(
                    train_idxs=inner_idxs[inner_train_idxs].tolist(),
                    val_idxs=inner_idxs[inner_val_idxs].tolist(),
                )
                inner_fold_splits.append(inner_fold)

                # False if empty
                assert not bool(
                    set(inner_train_idxs)
                    & set(inner_val_idxs)
                    & set(test_idxs)
                )
                assert not bool(
                    set(inner_idxs[inner_train_idxs])
                    & set(inner_idxs[inner_val_idxs])
                    & set(test_idxs)
                )

            self.inner_folds.append(inner_fold_splits)

            # Obtain outer val from outer train in an holdout fashion
            outer_val_splitter = self._get_splitter(
                n_splits=1,
                stratified=stratified,
                eval_ratio=self.outer_val_ratio,
            )
            outer_train_idxs, outer_val_idxs = list(
                outer_val_splitter.split(inner_idxs, y=inner_targets)
            )[0]

            # False if empty
            assert not bool(
                set(outer_train_idxs) & set(outer_val_idxs) & set(test_idxs)
            )
            assert not bool(
                set(outer_train_idxs) & set(outer_val_idxs) & set(test_idxs)
            )
            assert not bool(
                set(inner_idxs[outer_train_idxs])
                & set(inner_idxs[outer_val_idxs])
                & set(test_idxs)
            )

            np.random.shuffle(outer_train_idxs)
            np.random.shuffle(outer_val_idxs)
            np.random.shuffle(test_idxs)
            outer_fold = OuterFold(
                train_idxs=inner_idxs[outer_train_idxs].tolist(),
                val_idxs=inner_idxs[outer_val_idxs].tolist(),
                test_idxs=outer_idxs[test_idxs].tolist(),
            )
            self.outer_folds.append(outer_fold)
