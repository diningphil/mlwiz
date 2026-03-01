"""Custom samplers used by MLWiz data loaders.

Includes :class:`~mlwiz.data.sampler.RandomSampler`, which records the applied permutation.
"""

import math

import torch
from torch.utils.data import sampler
from torch.utils.data.distributed import DistributedSampler

from mlwiz.data.dataset import DatasetInterface


class RandomSampler(sampler.RandomSampler):
    """
    This sampler wraps the dataset and saves the random permutation applied
    to the samples, so that it will be available
    for further use (e.g. for saving embeddings in the original samples order).
    The permutation is saved in the 'permutation' attribute.

    Args:
        data_source (:class:`mlwiz.data.DatasetInterface`): the dataset object
    """

    def __init__(self, data_source: DatasetInterface):
        """
        Initialize the sampler and reset the stored permutation.

        Args:
            data_source (DatasetInterface): Dataset to sample from.

        Side effects:
            Sets ``self.permutation`` to ``None``; it will be populated on the
            next call to ``__iter__``.
        """
        super().__init__(data_source)
        self.permutation = None

    def __iter__(self):
        """
        Iterates over the samples according to a pre-determined permutation.

        Returns:
             An iterable with permuted indices.

        """
        n = len(self.data_source)
        self.permutation = torch.randperm(n).tolist()
        return iter(self.permutation)


class DistributedRandomSampler(DistributedSampler):
    """
    Distributed sampler that stores both global and rank-local permutations.

    The ``permutation`` attribute mirrors :class:`RandomSampler` semantics and
    contains the global sampled order (potentially padded when ``drop_last`` is
    ``False``). ``local_permutation`` contains only the indices assigned to the
    current rank.
    """

    def __init__(
        self,
        dataset: DatasetInterface,
        num_replicas: int = None,
        rank: int = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ):
        """
        Initialize sampler and reset stored permutations.
        """
        super().__init__(
            dataset=dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
        )
        self.permutation = None
        self.local_permutation = None

    def __iter__(self):
        """
        Iterate over rank-local indices and store both global/local permutations.
        """
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        if not self.drop_last:
            padding_size = self.total_size - len(indices)
            if padding_size > 0:
                if padding_size <= len(indices):
                    indices += indices[:padding_size]
                else:
                    indices += (
                        indices * math.ceil(padding_size / len(indices))
                    )[:padding_size]
        else:
            indices = indices[: self.total_size]

        if len(indices) != self.total_size:
            raise RuntimeError(
                "DistributedRandomSampler computed an invalid global permutation "
                f"length: expected {self.total_size}, got {len(indices)}."
            )

        local_indices = indices[self.rank : self.total_size : self.num_replicas]
        if len(local_indices) != self.num_samples:
            raise RuntimeError(
                "DistributedRandomSampler computed an invalid local permutation "
                f"length: expected {self.num_samples}, got {len(local_indices)}."
            )

        self.permutation = list(indices)
        self.local_permutation = list(local_indices)

        return iter(local_indices)
