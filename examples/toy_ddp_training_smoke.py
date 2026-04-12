"""
Tiny DDP smoke test built on MLWiz training routines.

Run:
    python3 examples/toy_ddp_training_smoke.py
"""

import os
import socket

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler

from mlwiz.model.interface import ModelInterface
from mlwiz.static import MAIN_LOSS
from mlwiz.training.callback.engine_callback import EngineCallback
from mlwiz.training.callback.metric import MeanAbsoluteError, MeanSquareError
from mlwiz.training.callback.optimizer import Optimizer
from mlwiz.training.engine import TrainingEngine


class TinyRegressor(ModelInterface):
    """Small regression model for smoke testing."""

    def __init__(self, dim_input_features=4, dim_target=1):
        super().__init__(dim_input_features, dim_target, config={})
        self.net = torch.nn.Linear(dim_input_features, dim_target)

    def forward(self, x):
        out = self.net(x)
        return out, out


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _worker(rank: int, world_size: int, master_port: int):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    try:
        # Same dataset on every rank; DistributedSampler shards it.
        torch.manual_seed(7)
        x = torch.randn(1024, 4)
        y = (x.sum(dim=1, keepdim=True) * 0.5) + 0.1
        dataset = TensorDataset(x, y)

        sampler = DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=True
        )
        loader = DataLoader(dataset, batch_size=64, sampler=sampler)

        device = f"cuda:{rank}"
        model = TinyRegressor().to(device)
        ddp_model = DDP(model, device_ids=[rank], output_device=rank)

        loss = MeanSquareError(use_as_loss=True, device=device)
        scorer = MeanAbsoluteError(use_as_loss=False, device=device)
        optimizer = Optimizer(
            ddp_model, optimizer_class_name="torch.optim.SGD", lr=0.1
        )

        engine = TrainingEngine(
            engine_callback=EngineCallback,
            model=ddp_model,
            loss=loss,
            optimizer=optimizer,
            scorer=scorer,
            device=device,
            exp_path="/tmp/mlwiz_ddp_smoke",
        )

        train_loss, _, *_ = engine.train(
            train_loader=loader,
            validation_loader=None,
            test_loader=None,
            max_epochs=5,
            logger=None,
        )

        if rank == 0:
            print(
                "DDP smoke test finished. Final training loss:",
                float(train_loss[MAIN_LOSS]),
            )
    finally:
        dist.destroy_process_group()


def main():
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        print("Need at least 2 visible CUDA GPUs for this smoke test.")
        return

    world_size = 2
    master_port = _free_port()
    mp.spawn(_worker, args=(world_size, master_port), nprocs=world_size)


if __name__ == "__main__":
    main()
