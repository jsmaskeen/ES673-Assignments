import random
import time
from typing import Any, Optional

import torch
from torch import Tensor, nn


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def accuracy(prediction: Tensor, target: Tensor) -> float:
    return (prediction == target).float().mean().item()


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def get_device(use_gpu: bool = True) -> torch.device:
    if use_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def timed_run(
    name: str, fn, model: nn.Module, device: Optional[torch.device] = None
) -> tuple[float, dict[str, float], Any]:
    from utils.energy_tracker import EnergyTracker

    tracker = EnergyTracker(device=device)
    with tracker:
        result = fn()
    metrics = tracker.summary()
    print(
        f"[{name}] elapsed={metrics['elapsed_s']:.2f}s energy={metrics['energy_joules']:.2f}J avg_power={metrics['avg_power_w']:.2f}W peak_vram={metrics['peak_memory_mb']:.2f}MB params={parameter_count(model):,}"
    )
    return metrics["elapsed_s"], metrics, result
