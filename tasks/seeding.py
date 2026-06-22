"""Standardized seeding for full experiment reproducibility.

Every runner calls `set_seed(seed)` once at startup (seeds Python `random`, NumPy's
global RNG, torch CPU + all CUDA devices) and draws task data through explicitly
seeded generators (`numpy_rng` / `torch_generator`) rather than the global state,
so a (seed) fully determines a run.

We deliberately do NOT call torch.use_deterministic_algorithms(True): the triton /
mamba-ssm / fla kernels the baselines rely on have no deterministic variant, and
forcing it would error or fall back to far slower paths. Seeding the RNGs gives
run-to-run reproducibility modulo the small nondeterminism of those fused kernels.
"""
from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def numpy_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(int(seed))


def torch_generator(seed: int, device: str = "cpu") -> torch.Generator:
    return torch.Generator(device=device).manual_seed(int(seed))


__all__ = ["set_seed", "numpy_rng", "torch_generator"]
