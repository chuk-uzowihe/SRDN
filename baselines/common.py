"""Shared bits for the baseline wrappers.

RMSNorm here matches the one the baselines were validated with (eps 1e-6); SRDN
itself uses srdn.norm.RMSNorm. Each architecture keeps its own validated norm.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
REFS = REPO_ROOT / "refs"


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        return (xf * torch.rsqrt(xf.square().mean(dim=-1, keepdim=True) + self.eps) * self.weight).to(x.dtype)


__all__ = ["RMSNorm", "REPO_ROOT", "REFS"]
