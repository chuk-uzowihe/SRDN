"""Depthwise causal short conv + per-projection front-end (used by the SRDN mixer).

Exact forward / chunk-continuation / single-step parity via a (K-1)-frame ring buffer.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalDWConv1d(nn.Module):
    def __init__(self, channels: int, kernel: int) -> None:
        super().__init__()
        self.kernel = int(kernel)
        self.channels = int(channels)
        self.conv = nn.Conv1d(channels, channels, self.kernel, groups=channels,
                              padding=self.kernel - 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:        # [B,T,C]
        T = x.shape[1]
        return self.conv(x.transpose(1, 2))[..., :T].transpose(1, 2)

    def forward_cont(self, x: torch.Tensor, hist: torch.Tensor):
        """Chunk-correct continuation: prepend the carried (K-1)-frame history as real
        left-context, valid-conv to exactly T outputs, return the new history."""
        xin = torch.cat([hist, x], dim=1)
        y = F.conv1d(xin.transpose(1, 2), self.conv.weight, self.conv.bias,
                     padding=0, groups=self.channels)
        return y.transpose(1, 2), xin[:, -(self.kernel - 1):, :]

    def init_hist(self, B: int, device) -> torch.Tensor:
        return torch.zeros(B, self.kernel - 1, self.channels, device=device)

    def step(self, x_t: torch.Tensor, hist: torch.Tensor):     # x_t [B,C]
        window = torch.cat([hist, x_t.unsqueeze(1)], dim=1)
        y = torch.einsum("bkc,ck->bc", window, self.conv.weight[:, 0, :]) + self.conv.bias
        return y, window[:, 1:, :]


class QKVFeature(nn.Module):
    """Linear -> optional depthwise causal short conv. Returns LINEAR float features
    [..., H, dh]; SiLU + L2 + state-conditioning are applied in the mixer step."""

    def __init__(self, d_in: int, H: int, dh: int, *, short_conv: bool, conv_size: int) -> None:
        super().__init__()
        self.H, self.dh = H, dh
        self.proj = nn.Linear(d_in, H * dh, bias=False)
        self.conv = CausalDWConv1d(H * dh, conv_size) if short_conv else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.proj(x)
        if self.conv is not None:
            z = self.conv(z)
        return z.view(*x.shape[:-1], self.H, self.dh).float()

    def forward_with_hist(self, x: torch.Tensor, hist):
        z = self.proj(x)
        if self.conv is not None:
            z, hist = self.conv.forward_cont(z, hist)
        return z.view(*x.shape[:-1], self.H, self.dh).float(), hist

    def init_hist(self, B: int, device):
        return None if self.conv is None else self.conv.init_hist(B, device)

    def step(self, x_t: torch.Tensor, hist):
        z = self.proj(x_t)
        if self.conv is not None:
            z, hist = self.conv.step(z, hist)
        return z.view(x_t.shape[0], self.H, self.dh).float(), hist


__all__ = ["CausalDWConv1d", "QKVFeature"]
