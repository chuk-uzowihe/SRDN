"""enwik8 char-level (byte) language-model data.

enwik8 = first 100M bytes of English Wikipedia. Standard 90/5/5 split and metric
bits-per-byte (bpc) = CE_nats / ln 2. Byte-level, so vocab is exactly 256.
Download: http://mattmahoney.net/dc/enwik8.zip -> artifacts/data/enwik8
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

ENWIK8 = Path(__file__).resolve().parents[1] / "artifacts" / "data" / "enwik8"
VOCAB_SIZE = 256
_SPLITS = (90_000_000, 95_000_000)


def load_enwik8(path: Path = ENWIK8):
    """Return (train, val, test) as uint8 numpy arrays."""
    if not path.exists():
        raise FileNotFoundError(f"{path} missing -- download enwik8 (mattmahoney.net/dc/enwik8.zip)")
    data = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    a, b = _SPLITS
    return data[:a], data[a:b], data[b:]


def batch(split: np.ndarray, batch_size: int, seq_len: int, gen: torch.Generator, device) -> torch.Tensor:
    """Random contiguous windows -> long [B, seq_len+1] (input + shifted target)."""
    hi = len(split) - seq_len - 1
    ix = torch.randint(0, hi, (batch_size,), generator=gen).tolist()
    win = np.stack([split[i:i + seq_len + 1] for i in ix])
    return torch.from_numpy(win.astype(np.int64)).to(device)


__all__ = ["VOCAB_SIZE", "load_enwik8", "batch"]
