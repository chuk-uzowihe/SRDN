"""FRJT (Flip-Register Jump Table): a non-associative single-cell state-tracking
diagnostic from arXiv:2510.06828.

A program is a list of blocks; each block flips a 1-bit register then conditionally
jumps by an offset that DEPENDS on the current register value. Predicting the halt
class requires actually tracking the register through the data-dependent control
flow -- a recurrence-complete computation that chunk-parallel (TC0) models cannot do
as the jump-table depth grows. The model reads the token program and must output the
halt class at the final position (optional dense supervision labels the register
after every block).

Vocab: PAD, START, INIT0/1, FLIP0/1, then `max_jump` JUMP tokens.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

FRJT_PAD = 0
FRJT_START = 1
FRJT_INIT0 = 2
FRJT_INIT1 = 3
FRJT_FLIP0 = 4
FRJT_FLIP1 = 5
FRJT_JUMP_BASE = 6


@dataclass(frozen=True)
class FRJTTaskConfig:
    depth_min: int
    depth_max: int
    max_jump: int
    direct_halt_prob: float = 0.0
    dense_supervision: bool = False


@dataclass
class TaskBatch:
    inputs: torch.Tensor          # [B, T] token ids
    targets: torch.Tensor         # [B, T] class labels (ignore_index off-supervision)
    final_indices: torch.Tensor   # [B] position of the halt-class target
    final_targets: torch.Tensor   # [B] halt class
    stats: dict[str, float]


def frjt_vocab_size(max_jump: int) -> int:
    return FRJT_JUMP_BASE + int(max_jump)


def _sample_offsets(*, remaining, max_jump, direct_halt_prob, rng):
    max_local = max(1, min(int(max_jump), int(remaining) + 1))
    options = np.arange(1, max_local + 1)
    weights = 1.0 / options.astype(np.float64)
    if options.size >= remaining:
        weights[max(0, remaining - 1)] *= 1.5
    if options.size >= (remaining + 1):
        weights[remaining] *= 1.25
    if rng.random() < float(direct_halt_prob):
        if options.size >= remaining:
            weights[max(0, remaining - 1)] *= 4.0
        if options.size >= (remaining + 1):
            weights[remaining] *= 3.0
    weights = weights / np.maximum(weights.sum(), 1e-12)
    return int(rng.choice(options, p=weights)), int(rng.choice(options, p=weights))


def _execute(blocks, init_state):
    pc, reg, depth, executed = 0, int(init_state), len(blocks), 0
    while pc < depth:
        flip, off0, off1 = blocks[pc]
        reg ^= int(flip)
        pc = pc + (off1 if reg == 1 else off0)
        executed += 1
    return int((reg ^ (pc & 1)) & 1), executed, pc


def _scan_registers(blocks, init_state):
    """Path-independent left-to-right register labels (dense supervision must not
    reveal the executed jump path)."""
    reg, labels = int(init_state), []
    for flip, _, _ in blocks:
        reg ^= int(flip)
        labels.append(int(reg))
    return labels


def generate_frjt_batch(*, cfg: FRJTTaskConfig, batch_size: int, seed: int,
                        ignore_index: int = -100, depth_override: int | None = None) -> TaskBatch:
    rng = np.random.default_rng(seed)
    if depth_override is None:
        depths = rng.integers(cfg.depth_min, cfg.depth_max + 1, size=batch_size, dtype=np.int32)
    else:
        depths = np.full((batch_size,), int(depth_override), dtype=np.int32)
    max_len = 2 + 3 * int(np.max(depths))
    x = np.full((batch_size, max_len), FRJT_PAD, dtype=np.int64)
    y = np.full((batch_size, max_len), ignore_index, dtype=np.int64)
    final_idx = np.zeros((batch_size,), dtype=np.int64)
    final_tgt = np.zeros((batch_size,), dtype=np.int64)
    exec_counts = []

    for b in range(batch_size):
        depth = int(depths[b])
        init_state = int(rng.integers(0, 2))
        blocks = []
        for i in range(depth):
            flip = int(rng.integers(0, 2))
            off0, off1 = _sample_offsets(remaining=depth - i, max_jump=cfg.max_jump,
                                         direct_halt_prob=cfg.direct_halt_prob, rng=rng)
            blocks.append((flip, off0, off1))
        label, executed, _ = _execute(blocks, init_state)
        exec_counts.append(executed)
        tokens = [FRJT_START, FRJT_INIT1 if init_state == 1 else FRJT_INIT0]
        for flip, off0, off1 in blocks:
            tokens.append(FRJT_FLIP1 if flip == 1 else FRJT_FLIP0)
            tokens.append(FRJT_JUMP_BASE + max(1, min(cfg.max_jump, off0)) - 1)
            tokens.append(FRJT_JUMP_BASE + max(1, min(cfg.max_jump, off1)) - 1)
        L = len(tokens)
        x[b, :L] = np.asarray(tokens, dtype=np.int64)
        if cfg.dense_supervision:
            for i, reg_after in enumerate(_scan_registers(blocks, init_state)):
                y[b, 2 + i * 3] = int(reg_after)
        y[b, L - 1] = int(label)
        final_idx[b] = int(L - 1)
        final_tgt[b] = int(label)

    stats = {"mean_depth": float(np.mean(depths)),
             "mean_effective_depth": float(np.mean(exec_counts)),
             "class_balance": float(np.mean(final_tgt))}
    return TaskBatch(inputs=torch.from_numpy(x).long(), targets=torch.from_numpy(y).long(),
                     final_indices=torch.from_numpy(final_idx).long(),
                     final_targets=torch.from_numpy(final_tgt).long(), stats=stats)


__all__ = ["FRJTTaskConfig", "TaskBatch", "frjt_vocab_size", "generate_frjt_batch",
           "FRJT_PAD", "FRJT_START", "FRJT_INIT0", "FRJT_INIT1", "FRJT_FLIP0", "FRJT_FLIP1",
           "FRJT_JUMP_BASE"]
