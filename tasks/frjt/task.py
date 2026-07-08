"""FRJT (Flip-Register Jump Table): a non-associative state-tracking diagnostic
following arXiv:2510.06828.

A program is a list of blocks; each block flips (XORs) a 1-bit register then jumps by
one of its two forward offsets, selected by the current register value ("each block
performs two jumps, only one of which will execute depending on a condition
evaluation"). The halt class is which terminal the program counter lands in (pc ==
depth vs depth+1). Predicting it requires tracking (pc, register) through the
data-dependent control flow -- the state space grows with depth, so chunk-parallel
(TC0) models hit accuracy cliffs as depth grows. Paper-faithful training mixes
program depths uniformly in [1, depth_max]: short programs give training signal for
the transition function at every scale (the paper's substitute for register
supervision); dense_supervision is available but off in the paper-faithful protocol.

NOTE (accuracy floor): the label is necessarily a function of visible tokens, and the
FINAL block's offsets alone give ~0.64 accuracy without any state tracking (when its
two offsets agree, the outcome is register-independent; when they differ, it halts A
with the base rate). Read accuracies against this ~0.64 shortcut floor, not 0.5 --
a model at ~0.63 deep has learned the local shortcut, not partial state tracking.
`shortcut_floor()` measures it for a config.

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


def _sample_offsets(*, remaining, max_jump, rng):
    """Exponentially decaying jump length, decay 2/3 per step (closer jumps likelier ->
    maximize true depth; mean offset ~2.0 -> ~50% code coverage, scaling LINEARLY with
    depth -- no premature-halt mechanism, so effective depth never asymptotes as nominal
    depth grows). The two HALTING offsets, when in range (exact halt -> terminal A at
    pc==depth; overshoot -> terminal B at pc==depth+1), are held at EQUAL weight so the
    halt class stays ~50/50."""
    max_local = max(1, min(int(max_jump), int(remaining) + 1))
    options = np.arange(1, max_local + 1)
    weights = np.power(2.0 / 3.0, options.astype(np.float64) - 1.0)
    if options.size >= (remaining + 1):
        weights[remaining] = weights[max(0, remaining - 1)]
    weights = weights / np.maximum(weights.sum(), 1e-12)
    return int(rng.choice(options, p=weights)), int(rng.choice(options, p=weights))


def _execute(blocks, init_state):
    pc, reg, depth, executed = 0, int(init_state), len(blocks), 0
    while pc < depth:
        flip, off0, off1 = blocks[pc]
        reg ^= int(flip)
        pc = pc + (off1 if reg == 1 else off0)
        executed += 1
    # halt class = which terminal the program counter lands in. Offsets are capped at
    # remaining+1, so the halting pc is exactly depth (state A) or depth+1 (state B) --
    # "the program halts in either state A or B depending on the final location of the
    # program counter" (arXiv:2510.06828). The register drives the control flow only.
    return int(pc - depth), executed, pc


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
                                         rng=rng)
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


def shortcut_floor(*, cfg: FRJTTaskConfig, depth: int, n: int = 20_000, seed: int = 0) -> float:
    """Accuracy of the best LOCAL shortcut: majority-vote the label from the FINAL block's
    offset pair alone (no state tracking). ~0.64 at every depth -- the floor "chance"
    accuracy should be read against (see module docstring)."""
    rng = np.random.default_rng(seed)
    from collections import Counter, defaultdict
    votes, outcomes = defaultdict(Counter), []
    for _ in range(int(n)):
        init_state = int(rng.integers(0, 2))
        blocks = [(int(rng.integers(0, 2)),
                   *_sample_offsets(remaining=depth - i, max_jump=cfg.max_jump, rng=rng))
                  for i in range(int(depth))]
        label, _, _ = _execute(blocks, init_state)
        key = blocks[-1][1:]                       # the final block's (off0, off1)
        votes[key][label] += 1
        outcomes.append((key, label))
    majority = {k: c.most_common(1)[0][0] for k, c in votes.items()}
    return float(np.mean([majority[k] == lab for k, lab in outcomes]))


__all__ = ["FRJTTaskConfig", "TaskBatch", "frjt_vocab_size", "generate_frjt_batch",
           "shortcut_floor",
           "FRJT_PAD", "FRJT_START", "FRJT_INIT0", "FRJT_INIT1", "FRJT_FLIP0", "FRJT_FLIP1",
           "FRJT_JUMP_BASE"]
