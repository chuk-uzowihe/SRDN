#!/usr/bin/env python3
"""PyTorch/xma graph-RL runner for no-conv M2RNN."""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import srdn
from tasks.graph_rl import (
    GraphRLState,
    graph_shortest_distances,
    discounted_action_returns,
    final_suffix_action_weights,
    initial_tokens,
    legal_action_tokens,
    observation_tokens,
    position_group_advantages,
    sample_center_curriculum_initial_state,
    sample_episode_graphs,
    sample_graph_pool,
    sample_num_nodes,
    sample_initial_state,
    step_state,
)
from tasks.graph_traversal import GRAPH_GOAL, GRAPH_PAD, GraphTraversalVocab


def parse_int_list(value: str) -> list[int]:
    return [int(x) for x in value.split(",") if x.strip()]


def mean_recent_delta(values: list[float], window: int) -> tuple[float, float, float]:
    recent = values[-int(window) :]
    split = max(1, len(recent) // 2)
    early = recent[:split]
    late = recent[split:]
    early_mean = float(sum(early) / max(1, len(early)))
    late_mean = float(sum(late) / max(1, len(late)))
    return early_mean, late_mean, late_mean - early_mean


def process_token_lists(
    model: srdn.SRDNLM,
    states: list[torch.Tensor],
    token_lists: list[list[int]],
    *,
    pad_token: int,
    device: torch.device,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    bsz = len(token_lists)
    max_len = max((len(xs) for xs in token_lists), default=0)
    last_logits = torch.zeros((bsz, model.vocab_size), device=device)
    for pos in range(max_len):
        tok = torch.full((bsz,), int(pad_token), device=device, dtype=torch.long)
        mask = torch.zeros((bsz,), device=device, dtype=torch.bool)
        for row, toks in enumerate(token_lists):
            if pos < len(toks):
                tok[row] = int(toks[pos])
                mask[row] = True
        logits, states = model.step(tok, states, mask)
        last_logits = torch.where(mask[:, None], logits, last_logits)
    return last_logits, states


def sample_actions(logits: torch.Tensor, legal_lists: list[list[int]], temperature: float, greedy: bool) -> list[int]:
    actions: list[int] = []
    for row, legal in enumerate(legal_lists):
        legal_t = torch.tensor(legal, device=logits.device, dtype=torch.long)
        vals = logits[row, legal_t].float()
        if greedy or temperature <= 0.0:
            actions.append(int(legal_t[torch.argmax(vals)].item()))
        else:
            probs = torch.softmax(vals / float(temperature), dim=0)
            actions.append(int(legal_t[torch.multinomial(probs, 1)].item()))
    return actions


def add_node_bucket_metrics(
    metrics: dict[str, float],
    *,
    prefix: str,
    node_counts: np.ndarray,
    goals_by_episode: np.ndarray,
    action_counts: np.ndarray,
    unique_nodes: np.ndarray,
    split: int,
) -> None:
    nodes = np.asarray(node_counts, dtype=np.float32)
    goals = np.asarray(goals_by_episode, dtype=np.float32)
    actions = np.maximum(np.asarray(action_counts, dtype=np.float32), 1.0)
    unique = np.asarray(unique_nodes, dtype=np.float32)
    masks = {
        "small": nodes <= float(split),
        "large": nodes > float(split),
    }
    for name, mask in masks.items():
        count = int(mask.sum())
        metrics[f"{prefix}/{name}_episodes"] = float(count)
        if count == 0:
            continue
        metrics[f"{prefix}/{name}_mean_num_nodes"] = float(nodes[mask].mean())
        metrics[f"{prefix}/{name}_goals_reached_per_action"] = float((goals[mask] / actions[mask]).mean())
        metrics[f"{prefix}/{name}_goals_reached_per_episode"] = float(goals[mask].mean())
        metrics[f"{prefix}/{name}_unique_nodes_visited_per_episode"] = float(unique[mask].mean())
        metrics[f"{prefix}/{name}_unique_nodes_visited_per_action"] = float((unique[mask] / actions[mask]).mean())


def generate_rollout_batch(
    model: srdn.SRDNLM,
    *,
    graph_pool: list[Any] | None,
    graph_backend: str,
    rng: np.random.Generator,
    batch_size: int,
    num_nodes: int,
    node_sampling: str,
    node_log2_min: float,
    node_log2_max: float,
    min_episode_len: int,
    max_episode_len: int,
    task_mode: str,
    initial_goal_distance_scale: float,
    goal_distance_scale_growth: float,
    gamma: float,
    advantage_mode: str,
    tail_downweight_actions: int,
    tail_min_weight: float,
    temperature: float,
    greedy: bool,
    seq_pad_multiple: int,
    node_bucket_split: int,
    device: torch.device,
) -> dict[str, Any]:
    vocab = GraphTraversalVocab(num_nodes=int(num_nodes))
    states = model.init_states(int(batch_size), device)
    episode_lengths = rng.integers(int(min_episode_len), int(max_episode_len) + 1, size=(int(batch_size),), dtype=np.int32)
    node_counts = sample_num_nodes(
        rng,
        batch_size=int(batch_size),
        mode=str(node_sampling),
        fixed_num_nodes=int(num_nodes),
        log2_min=float(node_log2_min),
        log2_max=float(node_log2_max),
    )
    graph_sample_start = time.perf_counter()
    graphs = sample_episode_graphs(
        num_nodes=int(num_nodes) if str(node_sampling) == "fixed" else node_counts.tolist(),
        batch_size=int(batch_size),
        rng=rng,
        graph_pool=graph_pool,
        graph_backend=str(graph_backend),
    )
    graph_sample_sec = time.perf_counter() - graph_sample_start
    if str(task_mode) == "uniform":
        env_states: list[Any] = [sample_initial_state(graph, rng=rng) for graph in graphs]
    elif str(task_mode) == "center_curriculum":
        env_states = [
            sample_center_curriculum_initial_state(
                graph,
                rng=rng,
                initial_goal_distance_scale=float(initial_goal_distance_scale),
                goal_distance_scale_growth=float(goal_distance_scale_growth),
            )
            for graph in graphs
        ]
    else:
        raise ValueError(f"unsupported task_mode: {task_mode}")
    visited_nodes = [{int(s.current)} for s in env_states]
    token_lists = [initial_tokens(s, vocab) for s in env_states]
    _, states = process_token_lists(model, states, token_lists, pad_token=GRAPH_PAD, device=device)

    action_positions: list[list[int]] = [[] for _ in range(int(batch_size))]
    allowed_by_action: list[list[list[int]]] = [[] for _ in range(int(batch_size))]
    rewards_by_action = np.zeros((int(batch_size), int(max_episode_len)), dtype=np.float32)
    action_valid = np.zeros_like(rewards_by_action, dtype=bool)
    action_counts = np.zeros((int(batch_size),), dtype=np.int32)

    for decision in range(int(max_episode_len)):
        active = episode_lengths > decision
        obs_lists = [observation_tokens(s, vocab) if bool(active[row]) else [] for row, s in enumerate(env_states)]
        for row, obs in enumerate(obs_lists):
            token_lists[row].extend(obs)
        logits, states = process_token_lists(model, states, obs_lists, pad_token=GRAPH_PAD, device=device)
        legal_lists = [legal_action_tokens(env_states[row], vocab) if bool(active[row]) else [GRAPH_PAD] for row in range(int(batch_size))]
        sampled = sample_actions(logits, legal_lists, temperature=float(temperature), greedy=bool(greedy))
        action_step_tokens = [[] for _ in range(int(batch_size))]
        post_action_tokens = [[] for _ in range(int(batch_size))]
        for row in range(int(batch_size)):
            if not bool(active[row]):
                continue
            action = int(sampled[row])
            chosen = env_states[row].graph.internal_for_label(vocab.token_node_label(action))
            action_token_pos = len(token_lists[row])
            token_lists[row].append(action)
            action_positions[row].append(action_token_pos - 1)
            allowed_by_action[row].append(legal_lists[row])
            env_states[row], reward = step_state(env_states[row], action, vocab, rng=rng)
            visited_nodes[row].add(int(chosen))
            rewards_by_action[row, decision] = float(reward)
            action_valid[row, decision] = True
            action_counts[row] += 1
            action_step_tokens[row] = [action]
            if reward > 0.0:
                post = [GRAPH_GOAL, env_states[row].graph.token_for_internal(env_states[row].goal, vocab)]
                token_lists[row].extend(post)
                post_action_tokens[row] = post
        _, states = process_token_lists(model, states, action_step_tokens, pad_token=GRAPH_PAD, device=device)
        _, states = process_token_lists(model, states, post_action_tokens, pad_token=GRAPH_PAD, device=device)

    returns = discounted_action_returns(rewards_by_action, action_valid, gamma=float(gamma))
    advantages_by_action = position_group_advantages(returns, action_valid, mode=str(advantage_mode))
    weights_by_action = final_suffix_action_weights(
        action_valid,
        suffix_actions=int(tail_downweight_actions),
        min_weight=float(tail_min_weight),
        gamma=float(gamma),
    )
    max_tokens = max(len(xs) for xs in token_lists)
    if seq_pad_multiple > 1:
        max_tokens = int(math.ceil(float(max_tokens) / float(seq_pad_multiple)) * int(seq_pad_multiple))
    tokens = np.full((int(batch_size), max_tokens), GRAPH_PAD, dtype=np.int64)
    action_mask = np.zeros((int(batch_size), max_tokens - 1), dtype=np.float32)
    allowed_mask = np.zeros((int(batch_size), max_tokens - 1, vocab.vocab_size), dtype=bool)
    advantages = np.zeros((int(batch_size), max_tokens - 1), dtype=np.float32)
    action_weights = np.zeros((int(batch_size), max_tokens - 1), dtype=np.float32)
    action_indices = np.full((int(batch_size), max_tokens - 1), -1, dtype=np.int64)
    for row, toks in enumerate(token_lists):
        tokens[row, : len(toks)] = np.asarray(toks, dtype=np.int64)
        for aidx, target_pos in enumerate(action_positions[row]):
            action_mask[row, target_pos] = 1.0
            allowed_mask[row, target_pos, np.asarray(allowed_by_action[row][aidx], dtype=np.int64)] = True
            advantages[row, target_pos] = advantages_by_action[row, aidx]
            action_weights[row, target_pos] = weights_by_action[row, aidx]
            action_indices[row, target_pos] = int(aidx)

    goals_by_episode = rewards_by_action.sum(axis=1)
    goals_per_action = goals_by_episode / np.maximum(action_counts.astype(np.float32), 1.0)
    unique = np.asarray([len(x) for x in visited_nodes], dtype=np.float32)
    center_goal_distances = []
    current_goal_distances = []
    goal_distance_scales = []
    for state in env_states:
        center = getattr(state, "center", None)
        if center is None:
            continue
        distances = graph_shortest_distances(state.graph, int(center))
        center_goal_distances.append(float(distances[int(state.goal)]))
        goal_distance_scales.append(float(state.goal_distance_scale))
        current_distances = graph_shortest_distances(state.graph, int(state.current))
        current_goal_distances.append(float(current_distances[int(state.goal)]))
    metrics = {
        "rollout/goals_reached_per_action": float(goals_per_action.mean()),
        "rollout/goals_reached_per_episode": float(goals_by_episode.mean()),
        "rollout/reward_per_action": float(goals_per_action.mean()),
        "rollout/reward_per_episode": float(goals_by_episode.mean()),
        "rollout/unique_nodes_visited_per_episode": float(unique.mean()),
        "rollout/unique_nodes_visited_per_action": float((unique / np.maximum(action_counts.astype(np.float32), 1.0)).mean()),
        "rollout/mean_episode_len": float(episode_lengths.mean()),
        "rollout/mean_num_nodes": float(node_counts.mean()),
        "rollout/min_num_nodes": float(node_counts.min()),
        "rollout/max_num_nodes": float(node_counts.max()),
        "rollout/mean_tokens": float(np.mean([len(xs) for xs in token_lists])),
        "rollout/max_tokens": float(max_tokens),
        "rollout/action_tokens": float(action_mask.sum()),
        "rollout/graph_sample_sec": float(graph_sample_sec),
        "rollout/graph_sample_ms_per_episode": float(1000.0 * graph_sample_sec / max(1, int(batch_size))),
        "rollout/adv_mean": float(advantages[action_mask > 0].mean()) if np.any(action_mask > 0) else 0.0,
        "rollout/adv_std": float(advantages[action_mask > 0].std()) if np.any(action_mask > 0) else 0.0,
        "rollout/tail_weight_mean": float(action_weights[action_mask > 0].mean()) if np.any(action_mask > 0) else 0.0,
    }
    if center_goal_distances:
        metrics["rollout/mean_goal_distance_scale"] = float(np.mean(goal_distance_scales))
        metrics["rollout/mean_goal_center_distance"] = float(np.mean(center_goal_distances))
        metrics["rollout/max_goal_center_distance"] = float(np.max(center_goal_distances))
        metrics["rollout/mean_goal_current_distance"] = float(np.mean(current_goal_distances))
        metrics["rollout/max_goal_current_distance"] = float(np.max(current_goal_distances))
    add_node_bucket_metrics(
        metrics,
        prefix="rollout",
        node_counts=node_counts,
        goals_by_episode=goals_by_episode,
        action_counts=action_counts,
        unique_nodes=unique,
        split=int(node_bucket_split),
    )
    return {
        "tokens": torch.as_tensor(tokens, device=device, dtype=torch.long),
        "action_mask": torch.as_tensor(action_mask, device=device),
        "allowed_mask": torch.as_tensor(allowed_mask, device=device),
        "advantages": torch.as_tensor(advantages, device=device),
        "action_weights": torch.as_tensor(action_weights, device=device),
        "action_indices": torch.as_tensor(action_indices, device=device),
        "rewards_by_action": rewards_by_action,
        "action_valid": action_valid,
        "node_counts": node_counts,
        "unique_nodes": unique,
        "metrics": metrics,
    }


def move_batch_tensors(batch: dict[str, Any], device: torch.device | str) -> dict[str, Any]:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def micro_batch_sizes(global_batch_size: int, micro_batch_size: int) -> list[int]:
    global_batch_size = int(global_batch_size)
    micro_batch_size = int(micro_batch_size)
    if global_batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if micro_batch_size <= 0:
        micro_batch_size = global_batch_size
    if micro_batch_size > global_batch_size:
        micro_batch_size = global_batch_size
    return [min(micro_batch_size, global_batch_size - start) for start in range(0, global_batch_size, micro_batch_size)]


def resolve_micro_batch_size(
    *,
    global_batch_size: int,
    max_micro_batch_size: int,
    stage_len: int,
    auto_micro_batch: bool,
    micro_batch_action_budget: int,
    min_micro_batch_size: int,
) -> int:
    cap = int(max_micro_batch_size)
    if cap <= 0:
        cap = int(global_batch_size)
    cap = max(1, min(int(global_batch_size), cap))
    if not bool(auto_micro_batch):
        return cap
    budget = max(1, int(micro_batch_action_budget))
    by_length = budget // max(1, int(stage_len))
    floor = max(1, int(min_micro_batch_size))
    return max(1, min(cap, max(floor, by_length)))


def batch_weighted_action_count(batch: dict[str, Any]) -> torch.Tensor:
    return (batch["action_mask"] * batch["action_weights"]).sum()


def aggregate_micro_metrics(records: list[tuple[int, dict[str, float]]]) -> dict[str, float]:
    if not records:
        return {}
    total_batch = float(sum(size for size, _ in records))
    weighted_mean_keys = {
        "rollout/goals_reached_per_action",
        "rollout/goals_reached_per_episode",
        "rollout/reward_per_action",
        "rollout/reward_per_episode",
        "rollout/unique_nodes_visited_per_episode",
        "rollout/unique_nodes_visited_per_action",
        "rollout/mean_episode_len",
        "rollout/mean_num_nodes",
        "rollout/mean_tokens",
        "rollout/adv_mean",
        "rollout/adv_std",
        "rollout/tail_weight_mean",
        "rollout/mean_goal_distance_scale",
        "rollout/mean_goal_center_distance",
        "rollout/mean_goal_current_distance",
        "rollout/graph_sample_ms_per_episode",
    }
    sum_keys = {
        "loss",
        "pg_loss",
        "entropy",
        "mean_logp",
        "weighted_actions",
        "rollout/action_tokens",
        "rollout/graph_sample_sec",
        "entropy_penultimate_chunk_actions",
    }
    max_keys = {
        "rollout/max_tokens",
        "rollout/max_num_nodes",
        "rollout/max_goal_center_distance",
        "rollout/max_goal_current_distance",
    }
    min_keys = {
        "rollout/min_num_nodes",
    }
    out: dict[str, float] = {}
    all_keys = sorted({key for _, metrics in records for key in metrics})
    for key in all_keys:
        vals = [(size, metrics[key]) for size, metrics in records if key in metrics and math.isfinite(float(metrics[key]))]
        if not vals:
            continue
        if key == "entropy_penultimate_chunk":
            denom = sum(
                float(metrics.get("entropy_penultimate_chunk_actions", 0.0))
                for _, metrics in records
                if "entropy_penultimate_chunk" in metrics
            )
            if denom > 0.0:
                out[key] = float(
                    sum(
                        float(metrics["entropy_penultimate_chunk"]) * float(metrics.get("entropy_penultimate_chunk_actions", 0.0))
                        for _, metrics in records
                        if "entropy_penultimate_chunk" in metrics
                    )
                    / denom
                )
            else:
                out[key] = float(vals[0][1])
        elif key in sum_keys or key.endswith("_episodes"):
            out[key] = float(sum(float(value) for _, value in vals))
        elif key in max_keys:
            out[key] = float(max(float(value) for _, value in vals))
        elif key in min_keys:
            out[key] = float(min(float(value) for _, value in vals))
        elif key in weighted_mean_keys or key.endswith("_episodes") or key.endswith("_mean_num_nodes") or key.endswith("_goals_reached_per_action") or key.endswith("_goals_reached_per_episode") or key.endswith("_unique_nodes_visited_per_episode") or key.endswith("_unique_nodes_visited_per_action"):
            out[key] = float(sum(float(size) * float(value) for size, value in vals) / max(1.0, total_batch))
        else:
            out[key] = float(vals[0][1])
    return out


def policy_loss(
    model: srdn.SRDNLM,
    batch: dict[str, Any],
    *,
    entropy_coef: float,
    use_xma: bool,
    train_sequence_chunk_size: int,
    train_detach_boundaries: bool,
    train_remat_chunks: bool,
    stage_len: int,
    length_schedule_chunk_actions: int,
    loss_denominator: torch.Tensor | float | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    tokens = batch["tokens"]
    inputs = tokens[:, :-1]
    labels = tokens[:, 1:].clamp(0, model.vocab_size - 1)
    logits = model.chunked_logits(
        inputs,
        use_xma=use_xma,
        chunk_size=int(train_sequence_chunk_size),
        detach_boundaries=bool(train_detach_boundaries),
        remat_chunks=bool(train_remat_chunks),
    )
    allowed_mask = batch["allowed_mask"]
    masked_logits = logits.masked_fill(~allowed_mask, -1.0e9)
    log_probs = masked_logits - torch.logsumexp(masked_logits, dim=-1, keepdim=True)
    chosen = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    probs = torch.where(allowed_mask, log_probs.exp(), torch.zeros_like(log_probs))
    entropy = -(probs * torch.where(allowed_mask, log_probs, torch.zeros_like(log_probs))).sum(dim=-1)
    weights = batch["action_mask"] * batch["action_weights"]
    local_denom = weights.sum()
    denom = local_denom.clamp_min(1.0)
    if loss_denominator is not None:
        denom = torch.as_tensor(loss_denominator, device=weights.device, dtype=weights.dtype).clamp_min(1.0)
    pg = -(chosen * batch["advantages"] * weights).sum() / denom
    ent = (entropy * weights).sum() / denom
    loss = pg - float(entropy_coef) * ent
    chunk_actions = max(1, int(length_schedule_chunk_actions))
    current_chunks = max(1, int(math.ceil(float(max(1, int(stage_len))) / float(chunk_actions))))
    penultimate_chunk = max(0, current_chunks - 2)
    chunk_start = int(penultimate_chunk * chunk_actions)
    chunk_end = int(min(max(1, int(stage_len)), chunk_start + chunk_actions))
    action_indices = batch["action_indices"]
    chunk_mask = (action_indices >= chunk_start) & (action_indices < chunk_end)
    chunk_weights = weights * chunk_mask.float()
    chunk_denom = chunk_weights.sum()
    chunk_entropy = (entropy * chunk_weights).sum() / chunk_denom.clamp_min(1.0)
    return loss, {
        "loss": float(loss.detach().cpu()),
        "pg_loss": float(pg.detach().cpu()),
        "entropy": float(ent.detach().cpu()),
        "entropy_penultimate_chunk": float(chunk_entropy.detach().cpu()),
        "entropy_penultimate_chunk_actions": float(chunk_denom.detach().cpu()),
        "entropy_penultimate_chunk_start": float(chunk_start),
        "entropy_penultimate_chunk_end": float(chunk_end),
        "mean_logp": float(((chosen * weights).sum() / denom).detach().cpu()),
        "weighted_actions": float(local_denom.detach().cpu()),
    }


def evaluate(model: srdn.SRDNLM, *, graph_pool: list[Any] | None, graph_backend: str, rng: np.random.Generator, num_nodes: int, node_sampling: str, node_log2_min: float, node_log2_max: float, node_bucket_split: int, episode_lengths: list[int], episodes: int, task_mode: str, initial_goal_distance_scale: float, goal_distance_scale_growth: float, gamma: float, temperature: float, greedy: bool, prefix: str, device: torch.device) -> dict[str, float]:
    out: dict[str, float] = {}
    for length in episode_lengths:
        batch = generate_rollout_batch(
            model,
            graph_pool=graph_pool,
            graph_backend=str(graph_backend),
            rng=rng,
            batch_size=int(episodes),
            num_nodes=int(num_nodes),
            node_sampling=str(node_sampling),
            node_log2_min=float(node_log2_min),
            node_log2_max=float(node_log2_max),
            min_episode_len=int(length),
            max_episode_len=int(length),
            task_mode=str(task_mode),
            initial_goal_distance_scale=float(initial_goal_distance_scale),
            goal_distance_scale_growth=float(goal_distance_scale_growth),
            gamma=float(gamma),
            advantage_mode="none",
            tail_downweight_actions=0,
            tail_min_weight=1.0,
            temperature=float(temperature),
            greedy=bool(greedy),
            seq_pad_multiple=1,
            node_bucket_split=int(node_bucket_split),
            device=device,
        )
        goals = batch["rewards_by_action"].sum(axis=1)
        action_counts = batch["action_valid"].sum(axis=1)
        per_action = goals / np.maximum(action_counts, 1)
        out[f"{prefix}_len_{length}/goals_reached_per_action"] = float(per_action.mean())
        out[f"{prefix}_len_{length}/goals_reached_per_episode"] = float(goals.mean())
        out[f"{prefix}_len_{length}/reward_per_action"] = float(per_action.mean())
        out[f"{prefix}_len_{length}/reward_per_episode"] = float(goals.mean())
        out[f"{prefix}_len_{length}/unique_nodes_visited_per_episode"] = batch["metrics"]["rollout/unique_nodes_visited_per_episode"]
        out[f"{prefix}_len_{length}/unique_nodes_visited_per_action"] = batch["metrics"]["rollout/unique_nodes_visited_per_action"]
        out[f"{prefix}_len_{length}/mean_num_nodes"] = batch["metrics"]["rollout/mean_num_nodes"]
        out[f"{prefix}_len_{length}/max_tokens"] = float(batch["tokens"].shape[1])
        bucket_metrics: dict[str, float] = {}
        add_node_bucket_metrics(
            bucket_metrics,
            prefix=f"{prefix}_len_{length}",
            node_counts=batch["node_counts"],
            goals_by_episode=goals,
            action_counts=action_counts,
            unique_nodes=batch["unique_nodes"],
            split=int(node_bucket_split),
        )
        out.update(bucket_metrics)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="graph_rl_m2rnn_torch")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "artifacts" / "graph-rl-m2rnn-torch")
    parser.add_argument("--num-nodes", type=int, default=8)
    parser.add_argument("--node-sampling", choices=["fixed", "log_uniform_int"], default="fixed")
    parser.add_argument("--node-log2-min", type=float, default=3.0)
    parser.add_argument("--node-log2-max", type=float, default=6.0)
    parser.add_argument("--node-bucket-split", type=int, default=16)
    parser.add_argument("--graph-source", choices=["fresh", "pool"], default="fresh")
    parser.add_argument("--graph-backend", choices=["scipy", "python"], default="scipy")
    parser.add_argument("--graph-pool-size", type=int, default=64)
    parser.add_argument("--task-mode", choices=["uniform", "center_curriculum"], default="uniform")
    parser.add_argument("--initial-goal-distance-scale", type=float, default=0.25)
    parser.add_argument("--goal-distance-scale-growth", type=float, default=0.10)
    parser.add_argument("--train-steps", type=int, default=8000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--micro-batch-size", type=int, default=0)
    parser.add_argument("--auto-micro-batch", action="store_true")
    parser.add_argument("--micro-batch-action-budget", type=int, default=0)
    parser.add_argument("--min-micro-batch-size", type=int, default=1)
    parser.add_argument("--eval-episodes", type=int, default=128)
    parser.add_argument("--sampled-eval-episodes", type=int, default=128)
    parser.add_argument("--curriculum-lengths", default="32")
    parser.add_argument("--eval-lengths", default="32,64,128,256,512")
    parser.add_argument("--quick-eval-lengths", default="")
    parser.add_argument("--variable-length-frac", type=float, default=0.25)
    parser.add_argument("--length-schedule-mode", choices=["plateau", "penultimate_entropy", "score_threshold"], default="plateau")
    parser.add_argument("--length-schedule-chunk-actions", type=int, default=16)
    parser.add_argument("--max-curriculum-length", type=int, default=0)
    parser.add_argument("--entropy-advance-threshold", type=float, default=1.0)
    parser.add_argument("--advance-margin", type=float, default=0.02)
    parser.add_argument("--plateau-min-steps", type=int, default=1000)
    parser.add_argument("--plateau-window", type=int, default=5)
    parser.add_argument("--plateau-min-score", type=float, default=0.05)
    parser.add_argument("--plateau-score-epsilon", type=float, default=0.02)
    parser.add_argument("--plateau-entropy-epsilon", type=float, default=0.08)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--advantage-mode", choices=["none", "z", "loo", "loo_z"], default="loo_z")
    parser.add_argument("--tail-downweight-actions", type=int, default=4)
    parser.add_argument("--tail-min-weight", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--optimizer", choices=["adamw", "muon"], default="adamw")
    parser.add_argument("--architecture", choices=["srdn", "transformer", "mamba3", "m2rnn", "gdn2"], default="srdn")
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--ffn-mult", type=float, default=2.0)
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--srdn-short-conv", action="store_true",
                        help="fla-style depthwise short conv on srdn q/k/v (conv-on parity with gdn2/m2rnn)")
    parser.add_argument("--mamba-state", type=int, default=64)
    parser.add_argument("--mamba-head-dim", type=int, default=32)
    parser.add_argument("--m2rnn-head-dim", type=int, default=50)
    parser.add_argument("--m2rnn-kernel-size", type=int, default=4)
    parser.add_argument("--gdn2-head-dim", type=int, default=28)
    parser.add_argument("--gdn2-expand-v", type=float, default=1.0)
    parser.add_argument("--gdn2-repo", default=str(PROJECT_ROOT / "refs" / "GatedDeltaNet-2"))
    parser.add_argument("--seq-pad-multiple", type=int, default=128)
    parser.add_argument("--train-sequence-chunk-size", type=int, default=0)
    parser.add_argument("--train-remat-chunks", action="store_true")
    parser.add_argument("--train-detach-boundaries", action="store_true")
    parser.add_argument("--chunked-train-no-xma", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--full-eval-every", type=int, default=500)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--checkpoint-steps", default="")
    parser.add_argument("--max-elapsed-sec", type=float, default=0.0)
    parser.add_argument("--no-xma", action="store_true")
    parser.add_argument("--wandb-project", default="smalltime-graph-rl")
    parser.add_argument("--wandb-entity", default="")
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
    args = parser.parse_args()

    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    architecture = str(args.architecture)
    # Each mixer selects its own kernel internally (the M2RNN twin uses the xma kernel
    # on CUDA; GDN-2 forces its chunk kernel; SRDN is a pure scan). use_xma is kept as
    # a no-op flag on the policy surface for interface compatibility.
    use_xma = train_use_xma = False
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / f"{args.run_name}.jsonl"

    curriculum_lengths = parse_int_list(args.curriculum_lengths)
    if not curriculum_lengths:
        raise ValueError("--curriculum-lengths must contain at least one length")
    eval_lengths = parse_int_list(args.eval_lengths)
    quick_eval_lengths = parse_int_list(args.quick_eval_lengths)
    checkpoint_steps = set(parse_int_list(args.checkpoint_steps))
    length_schedule_chunk_actions = (
        int(args.length_schedule_chunk_actions)
        if int(args.length_schedule_chunk_actions) > 0
        else int(curriculum_lengths[0])
    )
    initial_curriculum_len = int(curriculum_lengths[0])
    max_curriculum_len = int(args.max_curriculum_length)
    if max_curriculum_len > 0 and max_curriculum_len < initial_curriculum_len:
        raise ValueError("--max-curriculum-length must be 0 or >= the initial curriculum length")
    max_micro_batch_size = int(args.micro_batch_size) if int(args.micro_batch_size) > 0 else int(args.batch_size)
    max_micro_batch_size = max(1, min(int(args.batch_size), max_micro_batch_size))
    micro_batch_action_budget = int(args.micro_batch_action_budget)
    if bool(args.auto_micro_batch) and micro_batch_action_budget <= 0:
        micro_batch_action_budget = int(max_micro_batch_size) * int(initial_curriculum_len)

    def curriculum_stage_len(index: int) -> int:
        if str(args.length_schedule_mode) in {"plateau", "penultimate_entropy"}:
            length = initial_curriculum_len + int(index) * int(length_schedule_chunk_actions)
            return min(length, max_curriculum_len) if max_curriculum_len > 0 else length
        return int(curriculum_lengths[int(index)])

    def has_next_curriculum_stage(index: int) -> bool:
        if str(args.length_schedule_mode) in {"plateau", "penultimate_entropy"}:
            return max_curriculum_len <= 0 or curriculum_stage_len(index) < max_curriculum_len
        return int(index) + 1 < len(curriculum_lengths)

    if str(args.length_schedule_mode) in {"plateau", "penultimate_entropy"}:
        if max_curriculum_len > 0:
            expanded = list(range(initial_curriculum_len, max_curriculum_len + 1, int(length_schedule_chunk_actions)))
            if expanded[-1] != max_curriculum_len:
                expanded.append(max_curriculum_len)
            expanded_curriculum_lengths = ",".join(str(x) for x in sorted(set(int(x) for x in expanded)))
        else:
            expanded_curriculum_lengths = f"{initial_curriculum_len},+{int(length_schedule_chunk_actions)}(unbounded)"
    else:
        expanded_curriculum_lengths = ",".join(str(x) for x in curriculum_lengths)
    if args.node_sampling != "fixed" and args.graph_source != "fresh":
        raise ValueError("variable node sampling requires --graph-source fresh")
    if args.node_sampling != "fixed" and int(args.num_nodes) < int(math.floor(2.0 ** float(args.node_log2_max))):
        raise ValueError("--num-nodes is the maximum vocab/graph size and must cover node-log2-max")
    vocab = GraphTraversalVocab(num_nodes=int(args.num_nodes))
    arch = str(args.architecture)
    V, d, L, H, ffn = vocab.vocab_size, int(args.d_model), int(args.layers), int(args.heads), float(args.ffn_mult)
    if arch in ("mamba3", "m2rnn", "gdn2") and device.type != "cuda":
        raise RuntimeError(f"--architecture {arch} requires CUDA (kernels are CUDA-only)")
    if arch == "srdn":
        model = srdn.build_srdn(V, d, L, H, d // H, ffn, short_conv=bool(args.srdn_short_conv))
    elif arch == "transformer":
        model = srdn.build_transformer(V, d, L, H, ffn, max_seq_len=int(args.max_seq_len))
    elif arch == "mamba3":
        model = srdn.build_mamba3(V, d, L, ffn, state_size=int(args.mamba_state),
                                  head_dim=int(args.mamba_head_dim))
    elif arch == "m2rnn":
        model = srdn.build_m2rnn(V, d, L, H, int(args.m2rnn_head_dim), ffn,
                                 kernel_size=int(args.m2rnn_kernel_size))
    elif arch == "gdn2":
        model = srdn.build_gdn2(V, d, L, H, int(args.gdn2_head_dim), ffn, expand_v=float(args.gdn2_expand_v),
                                gdn2_repo=args.gdn2_repo)
    else:
        raise ValueError(f"unknown architecture {arch}")
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    non_embedding_params = sum(p.numel() for name, p in model.named_parameters() if not name.startswith("embed."))
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.learning_rate), weight_decay=float(args.weight_decay))
    config = {
        **vars(args),
        "device": str(device),
        "use_xma": use_xma,
        "train_use_xma": train_use_xma,
        "vocab_size": vocab.vocab_size,
        "total_params": int(total_params),
        "non_embedding_params": int(non_embedding_params),
        "expanded_curriculum_lengths": expanded_curriculum_lengths,
        "resolved_length_schedule_chunk_actions": int(length_schedule_chunk_actions),
        "max_micro_batch_size": int(max_micro_batch_size),
        "micro_batch_action_budget": int(micro_batch_action_budget),
        "initial_micro_batch_size": int(
            resolve_micro_batch_size(
                global_batch_size=int(args.batch_size),
                max_micro_batch_size=int(max_micro_batch_size),
                stage_len=int(initial_curriculum_len),
                auto_micro_batch=bool(args.auto_micro_batch),
                micro_batch_action_budget=int(micro_batch_action_budget),
                min_micro_batch_size=int(args.min_micro_batch_size),
            )
        ),
        "initial_grad_accum_steps": int(
            len(
                micro_batch_sizes(
                    int(args.batch_size),
                    resolve_micro_batch_size(
                        global_batch_size=int(args.batch_size),
                        max_micro_batch_size=int(max_micro_batch_size),
                        stage_len=int(initial_curriculum_len),
                        auto_micro_batch=bool(args.auto_micro_batch),
                        micro_batch_action_budget=int(micro_batch_action_budget),
                        min_micro_batch_size=int(args.min_micro_batch_size),
                    ),
                )
            )
        ),
        "true_batch_size": int(args.batch_size),
    }
    with open(args.output_dir / f"{args.run_name}_config.json", "w") as f:
        json.dump({k: str(v) if isinstance(v, Path) else v for k, v in config.items()}, f, indent=2, sort_keys=True)
    print(json.dumps({"event": "config", **{k: str(v) if isinstance(v, Path) else v for k, v in config.items()}}), flush=True)

    rng = np.random.default_rng(int(args.seed))
    eval_rng = np.random.default_rng(int(args.seed) + 10_000)
    graph_pool = None
    if args.graph_source == "pool":
        graph_pool = sample_graph_pool(
            num_nodes=int(args.num_nodes),
            pool_size=int(args.graph_pool_size),
            seed=int(args.seed) + 1,
            graph_backend=str(args.graph_backend),
        )
    run = None
    if args.wandb_mode != "disabled":
        import wandb

        os.environ["WANDB_MODE"] = args.wandb_mode
        run = wandb.init(project=args.wandb_project, entity=args.wandb_entity or None, name=args.run_name, config=config, dir=str(args.output_dir))

    stage_idx = 0
    stage_start_step = 1
    stage_eval_scores: list[float] = []
    stage_eval_entropies: list[float] = []
    last_eval: dict[str, float] = {}
    start = time.time()
    for step in range(1, int(args.train_steps) + 1):
        model.eval()
        stage_len = int(curriculum_stage_len(stage_idx))
        min_len = max(1, int(math.floor(stage_len * (1.0 - float(args.variable_length_frac)))))
        step_micro_batch_size = resolve_micro_batch_size(
            global_batch_size=int(args.batch_size),
            max_micro_batch_size=int(max_micro_batch_size),
            stage_len=int(stage_len),
            auto_micro_batch=bool(args.auto_micro_batch),
            micro_batch_action_budget=int(micro_batch_action_budget),
            min_micro_batch_size=int(args.min_micro_batch_size),
        )
        train_micro_sizes = micro_batch_sizes(int(args.batch_size), int(step_micro_batch_size))
        grad_accum_steps = len(train_micro_sizes)
        if grad_accum_steps == 1:
            batch = generate_rollout_batch(
                model,
                graph_pool=graph_pool,
                graph_backend=str(args.graph_backend),
                rng=rng,
                batch_size=int(args.batch_size),
                num_nodes=int(args.num_nodes),
                node_sampling=str(args.node_sampling),
                node_log2_min=float(args.node_log2_min),
                node_log2_max=float(args.node_log2_max),
                min_episode_len=min_len,
                max_episode_len=stage_len,
                task_mode=str(args.task_mode),
                initial_goal_distance_scale=float(args.initial_goal_distance_scale),
                goal_distance_scale_growth=float(args.goal_distance_scale_growth),
                gamma=float(args.gamma),
                advantage_mode=args.advantage_mode,
                tail_downweight_actions=int(args.tail_downweight_actions),
                tail_min_weight=float(args.tail_min_weight),
                temperature=float(args.temperature),
                greedy=False,
                seq_pad_multiple=int(args.seq_pad_multiple),
                node_bucket_split=int(args.node_bucket_split),
                device=device,
            )
            model.train()
            opt.zero_grad(set_to_none=True)
            loss, loss_metrics = policy_loss(
                model,
                batch,
                entropy_coef=float(args.entropy_coef),
                use_xma=train_use_xma,
                train_sequence_chunk_size=int(args.train_sequence_chunk_size),
                train_detach_boundaries=bool(args.train_detach_boundaries),
                train_remat_chunks=bool(args.train_remat_chunks),
                stage_len=stage_len,
                length_schedule_chunk_actions=int(length_schedule_chunk_actions),
            )
            loss.backward()
            train_metrics = {**loss_metrics, **batch["metrics"]}
        else:
            micro_batches: list[tuple[int, dict[str, Any], dict[str, float]]] = []
            total_weighted_actions = 0.0
            for micro_size in train_micro_sizes:
                batch = generate_rollout_batch(
                    model,
                    graph_pool=graph_pool,
                    graph_backend=str(args.graph_backend),
                    rng=rng,
                    batch_size=int(micro_size),
                    num_nodes=int(args.num_nodes),
                    node_sampling=str(args.node_sampling),
                    node_log2_min=float(args.node_log2_min),
                    node_log2_max=float(args.node_log2_max),
                    min_episode_len=min_len,
                    max_episode_len=stage_len,
                    task_mode=str(args.task_mode),
                    initial_goal_distance_scale=float(args.initial_goal_distance_scale),
                    goal_distance_scale_growth=float(args.goal_distance_scale_growth),
                    gamma=float(args.gamma),
                    advantage_mode=args.advantage_mode,
                    tail_downweight_actions=int(args.tail_downweight_actions),
                    tail_min_weight=float(args.tail_min_weight),
                    temperature=float(args.temperature),
                    greedy=False,
                    seq_pad_multiple=int(args.seq_pad_multiple),
                    node_bucket_split=int(args.node_bucket_split),
                    device=device,
                )
                total_weighted_actions += float(batch_weighted_action_count(batch).detach().cpu())
                micro_batches.append((int(micro_size), move_batch_tensors(batch, torch.device("cpu")), dict(batch["metrics"])))
                del batch
            if device.type == "cuda":
                torch.cuda.empty_cache()
            model.train()
            opt.zero_grad(set_to_none=True)
            loss_metric_records: list[tuple[int, dict[str, float]]] = []
            loss_denominator = torch.tensor(float(total_weighted_actions), device=device, dtype=torch.float32).clamp_min(1.0)
            for micro_size, batch_cpu, rollout_metrics in micro_batches:
                batch = move_batch_tensors(batch_cpu, device)
                loss, loss_metrics = policy_loss(
                    model,
                    batch,
                    entropy_coef=float(args.entropy_coef),
                    use_xma=train_use_xma,
                    train_sequence_chunk_size=int(args.train_sequence_chunk_size),
                    train_detach_boundaries=bool(args.train_detach_boundaries),
                    train_remat_chunks=bool(args.train_remat_chunks),
                    stage_len=stage_len,
                    length_schedule_chunk_actions=int(length_schedule_chunk_actions),
                    loss_denominator=loss_denominator,
                )
                loss.backward()
                loss_metric_records.append((int(micro_size), {**loss_metrics, **rollout_metrics}))
                del batch, loss
            train_metrics = aggregate_micro_metrics(loss_metric_records)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        lr_scale = min(1.0, float(step) / max(1, int(args.warmup_steps)))
        for group in opt.param_groups:
            group["lr"] = float(args.learning_rate) * lr_scale
        opt.step()
        record = {
            "step": step,
            "elapsed_sec": time.time() - start,
            "curriculum/stage_idx": stage_idx,
            "curriculum/stage_len": stage_len,
            "curriculum/min_episode_len": min_len,
            "learning_rate": float(args.learning_rate) * lr_scale,
            "train/global_batch_size": float(args.batch_size),
            "train/micro_batch_size": float(step_micro_batch_size),
            "train/grad_accum_steps": float(grad_accum_steps),
            "train/micro_batch_action_budget": float(micro_batch_action_budget),
            **train_metrics,
        }
        record["curriculum/length_schedule_mode"] = str(args.length_schedule_mode)
        record["curriculum/length_schedule_chunk_actions"] = float(length_schedule_chunk_actions)
        if step % int(args.eval_every) == 0 or step == 1:
            model.eval()
            full_eval = (
                step == 1
                or step == int(args.train_steps)
                or (int(args.full_eval_every) > 0 and step % int(args.full_eval_every) == 0)
            )
            if full_eval:
                eval_lengths_this = eval_lengths
            else:
                eval_lengths_this = sorted(set([stage_len, *quick_eval_lengths]))
            with torch.no_grad():
                greedy_eval = evaluate(model, graph_pool=graph_pool, graph_backend=str(args.graph_backend), rng=eval_rng, num_nodes=int(args.num_nodes), node_sampling=str(args.node_sampling), node_log2_min=float(args.node_log2_min), node_log2_max=float(args.node_log2_max), node_bucket_split=int(args.node_bucket_split), episode_lengths=eval_lengths_this, episodes=int(args.eval_episodes), task_mode=str(args.task_mode), initial_goal_distance_scale=float(args.initial_goal_distance_scale), goal_distance_scale_growth=float(args.goal_distance_scale_growth), gamma=float(args.gamma), temperature=float(args.temperature), greedy=True, prefix="greedy_eval", device=device)
                sampled_eval = evaluate(model, graph_pool=graph_pool, graph_backend=str(args.graph_backend), rng=eval_rng, num_nodes=int(args.num_nodes), node_sampling=str(args.node_sampling), node_log2_min=float(args.node_log2_min), node_log2_max=float(args.node_log2_max), node_bucket_split=int(args.node_bucket_split), episode_lengths=eval_lengths_this, episodes=int(args.sampled_eval_episodes), task_mode=str(args.task_mode), initial_goal_distance_scale=float(args.initial_goal_distance_scale), goal_distance_scale_growth=float(args.goal_distance_scale_growth), gamma=float(args.gamma), temperature=float(args.temperature), greedy=False, prefix="sampled_eval", device=device)
            last_eval = {**greedy_eval, **sampled_eval}
            record.update(last_eval)
            record["eval/full"] = float(full_eval)
            record["eval/num_lengths"] = float(len(eval_lengths_this))
            current = last_eval.get(f"sampled_eval_len_{stage_len}/goals_reached_per_action", 0.0)
            record["curriculum/current_score"] = float(current)
            record["curriculum/advance_threshold"] = float(args.advance_margin)
            if str(args.length_schedule_mode) == "plateau":
                stage_eval_scores.append(float(current))
                stage_eval_entropies.append(float(train_metrics["entropy_penultimate_chunk"]))
        advance_reason = None
        if has_next_curriculum_stage(stage_idx):
            if str(args.length_schedule_mode) in {"plateau", "penultimate_entropy"}:
                current_entropy = float(train_metrics["entropy_penultimate_chunk"])
                entropy_actions = float(train_metrics["entropy_penultimate_chunk_actions"])
                record["curriculum/current_entropy"] = current_entropy
                record["curriculum/entropy_advance_threshold"] = float(args.entropy_advance_threshold)
                if str(args.length_schedule_mode) == "penultimate_entropy" and entropy_actions > 0.0 and current_entropy <= float(args.entropy_advance_threshold):
                    advance_reason = "penultimate_entropy"
                elif str(args.length_schedule_mode) == "plateau" and (step % int(args.eval_every) == 0 or step == 1):
                    stage_steps = int(step) - int(stage_start_step) + 1
                    window = max(2, int(args.plateau_window))
                    score_ready = (
                        bool(stage_eval_scores)
                        and float(stage_eval_scores[-1]) >= float(args.plateau_min_score)
                    )
                    min_steps_ready = stage_steps >= int(args.plateau_min_steps)
                    record["curriculum/stage_steps"] = float(stage_steps)
                    record["curriculum/plateau_min_steps"] = float(args.plateau_min_steps)
                    record["curriculum/plateau_window"] = float(window)
                    record["curriculum/plateau_min_score"] = float(args.plateau_min_score)
                    record["curriculum/plateau_score_epsilon"] = float(args.plateau_score_epsilon)
                    record["curriculum/plateau_entropy_epsilon"] = float(args.plateau_entropy_epsilon)
                    if len(stage_eval_scores) >= window:
                        score_early, score_late, score_delta = mean_recent_delta(stage_eval_scores, window)
                        entropy_early, entropy_late, entropy_delta = mean_recent_delta(stage_eval_entropies, window)
                        entropy_drop = -float(entropy_delta)
                        score_plateau = abs(float(score_delta)) <= float(args.plateau_score_epsilon)
                        entropy_plateau = abs(float(entropy_delta)) <= float(args.plateau_entropy_epsilon)
                        record["curriculum/plateau_score_early_mean"] = score_early
                        record["curriculum/plateau_score_late_mean"] = score_late
                        record["curriculum/plateau_score_delta"] = score_delta
                        record["curriculum/plateau_entropy_early_mean"] = entropy_early
                        record["curriculum/plateau_entropy_late_mean"] = entropy_late
                        record["curriculum/plateau_entropy_delta"] = entropy_delta
                        record["curriculum/plateau_entropy_drop"] = entropy_drop
                        record["curriculum/plateau_score_ready"] = float(score_ready)
                        record["curriculum/plateau_min_steps_ready"] = float(min_steps_ready)
                        record["curriculum/plateau_score_plateau"] = float(score_plateau)
                        record["curriculum/plateau_entropy_plateau"] = float(entropy_plateau)
                        if (
                            entropy_actions > 0.0
                            and min_steps_ready
                            and score_ready
                            and current_entropy <= float(args.entropy_advance_threshold)
                        ):
                            advance_reason = "plateau_entropy_floor"
                        elif min_steps_ready and score_ready and score_plateau and entropy_plateau:
                            advance_reason = "plateau"
            elif step % int(args.eval_every) == 0 or step == 1:
                current = last_eval.get(f"sampled_eval_len_{stage_len}/goals_reached_per_action", 0.0)
                if current >= float(args.advance_margin):
                    advance_reason = "score_threshold"
        if advance_reason is not None:
            stage_idx += 1
            stage_start_step = int(step) + 1
            stage_eval_scores = []
            stage_eval_entropies = []
            record["curriculum/advanced_to_stage_idx"] = stage_idx
            record["curriculum/advance_reason"] = advance_reason
            print(
                json.dumps(
                    {
                        "event": "curriculum_advance",
                        "step": step,
                        "new_stage_idx": stage_idx,
                        "new_stage_len": curriculum_stage_len(stage_idx),
                        "reason": advance_reason,
                        "entropy": float(train_metrics["entropy_penultimate_chunk"]),
                        "entropy_threshold": float(args.entropy_advance_threshold),
                    }
                ),
                flush=True,
            )
        if step % int(args.log_every) == 0 or step == 1:
            with open(metrics_path, "a") as f:
                f.write(json.dumps(record, sort_keys=True) + "\n")
            print(json.dumps(record, sort_keys=True), flush=True)
            if run is not None:
                run.log(record, step=step)
        should_checkpoint = (
            step == int(args.train_steps)
            or step in checkpoint_steps
            or (int(args.checkpoint_every) > 0 and step % int(args.checkpoint_every) == 0)
        )
        if should_checkpoint:
            ckpt_dir = args.output_dir / "checkpoints" / args.run_name
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            path = ckpt_dir / f"step_{step:08d}.pt"
            torch.save({"step": step, "model": model.state_dict(), "optimizer": opt.state_dict(), "args": vars(args)}, path)
            if run is not None:
                run.summary["latest_checkpoint"] = str(path)
        if float(args.max_elapsed_sec) > 0.0 and (time.time() - start) >= float(args.max_elapsed_sec):
            print(json.dumps({"event": "max_elapsed_sec_reached", "step": step, "elapsed_sec": time.time() - start}), flush=True)
            break

    final = {"event": "done", "steps": int(step), "elapsed_sec": time.time() - start, "final_stage_idx": stage_idx, "final_stage_len": int(curriculum_stage_len(stage_idx)), **last_eval}
    print(json.dumps(final, sort_keys=True), flush=True)
    with open(args.output_dir / f"{args.run_name}_final.json", "w") as f:
        json.dump(final, f, indent=2, sort_keys=True)
    if run is not None:
        run.summary.update(final)
        run.finish()


if __name__ == "__main__":
    main()
