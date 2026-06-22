"""RL utilities for graph traversal experiments."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from collections import deque

import numpy as np

from tasks.graph_traversal import (
    GRAPH_ADJ_CLOSE,
    GRAPH_ADJ_OPEN,
    GRAPH_GOAL,
    DelaunayGraph,
    GraphTraversalVocab,
    generate_delaunay_graph,
)


@dataclass(frozen=True)
class GraphRLState:
    graph: DelaunayGraph
    current: int
    goal: int


@dataclass(frozen=True)
class CenterGoalCurriculumState:
    graph: DelaunayGraph
    current: int
    goal: int
    center: int
    goals_reached: int
    initial_goal_distance_scale: float = 0.25
    goal_distance_scale_growth: float = 0.10

    @property
    def goal_distance_scale(self) -> float:
        return float(self.initial_goal_distance_scale) + float(self.goals_reached) * float(self.goal_distance_scale_growth)

def sample_graph_pool(
    *,
    num_nodes: int,
    pool_size: int,
    seed: int,
    randomize_labels: bool = True,
    graph_backend: str = "scipy",
) -> list[DelaunayGraph]:
    """Generate a reusable pool of random Delaunay graphs."""
    rng = np.random.default_rng(int(seed))
    return [
        generate_delaunay_graph(
            num_nodes=int(num_nodes),
            rng=rng,
            randomize_labels=bool(randomize_labels),
            backend=str(graph_backend),
        )
        for _ in range(int(pool_size))
    ]


def sample_num_nodes(
    rng: np.random.Generator,
    *,
    batch_size: int,
    mode: str,
    fixed_num_nodes: int,
    log2_min: float = 3.0,
    log2_max: float = 6.0,
) -> np.ndarray:
    """Sample per-episode graph sizes; fixed_num_nodes is also the max vocab size."""
    if mode == "fixed":
        return np.full((int(batch_size),), int(fixed_num_nodes), dtype=np.int32)
    if mode != "log_uniform_int":
        raise ValueError(f"unsupported node sampling mode: {mode}")
    if float(log2_min) > float(log2_max):
        raise ValueError("log2_min must be <= log2_max")
    values = np.floor(np.exp2(rng.uniform(float(log2_min), float(log2_max), size=(int(batch_size),)))).astype(np.int32)
    return np.clip(values, 3, int(fixed_num_nodes))


def sample_episode_graphs(
    *,
    num_nodes: int | Sequence[int],
    batch_size: int,
    rng: np.random.Generator,
    graph_pool: list[DelaunayGraph] | None = None,
    randomize_labels: bool = True,
    graph_backend: str = "scipy",
) -> list[DelaunayGraph]:
    """Sample one graph per episode, either fresh or from a reusable pool."""
    if graph_pool is not None:
        if not isinstance(num_nodes, (int, np.integer)):
            raise ValueError("variable num_nodes is only supported for fresh graph sampling")
        indices = rng.integers(0, len(graph_pool), size=(int(batch_size),), dtype=np.int32)
        return [graph_pool[int(i)] for i in indices]
    if isinstance(num_nodes, (int, np.integer)):
        counts = [int(num_nodes)] * int(batch_size)
    else:
        counts = [int(x) for x in num_nodes]
        if len(counts) != int(batch_size):
            raise ValueError(f"num_nodes sequence length {len(counts)} must match batch_size {batch_size}")
    return [
        generate_delaunay_graph(
            num_nodes=count,
            rng=rng,
            randomize_labels=bool(randomize_labels),
            backend=str(graph_backend),
        )
        for count in counts
    ]


def sample_initial_state(
    graph: DelaunayGraph,
    *,
    rng: np.random.Generator,
) -> GraphRLState:
    current = int(rng.integers(0, graph.num_nodes))
    goal = sample_new_goal(graph.num_nodes, current=current, rng=rng)
    return GraphRLState(graph=graph, current=current, goal=goal)


def graph_center_node(graph: DelaunayGraph) -> int:
    """Return the node closest to the geometric center of the point cloud."""
    points = np.asarray(graph.points, dtype=np.float64)
    center = points.mean(axis=0)
    distances = np.sum((points - center[None, :]) ** 2, axis=1)
    return int(np.argmin(distances))


def graph_shortest_distances(graph: DelaunayGraph, source: int) -> np.ndarray:
    """Unweighted graph distances from `source`; connected Delaunay graphs are finite."""
    source_i = int(source)
    distances = np.full((graph.num_nodes,), -1, dtype=np.int32)
    distances[source_i] = 0
    q: deque[int] = deque([source_i])
    while q:
        node = q.popleft()
        next_dist = int(distances[node]) + 1
        for nbr in graph.neighbors[node]:
            nbr_i = int(nbr)
            if distances[nbr_i] < 0:
                distances[nbr_i] = next_dist
                q.append(nbr_i)
    return distances


def sample_center_curriculum_goal(
    graph: DelaunayGraph,
    *,
    current: int,
    distance_scale: float = 0.25,
    rng: np.random.Generator,
) -> int:
    """Sample a goal from all nodes, favoring nodes near `current`.

    Weights are exponential in graph-hop distance. Small scales concentrate
    mass on one-hop goals; as the scale grows, weights approach a uniform
    distribution over all non-current nodes.
    """
    distances = graph_shortest_distances(graph, int(current))
    eligible = np.flatnonzero((distances > 0) & (np.arange(graph.num_nodes) != int(current)))
    if eligible.size == 0:
        raise ValueError("connected graph had no eligible non-current goal")
    scale = max(float(distance_scale), 1.0e-6)
    logits = -(distances[eligible].astype(np.float64) - 1.0) / scale
    logits -= float(logits.max())
    weights = np.exp(logits)
    probs = weights / float(weights.sum())
    return int(eligible[int(rng.choice(eligible.size, p=probs))])


def sample_center_curriculum_initial_state(
    graph: DelaunayGraph,
    *,
    rng: np.random.Generator,
    initial_goal_distance_scale: float = 0.25,
    goal_distance_scale_growth: float = 0.10,
) -> CenterGoalCurriculumState:
    center = graph_center_node(graph)
    goal = sample_center_curriculum_goal(
        graph,
        current=center,
        distance_scale=float(initial_goal_distance_scale),
        rng=rng,
    )
    return CenterGoalCurriculumState(
        graph=graph,
        current=center,
        goal=goal,
        center=center,
        goals_reached=0,
        initial_goal_distance_scale=float(initial_goal_distance_scale),
        goal_distance_scale_growth=float(goal_distance_scale_growth),
    )


def sample_new_goal(num_nodes: int, *, current: int, rng: np.random.Generator) -> int:
    choices = np.arange(int(num_nodes), dtype=np.int64)
    choices = choices[choices != int(current)]
    return int(choices[int(rng.integers(0, choices.shape[0]))])


def initial_tokens(state: GraphRLState | CenterGoalCurriculumState, vocab: GraphTraversalVocab) -> list[int]:
    return [
        state.graph.token_for_internal(state.current, vocab),
        GRAPH_GOAL,
        state.graph.token_for_internal(state.goal, vocab),
    ]


def observation_tokens(state: GraphRLState | CenterGoalCurriculumState, vocab: GraphTraversalVocab) -> list[int]:
    neighbor_tokens = [state.graph.token_for_internal(nbr, vocab) for nbr in state.graph.neighbors[state.current]]
    neighbor_tokens.sort()
    return [GRAPH_ADJ_OPEN, *neighbor_tokens, GRAPH_ADJ_CLOSE]


def legal_action_tokens(state: GraphRLState | CenterGoalCurriculumState, vocab: GraphTraversalVocab) -> list[int]:
    toks = [state.graph.token_for_internal(nbr, vocab) for nbr in state.graph.neighbors[state.current]]
    toks.sort()
    return toks


def step_state(
    state: GraphRLState | CenterGoalCurriculumState,
    action_token: int,
    vocab: GraphTraversalVocab,
    *,
    rng: np.random.Generator,
) -> tuple[GraphRLState | CenterGoalCurriculumState, float]:
    label = vocab.token_node_label(int(action_token))
    chosen = state.graph.internal_for_label(label)
    if chosen not in state.graph.neighbors[state.current]:
        raise ValueError(f"illegal graph action token {action_token}")
    reward = 1.0 if int(chosen) == int(state.goal) else 0.0
    if isinstance(state, CenterGoalCurriculumState):
        goals_reached = int(state.goals_reached) + int(reward > 0.0)
        distance_scale = float(state.initial_goal_distance_scale) + float(goals_reached) * float(state.goal_distance_scale_growth)
        goal = (
            sample_center_curriculum_goal(
                state.graph,
                current=chosen,
                distance_scale=distance_scale,
                rng=rng,
            )
            if reward > 0.0
            else int(state.goal)
        )
        return (
            CenterGoalCurriculumState(
                graph=state.graph,
                current=int(chosen),
                goal=int(goal),
                center=int(state.center),
                goals_reached=goals_reached,
                initial_goal_distance_scale=float(state.initial_goal_distance_scale),
                goal_distance_scale_growth=float(state.goal_distance_scale_growth),
            ),
            reward,
        )
    goal = sample_new_goal(state.graph.num_nodes, current=chosen, rng=rng) if reward > 0.0 else int(state.goal)
    return GraphRLState(graph=state.graph, current=int(chosen), goal=int(goal)), reward


def discounted_action_returns(
    rewards: np.ndarray,
    action_mask: np.ndarray,
    *,
    gamma: float,
) -> np.ndarray:
    """Discount over action steps, not raw token distance."""
    rewards = np.asarray(rewards, dtype=np.float32)
    mask = np.asarray(action_mask, dtype=bool)
    if rewards.shape != mask.shape:
        raise ValueError(f"rewards shape {rewards.shape} must match action_mask shape {mask.shape}")
    out = np.zeros_like(rewards, dtype=np.float32)
    running = np.zeros((rewards.shape[0],), dtype=np.float32)
    gamma_f = np.float32(gamma)
    for t in range(rewards.shape[1] - 1, -1, -1):
        running = rewards[:, t] + gamma_f * running
        running = np.where(mask[:, t], running, 0.0)
        out[:, t] = np.where(mask[:, t], running, 0.0)
    return out


def position_group_advantages(
    returns: np.ndarray,
    action_mask: np.ndarray,
    *,
    mode: str = "loo_z",
    eps: float = 1e-6,
) -> np.ndarray:
    """Normalize returns across rollout group members at the same action offset.

    `loo_z` uses a leave-one-out baseline where at least two samples exist, then
    scales by the group standard deviation. Singleton positions get zero
    advantage because no group baseline is available.
    """
    returns = np.asarray(returns, dtype=np.float32)
    mask = np.asarray(action_mask, dtype=bool)
    if returns.shape != mask.shape:
        raise ValueError(f"returns shape {returns.shape} must match action_mask shape {mask.shape}")
    if mode == "none":
        return np.where(mask, returns, 0.0).astype(np.float32)
    if mode not in {"z", "loo", "loo_z"}:
        raise ValueError(f"unsupported advantage normalization mode: {mode}")

    masked = np.where(mask, returns, 0.0).astype(np.float32)
    count = mask.sum(axis=0, keepdims=True).astype(np.float32)
    total = masked.sum(axis=0, keepdims=True)
    mean = total / np.maximum(count, 1.0)
    centered = np.where(mask, returns - mean, 0.0)
    var = (centered * centered).sum(axis=0, keepdims=True) / np.maximum(count, 1.0)
    std = np.sqrt(var + np.float32(eps))

    if mode == "z":
        adv = centered / std
    else:
        loo_count = np.maximum(count - 1.0, 1.0)
        loo_mean = (total - masked) / loo_count
        adv = np.where(mask, returns - loo_mean, 0.0)
        if mode == "loo_z":
            adv = adv / std
    adv = np.where((mask & (count >= 2.0)), adv, 0.0)
    return adv.astype(np.float32)


def final_suffix_action_weights(
    action_mask: np.ndarray,
    *,
    suffix_actions: int,
    min_weight: float,
    gamma: float = 0.99,
) -> np.ndarray:
    """Down-weight final valid actions by their visible discounted evidence mass."""
    mask = np.asarray(action_mask, dtype=bool)
    weights = mask.astype(np.float32)
    suffix = int(suffix_actions)
    if suffix <= 0:
        return weights
    min_w = float(min_weight)
    if min_w < 0.0 or min_w > 1.0:
        raise ValueError("min_weight must be in [0, 1]")
    gamma_f = float(gamma)
    if gamma_f < 0.0 or gamma_f > 1.0:
        raise ValueError("gamma must be in [0, 1]")
    action_idx = np.cumsum(mask.astype(np.int32), axis=1) - 1
    valid_counts = mask.sum(axis=1, keepdims=True).astype(np.int32)
    remaining_after = valid_counts - 1 - action_idx
    evidence = np.sqrt(
        np.maximum(
            0.0,
            1.0 - np.power(gamma_f, 2.0 * (remaining_after.astype(np.float32) + 1.0)),
        )
    ).astype(np.float32)
    tail = remaining_after < suffix
    weights = np.where(tail, evidence, weights)
    return np.where(mask, weights, 0.0).astype(np.float32)
