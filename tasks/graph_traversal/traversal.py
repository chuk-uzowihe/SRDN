"""Delaunay graph traversal traces for latent-compute experiments."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import exp

import numpy as np
import torch
from scipy.spatial import Delaunay

GRAPH_PAD = 0
GRAPH_PAUSE = 1
GRAPH_GOAL = 2
GRAPH_ADJ_OPEN = 3
GRAPH_ADJ_CLOSE = 4
GRAPH_NODE_BASE = 5


@dataclass(frozen=True)
class GraphTraversalVocab:
    """Token layout for generated graph traces."""

    num_nodes: int

    @property
    def vocab_size(self) -> int:
        return GRAPH_NODE_BASE + int(self.num_nodes)

    def node_token(self, node_label: int) -> int:
        node_label = int(node_label)
        if node_label < 0 or node_label >= self.num_nodes:
            raise ValueError(f"node_label must be in [0, {self.num_nodes}), got {node_label}")
        return GRAPH_NODE_BASE + node_label

    def token_node_label(self, token_id: int) -> int:
        token_id = int(token_id)
        label = token_id - GRAPH_NODE_BASE
        if label < 0 or label >= self.num_nodes:
            raise ValueError(f"token_id {token_id} is not a graph node token")
        return label

    def token_text(self, token_id: int) -> str:
        token_id = int(token_id)
        if token_id == GRAPH_PAD:
            return "<pad>"
        if token_id == GRAPH_PAUSE:
            return "<pause>"
        if token_id == GRAPH_GOAL:
            return "goal"
        if token_id == GRAPH_ADJ_OPEN:
            return "(adjacent"
        if token_id == GRAPH_ADJ_CLOSE:
            return ")"
        return str(self.token_node_label(token_id) + 1)

    def detokenize(self, token_ids: np.ndarray | list[int]) -> str:
        return " ".join(self.token_text(int(tok)) for tok in token_ids)


@dataclass(frozen=True)
class GraphTraversalPolicy:
    """The six Phase 1 trace-shaping variables."""

    memory_rate: float
    mistake_rate: float
    pause_budget: float
    deliberation_shape: float
    random_move_floor: float
    pause_noise: float
    max_pauses_per_decision: int = 8


@dataclass(frozen=True)
class DelaunayGraph:
    points: np.ndarray
    neighbors: tuple[tuple[int, ...], ...]
    labels: tuple[int, ...]

    @property
    def num_nodes(self) -> int:
        return len(self.neighbors)

    def token_for_internal(self, node: int, vocab: GraphTraversalVocab) -> int:
        return vocab.node_token(self.labels[int(node)])

    def internal_for_label(self, label: int) -> int:
        for node, node_label in enumerate(self.labels):
            if int(node_label) == int(label):
                return int(node)
        raise ValueError(f"unknown node label {label}")


@dataclass(frozen=True)
class GraphTraversalTrace:
    tokens: np.ndarray
    next_token_mask: np.ndarray
    action_positions: np.ndarray
    reward_positions: np.ndarray
    graph: DelaunayGraph
    policy: GraphTraversalPolicy
    stats: dict[str, float]


@dataclass(frozen=True)
class GraphTraversalBatch:
    inputs: torch.Tensor
    targets: torch.Tensor
    next_token_mask: torch.Tensor
    action_target_mask: torch.Tensor
    reward_target_mask: torch.Tensor
    stats: dict[str, float]


def graph_traversal_vocab_size(num_nodes: int) -> int:
    return GraphTraversalVocab(num_nodes=int(num_nodes)).vocab_size


def _circumcircle(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    *,
    eps: float = 1e-12,
) -> tuple[np.ndarray, float] | None:
    ax, ay = float(a[0]), float(a[1])
    bx, by = float(b[0]), float(b[1])
    cx, cy = float(c[0]), float(c[1])
    denom = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(denom) < eps:
        return None
    a2 = ax * ax + ay * ay
    b2 = bx * bx + by * by
    c2 = cx * cx + cy * cy
    ux = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / denom
    uy = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / denom
    center = np.asarray([ux, uy], dtype=np.float64)
    radius2 = float(np.sum((center - a) ** 2))
    return center, radius2


def _is_connected(neighbors: list[set[int]]) -> bool:
    if not neighbors:
        return False
    seen = {0}
    q: deque[int] = deque([0])
    while q:
        node = q.popleft()
        for nxt in neighbors[node]:
            if nxt not in seen:
                seen.add(nxt)
                q.append(nxt)
    return len(seen) == len(neighbors)


def _delaunay_neighbors(points: np.ndarray) -> tuple[tuple[int, ...], ...]:
    n = int(points.shape[0])
    if n < 3:
        raise ValueError("Delaunay graph traversal requires at least 3 nodes")
    edges: set[tuple[int, int]] = set()
    eps = 1e-10
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                circle = _circumcircle(points[i], points[j], points[k])
                if circle is None:
                    continue
                center, radius2 = circle
                empty = True
                for m in range(n):
                    if m == i or m == j or m == k:
                        continue
                    dist2 = float(np.sum((points[m] - center) ** 2))
                    if dist2 < radius2 - eps:
                        empty = False
                        break
                if empty:
                    edges.add((i, j))
                    edges.add((i, k))
                    edges.add((j, k))

    neighbors = [set() for _ in range(n)]
    for a, b in edges:
        neighbors[a].add(b)
        neighbors[b].add(a)
    return tuple(tuple(sorted(nbrs)) for nbrs in neighbors)


def _scipy_delaunay_neighbors(points: np.ndarray) -> tuple[tuple[int, ...], ...]:
    n = int(points.shape[0])
    if n < 3:
        raise ValueError("Delaunay graph traversal requires at least 3 nodes")
    tri = Delaunay(points)
    neighbors = [set() for _ in range(n)]
    for simplex in tri.simplices:
        a, b, c = (int(simplex[0]), int(simplex[1]), int(simplex[2]))
        neighbors[a].add(b)
        neighbors[a].add(c)
        neighbors[b].add(a)
        neighbors[b].add(c)
        neighbors[c].add(a)
        neighbors[c].add(b)
    return tuple(tuple(sorted(nbrs)) for nbrs in neighbors)


def generate_delaunay_graph(
    *,
    num_nodes: int,
    rng: np.random.Generator,
    randomize_labels: bool = True,
    max_attempts: int = 32,
    backend: str = "scipy",
) -> DelaunayGraph:
    """Generate a connected random-label Delaunay graph on the unit square."""
    num_nodes = int(num_nodes)
    if num_nodes < 3:
        raise ValueError("num_nodes must be at least 3")
    if backend not in {"scipy", "python"}:
        raise ValueError(f"unsupported Delaunay backend: {backend}")

    for _ in range(int(max_attempts)):
        points = rng.random((num_nodes, 2), dtype=np.float64)
        neighbors_tuple = _scipy_delaunay_neighbors(points) if backend == "scipy" else _delaunay_neighbors(points)
        neighbors = [set(nbrs) for nbrs in neighbors_tuple]
        if all(neighbors) and _is_connected(neighbors):
            if randomize_labels:
                labels = tuple(int(x) for x in rng.permutation(num_nodes))
            else:
                labels = tuple(range(num_nodes))
            return DelaunayGraph(
                points=points,
                neighbors=tuple(tuple(sorted(nbrs)) for nbrs in neighbors),
                labels=labels,
            )
    raise RuntimeError(f"failed to generate a connected Delaunay graph after {max_attempts} attempts")


def sample_graph_traversal_policy(
    rng: np.random.Generator,
    *,
    pause_budget_scale: float = 1.0,
    max_pauses_per_decision: int = 8,
) -> GraphTraversalPolicy:
    """Sample the six trace variables from conservative default ranges."""
    pause_budget_scale = max(0.0, float(pause_budget_scale))
    max_pauses_per_decision = max(0, int(max_pauses_per_decision))
    pause_budget = float(rng.uniform(0.0, 2.5)) * pause_budget_scale
    return GraphTraversalPolicy(
        memory_rate=float(rng.uniform(0.03, 0.22)),
        mistake_rate=float(rng.uniform(0.02, 0.22)),
        pause_budget=pause_budget,
        deliberation_shape=float(rng.uniform(0.0, 1.0)),
        random_move_floor=float(rng.uniform(0.01, 0.12)),
        pause_noise=float(rng.uniform(0.15, 1.25)) if pause_budget > 0.0 and max_pauses_per_decision > 0 else 0.0,
        max_pauses_per_decision=max_pauses_per_decision,
    )


def _shortest_path_next(
    known_neighbors: list[set[int]],
    start: int,
    goal: int,
) -> tuple[int, int] | None:
    start = int(start)
    goal = int(goal)
    if start == goal:
        return None
    parent = {start: -1}
    q: deque[int] = deque([start])
    while q:
        cur = q.popleft()
        if cur == goal:
            break
        for nxt in sorted(known_neighbors[cur]):
            if nxt not in parent:
                parent[nxt] = cur
                q.append(nxt)
    if goal not in parent:
        return None
    path = [goal]
    while path[-1] != start:
        path.append(parent[path[-1]])
    path.reverse()
    return int(path[1]), len(path) - 1


def _choose_exploration_neighbor(
    *,
    current: int,
    graph: DelaunayGraph,
    visit_counts: np.ndarray,
    rng: np.random.Generator,
) -> int:
    neighbors = np.asarray(graph.neighbors[int(current)], dtype=np.int64)
    weights = 1.0 / (1.0 + visit_counts[neighbors].astype(np.float64))
    weights = weights / np.maximum(float(weights.sum()), 1e-12)
    return int(rng.choice(neighbors, p=weights))


def _choice(values: list[int] | tuple[int, ...] | np.ndarray, rng: np.random.Generator) -> int:
    values_arr = np.asarray(values, dtype=np.int64)
    return int(values_arr[int(rng.integers(0, len(values_arr)))])


def _sample_pause_count(
    *,
    policy: GraphTraversalPolicy,
    decision_idx: int,
    degree: int,
    path_len: int | None,
    post_mistake: bool,
    memory_progress: float,
    rng: np.random.Generator,
) -> int:
    shape = min(max(float(policy.deliberation_shape), 0.0), 1.0)
    degree_term = min(1.0, max(0.0, (float(degree) - 2.0) / 5.0))
    path_term = 0.0 if path_len is None else min(1.0, max(0.0, (float(path_len) - 1.0) / 6.0))
    recovery_term = 1.0 if post_mistake else 0.0
    time_term = min(1.0, max(0.0, float(memory_progress)))
    structured = 0.25 + 0.30 * time_term + 0.25 * path_term + 0.10 * degree_term + 0.10 * recovery_term
    mean_pauses = float(policy.pause_budget) * ((1.0 - shape) + shape * 2.0 * structured)
    mean_pauses = max(0.0, mean_pauses)
    noise = max(0.0, float(policy.pause_noise))
    if noise <= 1e-8:
        pauses = int(round(mean_pauses))
    else:
        gamma_shape = 1.0 / (noise * noise)
        gamma_scale = mean_pauses * noise * noise
        lam = float(rng.gamma(gamma_shape, gamma_scale)) if mean_pauses > 0.0 else 0.0
        pauses = int(rng.poisson(lam))
    return int(min(max(pauses, 0), int(policy.max_pauses_per_decision)))


def _mask(vocab_size: int, allowed_tokens: set[int] | list[int] | tuple[int, ...]) -> np.ndarray:
    out = np.zeros((vocab_size,), dtype=bool)
    out[np.asarray(list(allowed_tokens), dtype=np.int64)] = True
    return out


def generate_graph_traversal_trace(
    *,
    num_nodes: int,
    decision_count: int,
    seed: int,
    policy: GraphTraversalPolicy | None = None,
    graph: DelaunayGraph | None = None,
    randomize_labels: bool = True,
    pause_budget_scale: float = 1.0,
    max_pauses_per_decision: int = 8,
    allow_pause_actions: bool = True,
) -> GraphTraversalTrace:
    """Generate one Phase 1 graph traversal trace."""
    rng = np.random.default_rng(seed)
    vocab = GraphTraversalVocab(num_nodes=int(num_nodes))
    graph = graph or generate_delaunay_graph(
        num_nodes=num_nodes,
        rng=rng,
        randomize_labels=randomize_labels,
    )
    policy = policy or sample_graph_traversal_policy(
        rng,
        pause_budget_scale=pause_budget_scale,
        max_pauses_per_decision=max_pauses_per_decision,
    )

    tokens: list[int] = []
    next_masks: list[np.ndarray] = []
    action_positions: list[bool] = []
    reward_positions: list[bool] = []

    def append(token_id: int, allowed: set[int] | list[int] | tuple[int, ...] | None, *, action: bool = False, reward: bool = False) -> None:
        if tokens:
            if allowed is None:
                allowed = {int(token_id)}
            next_masks.append(_mask(vocab.vocab_size, allowed))
        tokens.append(int(token_id))
        action_positions.append(bool(action))
        reward_positions.append(bool(reward))

    all_node_tokens = {vocab.node_token(label) for label in range(num_nodes)}
    current = int(rng.integers(0, num_nodes))
    goal = _choice([node for node in range(num_nodes) if node != current], rng)
    known_neighbors = [set() for _ in range(num_nodes)]
    visit_counts = np.zeros((num_nodes,), dtype=np.int64)
    visit_counts[current] += 1

    append(graph.token_for_internal(current, vocab), None)
    append(GRAPH_GOAL, {GRAPH_GOAL})
    append(graph.token_for_internal(goal, vocab), all_node_tokens - {graph.token_for_internal(current, vocab)})

    goals_hit = 0
    random_moves = 0
    mistakes = 0
    planned_moves = 0
    total_pauses = 0
    post_mistake = False

    for decision_idx in range(int(decision_count)):
        current_neighbors = tuple(int(nbr) for nbr in graph.neighbors[current])
        for nbr in current_neighbors:
            known_neighbors[current].add(nbr)
            known_neighbors[nbr].add(current)

        append(GRAPH_ADJ_OPEN, {GRAPH_ADJ_OPEN})
        neighbor_tokens = [graph.token_for_internal(nbr, vocab) for nbr in current_neighbors]
        neighbor_tokens.sort()
        for tok in neighbor_tokens:
            append(tok, set(neighbor_tokens) | {GRAPH_ADJ_CLOSE})
        append(GRAPH_ADJ_CLOSE, {GRAPH_ADJ_CLOSE})

        planned = _shortest_path_next(known_neighbors, current, goal)
        planned_next = None if planned is None else int(planned[0])
        path_len = None if planned is None else int(planned[1])
        memory_progress = 1.0 - exp(-float(policy.memory_rate) * float(decision_idx + 1))
        action_allowed = {graph.token_for_internal(nbr, vocab) for nbr in current_neighbors}
        if allow_pause_actions:
            action_allowed.add(GRAPH_PAUSE)
        pauses = _sample_pause_count(
            policy=policy,
            decision_idx=decision_idx,
            degree=len(current_neighbors),
            path_len=path_len,
            post_mistake=post_mistake,
            memory_progress=memory_progress,
            rng=rng,
        )
        for _ in range(pauses):
            if not allow_pause_actions:
                raise ValueError("policy sampled pauses but allow_pause_actions=False")
            append(GRAPH_PAUSE, action_allowed, action=True)
        total_pauses += pauses

        choose_random = bool(rng.random() < float(policy.random_move_floor))
        chose_mistake = False
        used_plan = False
        if choose_random:
            chosen = _choice(current_neighbors, rng)
            random_moves += 1
        elif planned_next is not None and rng.random() < memory_progress:
            alternatives = [nbr for nbr in current_neighbors if nbr != planned_next]
            if alternatives and rng.random() < float(policy.mistake_rate):
                chosen = _choice(alternatives, rng)
                mistakes += 1
                chose_mistake = True
            else:
                chosen = planned_next
                planned_moves += 1
                used_plan = True
        else:
            chosen = _choose_exploration_neighbor(
                current=current,
                graph=graph,
                visit_counts=visit_counts,
                rng=rng,
            )

        reached_goal = int(chosen) == int(goal)
        append(graph.token_for_internal(chosen, vocab), action_allowed, action=True, reward=reached_goal)
        current = int(chosen)
        visit_counts[current] += 1
        post_mistake = bool(chose_mistake or (planned_next is not None and not used_plan and not reached_goal))

        if reached_goal:
            goals_hit += 1
            goal = _choice([node for node in range(num_nodes) if node != current], rng)
            append(GRAPH_GOAL, {GRAPH_GOAL})
            append(
                graph.token_for_internal(goal, vocab),
                all_node_tokens - {graph.token_for_internal(current, vocab)},
            )

    token_arr = np.asarray(tokens, dtype=np.int64)
    stats = {
        "decisions": float(decision_count),
        "goals_hit": float(goals_hit),
        "goal_hit_rate": float(goals_hit / max(1, int(decision_count))),
        "mean_pauses": float(total_pauses / max(1, int(decision_count))),
        "random_moves": float(random_moves),
        "mistakes": float(mistakes),
        "planned_moves": float(planned_moves),
        "memory_rate": float(policy.memory_rate),
        "mistake_rate": float(policy.mistake_rate),
        "pause_budget": float(policy.pause_budget),
        "deliberation_shape": float(policy.deliberation_shape),
        "random_move_floor": float(policy.random_move_floor),
        "pause_noise": float(policy.pause_noise),
    }
    return GraphTraversalTrace(
        tokens=token_arr,
        next_token_mask=np.stack(next_masks, axis=0) if next_masks else np.zeros((0, vocab.vocab_size), dtype=bool),
        action_positions=np.asarray(action_positions, dtype=bool),
        reward_positions=np.asarray(reward_positions, dtype=bool),
        graph=graph,
        policy=policy,
        stats=stats,
    )


def build_graph_traversal_batch(
    traces: list[GraphTraversalTrace],
    *,
    num_nodes: int,
    context_length: int | None = None,
    ignore_index: int = -100,
) -> GraphTraversalBatch:
    """Pack traces into shifted LM tensors plus grammar/action/reward masks."""
    if not traces:
        raise ValueError("traces must be non-empty")
    vocab = GraphTraversalVocab(num_nodes=int(num_nodes))
    max_trace_len = max(int(trace.tokens.shape[0]) for trace in traces)
    if context_length is None:
        seq_len = max_trace_len
        out_len = max(0, seq_len - 1)
    else:
        out_len = int(context_length)
        seq_len = out_len + 1

    batch_size = len(traces)
    x = np.full((batch_size, out_len), GRAPH_PAD, dtype=np.int64)
    y = np.full((batch_size, out_len), ignore_index, dtype=np.int64)
    next_token_mask = np.zeros((batch_size, out_len, vocab.vocab_size), dtype=bool)
    action_target_mask = np.zeros((batch_size, out_len), dtype=bool)
    reward_target_mask = np.zeros((batch_size, out_len), dtype=bool)
    pad_mask = _mask(vocab.vocab_size, {GRAPH_PAD})

    for b, trace in enumerate(traces):
        trace_tokens = trace.tokens[:seq_len]
        if trace_tokens.shape[0] < 2:
            continue
        L = min(out_len, int(trace_tokens.shape[0] - 1))
        x[b, :L] = trace_tokens[:L]
        y[b, :L] = trace_tokens[1 : L + 1]
        next_token_mask[b, :L] = trace.next_token_mask[:L]
        next_token_mask[b, L:] = pad_mask
        action_target_mask[b, :L] = trace.action_positions[1 : L + 1]
        reward_target_mask[b, :L] = trace.reward_positions[1 : L + 1]

    stats: dict[str, float] = {}
    for key in traces[0].stats:
        stats[key] = float(np.mean([trace.stats[key] for trace in traces]))

    return GraphTraversalBatch(
        inputs=torch.from_numpy(x).long(),
        targets=torch.from_numpy(y).long(),
        next_token_mask=torch.from_numpy(next_token_mask),
        action_target_mask=torch.from_numpy(action_target_mask),
        reward_target_mask=torch.from_numpy(reward_target_mask),
        stats=stats,
    )


def generate_graph_traversal_batch(
    *,
    num_nodes: int,
    decision_count: int,
    batch_size: int,
    seed: int,
    context_length: int | None = None,
    ignore_index: int = -100,
    policy: GraphTraversalPolicy | None = None,
    randomize_labels: bool = True,
    pause_budget_scale: float = 1.0,
    max_pauses_per_decision: int = 8,
    allow_pause_actions: bool = True,
) -> GraphTraversalBatch:
    """Generate a batch of independent graph traversal traces."""
    traces = [
        generate_graph_traversal_trace(
            num_nodes=num_nodes,
            decision_count=decision_count,
            seed=seed + 1_000_003 * b,
            policy=policy,
            randomize_labels=randomize_labels,
            pause_budget_scale=pause_budget_scale,
            max_pauses_per_decision=max_pauses_per_decision,
            allow_pause_actions=allow_pause_actions,
        )
        for b in range(int(batch_size))
    ]
    return build_graph_traversal_batch(
        traces,
        num_nodes=num_nodes,
        context_length=context_length,
        ignore_index=ignore_index,
    )
