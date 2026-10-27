"""Policy-regret diagnostics built around stochastic joint-action MCTS.

The JAX basketball policy emits one action for every controlled player at the
same environment tick.  A planner action is therefore a *joint* team action,
not the single-player override used by the removed SB3 advisor.  This module
owns the dependency-light search and statistical pieces of the JAX-native
counterfactual evaluator.  Environment/checkpoint wiring lives separately so
the search can be tested without restoring a model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np


MCTS_REGRET_SCHEMA_VERSION = 1
DEFAULT_SEARCH_DEPTHS = (1, 3, 5)
DEFAULT_SEARCH_DEPTH = 5
DEFAULT_SIMULATIONS_PER_ACTION = 512
DEFAULT_CALIBRATION_BUDGETS = (64, 128, 256, 512, 1024)

JointAction = tuple[int, ...]


@dataclass(frozen=True)
class MctsRegretConfig:
    """Validated protocol settings for one policy-regret study."""

    search_depths: tuple[int, ...] = DEFAULT_SEARCH_DEPTHS
    primary_search_depth: int = DEFAULT_SEARCH_DEPTH
    simulations_per_action: int = DEFAULT_SIMULATIONS_PER_ACTION
    calibration_budgets: tuple[int, ...] = DEFAULT_CALIBRATION_BUDGETS
    exploration_c: float = 1.4
    progressive_widening_c: float = 2.0
    progressive_widening_alpha: float = 0.5
    max_joint_actions_per_node: int = 64
    discount: float = 1.0

    def __post_init__(self) -> None:
        depths = tuple(int(value) for value in self.search_depths)
        budgets = tuple(int(value) for value in self.calibration_budgets)
        if not depths or any(value < 1 for value in depths):
            raise ValueError("search_depths must contain positive integers.")
        if tuple(sorted(set(depths))) != depths:
            raise ValueError("search_depths must be unique and increasing.")
        if int(self.primary_search_depth) not in depths:
            raise ValueError("primary_search_depth must be present in search_depths.")
        if int(self.simulations_per_action) < 1:
            raise ValueError("simulations_per_action must be positive.")
        if not budgets or any(value < 1 for value in budgets):
            raise ValueError("calibration_budgets must contain positive integers.")
        if tuple(sorted(set(budgets))) != budgets:
            raise ValueError("calibration_budgets must be unique and increasing.")
        if float(self.exploration_c) < 0.0:
            raise ValueError("exploration_c must be non-negative.")
        if float(self.progressive_widening_c) <= 0.0:
            raise ValueError("progressive_widening_c must be positive.")
        if not 0.0 < float(self.progressive_widening_alpha) <= 1.0:
            raise ValueError("progressive_widening_alpha must be in (0, 1].")
        if int(self.max_joint_actions_per_node) < 1:
            raise ValueError("max_joint_actions_per_node must be positive.")
        if not 0.0 <= float(self.discount) <= 1.0:
            raise ValueError("discount must be in [0, 1].")


@dataclass(frozen=True)
class JointActionProposal:
    action: JointAction
    prior: float

    def __post_init__(self) -> None:
        if not self.action:
            raise ValueError("A joint action must contain at least one player action.")
        if any(int(value) < 0 for value in self.action):
            raise ValueError("Joint action indices must be non-negative.")
        if not math.isfinite(float(self.prior)) or float(self.prior) < 0.0:
            raise ValueError("Joint action prior must be finite and non-negative.")


@dataclass(frozen=True)
class MctsTransition:
    state: Any
    reward: float
    terminal: bool


@dataclass
class _ActionEdge:
    action: JointAction
    prior: float
    visits: int = 0
    value_sum: float = 0.0

    @property
    def mean_value(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


@dataclass
class _SearchNode:
    visits: int = 0
    value_sum: float = 0.0
    children: dict[JointAction, _ActionEdge] = field(default_factory=dict)

    @property
    def mean_value(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


@dataclass(frozen=True)
class MctsActionStats:
    action: JointAction
    prior: float
    visits: int
    mean_value: float


@dataclass(frozen=True)
class MctsSearchResult:
    selected_action: JointAction
    selected_value: float
    simulations: int
    max_depth_reached: int
    nodes_expanded: int
    action_stats: tuple[MctsActionStats, ...]


@dataclass(frozen=True)
class PairedActionComparison:
    policy_action: JointAction
    planner_action: JointAction
    rollout_seeds: tuple[int, ...]
    policy_returns: tuple[float, ...]
    planner_returns: tuple[float, ...]
    regret_samples: tuple[float, ...]
    policy_value: float
    planner_value: float
    regret_mean: float
    regret_ci95_low: float
    regret_ci95_high: float


StateKeyFn = Callable[[Any], Any]
ProposalFn = Callable[
    [Any, np.random.Generator, int, frozenset[JointAction]],
    Sequence[JointActionProposal],
]
TransitionFn = Callable[[Any, JointAction, np.random.Generator], MctsTransition]
RolloutFn = Callable[[Any, np.random.Generator], float]


def _validated_policy_rows(
    probabilities: np.ndarray,
    legal_mask: np.ndarray,
) -> np.ndarray:
    probs = np.asarray(probabilities, dtype=np.float64)
    legal = np.asarray(legal_mask, dtype=bool)
    if probs.ndim != 2:
        raise ValueError(
            "Joint-action policy probabilities must have shape (players, actions)."
        )
    if legal.shape != probs.shape:
        raise ValueError("legal_mask must have the same shape as probabilities.")
    if not np.all(np.isfinite(probs)) or np.any(probs < 0.0):
        raise ValueError("Policy probabilities must be finite and non-negative.")
    normalized = np.zeros_like(probs, dtype=np.float64)
    for player_index in range(probs.shape[0]):
        row_legal = legal[player_index]
        if not np.any(row_legal):
            raise ValueError(f"Player row {player_index} has no legal action.")
        row = np.where(row_legal, probs[player_index], 0.0)
        total = float(row.sum())
        if total <= 0.0:
            row = row_legal.astype(np.float64)
            total = float(row.sum())
        normalized[player_index] = row / total
    return normalized


def joint_action_probability(
    action: Sequence[int],
    probabilities: np.ndarray,
    legal_mask: np.ndarray,
) -> float:
    """Return the factorized policy probability of one complete team action."""

    normalized = _validated_policy_rows(probabilities, legal_mask)
    values = tuple(int(value) for value in action)
    if len(values) != normalized.shape[0]:
        raise ValueError("Joint action width must equal the controlled player count.")
    log_probability = 0.0
    for player_index, action_index in enumerate(values):
        if not 0 <= action_index < normalized.shape[1]:
            raise ValueError(
                f"Action {action_index} is outside the policy action space."
            )
        probability = float(normalized[player_index, action_index])
        if probability <= 0.0:
            return 0.0
        log_probability += math.log(probability)
    return float(math.exp(log_probability))


def sample_factorized_joint_actions(
    probabilities: np.ndarray,
    legal_mask: np.ndarray,
    *,
    count: int,
    rng: np.random.Generator,
    excluded: Iterable[JointAction] = (),
    include_deterministic: bool = True,
) -> tuple[JointActionProposal, ...]:
    """Sample unique, complete team actions from factorized policy rows.

    The deterministic per-player argmax is proposed first.  Remaining actions
    are sampled from every controlled player's masked categorical.  This keeps
    the full simultaneous-action semantics while progressive widening avoids
    exhaustive enumeration of the Cartesian product.
    """

    requested = int(count)
    if requested < 0:
        raise ValueError("count must be non-negative.")
    if requested == 0:
        return ()
    normalized = _validated_policy_rows(probabilities, legal_mask)
    excluded_set = {tuple(int(value) for value in action) for action in excluded}
    proposals: list[JointActionProposal] = []
    seen = set(excluded_set)

    def _append(action: JointAction) -> None:
        if action in seen or len(proposals) >= requested:
            return
        seen.add(action)
        proposals.append(
            JointActionProposal(
                action=action,
                prior=joint_action_probability(action, normalized, legal_mask),
            )
        )

    if include_deterministic:
        _append(tuple(int(value) for value in np.argmax(normalized, axis=1)))

    # Duplicate joint samples are common once a policy becomes sharp.  Use a
    # bounded retry count so the planner remains predictable even when fewer
    # unique joint actions have nonzero support than requested.
    max_attempts = max(32, requested * 32)
    for _ in range(max_attempts):
        if len(proposals) >= requested:
            break
        action = tuple(
            int(rng.choice(normalized.shape[1], p=normalized[player_index]))
            for player_index in range(normalized.shape[0])
        )
        _append(action)
    return tuple(proposals)


def _progressive_widening_limit(config: MctsRegretConfig, visits: int) -> int:
    proposed = int(
        math.ceil(
            float(config.progressive_widening_c)
            * (max(1, int(visits)) ** float(config.progressive_widening_alpha))
        )
    )
    return max(1, min(int(config.max_joint_actions_per_node), proposed))


def _select_edge(
    node: _SearchNode,
    *,
    exploration_c: float,
) -> _ActionEdge:
    if not node.children:
        raise ValueError("Cannot select an action from an unexpanded MCTS node.")
    prior_total = sum(max(float(edge.prior), 0.0) for edge in node.children.values())
    uniform_prior = 1.0 / float(len(node.children))
    parent_scale = math.sqrt(float(max(1, node.visits)))

    def _score(edge: _ActionEdge) -> tuple[float, float, JointAction]:
        normalized_prior = (
            max(float(edge.prior), 0.0) / prior_total
            if prior_total > 0.0
            else uniform_prior
        )
        exploration = (
            float(exploration_c)
            * normalized_prior
            * parent_scale
            / float(1 + edge.visits)
        )
        # The action tuple supplies deterministic tie-breaking.
        return edge.mean_value + exploration, normalized_prior, edge.action

    return max(node.children.values(), key=_score)


def run_joint_action_mcts(
    root_state: Any,
    *,
    state_key: StateKeyFn,
    propose_actions: ProposalFn,
    transition: TransitionFn,
    rollout_value: RolloutFn,
    simulations: int,
    max_depth: int,
    seed: int,
    config: MctsRegretConfig | None = None,
) -> MctsSearchResult:
    """Run stochastic MCTS while treating the opponent as environment policy.

    Nodes are keyed by realized states rather than by a single cached child
    state.  Re-visiting an edge therefore re-samples opponent actions and game
    randomness, and different chance outcomes receive distinct nodes.  The
    transition callback owns the fixed opponent policy.
    """

    protocol = config or MctsRegretConfig()
    simulation_count = int(simulations)
    depth_limit = int(max_depth)
    if simulation_count < 1:
        raise ValueError("simulations must be positive.")
    if depth_limit < 1:
        raise ValueError("max_depth must be positive.")

    rng = np.random.default_rng(int(seed))
    nodes: dict[tuple[int, Any], _SearchNode] = {}
    root_key = (0, state_key(root_state))
    root = nodes.setdefault(root_key, _SearchNode())
    max_depth_reached = 0

    for _ in range(simulation_count):
        state = root_state
        node = root
        path: list[tuple[_SearchNode, _ActionEdge, float]] = []
        leaf_value = 0.0

        for depth in range(depth_limit):
            max_depth_reached = max(max_depth_reached, depth + 1)
            child_limit = _progressive_widening_limit(protocol, node.visits + 1)
            missing = max(0, child_limit - len(node.children))
            if missing:
                proposals = propose_actions(
                    state,
                    rng,
                    missing,
                    frozenset(node.children),
                )
                for proposal in proposals:
                    action = tuple(int(value) for value in proposal.action)
                    node.children.setdefault(
                        action,
                        _ActionEdge(action=action, prior=float(proposal.prior)),
                    )
            if not node.children:
                leaf_value = float(rollout_value(state, rng))
                break

            edge = _select_edge(node, exploration_c=float(protocol.exploration_c))
            outcome = transition(state, edge.action, rng)
            reward = float(outcome.reward)
            if not math.isfinite(reward):
                raise ValueError("MCTS transition returned a non-finite reward.")
            path.append((node, edge, reward))
            state = outcome.state
            if bool(outcome.terminal):
                leaf_value = 0.0
                break

            next_key = (depth + 1, state_key(state))
            next_node = nodes.get(next_key)
            if next_node is None:
                next_node = _SearchNode()
                nodes[next_key] = next_node
                leaf_value = float(rollout_value(state, rng))
                node = next_node
                break
            node = next_node
        else:
            leaf_value = float(rollout_value(state, rng))

        if not math.isfinite(leaf_value):
            raise ValueError("MCTS rollout returned a non-finite value.")
        total = float(leaf_value)
        if not path:
            node.visits += 1
            node.value_sum += total
            continue
        for visited_node, edge, reward in reversed(path):
            total = reward + (float(protocol.discount) * total)
            edge.visits += 1
            edge.value_sum += total
            visited_node.visits += 1
            visited_node.value_sum += total

    visited_edges = [edge for edge in root.children.values() if edge.visits > 0]
    if not visited_edges:
        raise RuntimeError("MCTS completed without visiting a root action.")
    selected = max(
        visited_edges,
        key=lambda edge: (edge.visits, edge.mean_value, edge.action),
    )
    stats = tuple(
        MctsActionStats(
            action=edge.action,
            prior=float(edge.prior),
            visits=int(edge.visits),
            mean_value=float(edge.mean_value),
        )
        for edge in sorted(
            visited_edges,
            key=lambda item: (-item.visits, -item.mean_value, item.action),
        )
    )
    return MctsSearchResult(
        selected_action=selected.action,
        selected_value=float(selected.mean_value),
        simulations=simulation_count,
        max_depth_reached=int(max_depth_reached),
        nodes_expanded=len(nodes),
        action_stats=stats,
    )


def evaluate_action_return(
    state: Any,
    action: JointAction,
    *,
    transition: TransitionFn,
    rollout_value: RolloutFn,
    seed: int,
    discount: float = 1.0,
) -> float:
    """Evaluate one forced root action followed by fixed-policy continuation."""

    if not 0.0 <= float(discount) <= 1.0:
        raise ValueError("discount must be in [0, 1].")
    rng = np.random.default_rng(int(seed))
    outcome = transition(state, tuple(int(value) for value in action), rng)
    reward = float(outcome.reward)
    if not math.isfinite(reward):
        raise ValueError("Transition returned a non-finite reward.")
    continuation = 0.0
    if not bool(outcome.terminal):
        continuation = float(rollout_value(outcome.state, rng))
        if not math.isfinite(continuation):
            raise ValueError("Rollout returned a non-finite value.")
    return reward + (float(discount) * continuation)


def compare_actions_with_paired_rollouts(
    state: Any,
    *,
    policy_action: JointAction,
    planner_action: JointAction,
    transition: TransitionFn,
    rollout_value: RolloutFn,
    rollout_seeds: Sequence[int],
    discount: float = 1.0,
    bootstrap_seed: int = 0,
    bootstrap_samples: int = 2000,
) -> PairedActionComparison:
    """Compare root actions using the same chance stream for every seed."""

    seeds = tuple(int(seed) for seed in rollout_seeds)
    if not seeds:
        raise ValueError("rollout_seeds must contain at least one seed.")
    policy_returns = tuple(
        evaluate_action_return(
            state,
            policy_action,
            transition=transition,
            rollout_value=rollout_value,
            seed=seed,
            discount=discount,
        )
        for seed in seeds
    )
    planner_returns = tuple(
        evaluate_action_return(
            state,
            planner_action,
            transition=transition,
            rollout_value=rollout_value,
            seed=seed,
            discount=discount,
        )
        for seed in seeds
    )
    regrets = tuple(
        float(planner) - float(policy)
        for policy, planner in zip(policy_returns, planner_returns, strict=True)
    )
    low, high = paired_bootstrap_mean_interval(
        regrets,
        seed=int(bootstrap_seed),
        samples=int(bootstrap_samples),
    )
    return PairedActionComparison(
        policy_action=tuple(int(value) for value in policy_action),
        planner_action=tuple(int(value) for value in planner_action),
        rollout_seeds=seeds,
        policy_returns=policy_returns,
        planner_returns=planner_returns,
        regret_samples=regrets,
        policy_value=float(np.mean(policy_returns)),
        planner_value=float(np.mean(planner_returns)),
        regret_mean=float(np.mean(regrets)),
        regret_ci95_low=float(low),
        regret_ci95_high=float(high),
    )


@dataclass(frozen=True)
class RegretRecord:
    state_id: str
    policy_value: float
    planner_value: float
    policy_action: JointAction
    planner_action: JointAction

    @property
    def regret(self) -> float:
        return float(self.planner_value) - float(self.policy_value)

    @property
    def exact_action_agreement(self) -> bool:
        return tuple(self.policy_action) == tuple(self.planner_action)


def paired_bootstrap_mean_interval(
    values: Sequence[float],
    *,
    seed: int = 0,
    samples: int = 2000,
    confidence: float = 0.95,
) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("values must be a non-empty one-dimensional sequence.")
    if not np.all(np.isfinite(array)):
        raise ValueError("values must be finite.")
    if int(samples) < 1:
        raise ValueError("samples must be positive.")
    if not 0.0 < float(confidence) < 1.0:
        raise ValueError("confidence must be in (0, 1).")
    if array.size == 1:
        value = float(array[0])
        return value, value
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, array.size, size=(int(samples), array.size))
    means = array[indices].mean(axis=1)
    tail = (1.0 - float(confidence)) / 2.0
    return float(np.quantile(means, tail)), float(np.quantile(means, 1.0 - tail))


def summarize_regret_records(
    records: Sequence[RegretRecord],
    *,
    bootstrap_seed: int = 0,
    bootstrap_samples: int = 2000,
) -> dict[str, Any]:
    if not records:
        raise ValueError("At least one regret record is required.")
    regrets = np.asarray([record.regret for record in records], dtype=np.float64)
    low, high = paired_bootstrap_mean_interval(
        regrets,
        seed=int(bootstrap_seed),
        samples=int(bootstrap_samples),
    )
    agreements = np.asarray(
        [record.exact_action_agreement for record in records], dtype=np.float64
    )
    return {
        "schema_version": MCTS_REGRET_SCHEMA_VERSION,
        "state_count": int(regrets.size),
        "exact_action_agreement_rate": float(agreements.mean()),
        "regret_mean": float(regrets.mean()),
        "regret_median": float(np.median(regrets)),
        "regret_p10": float(np.quantile(regrets, 0.10)),
        "regret_p90": float(np.quantile(regrets, 0.90)),
        "regret_positive_fraction": float(np.mean(regrets > 0.0)),
        "regret_mean_ci95_low": low,
        "regret_mean_ci95_high": high,
    }


def search_result_payload(result: MctsSearchResult) -> dict[str, Any]:
    """Return a JSON-ready representation of one MCTS decision."""

    return {
        "selected_action": list(result.selected_action),
        "selected_value": float(result.selected_value),
        "simulations": int(result.simulations),
        "max_depth_reached": int(result.max_depth_reached),
        "nodes_expanded": int(result.nodes_expanded),
        "action_stats": [
            {
                "action": list(item.action),
                "prior": float(item.prior),
                "visits": int(item.visits),
                "mean_value": float(item.mean_value),
            }
            for item in result.action_stats
        ],
    }


def paired_comparison_payload(result: PairedActionComparison) -> dict[str, Any]:
    return {
        "policy_action": list(result.policy_action),
        "planner_action": list(result.planner_action),
        "rollout_seeds": list(result.rollout_seeds),
        "policy_returns": list(result.policy_returns),
        "planner_returns": list(result.planner_returns),
        "regret_samples": list(result.regret_samples),
        "policy_value": float(result.policy_value),
        "planner_value": float(result.planner_value),
        "regret_mean": float(result.regret_mean),
        "regret_ci95_low": float(result.regret_ci95_low),
        "regret_ci95_high": float(result.regret_ci95_high),
    }


def calibration_protocol_payload(
    config: MctsRegretConfig | None = None,
) -> Mapping[str, Any]:
    protocol = config or MctsRegretConfig()
    return {
        "schema_version": MCTS_REGRET_SCHEMA_VERSION,
        "search_depths": list(protocol.search_depths),
        "primary_search_depth": int(protocol.primary_search_depth),
        "simulations_per_action": int(protocol.simulations_per_action),
        "calibration_budgets": list(protocol.calibration_budgets),
        "exploration_c": float(protocol.exploration_c),
        "progressive_widening_c": float(protocol.progressive_widening_c),
        "progressive_widening_alpha": float(protocol.progressive_widening_alpha),
        "max_joint_actions_per_node": int(protocol.max_joint_actions_per_node),
        "discount": float(protocol.discount),
    }
