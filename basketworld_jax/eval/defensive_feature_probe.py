"""Offline probes for defensive MCTS action labels.

The probe consumes replayable states and completed root-action comparisons from
the JAX MCTS regret evaluator.  It never runs search.  Its purpose is to
separate three explanations for defensive policy regret:

* the current observation already contains learnable information, but PPO has
  weak credit assignment;
* explicit defensive geometry makes the target substantially easier to learn;
* the MCTS target is unstable or not representable by a small policy-like
  model.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from basketworld_jax.config.environment import JaxEnvironmentConfig
from basketworld_jax.env.minimal import (
    TEAM_A,
    KernelState,
    build_action_masks_batch,
    build_kernel_static_from_config,
    build_policy_observation_batch_with_role_flag,
    build_shot_profile_batch,
)
from basketworld_jax.eval.mcts_regret_jax import load_persisted_sample_states
from basketworld_jax.provenance import capture_source_provenance
from basketworld_jax.train.cli import ensure_jax_available


PROBE_SCHEMA_VERSION = 1
DEFENSIVE_ACTION_COUNT = 7
FEATURE_VARIANTS = (
    "current_observation",
    "directional_geometry",
    "threat_and_responsibility",
)
DIRECTIONAL_FEATURE_LABELS = (
    "ball_handler_delta_q_norm",
    "ball_handler_delta_r_norm",
    "nearest_offense_delta_q_norm",
    "nearest_offense_delta_r_norm",
    "ball_handler_valid",
    "nearest_offense_valid",
)
THREAT_FEATURE_LABELS = (
    "nearest_offense_expected_shot_value_norm",
    "closest_defender_to_ball_handler",
    "closest_defender_to_highest_expected_shot_value",
)


@dataclass(frozen=True)
class ProbeTrainingConfig:
    split_seed: int = 17
    training_seeds: tuple[int, ...] = (1, 2, 3)
    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    learning_rate: float = 1.0e-3
    epochs: int = 300
    patience: int = 40
    embed_dim: int = 64
    token_mlp_dim: int = 64
    num_heads: int = 4
    num_layers: int = 1
    bootstrap_samples: int = 2000
    high_regret_quantile: float = 0.75
    learning_curve_fractions: tuple[float, ...] = (0.25, 0.50, 1.0)

    def validate(self) -> None:
        if not 0.0 < float(self.train_fraction) < 1.0:
            raise ValueError("train_fraction must be between zero and one.")
        if not 0.0 < float(self.validation_fraction) < 1.0:
            raise ValueError("validation_fraction must be between zero and one.")
        if float(self.train_fraction) + float(self.validation_fraction) >= 1.0:
            raise ValueError("train and validation fractions must leave a test split.")
        if not self.training_seeds:
            raise ValueError("At least one training seed is required.")
        if int(self.epochs) < 1 or int(self.patience) < 1:
            raise ValueError("epochs and patience must be positive.")
        if int(self.embed_dim) < 1 or int(self.token_mlp_dim) < 1:
            raise ValueError("Probe dimensions must be positive.")
        if int(self.num_heads) < 1 or int(self.num_layers) < 1:
            raise ValueError("Probe attention settings must be positive.")
        if int(self.embed_dim) % int(self.num_heads) != 0:
            raise ValueError("embed_dim must be divisible by num_heads.")
        if int(self.bootstrap_samples) < 1:
            raise ValueError("bootstrap_samples must be positive.")
        if not 0.0 < float(self.high_regret_quantile) < 1.0:
            raise ValueError("high_regret_quantile must be between zero and one.")
        if not self.learning_curve_fractions:
            raise ValueError("At least one learning-curve fraction is required.")
        if any(
            not 0.0 < float(value) <= 1.0 for value in self.learning_curve_fractions
        ):
            raise ValueError("Learning-curve fractions must be in (0, 1].")


@dataclass(frozen=True)
class DefensiveProbeDataset:
    state_ids: tuple[str, ...]
    base_player_tokens: np.ndarray
    global_features: np.ndarray
    extra_features: Mapping[str, np.ndarray]
    action_masks: np.ndarray
    labels: np.ndarray
    policy_actions: np.ndarray
    regret: np.ndarray
    stable_labels: np.ndarray
    stability_available: bool
    split: np.ndarray
    source_payload: Mapping[str, Any]
    source_metadata: Mapping[str, Any]
    feature_labels: Mapping[str, tuple[str, ...]]


def _json_payload(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {source}.")
    return payload


def _resolve_artifact_path(raw_path: str | Path, *, relative_to: Path) -> Path:
    candidate = Path(raw_path).expanduser()
    if candidate.exists():
        return candidate.resolve()
    local = relative_to / candidate.name
    if local.exists():
        return local.resolve()
    raise FileNotFoundError(f"Unable to resolve study artifact {raw_path!s}.")


def _stack_states(states: Sequence[KernelState]) -> KernelState:
    if not states:
        raise ValueError("At least one replay state is required.")
    values = []
    for field_index, _ in enumerate(KernelState._fields):
        values.append(
            np.concatenate(
                [np.asarray(state[field_index]) for state in states],
                axis=0,
            )
        )
    return KernelState(*values)


def make_state_splits(
    state_ids: Sequence[str],
    *,
    seed: int,
    train_fraction: float,
    validation_fraction: float,
) -> np.ndarray:
    """Return deterministic train/validation/test labels grouped by state."""

    count = len(state_ids)
    if count < 3:
        raise ValueError("At least three states are required for three splits.")
    if len(set(str(value) for value in state_ids)) != count:
        raise ValueError("State IDs must be unique before splitting.")
    if not 0.0 < float(train_fraction) < 1.0:
        raise ValueError("train_fraction must be between zero and one.")
    if not 0.0 < float(validation_fraction) < 1.0:
        raise ValueError("validation_fraction must be between zero and one.")
    if float(train_fraction) + float(validation_fraction) >= 1.0:
        raise ValueError("Split fractions must leave a non-empty test set.")

    rng = np.random.default_rng(int(seed))
    order = rng.permutation(count)
    train_count = max(1, int(np.floor(count * float(train_fraction))))
    validation_count = max(1, int(np.floor(count * float(validation_fraction))))
    if train_count + validation_count >= count:
        validation_count = max(1, count - train_count - 1)
    split = np.full(count, "test", dtype="<U10")
    split[order[:train_count]] = "train"
    split[order[train_count : train_count + validation_count]] = "validation"
    return split


def _hex_distance(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    delta = np.asarray(right, dtype=np.int32) - np.asarray(left, dtype=np.int32)
    return (
        np.abs(delta[..., 0])
        + np.abs(delta[..., 1])
        + np.abs(delta[..., 0] + delta[..., 1])
    ) // 2


def compute_defensive_relational_features(
    *,
    positions: np.ndarray,
    ball_holder: np.ndarray,
    offense_ids: np.ndarray,
    defense_ids: np.ndarray,
    expected_shot_values: np.ndarray,
    court_norm_den: float,
    offense_on_court: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute direction and responsibility features in active roster order.

    ``expected_shot_values`` is indexed by stable player ID.  The legacy shot
    profile implementation calls this quantity ``expected_points`` internally;
    this study uses the current domain term, expected shot value.
    """

    positions = np.asarray(positions, dtype=np.int32)
    ball_holder = np.asarray(ball_holder, dtype=np.int32)
    offense_ids = np.asarray(offense_ids, dtype=np.int32)
    defense_ids = np.asarray(defense_ids, dtype=np.int32)
    expected_shot_values = np.asarray(expected_shot_values, dtype=np.float32)
    if positions.ndim != 3 or positions.shape[-1] != 2:
        raise ValueError("positions must have shape (states, players, 2).")
    state_count = positions.shape[0]
    if offense_ids.shape[0] != state_count or defense_ids.shape[0] != state_count:
        raise ValueError("Active roster IDs must have one row per state.")
    if offense_ids.shape != defense_ids.shape:
        raise ValueError("Offense and defense roster widths must match.")
    if float(court_norm_den) <= 0.0:
        raise ValueError("court_norm_den must be positive.")

    row_index = np.arange(state_count, dtype=np.int32)[:, None]
    offense_positions = positions[row_index, offense_ids]
    defense_positions = positions[row_index, defense_ids]
    if offense_on_court is None:
        offense_valid = np.ones(offense_ids.shape, dtype=bool)
    else:
        offense_valid = np.asarray(offense_on_court, dtype=bool)
        if offense_valid.shape != offense_ids.shape:
            raise ValueError("offense_on_court must match offense_ids shape.")

    pair_distances = _hex_distance(
        defense_positions[:, :, None, :],
        offense_positions[:, None, :, :],
    ).astype(np.float32)
    masked_pair_distances = np.where(offense_valid[:, None, :], pair_distances, np.inf)
    nearest_slots = np.argmin(masked_pair_distances, axis=2)
    nearest_valid = np.any(offense_valid, axis=1)
    nearest_positions = offense_positions[
        np.arange(state_count)[:, None], nearest_slots
    ]
    nearest_delta = nearest_positions - defense_positions
    nearest_delta = np.where(nearest_valid[:, None, None], nearest_delta, 0).astype(
        np.float32
    )

    holder_matches = offense_ids == ball_holder[:, None]
    holder_valid = np.any(holder_matches & offense_valid, axis=1)
    safe_holder_slots = np.argmax(holder_matches, axis=1)
    holder_positions = offense_positions[np.arange(state_count), safe_holder_slots]
    holder_delta = holder_positions[:, None, :] - defense_positions
    holder_delta = np.where(holder_valid[:, None, None], holder_delta, 0).astype(
        np.float32
    )

    directional = np.concatenate(
        [
            holder_delta / float(court_norm_den),
            nearest_delta / float(court_norm_den),
            np.broadcast_to(
                holder_valid[:, None, None],
                (*defense_ids.shape, 1),
            ).astype(np.float32),
            np.broadcast_to(
                nearest_valid[:, None, None],
                (*defense_ids.shape, 1),
            ).astype(np.float32),
        ],
        axis=2,
    )

    offense_shot_values = expected_shot_values[row_index, offense_ids]
    nearest_shot_values = np.take_along_axis(offense_shot_values, nearest_slots, axis=1)
    nearest_shot_values = np.where(nearest_valid[:, None], nearest_shot_values, 0.0)

    holder_distances = _hex_distance(
        defense_positions,
        holder_positions[:, None, :],
    ).astype(np.float32)
    holder_min = np.min(holder_distances, axis=1, keepdims=True)
    closest_to_holder = (
        holder_valid[:, None] & (holder_distances == holder_min)
    ).astype(np.float32)

    masked_shot_values = np.where(offense_valid, offense_shot_values, -np.inf)
    highest_value_slots = np.argmax(masked_shot_values, axis=1)
    highest_value_positions = offense_positions[
        np.arange(state_count), highest_value_slots
    ]
    highest_value_distances = _hex_distance(
        defense_positions,
        highest_value_positions[:, None, :],
    ).astype(np.float32)
    highest_min = np.min(highest_value_distances, axis=1, keepdims=True)
    closest_to_highest = (
        nearest_valid[:, None] & (highest_value_distances == highest_min)
    ).astype(np.float32)

    threat = np.stack(
        [
            nearest_shot_values / 3.0,
            closest_to_holder,
            closest_to_highest,
        ],
        axis=2,
    ).astype(np.float32)
    return directional.astype(np.float32), threat


def _observation_kwargs(policy_spec: Mapping[str, Any]) -> dict[str, Any]:
    schema_version = int(policy_spec.get("observation_schema_version", 1))
    return {
        "model_type": str(policy_spec.get("model_type", "attention")),
        "rebound_win_prob_features": bool(
            policy_spec.get("rebound_win_prob_features", False)
        ),
        "rebound_target_observation_features": bool(
            policy_spec.get("rebound_target_observation_features", True)
        ),
        "multi_possession_features": bool(
            policy_spec.get("multi_possession_features", False)
        ),
        "check_clearance_features": schema_version >= 5,
        "team_clearance_features": schema_version >= 6,
    }


def _planner_actions_by_state(
    payload: Mapping[str, Any], *, depth: int
) -> dict[str, tuple[int, ...]]:
    actions: dict[str, tuple[int, ...]] = {}
    for record in payload.get("records", []):
        sample = dict(record.get("sample", {}))
        state_id = str(sample.get("state_id", ""))
        depth_result = dict(record.get("depth_results", {})).get(str(int(depth)))
        if not state_id or not isinstance(depth_result, Mapping):
            continue
        comparison = depth_result.get("paired_comparison", {})
        action = tuple(int(value) for value in comparison.get("planner_action", []))
        if action:
            actions[state_id] = action
    return actions


def _validate_stability_payload(
    source: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    state_ids: Sequence[str],
    depth: int,
) -> dict[str, tuple[int, ...]]:
    """Require a comparison artifact to differ only in search sampling."""

    source_provenance = dict(source.get("provenance", {}))
    candidate_provenance = dict(candidate.get("provenance", {}))
    identity_fields = (
        "evaluated_checkpoint_update_index",
        "opponent_checkpoint_update_index",
        "evaluated_policy_spec",
        "opponent_policy_spec",
        "evaluated_stable_team",
        "opponent_stable_team",
        "evaluated_action_mode",
        "opponent_action_mode",
        "resolved_environment_config",
        "value_target",
    )
    mismatches = [
        field
        for field in identity_fields
        if source_provenance.get(field) != candidate_provenance.get(field)
    ]
    if mismatches:
        raise ValueError(
            "Stability artifact policy/environment provenance differs for: "
            + ", ".join(mismatches)
        )
    actions = _planner_actions_by_state(candidate, depth=depth)
    missing = [state_id for state_id in state_ids if state_id not in actions]
    if missing:
        raise ValueError(
            "Stability artifact is not complete over the source states; "
            f"missing {len(missing)} of {len(state_ids)} states."
        )
    return actions


def stable_action_mask(
    state_ids: Sequence[str],
    source_actions: Mapping[str, Sequence[int]],
    comparison_actions: Sequence[Mapping[str, Sequence[int]]],
) -> tuple[np.ndarray, bool]:
    """Return joint-action stability across at least two completed artifacts."""

    if not comparison_actions:
        return np.zeros(len(state_ids), dtype=bool), False
    stable = []
    for state_id in state_ids:
        source = tuple(int(value) for value in source_actions.get(str(state_id), ()))
        matches = bool(source)
        for candidate_map in comparison_actions:
            candidate = tuple(
                int(value) for value in candidate_map.get(str(state_id), ())
            )
            matches = matches and bool(candidate) and candidate == source
        stable.append(matches)
    return np.asarray(stable, dtype=bool), True


def prepare_probe_dataset(
    results_path: str | Path,
    *,
    depth: int | None = None,
    stability_result_paths: Sequence[str | Path] = (),
    split_seed: int = 17,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> DefensiveProbeDataset:
    """Load completed MCTS outputs and construct policy-like probe inputs."""

    results_file = Path(results_path).expanduser().resolve()
    source = _json_payload(results_file)
    if not source.get("records"):
        raise ValueError("MCTS results contain no completed per-state records.")
    primary_depth = int(
        depth
        if depth is not None
        else source.get("aggregate", {}).get("primary_search_depth", 0)
    )
    if primary_depth < 1:
        raise ValueError("A positive search depth is required.")

    sample_artifacts = dict(source.get("sample_artifacts", {}))
    metadata_path = _resolve_artifact_path(
        sample_artifacts.get("metadata", "sampled_states.json"),
        relative_to=results_file.parent,
    )
    sample_records, states, sample_metadata = load_persisted_sample_states(
        metadata_path
    )
    state_index = {
        str(record.get("state_id", "")): index
        for index, record in enumerate(sample_records)
    }

    chosen_states: list[KernelState] = []
    state_ids: list[str] = []
    labels: list[list[int]] = []
    policy_actions: list[list[int]] = []
    regrets: list[float] = []
    for record in source.get("records", []):
        sample = dict(record.get("sample", {}))
        if str(sample.get("evaluated_role", "")) != "defense":
            continue
        state_id = str(sample.get("state_id", ""))
        if state_id not in state_index:
            raise ValueError(f"Missing replay state for MCTS record {state_id!r}.")
        depth_result = dict(record.get("depth_results", {})).get(str(primary_depth))
        if not isinstance(depth_result, Mapping):
            raise ValueError(
                f"MCTS record {state_id!r} has no depth-{primary_depth} result."
            )
        comparison = dict(depth_result.get("paired_comparison", {}))
        planner_action = [int(value) for value in comparison.get("planner_action", [])]
        policy_action = [int(value) for value in comparison.get("policy_action", [])]
        if not planner_action or len(planner_action) != len(policy_action):
            raise ValueError(f"Invalid action labels for state {state_id!r}.")
        if any(
            value < 0 or value >= DEFENSIVE_ACTION_COUNT for value in planner_action
        ):
            raise ValueError(
                f"State {state_id!r} contains a non-defensive MCTS action."
            )
        if any(value < 0 or value >= DEFENSIVE_ACTION_COUNT for value in policy_action):
            raise ValueError(
                f"State {state_id!r} contains a non-defensive policy action."
            )
        chosen_states.append(states[state_index[state_id]])
        state_ids.append(state_id)
        labels.append(planner_action)
        policy_actions.append(policy_action)
        regrets.append(float(comparison.get("regret_mean", 0.0)))

    if not chosen_states:
        raise ValueError("MCTS results contain no defensive states.")

    provenance = dict(source.get("provenance", sample_metadata.get("provenance", {})))
    policy_spec = dict(provenance.get("evaluated_policy_spec", {}))
    if str(policy_spec.get("model_type", "")) != "attention":
        raise ValueError(
            "The defensive feature probe currently requires token observations."
        )
    environment_values = dict(provenance.get("resolved_environment_config", {}))
    environment_config = JaxEnvironmentConfig.from_mapping(environment_values)
    jax, jnp = ensure_jax_available("defensive feature probe")
    static = build_kernel_static_from_config(environment_config, jnp)
    state_batch = _stack_states(chosen_states)
    state_batch = jax.tree.map(jnp.asarray, state_batch)
    role_flag = jnp.full((len(chosen_states),), -1.0, dtype=jnp.float32)
    observation = build_policy_observation_batch_with_role_flag(
        static,
        state_batch,
        role_flag,
        jnp,
        **_observation_kwargs(policy_spec),
    )
    observation_np = np.asarray(jax.device_get(observation), dtype=np.float32)

    token_player_count = int(policy_spec.get("token_player_count", 0))
    token_dim = int(policy_spec.get("token_dim", 0))
    global_dim = int(policy_spec.get("global_dim", 0))
    training_player_count = int(policy_spec.get("training_player_count", 0))
    expected_width = (token_player_count * token_dim) + global_dim + 1
    if observation_np.shape[1] != expected_width:
        raise ValueError(
            f"Packed observation width {observation_np.shape[1]} does not match "
            f"policy spec width {expected_width}."
        )
    if token_player_count != 2 * training_player_count:
        raise ValueError(
            "Probe expects offense and defense token blocks of equal width."
        )
    player_width = token_player_count * token_dim
    base_tokens = observation_np[:, :player_width].reshape(
        len(chosen_states), token_player_count, token_dim
    )
    global_features = observation_np[:, player_width : player_width + global_dim]

    offense_team = np.asarray(state_batch.offense_team, dtype=np.int32)
    team_a_ids = np.asarray(jax.device_get(static.offense_ids), dtype=np.int32)
    team_b_ids = np.asarray(jax.device_get(static.defense_ids), dtype=np.int32)
    offense_ids = np.where(
        offense_team[:, None] == int(TEAM_A),
        team_a_ids[None, :],
        team_b_ids[None, :],
    )
    defense_ids = np.where(
        offense_team[:, None] == int(TEAM_A),
        team_b_ids[None, :],
        team_a_ids[None, :],
    )
    evaluated_team = str(provenance.get("evaluated_stable_team", "team_a"))
    expected_evaluated_team = np.where(offense_team == int(TEAM_A), "team_a", "team_b")
    if np.any(expected_evaluated_team == evaluated_team):
        raise ValueError("At least one selected record is not a defensive decision.")

    all_action_masks = np.asarray(
        jax.device_get(build_action_masks_batch(static, state_batch, jnp)),
        dtype=bool,
    )
    controlled_ids = team_a_ids if evaluated_team == "team_a" else team_b_ids
    controlled_masks = all_action_masks[:, controlled_ids, :DEFENSIVE_ACTION_COUNT]

    positions = np.asarray(jax.device_get(state_batch.positions), dtype=np.int32)
    cell_coords = np.asarray(jax.device_get(static.cell_coords), dtype=np.int32)
    offense_positions = positions[np.arange(len(chosen_states))[:, None], offense_ids]
    offense_on_court = np.any(
        np.all(
            offense_positions[:, :, None, :] == cell_coords[None, None, :, :],
            axis=-1,
        ),
        axis=-1,
    )
    shot_profile = build_shot_profile_batch(static, state_batch, jnp)
    expected_shot_values = np.asarray(
        jax.device_get(shot_profile["expected_points"]), dtype=np.float32
    )
    directional, threat = compute_defensive_relational_features(
        positions=positions,
        ball_holder=np.asarray(jax.device_get(state_batch.ball_holder)),
        offense_ids=offense_ids,
        defense_ids=defense_ids,
        expected_shot_values=expected_shot_values,
        court_norm_den=float(np.asarray(jax.device_get(static.court_norm_den))),
        offense_on_court=offense_on_court,
    )
    zero_extra = np.zeros((len(chosen_states), token_player_count, 0), dtype=np.float32)
    directional_extra = np.zeros(
        (len(chosen_states), token_player_count, directional.shape[2]),
        dtype=np.float32,
    )
    directional_extra[:, training_player_count:, :] = directional
    threat_extra = np.zeros(
        (
            len(chosen_states),
            token_player_count,
            directional.shape[2] + threat.shape[2],
        ),
        dtype=np.float32,
    )
    threat_extra[:, training_player_count:, :] = np.concatenate(
        [directional, threat], axis=2
    )

    source_actions = _planner_actions_by_state(source, depth=primary_depth)
    comparison_actions = [
        _validate_stability_payload(
            source,
            _json_payload(path),
            state_ids=state_ids,
            depth=primary_depth,
        )
        for path in stability_result_paths
    ]
    stable_labels, stability_available = stable_action_mask(
        state_ids,
        source_actions,
        comparison_actions,
    )
    split = make_state_splits(
        state_ids,
        seed=int(split_seed),
        train_fraction=float(train_fraction),
        validation_fraction=float(validation_fraction),
    )
    return DefensiveProbeDataset(
        state_ids=tuple(state_ids),
        base_player_tokens=base_tokens,
        global_features=global_features,
        extra_features={
            "current_observation": zero_extra,
            "directional_geometry": directional_extra,
            "threat_and_responsibility": threat_extra,
        },
        action_masks=controlled_masks,
        labels=np.asarray(labels, dtype=np.int32),
        policy_actions=np.asarray(policy_actions, dtype=np.int32),
        regret=np.asarray(regrets, dtype=np.float32),
        stable_labels=stable_labels,
        stability_available=stability_available,
        split=split,
        source_payload=source,
        source_metadata=sample_metadata,
        feature_labels={
            "current_observation": (),
            "directional_geometry": DIRECTIONAL_FEATURE_LABELS,
            "threat_and_responsibility": (
                *DIRECTIONAL_FEATURE_LABELS,
                *THREAT_FEATURE_LABELS,
            ),
        },
    )


def _direction_bucket(actions: np.ndarray) -> np.ndarray:
    values = np.asarray(actions, dtype=np.int32)
    buckets = np.full(values.shape, -1, dtype=np.int32)
    buckets[np.isin(values, (2, 3))] = 0
    buckets[np.isin(values, (1, 4))] = 1
    buckets[np.isin(values, (5, 6))] = 2
    return buckets


def _bootstrap_ratio(
    numerator: np.ndarray,
    denominator: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    numerator = np.asarray(numerator, dtype=np.float64)
    denominator = np.asarray(denominator, dtype=np.float64)
    if numerator.shape != denominator.shape or numerator.ndim != 1:
        raise ValueError("Bootstrap contributions must be matching vectors.")
    valid = denominator > 0.0
    estimate_den = float(np.sum(denominator[valid]))
    if estimate_den <= 0.0:
        return {
            "estimate": None,
            "ci95_low": None,
            "ci95_high": None,
            "state_count": int(numerator.shape[0]),
            "decision_count": 0,
        }
    estimate = float(np.sum(numerator[valid]) / estimate_den)
    rng = np.random.default_rng(int(seed))
    draws = rng.integers(0, numerator.shape[0], size=(int(samples), numerator.shape[0]))
    drawn_num = np.sum(numerator[draws], axis=1)
    drawn_den = np.sum(denominator[draws], axis=1)
    values = drawn_num[drawn_den > 0.0] / drawn_den[drawn_den > 0.0]
    return {
        "estimate": estimate,
        "ci95_low": float(np.quantile(values, 0.025)),
        "ci95_high": float(np.quantile(values, 0.975)),
        "state_count": int(numerator.shape[0]),
        "decision_count": int(round(estimate_den)),
    }


def classification_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    """Return state-bootstrapped exact and directional probe metrics."""

    labels = np.asarray(labels, dtype=np.int32)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.shape != (*labels.shape, DEFENSIVE_ACTION_COUNT):
        raise ValueError("Probability shape does not match defensive labels.")
    predictions = np.argmax(probabilities, axis=-1).astype(np.int32)
    state_count, player_count = labels.shape
    state_den = np.full(state_count, player_count, dtype=np.float64)
    correct = predictions == labels
    clipped = np.clip(probabilities, 1.0e-12, 1.0)
    target_probability = np.take_along_axis(clipped, labels[..., None], axis=-1)[..., 0]
    nll = -np.log(target_probability)
    move_agreement = (predictions > 0) == (labels > 0)
    both_move = (predictions > 0) & (labels > 0)
    exact_move = both_move & (predictions == labels)
    bucket_move = both_move & (
        _direction_bucket(predictions) == _direction_bucket(labels)
    )
    sorted_actions = np.argsort(-probabilities, axis=-1)
    top2 = np.any(sorted_actions[..., :2] == labels[..., None], axis=-1)
    top3 = np.any(sorted_actions[..., :3] == labels[..., None], axis=-1)

    def ratio(num: np.ndarray, den: np.ndarray, offset: int) -> dict[str, Any]:
        return _bootstrap_ratio(
            np.asarray(num, dtype=np.float64),
            np.asarray(den, dtype=np.float64),
            samples=int(bootstrap_samples),
            seed=int(bootstrap_seed) + offset,
        )

    confusion = np.zeros(
        (DEFENSIVE_ACTION_COUNT, DEFENSIVE_ACTION_COUNT), dtype=np.int64
    )
    np.add.at(confusion, (labels.reshape(-1), predictions.reshape(-1)), 1)
    return {
        "exact_player_action_accuracy": ratio(correct.sum(axis=1), state_den, 1),
        "negative_log_likelihood": ratio(nll.sum(axis=1), state_den, 2),
        "exact_joint_action_accuracy": ratio(
            np.all(correct, axis=1).astype(np.float64),
            np.ones(state_count),
            3,
        ),
        "move_vs_noop_accuracy": ratio(move_agreement.sum(axis=1), state_den, 4),
        "exact_move_direction_accuracy": ratio(
            exact_move.sum(axis=1), both_move.sum(axis=1), 5
        ),
        "three_bucket_move_direction_accuracy": ratio(
            bucket_move.sum(axis=1), both_move.sum(axis=1), 6
        ),
        "top_2_action_accuracy": ratio(top2.sum(axis=1), state_den, 7),
        "top_3_action_accuracy": ratio(top3.sum(axis=1), state_den, 8),
        "confusion_matrix": confusion.tolist(),
    }


def _deterministic_action_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    probabilities = np.full(
        (*np.asarray(labels).shape, DEFENSIVE_ACTION_COUNT),
        1.0e-6,
        dtype=np.float64,
    )
    np.put_along_axis(
        probabilities,
        np.asarray(predictions, dtype=np.int32)[..., None],
        1.0,
        axis=-1,
    )
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    metrics = classification_metrics(
        labels,
        probabilities,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    metrics["negative_log_likelihood"] = {
        "estimate": None,
        "ci95_low": None,
        "ci95_high": None,
        "state_count": int(np.asarray(labels).shape[0]),
        "decision_count": 0,
        "reason": "Full baseline action probabilities were not persisted.",
    }
    return metrics


def _frequency_baseline(
    labels: np.ndarray,
    action_masks: np.ndarray,
    train_mask: np.ndarray,
) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int32)
    probabilities = np.ones((labels.shape[1], DEFENSIVE_ACTION_COUNT), dtype=np.float64)
    for player_index in range(labels.shape[1]):
        probabilities[player_index] += np.bincount(
            labels[train_mask, player_index], minlength=DEFENSIVE_ACTION_COUNT
        )
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    out = np.broadcast_to(
        probabilities[None, :, :],
        (*labels.shape, DEFENSIVE_ACTION_COUNT),
    ).copy()
    out = np.where(action_masks, out, 0.0)
    denominator = out.sum(axis=-1, keepdims=True)
    fallback = np.zeros_like(out)
    fallback[..., 0] = 1.0
    return np.where(denominator > 0.0, out / np.maximum(denominator, 1.0e-12), fallback)


def _build_probe_module(
    *,
    controlled_player_count: int,
    embed_dim: int,
    token_mlp_dim: int,
    num_heads: int,
    num_layers: int,
):
    from flax import linen as nn

    class DefensiveActionProbe(nn.Module):
        @nn.compact
        def __call__(self, player_tokens, global_features):
            import jax.numpy as jnp

            expanded = jnp.broadcast_to(
                global_features[:, None, :],
                (
                    player_tokens.shape[0],
                    player_tokens.shape[1],
                    global_features.shape[-1],
                ),
            )
            hidden = jnp.concatenate([player_tokens, expanded], axis=-1)
            hidden = nn.relu(nn.Dense(int(token_mlp_dim), name="token_mlp_0")(hidden))
            hidden = nn.Dense(int(embed_dim), name="token_mlp_1")(hidden)
            for layer_index in range(int(num_layers)):
                attended = nn.SelfAttention(
                    num_heads=int(num_heads),
                    qkv_features=int(embed_dim),
                    out_features=int(embed_dim),
                    deterministic=True,
                    name=f"self_attention_{layer_index}",
                )(hidden)
                hidden = nn.LayerNorm(name=f"attention_norm_{layer_index}")(
                    hidden + attended
                )
            controlled = hidden[:, -int(controlled_player_count) :, :]
            return nn.Dense(DEFENSIVE_ACTION_COUNT, name="defensive_action_head")(
                controlled
            )

    return DefensiveActionProbe()


def train_attention_probe(
    *,
    player_tokens: np.ndarray,
    global_features: np.ndarray,
    action_masks: np.ndarray,
    labels: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    seed: int,
    config: ProbeTrainingConfig,
) -> dict[str, Any]:
    """Train one full-batch attention probe and return all-state predictions."""

    import jax
    import jax.numpy as jnp
    import optax

    model = _build_probe_module(
        controlled_player_count=labels.shape[1],
        embed_dim=int(config.embed_dim),
        token_mlp_dim=int(config.token_mlp_dim),
        num_heads=int(config.num_heads),
        num_layers=int(config.num_layers),
    )
    tokens_device = jnp.asarray(player_tokens, dtype=jnp.float32)
    globals_device = jnp.asarray(global_features, dtype=jnp.float32)
    masks_device = jnp.asarray(action_masks, dtype=jnp.bool_)
    labels_device = jnp.asarray(labels, dtype=jnp.int32)
    train_indices_device = jnp.asarray(train_indices, dtype=jnp.int32)
    validation_indices_device = jnp.asarray(validation_indices, dtype=jnp.int32)
    params = model.init(
        jax.random.PRNGKey(int(seed)),
        tokens_device[train_indices_device[:1]],
        globals_device[train_indices_device[:1]],
    )["params"]
    optimizer = optax.adam(float(config.learning_rate))
    opt_state = optimizer.init(params)

    def masked_logits(current_params, indices):
        logits = model.apply(
            {"params": current_params},
            tokens_device[indices],
            globals_device[indices],
        )
        return jnp.where(
            masks_device[indices], logits, jnp.asarray(-1.0e9, dtype=jnp.float32)
        )

    def loss_value(current_params, indices):
        logits = masked_logits(current_params, indices)
        losses = optax.softmax_cross_entropy_with_integer_labels(
            logits, labels_device[indices]
        )
        return jnp.mean(losses)

    @jax.jit
    def train_step(current_params, current_opt_state):
        loss, grads = jax.value_and_grad(loss_value)(
            current_params, train_indices_device
        )
        updates, next_opt_state = optimizer.update(
            grads, current_opt_state, current_params
        )
        next_params = optax.apply_updates(current_params, updates)
        return next_params, next_opt_state, loss

    validation_loss = jax.jit(loss_value)
    best_params = params
    best_validation = float("inf")
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, float | int]] = []
    for epoch in range(1, int(config.epochs) + 1):
        params, opt_state, train_loss = train_step(params, opt_state)
        val_loss = float(np.asarray(validation_loss(params, validation_indices_device)))
        if epoch == 1 or epoch % 10 == 0:
            history.append(
                {
                    "epoch": int(epoch),
                    "train_nll": float(np.asarray(train_loss)),
                    "validation_nll": val_loss,
                }
            )
        if val_loss < best_validation - 1.0e-7:
            best_validation = val_loss
            best_params = jax.tree.map(lambda value: value.copy(), params)
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= int(config.patience):
            break

    all_indices = jnp.arange(labels.shape[0], dtype=jnp.int32)
    final_logits = masked_logits(best_params, all_indices)
    probabilities = np.asarray(jax.nn.softmax(final_logits, axis=-1))
    return {
        "probabilities": probabilities,
        "best_epoch": int(best_epoch),
        "epochs_trained": int(epoch),
        "best_validation_nll": float(best_validation),
        "history": history,
    }


def _subset_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    subset_masks: Mapping[str, np.ndarray],
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for offset, (name, mask) in enumerate(subset_masks.items()):
        selected = np.asarray(mask, dtype=bool)
        if not np.any(selected):
            result[name] = {
                "available": False,
                "state_count": 0,
            }
            continue
        result[name] = {
            "available": True,
            **classification_metrics(
                labels[selected],
                probabilities[selected],
                bootstrap_samples=int(bootstrap_samples),
                bootstrap_seed=int(bootstrap_seed) + (offset * 100),
            ),
        }
    return result


def _paired_primary_delta(
    labels: np.ndarray,
    left_probabilities: np.ndarray,
    right_probabilities: np.ndarray,
    test_mask: np.ndarray,
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    selected_labels = np.asarray(labels)[test_mask]
    left = np.asarray(left_probabilities)[test_mask]
    right = np.asarray(right_probabilities)[test_mask]
    left_pred = np.argmax(left, axis=-1)
    right_pred = np.argmax(right, axis=-1)
    state_count, player_count = selected_labels.shape
    left_exact = np.mean(left_pred == selected_labels, axis=1)
    right_exact = np.mean(right_pred == selected_labels, axis=1)
    left_joint = np.all(left_pred == selected_labels, axis=1).astype(np.float64)
    right_joint = np.all(right_pred == selected_labels, axis=1).astype(np.float64)
    left_nll = np.mean(
        -np.log(
            np.clip(
                np.take_along_axis(left, selected_labels[..., None], axis=-1)[..., 0],
                1.0e-12,
                1.0,
            )
        ),
        axis=1,
    )
    right_nll = np.mean(
        -np.log(
            np.clip(
                np.take_along_axis(right, selected_labels[..., None], axis=-1)[..., 0],
                1.0e-12,
                1.0,
            )
        ),
        axis=1,
    )

    def delta(left_values: np.ndarray, right_values: np.ndarray, offset: int):
        differences = np.asarray(right_values) - np.asarray(left_values)
        rng = np.random.default_rng(int(bootstrap_seed) + offset)
        draws = rng.integers(0, state_count, size=(int(bootstrap_samples), state_count))
        values = np.mean(differences[draws], axis=1)
        return {
            "estimate": float(np.mean(differences)),
            "ci95_low": float(np.quantile(values, 0.025)),
            "ci95_high": float(np.quantile(values, 0.975)),
            "state_count": int(state_count),
            "decisions_per_state": int(player_count),
        }

    return {
        "exact_player_action_accuracy": delta(left_exact, right_exact, 1),
        "exact_joint_action_accuracy": delta(left_joint, right_joint, 2),
        "negative_log_likelihood": delta(left_nll, right_nll, 3),
        "negative_log_likelihood_direction": "negative_is_better",
    }


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_defensive_feature_probe(
    results_path: str | Path,
    *,
    output_dir: str | Path,
    config: ProbeTrainingConfig = ProbeTrainingConfig(),
    depth: int | None = None,
    stability_result_paths: Sequence[str | Path] = (),
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Train all controlled variants and persist one reproducible study."""

    config.validate()
    source_path = Path(results_path).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"Output directory already exists: {destination}")
    report = progress_callback or (lambda _: None)
    report("loading replayable defensive states")
    dataset = prepare_probe_dataset(
        source_path,
        depth=depth,
        stability_result_paths=stability_result_paths,
        split_seed=int(config.split_seed),
        train_fraction=float(config.train_fraction),
        validation_fraction=float(config.validation_fraction),
    )
    train_indices = np.flatnonzero(dataset.split == "train")
    validation_indices = np.flatnonzero(dataset.split == "validation")
    test_mask = dataset.split == "test"
    regret_threshold = float(
        np.quantile(dataset.regret[train_indices], float(config.high_regret_quantile))
    )
    subset_masks = {
        "test": test_mask,
        "test_high_regret": test_mask & (dataset.regret >= regret_threshold),
    }
    if dataset.stability_available:
        subset_masks["test_stable_label"] = test_mask & dataset.stable_labels
        subset_masks["test_stable_high_regret"] = (
            test_mask & dataset.stable_labels & (dataset.regret >= regret_threshold)
        )

    variant_results: dict[str, Any] = {}
    ensemble_probabilities: dict[str, np.ndarray] = {}
    learning_curves: dict[str, Any] = {}
    curve_train_order = np.random.default_rng(int(config.split_seed) + 1).permutation(
        train_indices
    )
    for variant_index, variant in enumerate(FEATURE_VARIANTS):
        report(f"training {variant}")
        player_tokens = np.concatenate(
            [dataset.base_player_tokens, dataset.extra_features[variant]], axis=2
        )
        seed_runs = []
        probabilities = []
        for training_seed in config.training_seeds:
            trained = train_attention_probe(
                player_tokens=player_tokens,
                global_features=dataset.global_features,
                action_masks=dataset.action_masks,
                labels=dataset.labels,
                train_indices=train_indices,
                validation_indices=validation_indices,
                seed=int(training_seed),
                config=config,
            )
            probabilities.append(np.asarray(trained.pop("probabilities")))
            seed_runs.append({"seed": int(training_seed), **trained})
        ensemble = np.mean(np.stack(probabilities, axis=0), axis=0)
        ensemble_probabilities[variant] = ensemble
        variant_results[variant] = {
            "feature_labels": list(dataset.feature_labels[variant]),
            "player_token_dim": int(player_tokens.shape[2]),
            "training_runs": seed_runs,
            "metrics": _subset_metrics(
                dataset.labels,
                ensemble,
                subset_masks=subset_masks,
                bootstrap_samples=int(config.bootstrap_samples),
                bootstrap_seed=int(config.split_seed) + (variant_index * 1000),
            ),
        }

        curve_rows = []
        curve_seed = int(config.training_seeds[0])
        for fraction in sorted(set(float(v) for v in config.learning_curve_fractions)):
            subset_count = max(1, int(np.floor(train_indices.shape[0] * fraction)))
            if fraction >= 1.0:
                curve_probabilities = probabilities[0]
                curve_run = {
                    "reused_full_seed_run": True,
                    "training_seed": curve_seed,
                }
            else:
                curve_train_indices = curve_train_order[:subset_count]
                curve_trained = train_attention_probe(
                    player_tokens=player_tokens,
                    global_features=dataset.global_features,
                    action_masks=dataset.action_masks,
                    labels=dataset.labels,
                    train_indices=curve_train_indices,
                    validation_indices=validation_indices,
                    seed=curve_seed,
                    config=config,
                )
                curve_probabilities = np.asarray(curve_trained.pop("probabilities"))
                curve_run = {
                    "reused_full_seed_run": False,
                    "training_seed": curve_seed,
                    **curve_trained,
                }
            curve_metrics = classification_metrics(
                dataset.labels[test_mask],
                curve_probabilities[test_mask],
                bootstrap_samples=int(config.bootstrap_samples),
                bootstrap_seed=int(config.split_seed)
                + 5000
                + variant_index * 100
                + subset_count,
            )
            curve_rows.append(
                {
                    "train_fraction": fraction,
                    "training_state_count": int(subset_count),
                    "run": curve_run,
                    "test_exact_player_action_accuracy": curve_metrics[
                        "exact_player_action_accuracy"
                    ],
                    "test_negative_log_likelihood": curve_metrics[
                        "negative_log_likelihood"
                    ],
                }
            )
        learning_curves[variant] = curve_rows

    frequency_probabilities = _frequency_baseline(
        dataset.labels, dataset.action_masks, dataset.split == "train"
    )
    baselines = {
        "evaluated_policy": {
            "metrics": {
                name: (
                    {
                        "available": False,
                        "state_count": 0,
                    }
                    if not np.any(mask)
                    else {
                        "available": True,
                        **_deterministic_action_metrics(
                            dataset.labels[mask],
                            dataset.policy_actions[mask],
                            bootstrap_samples=int(config.bootstrap_samples),
                            bootstrap_seed=int(config.split_seed) + 7000 + offset,
                        ),
                    }
                )
                for offset, (name, mask) in enumerate(subset_masks.items())
            }
        },
        "train_frequency": {
            "metrics": _subset_metrics(
                dataset.labels,
                frequency_probabilities,
                subset_masks=subset_masks,
                bootstrap_samples=int(config.bootstrap_samples),
                bootstrap_seed=int(config.split_seed) + 8000,
            )
        },
    }

    paired_differences = {
        "current_to_directional": _paired_primary_delta(
            dataset.labels,
            ensemble_probabilities["current_observation"],
            ensemble_probabilities["directional_geometry"],
            test_mask,
            bootstrap_samples=int(config.bootstrap_samples),
            bootstrap_seed=int(config.split_seed) + 9000,
        ),
        "directional_to_threat": _paired_primary_delta(
            dataset.labels,
            ensemble_probabilities["directional_geometry"],
            ensemble_probabilities["threat_and_responsibility"],
            test_mask,
            bootstrap_samples=int(config.bootstrap_samples),
            bootstrap_seed=int(config.split_seed) + 10000,
        ),
        "current_to_threat": _paired_primary_delta(
            dataset.labels,
            ensemble_probabilities["current_observation"],
            ensemble_probabilities["threat_and_responsibility"],
            test_mask,
            bootstrap_samples=int(config.bootstrap_samples),
            bootstrap_seed=int(config.split_seed) + 11000,
        ),
    }
    directional_delta = paired_differences["current_to_directional"]
    threat_delta = paired_differences["current_to_threat"]

    def reliable_uplift(delta: Mapping[str, Any]) -> bool:
        accuracy = dict(delta["exact_player_action_accuracy"])
        nll = dict(delta["negative_log_likelihood"])
        return bool(float(accuracy["ci95_low"]) > 0.0 or float(nll["ci95_high"]) < 0.0)

    interpretation = {
        "label_stability": (
            "available" if dataset.stability_available else "unavailable"
        ),
        "directional_geometry_reliable_uplift": reliable_uplift(directional_delta),
        "threat_and_responsibility_reliable_uplift": reliable_uplift(threat_delta),
        "conclusion": (
            "inconclusive_without_same_state_stability_artifact"
            if not dataset.stability_available
            else (
                "explicit_relational_feature_uplift_detected"
                if reliable_uplift(directional_delta) or reliable_uplift(threat_delta)
                else "no_reliable_relational_feature_uplift_detected"
            )
        ),
    }

    per_state = []
    for index, state_id in enumerate(dataset.state_ids):
        variants = {}
        for variant in FEATURE_VARIANTS:
            probability = ensemble_probabilities[variant][index]
            variants[variant] = {
                "predicted_action": np.argmax(probability, axis=-1).tolist(),
                "action_probabilities": probability.tolist(),
            }
        per_state.append(
            {
                "state_id": state_id,
                "split": str(dataset.split[index]),
                "mcts_action": dataset.labels[index].tolist(),
                "policy_action": dataset.policy_actions[index].tolist(),
                "regret": float(dataset.regret[index]),
                "stable_label": (
                    bool(dataset.stable_labels[index])
                    if dataset.stability_available
                    else None
                ),
                "variants": variants,
            }
        )

    payload = {
        "schema_version": PROBE_SCHEMA_VERSION,
        "study": "offline_defensive_feature_probe",
        "source": {
            "mcts_results": str(source_path),
            "mcts_results_sha256": _sha256_file(source_path),
            "search_depth": int(
                depth
                if depth is not None
                else dataset.source_payload.get("aggregate", {}).get(
                    "primary_search_depth", 0
                )
            ),
            "simulations_per_action": dataset.source_payload.get("protocol", {}).get(
                "simulations_per_action"
            ),
            "stability_results": [
                str(Path(path).expanduser().resolve())
                for path in stability_result_paths
            ],
            "stability_available": bool(dataset.stability_available),
        },
        "probe_provenance": capture_source_provenance(),
        "configuration": {
            "split_seed": int(config.split_seed),
            "training_seeds": [int(value) for value in config.training_seeds],
            "train_fraction": float(config.train_fraction),
            "validation_fraction": float(config.validation_fraction),
            "learning_rate": float(config.learning_rate),
            "epochs": int(config.epochs),
            "patience": int(config.patience),
            "embed_dim": int(config.embed_dim),
            "token_mlp_dim": int(config.token_mlp_dim),
            "num_heads": int(config.num_heads),
            "num_layers": int(config.num_layers),
            "bootstrap_samples": int(config.bootstrap_samples),
            "high_regret_quantile": float(config.high_regret_quantile),
            "high_regret_threshold_from_train": regret_threshold,
            "learning_curve_fractions": [
                float(value) for value in config.learning_curve_fractions
            ],
        },
        "dataset": {
            "state_count": len(dataset.state_ids),
            "defender_decision_count": int(dataset.labels.size),
            "split_counts": {
                name: int(np.sum(dataset.split == name))
                for name in ("train", "validation", "test")
            },
            "stable_state_count": (
                int(np.sum(dataset.stable_labels))
                if dataset.stability_available
                else None
            ),
        },
        "feature_variants": variant_results,
        "baselines": baselines,
        "paired_feature_differences": paired_differences,
        "interpretation": interpretation,
        "learning_curves": learning_curves,
        "per_state_predictions": per_state,
    }
    destination.mkdir(parents=True, exist_ok=False)
    results_file = destination / "defensive_feature_probe.json"
    results_file.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "results": str(results_file),
        "payload": payload,
    }


__all__ = [
    "DEFENSIVE_ACTION_COUNT",
    "DIRECTIONAL_FEATURE_LABELS",
    "FEATURE_VARIANTS",
    "PROBE_SCHEMA_VERSION",
    "THREAT_FEATURE_LABELS",
    "DefensiveProbeDataset",
    "ProbeTrainingConfig",
    "classification_metrics",
    "compute_defensive_relational_features",
    "make_state_splits",
    "prepare_probe_dataset",
    "run_defensive_feature_probe",
    "stable_action_mask",
    "train_attention_probe",
]
