"""JAX environment and checkpoint adapter for policy-regret MCTS.

The search implementation in :mod:`basketworld_jax.eval.mcts_regret` is kept
independent of JAX.  This module binds it to the real basketball kernel, two
explicit policy checkpoints, and possession-terminal score-differential
returns.  The evaluated checkpoint owns stable Team A; because roles switch
between possessions, sampled Team A states naturally cover both offense and
defense decisions.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Literal, Mapping

import numpy as np

from basketworld_jax.env.minimal import (
    GAME_PHASE_LIVE,
    KernelState,
    TEAM_A,
    TEAM_B,
    assemble_full_actions_jax,
    build_action_masks_batch,
    build_kernel_static_from_config,
    build_policy_intent_context_batch_with_role_flag,
    build_policy_observation_batch_with_role_flag,
    reset_batch_minimal,
    step_batch_game_clock,
    step_batch_minimal,
)
from basketworld_jax.eval.mcts_regret import (
    JointAction,
    JointActionProposal,
    MctsTransition,
    sample_factorized_joint_actions,
)
from basketworld_jax.eval.native import (
    _apply_paired_starting_teams,
    _adapt_policy_observation_to_spec,
    _checkpoint_payload_params,
    _compatible_native_cross_play_specs,
    _load_checkpoint_params,
    _resolve_native_environment_config,
)
from basketworld_jax.models import actor_critic_forward, apply_action_mask
from basketworld_jax.provenance import capture_source_provenance
from basketworld_jax.train.cli import ensure_jax_available


PolicyActionMode = Literal["deterministic", "sampled"]


def _validated_action_mode(value: str, *, field_name: str) -> PolicyActionMode:
    normalized = str(value).strip().lower()
    if normalized not in {"deterministic", "sampled"}:
        raise ValueError(f"{field_name} must be 'deterministic' or 'sampled'.")
    return normalized  # type: ignore[return-value]


def _observation_kwargs(spec: Any) -> dict[str, Any]:
    schema_version = int(getattr(spec, "observation_schema_version", 1))
    return {
        "model_type": str(spec.model_type),
        "rebound_win_prob_features": bool(spec.rebound_win_prob_features),
        "rebound_target_observation_features": bool(
            getattr(spec, "rebound_target_observation_features", True)
        ),
        "multi_possession_features": bool(
            getattr(spec, "multi_possession_features", False)
        ),
        "check_clearance_features": schema_version >= 5,
        "team_clearance_features": schema_version >= 6,
    }


def _host_scalar(jax: Any, value: Any) -> Any:
    array = np.asarray(jax.device_get(value))
    if array.size != 1:
        raise ValueError(f"Expected a scalar JAX value, got shape {array.shape}.")
    return array.reshape(-1)[0].item()


def kernel_state_key(state: Any, *, jax: Any) -> str:
    """Return a stable content key for a batched ``KernelState``.

    Array names, dtypes, shapes, and bytes all participate so two states with
    equal raw bytes but different field layout cannot collide accidentally.
    """

    digest = hashlib.blake2b(digest_size=20)
    fields = getattr(type(state), "_fields", ())
    for index, value in enumerate(jax.device_get(state)):
        name = fields[index] if index < len(fields) else str(index)
        array = np.ascontiguousarray(np.asarray(value))
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def meaningful_joint_choice(action_mask: np.ndarray) -> bool:
    """Whether at least one controlled player has multiple legal actions."""

    mask = np.asarray(action_mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("action_mask must have shape (players, actions).")
    legal_counts = mask.sum(axis=1)
    if np.any(legal_counts < 1):
        return False
    return bool(np.any(legal_counts > 1))


@dataclass(frozen=True)
class PolicyDecision:
    probabilities: np.ndarray
    legal_mask: np.ndarray
    deterministic_action: JointAction


@dataclass(frozen=True)
class SampledDecision:
    episode_index: int
    episode_seed: int
    policy_seed: int
    selection_seed: int
    environment_step: int
    eligible_decision_index: int
    eligible_decision_count: int
    state_id: str
    evaluated_role: Literal["offense", "defense"]
    shot_clock: int
    game_clock_remaining: int
    team_a_score: float
    team_b_score: float
    ball_holder: int
    legal_action_counts: tuple[int, ...]
    policy_action: JointAction
    state: Any = field(repr=False, compare=False)

    def metadata(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("state", None)
        payload["legal_action_counts"] = list(self.legal_action_counts)
        payload["policy_action"] = list(self.policy_action)
        return payload


@dataclass(frozen=True)
class EpisodeSampleResult:
    sample: SampledDecision | None
    completed: bool
    environment_steps: int
    eligible_decisions: int
    exclusion_counts: Mapping[str, int]


def _state_field_arrays(state: Any, *, jax: Any) -> dict[str, np.ndarray]:
    names = getattr(type(state), "_fields", ())
    values = jax.device_get(state)
    if len(names) != len(values):
        raise ValueError("KernelState field names do not match its array values.")
    return {
        str(name): np.asarray(value) for name, value in zip(names, values, strict=True)
    }


def persist_sampled_decisions(
    output_dir: str | Path,
    samples: list[SampledDecision],
    *,
    provenance: Mapping[str, Any],
    sampling_summary: Mapping[str, Any],
    jax: Any,
) -> dict[str, str]:
    """Persist lossless replay state arrays plus human-readable metadata."""

    if not samples:
        raise ValueError("At least one sampled decision is required.")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    state_path = destination / "sampled_states.npz"
    metadata_path = destination / "sampled_states.json"

    first_fields = _state_field_arrays(samples[0].state, jax=jax)
    stacked: dict[str, np.ndarray] = {}
    for name in first_fields:
        rows = [_state_field_arrays(sample.state, jax=jax)[name] for sample in samples]
        stacked[f"state__{name}"] = np.concatenate(rows, axis=0)
    np.savez_compressed(state_path, **stacked)
    metadata_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "sample_count": len(samples),
                "state_archive": state_path.name,
                "samples": [sample.metadata() for sample in samples],
                "sampling_summary": dict(sampling_summary),
                "provenance": dict(provenance),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "state_archive": str(state_path),
        "metadata": str(metadata_path),
    }


def load_persisted_sample_states(
    metadata_path: str | Path,
) -> tuple[list[dict[str, Any]], list[KernelState], dict[str, Any]]:
    """Restore persisted one-row states without rebuilding or resampling them."""

    metadata_file = Path(metadata_path)
    payload = json.loads(metadata_file.read_text(encoding="utf-8"))
    archive_path = metadata_file.parent / str(payload["state_archive"])
    records = [dict(item) for item in payload.get("samples", [])]
    states: list[KernelState] = []
    with np.load(archive_path, allow_pickle=False) as archive:
        count = int(payload.get("sample_count", 0))
        for sample_index in range(count):
            values = [
                np.asarray(archive[f"state__{name}"])[sample_index : sample_index + 1]
                for name in KernelState._fields
            ]
            states.append(KernelState(*values))
    if len(records) != len(states):
        raise ValueError("Sample metadata and state archive counts do not match.")
    return records, states, payload


@dataclass
class JaxCounterfactualRuntime:
    """Host-facing callbacks over one-row JAX environment states."""

    jax: Any
    jnp: Any
    static: Any
    config: Any
    evaluated_params: Any
    evaluated_spec: Any
    opponent_params: Any
    opponent_spec: Any
    evaluated_action_mode: PolicyActionMode = "deterministic"
    opponent_action_mode: PolicyActionMode = "deterministic"
    max_rollout_steps: int = 128
    provenance: Mapping[str, Any] = field(default_factory=dict)
    _compiled_evaluated_policy: Any = field(init=False, repr=False)
    _compiled_opponent_policy: Any = field(init=False, repr=False)
    _compiled_forced_step: Any = field(init=False, repr=False)
    _compiled_policy_step: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.evaluated_action_mode = _validated_action_mode(
            self.evaluated_action_mode,
            field_name="evaluated_action_mode",
        )
        self.opponent_action_mode = _validated_action_mode(
            self.opponent_action_mode,
            field_name="opponent_action_mode",
        )
        if int(self.max_rollout_steps) < 1:
            raise ValueError("max_rollout_steps must be positive.")
        if bool(getattr(self.evaluated_spec, "intent_selector_enabled", False)):
            raise ValueError(
                "MCTS regret evaluation does not yet support evaluated checkpoints "
                "with an intent selector. No-intent policies are supported."
            )
        if bool(getattr(self.opponent_spec, "intent_selector_enabled", False)):
            raise ValueError(
                "MCTS regret evaluation does not yet support opponent checkpoints "
                "with an intent selector. No-intent policies are supported."
            )
        self._build_compiled_functions()

    @property
    def evaluated_ids(self) -> Any:
        return self.static.offense_ids.astype(self.jnp.int32)

    @property
    def opponent_ids(self) -> Any:
        return self.static.defense_ids.astype(self.jnp.int32)

    def _policy_outputs(self, state: Any, params: Any, spec: Any, team: int):
        team_ids = self.evaluated_ids if int(team) == int(TEAM_A) else self.opponent_ids
        role_flag = self.jnp.where(
            state.offense_team == int(team),
            self.jnp.asarray(1.0, dtype=self.jnp.float32),
            self.jnp.asarray(-1.0, dtype=self.jnp.float32),
        )
        observation = build_policy_observation_batch_with_role_flag(
            self.static,
            state,
            role_flag,
            self.jnp,
            **_observation_kwargs(spec),
        )
        observation = _adapt_policy_observation_to_spec(
            observation,
            self.static,
            spec,
            self.jnp,
        )
        intent_context = build_policy_intent_context_batch_with_role_flag(
            self.static,
            state,
            role_flag,
            self.jnp,
        )
        action_mask = build_action_masks_batch(self.static, state, self.jnp)[
            :, team_ids, :
        ]
        forward = actor_critic_forward(
            params,
            observation,
            spec,
            self.jnp,
            intent_context=intent_context,
        )
        masked = apply_action_mask(
            forward["flat_policy_logits"],
            action_mask,
            spec,
            self.jax,
            self.jnp,
        )
        return (
            masked["probs"],
            action_mask,
            masked["masked_logits"],
            masked["deterministic_actions"],
        )

    def _build_compiled_functions(self) -> None:
        jax = self.jax
        jnp = self.jnp
        n_players = int(self.static.role_encoding.shape[0])
        step_batch = (
            step_batch_game_clock
            if str(self.config.episode_end_mode) == "game_clock"
            else step_batch_minimal
        )
        evaluated_mode = self.evaluated_action_mode
        opponent_mode = self.opponent_action_mode

        def evaluated_policy(state):
            return self._policy_outputs(
                state,
                self.evaluated_params,
                self.evaluated_spec,
                TEAM_A,
            )

        def opponent_policy(state):
            return self._policy_outputs(
                state,
                self.opponent_params,
                self.opponent_spec,
                TEAM_B,
            )

        def choose_action(masked_logits, deterministic_actions, key, mode):
            if mode == "deterministic":
                return deterministic_actions
            return jax.random.categorical(key, masked_logits, axis=-1).astype(jnp.int32)

        def forced_step(state, evaluated_actions, key):
            opponent_key, environment_key = jax.random.split(key, 2)
            _, _, opponent_logits, opponent_deterministic = opponent_policy(state)
            opponent_actions = choose_action(
                opponent_logits,
                opponent_deterministic,
                opponent_key,
                opponent_mode,
            )
            full_actions = assemble_full_actions_jax(
                evaluated_actions,
                opponent_actions,
                self.evaluated_ids,
                self.opponent_ids,
                n_players,
                jnp,
            )
            environment_keys = jax.random.split(
                environment_key, state.positions.shape[0]
            )
            return step_batch(
                self.static,
                state,
                full_actions,
                environment_keys,
                jax,
                jnp,
            )

        def policy_step(state, key):
            evaluated_key, opponent_key, environment_key = jax.random.split(key, 3)
            _, _, evaluated_logits, evaluated_deterministic = evaluated_policy(state)
            _, _, opponent_logits, opponent_deterministic = opponent_policy(state)
            evaluated_actions = choose_action(
                evaluated_logits,
                evaluated_deterministic,
                evaluated_key,
                evaluated_mode,
            )
            opponent_actions = choose_action(
                opponent_logits,
                opponent_deterministic,
                opponent_key,
                opponent_mode,
            )
            full_actions = assemble_full_actions_jax(
                evaluated_actions,
                opponent_actions,
                self.evaluated_ids,
                self.opponent_ids,
                n_players,
                jnp,
            )
            environment_keys = jax.random.split(
                environment_key, state.positions.shape[0]
            )
            output = step_batch(
                self.static,
                state,
                full_actions,
                environment_keys,
                jax,
                jnp,
            )
            return output, evaluated_actions, opponent_actions

        self._compiled_evaluated_policy = jax.jit(evaluated_policy)
        self._compiled_opponent_policy = jax.jit(opponent_policy)
        self._compiled_forced_step = jax.jit(forced_step)
        self._compiled_policy_step = jax.jit(policy_step)

    def reset(self, seed: int, *, starting_team: int = TEAM_A) -> Any:
        keys = self.jax.random.PRNGKey(int(seed))[None, :]
        state = reset_batch_minimal(self.static, keys, self.jax, self.jnp)
        if bool(self.config.enable_multi_possession):
            if int(starting_team) not in {int(TEAM_A), int(TEAM_B)}:
                raise ValueError("starting_team must be TEAM_A or TEAM_B.")
            state = _apply_paired_starting_teams(
                self.static,
                state,
                episode_offset=(0 if int(starting_team) == int(TEAM_A) else 1),
                jnp=self.jnp,
            )
        return state

    def state_key(self, state: Any) -> str:
        return kernel_state_key(state, jax=self.jax)

    def policy_decision(self, state: Any) -> PolicyDecision:
        probabilities, legal_mask, _, deterministic = self.jax.device_get(
            self._compiled_evaluated_policy(state)
        )
        probabilities_host = np.asarray(probabilities, dtype=np.float64)
        legal_host = np.asarray(legal_mask, dtype=bool)
        deterministic_host = np.asarray(deterministic, dtype=np.int32)
        if probabilities_host.shape[0] != 1:
            raise ValueError("MCTS runtime requires a one-row KernelState.")
        return PolicyDecision(
            probabilities=probabilities_host[0],
            legal_mask=legal_host[0],
            deterministic_action=tuple(int(value) for value in deterministic_host[0]),
        )

    def propose_actions(
        self,
        state: Any,
        rng: np.random.Generator,
        count: int,
        existing: frozenset[JointAction],
    ) -> tuple[JointActionProposal, ...]:
        decision = self.policy_decision(state)
        return sample_factorized_joint_actions(
            decision.probabilities,
            decision.legal_mask,
            count=int(count),
            rng=rng,
            excluded=existing,
        )

    def is_eligible_live_decision(self, state: Any) -> tuple[bool, str]:
        if int(state.positions.shape[0]) != 1:
            return False, "batch_size_not_one"
        if bool(_host_scalar(self.jax, state.episode_ended)):
            return False, "terminal_state"
        if int(_host_scalar(self.jax, state.game_phase)) != int(GAME_PHASE_LIVE):
            return False, "not_live_play"
        decision = self.policy_decision(state)
        if not meaningful_joint_choice(decision.legal_mask):
            return False, "forced_joint_action"
        return True, "eligible"

    def _next_key(self, rng: np.random.Generator) -> Any:
        seed = int(rng.integers(0, np.iinfo(np.uint32).max, dtype=np.uint32))
        return self.jax.random.PRNGKey(seed)

    def _score_margin_delta(self, output: Any) -> float:
        team_a = float(_host_scalar(self.jax, output.team_a_score_delta))
        team_b = float(_host_scalar(self.jax, output.team_b_score_delta))
        return team_a - team_b

    def transition(
        self,
        state: Any,
        action: JointAction,
        rng: np.random.Generator,
    ) -> MctsTransition:
        expected_width = int(self.evaluated_spec.training_player_count)
        if len(action) != expected_width:
            raise ValueError(
                f"Expected a {expected_width}-player joint action, got {len(action)}."
            )
        action_array = self.jnp.asarray([action], dtype=self.jnp.int32)
        output = self._compiled_forced_step(state, action_array, self._next_key(rng))
        terminal = bool(
            _host_scalar(self.jax, output.possession_ended)
            or _host_scalar(self.jax, output.done)
            or _host_scalar(self.jax, output.truncated)
        )
        return MctsTransition(
            state=output.state,
            reward=self._score_margin_delta(output),
            terminal=terminal,
        )

    def policy_step(
        self,
        state: Any,
        rng: np.random.Generator,
    ) -> tuple[Any, JointAction, JointAction, float, bool]:
        output, evaluated_actions, opponent_actions = self._compiled_policy_step(
            state,
            self._next_key(rng),
        )
        evaluated_host, opponent_host = self.jax.device_get(
            (evaluated_actions, opponent_actions)
        )
        terminal = bool(
            _host_scalar(self.jax, output.possession_ended)
            or _host_scalar(self.jax, output.done)
            or _host_scalar(self.jax, output.truncated)
        )
        return (
            output.state,
            tuple(int(value) for value in np.asarray(evaluated_host)[0]),
            tuple(int(value) for value in np.asarray(opponent_host)[0]),
            self._score_margin_delta(output),
            terminal,
        )

    def rollout_value(self, state: Any, rng: np.random.Generator) -> float:
        total = 0.0
        rollout_state = state
        for _ in range(int(self.max_rollout_steps)):
            rollout_state, _, _, reward, terminal = self.policy_step(
                rollout_state,
                rng,
            )
            total += float(reward)
            if terminal:
                return total
        raise RuntimeError(
            "Policy continuation did not reach a possession boundary within "
            f"{self.max_rollout_steps} steps."
        )


def sample_episode_decision(
    runtime: JaxCounterfactualRuntime,
    *,
    episode_index: int,
    episode_seed: int,
    policy_seed: int,
    selection_seed: int,
    max_episode_steps: int = 10_000,
) -> EpisodeSampleResult:
    """Uniformly reservoir-sample one eligible decision from one episode."""

    if int(max_episode_steps) < 1:
        raise ValueError("max_episode_steps must be positive.")
    starting_team = TEAM_A if int(episode_index) % 2 == 0 else TEAM_B
    state = runtime.reset(int(episode_seed), starting_team=starting_team)
    policy_rng = np.random.default_rng(int(policy_seed))
    selection_rng = np.random.default_rng(int(selection_seed))
    exclusions: Counter[str] = Counter()
    selected: dict[str, Any] | None = None
    eligible_count = 0
    environment_steps = 0

    for step_index in range(int(max_episode_steps)):
        if bool(_host_scalar(runtime.jax, state.episode_ended)):
            break
        eligible, reason = runtime.is_eligible_live_decision(state)
        selected_this_step = False
        if eligible:
            eligible_count += 1
            if int(selection_rng.integers(0, eligible_count)) == 0:
                decision = runtime.policy_decision(state)
                offense_team = int(_host_scalar(runtime.jax, state.offense_team))
                selected_this_step = True
                selected = {
                    "state": state,
                    "environment_step": step_index,
                    "eligible_decision_index": eligible_count,
                    "evaluated_role": (
                        "offense" if offense_team == int(TEAM_A) else "defense"
                    ),
                    "shot_clock": int(_host_scalar(runtime.jax, state.shot_clock)),
                    "game_clock_remaining": int(
                        _host_scalar(runtime.jax, state.game_clock_remaining)
                    ),
                    "team_a_score": float(
                        _host_scalar(runtime.jax, state.team_a_score)
                    ),
                    "team_b_score": float(
                        _host_scalar(runtime.jax, state.team_b_score)
                    ),
                    "ball_holder": int(_host_scalar(runtime.jax, state.ball_holder)),
                    "legal_action_counts": tuple(
                        int(value) for value in decision.legal_mask.sum(axis=1).tolist()
                    ),
                }
        else:
            exclusions[str(reason)] += 1
        state, evaluated_action, _, _, _ = runtime.policy_step(state, policy_rng)
        if selected_this_step:
            assert selected is not None
            selected["policy_action"] = evaluated_action
        environment_steps = step_index + 1

    completed = bool(_host_scalar(runtime.jax, state.episode_ended))
    if not completed:
        exclusions["episode_step_limit"] += 1
    sample = None
    if selected is not None:
        selected_state = selected.pop("state")
        sample = SampledDecision(
            episode_index=int(episode_index),
            episode_seed=int(episode_seed),
            policy_seed=int(policy_seed),
            selection_seed=int(selection_seed),
            eligible_decision_count=int(eligible_count),
            state_id=runtime.state_key(selected_state),
            state=selected_state,
            **selected,
        )
    return EpisodeSampleResult(
        sample=sample,
        completed=completed,
        environment_steps=int(environment_steps),
        eligible_decisions=int(eligible_count),
        exclusion_counts=dict(sorted(exclusions.items())),
    )


def sample_policy_decisions(
    runtime: JaxCounterfactualRuntime,
    *,
    episode_count: int,
    seed: int,
    max_episode_steps: int = 10_000,
) -> tuple[list[SampledDecision], dict[str, Any]]:
    """Sample at most one replayable state from each independent episode."""

    if int(episode_count) < 1:
        raise ValueError("episode_count must be positive.")
    seed_sequence = np.random.SeedSequence(int(seed))
    episode_sequences = seed_sequence.spawn(int(episode_count))
    samples: list[SampledDecision] = []
    exclusion_counts: Counter[str] = Counter()
    completed_episodes = 0
    total_eligible = 0
    total_environment_steps = 0

    for episode_index, sequence in enumerate(episode_sequences):
        seeds = sequence.generate_state(3, dtype=np.uint32)
        result = sample_episode_decision(
            runtime,
            episode_index=episode_index,
            episode_seed=int(seeds[0]),
            policy_seed=int(seeds[1]),
            selection_seed=int(seeds[2]),
            max_episode_steps=int(max_episode_steps),
        )
        if result.sample is not None:
            samples.append(result.sample)
        if result.completed:
            completed_episodes += 1
        total_eligible += int(result.eligible_decisions)
        total_environment_steps += int(result.environment_steps)
        exclusion_counts.update(result.exclusion_counts)

    role_counts = Counter(sample.evaluated_role for sample in samples)
    summary = {
        "seed": int(seed),
        "episode_count": int(episode_count),
        "completed_episode_count": int(completed_episodes),
        "sample_count": len(samples),
        "episodes_without_sample": int(episode_count) - len(samples),
        "total_eligible_decisions": int(total_eligible),
        "total_environment_steps": int(total_environment_steps),
        "role_counts": dict(sorted(role_counts.items())),
        "exclusion_counts": dict(sorted(exclusion_counts.items())),
        "selection_method": "one-pass-uniform-reservoir-per-episode",
        "starting_team_schedule": "alternating-team-a-team-b",
    }
    return samples, summary


def load_jax_counterfactual_runtime(
    *,
    evaluated_checkpoint_path: str | Path,
    opponent_checkpoint_path: str | Path,
    evaluated_action_mode: str = "deterministic",
    opponent_action_mode: str = "deterministic",
    environment_overrides: Mapping[str, Any] | None = None,
    max_rollout_steps: int = 128,
    evaluated_checkpoint_payload: Mapping[str, Any] | None = None,
    opponent_checkpoint_payload: Mapping[str, Any] | None = None,
) -> JaxCounterfactualRuntime:
    """Restore two explicit checkpoints and construct the JAX MCTS adapter."""

    if evaluated_checkpoint_payload is None:
        evaluated_payload, evaluated_params, evaluated_spec = _load_checkpoint_params(
            evaluated_checkpoint_path
        )
    else:
        evaluated_payload, evaluated_params, evaluated_spec = (
            _checkpoint_payload_params(evaluated_checkpoint_payload)
        )
    if opponent_checkpoint_payload is None:
        opponent_payload, opponent_params, opponent_spec = _load_checkpoint_params(
            opponent_checkpoint_path
        )
    else:
        opponent_payload, opponent_params, opponent_spec = _checkpoint_payload_params(
            opponent_checkpoint_payload
        )
    if not _compatible_native_cross_play_specs(evaluated_spec, opponent_spec):
        raise ValueError(
            "Evaluated and opponent checkpoints are not JAX cross-play compatible."
        )

    config = _resolve_native_environment_config(
        required_params={},
        optional_params=dict(environment_overrides or {}),
        unified_payload=evaluated_payload,
        opponent_payload=opponent_payload,
    )
    expected_multi = bool(config.enable_multi_possession)
    checkpoint_multi = {
        bool(getattr(spec, "multi_possession_features", False))
        for spec in (evaluated_spec, opponent_spec)
    }
    if checkpoint_multi != {expected_multi}:
        raise ValueError(
            "Checkpoint multi-possession observation schemas do not match the "
            f"resolved environment (checkpoints={sorted(checkpoint_multi)}, "
            f"environment={expected_multi})."
        )

    jax, jnp = ensure_jax_available("basketworld_jax/eval/mcts_regret_jax.py")
    static = build_kernel_static_from_config(config, jnp)
    provenance = {
        "evaluated_checkpoint_path": str(evaluated_checkpoint_path),
        "evaluated_checkpoint_update_index": int(
            evaluated_payload.get("update_index", 0) or 0
        ),
        "evaluated_checkpoint_source_provenance": dict(
            evaluated_payload.get("source_provenance", {}) or {}
        ),
        "opponent_checkpoint_path": str(opponent_checkpoint_path),
        "opponent_checkpoint_update_index": int(
            opponent_payload.get("update_index", 0) or 0
        ),
        "opponent_checkpoint_source_provenance": dict(
            opponent_payload.get("source_provenance", {}) or {}
        ),
        "evaluated_policy_spec": asdict(evaluated_spec),
        "opponent_policy_spec": asdict(opponent_spec),
        "resolved_environment_config": config.as_metadata(),
        "evaluated_stable_team": "team_a",
        "opponent_stable_team": "team_b",
        "evaluated_action_mode": _validated_action_mode(
            evaluated_action_mode,
            field_name="evaluated_action_mode",
        ),
        "opponent_action_mode": _validated_action_mode(
            opponent_action_mode,
            field_name="opponent_action_mode",
        ),
        "value_target": "team_a_minus_team_b_points_to_possession_end",
        "evaluator_source_provenance": capture_source_provenance(),
    }
    return JaxCounterfactualRuntime(
        jax=jax,
        jnp=jnp,
        static=static,
        config=config,
        evaluated_params=jax.device_put(evaluated_params),
        evaluated_spec=evaluated_spec,
        opponent_params=jax.device_put(opponent_params),
        opponent_spec=opponent_spec,
        evaluated_action_mode=evaluated_action_mode,  # type: ignore[arg-type]
        opponent_action_mode=opponent_action_mode,  # type: ignore[arg-type]
        max_rollout_steps=int(max_rollout_steps),
        provenance=provenance,
    )
