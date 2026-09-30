from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from basketworld.envs.basketworld_env_v2 import HexagonBasketballEnv
from basketworld_jax.env.minimal import (
    TEAM_A,
    TEAM_B,
    build_kernel_static_from_env,
    build_token_observation_batch_with_role_flag,
    build_token_observation_components_batch,
    canonicalize_player_tokens_by_active_role,
    reset_batch_minimal,
)
from basketworld_jax.models.actor_critic import (
    ActorCriticSpec,
    _build_pointer_slot_target_ids,
)


def _static_and_state(*, offense_team: int):
    env = HexagonBasketballEnv(
        players=2,
        render_mode=None,
        pass_mode="pointer_targeted",
    )
    env.enable_multi_possession = True
    static = build_kernel_static_from_env(env, xp=jnp)
    state = reset_batch_minimal(
        static,
        jax.random.split(jax.random.PRNGKey(37), 1),
        jax,
        jnp,
    )._replace(offense_team=jnp.asarray([offense_team], dtype=jnp.int8))
    return static, state


@pytest.mark.parametrize("offense_team", [TEAM_A, TEAM_B])
@pytest.mark.parametrize("viewer_team", [TEAM_A, TEAM_B])
def test_policy_token_half_always_matches_controlled_stable_roster(
    offense_team: int,
    viewer_team: int,
):
    static, state = _static_and_state(offense_team=offense_team)
    player_count = int(static.role_encoding.shape[0])
    players_per_team = int(static.offense_ids.shape[0])
    stable_player_ids = jnp.arange(player_count, dtype=jnp.int32)[None, :, None]

    canonical = canonicalize_player_tokens_by_active_role(
        static,
        state,
        stable_player_ids,
        jnp,
    )
    viewer_is_offense = viewer_team == offense_team
    selected_start = 0 if viewer_is_offense else players_per_team
    selected_ids = np.asarray(
        canonical[0, selected_start : selected_start + players_per_team, 0],
        dtype=np.int32,
    )
    expected_ids = np.asarray(
        static.offense_ids if viewer_team == TEAM_A else static.defense_ids,
        dtype=np.int32,
    )

    np.testing.assert_array_equal(selected_ids, expected_ids)


@pytest.mark.parametrize("offense_team", [TEAM_A, TEAM_B])
def test_packed_attention_observation_uses_canonical_active_role_order(
    offense_team: int,
):
    static, state = _static_and_state(offense_team=offense_team)
    raw_players, globals_vec, _ = build_token_observation_components_batch(
        static,
        state,
        jnp.asarray([1.0], dtype=jnp.float32),
        jnp,
        multi_possession_features=True,
    )
    packed = build_token_observation_batch_with_role_flag(
        static,
        state,
        jnp.asarray([1.0], dtype=jnp.float32),
        jnp,
        multi_possession_features=True,
    )
    player_count = int(raw_players.shape[1])
    token_dim = int(raw_players.shape[2])
    packed_players = np.asarray(
        packed[:, : player_count * token_dim].reshape(
            (1, player_count, token_dim)
        )
    )
    expected = np.asarray(
        canonicalize_player_tokens_by_active_role(
            static,
            state,
            raw_players,
            jnp,
        )
    )

    np.testing.assert_allclose(packed_players, expected)
    assert packed.shape[1] == (player_count * token_dim) + int(globals_vec.shape[1]) + 1


@pytest.mark.parametrize("offense_team", [TEAM_A, TEAM_B])
def test_canonical_policy_tokens_preserve_pointer_pass_teammate_slots(
    offense_team: int,
):
    static, state = _static_and_state(offense_team=offense_team)
    player_count = int(static.role_encoding.shape[0])
    players_per_team = int(static.offense_ids.shape[0])
    stable_player_ids = jnp.arange(player_count, dtype=jnp.int32)[None, :, None]
    canonical_ids = np.asarray(
        canonicalize_player_tokens_by_active_role(
            static,
            state,
            stable_player_ids,
            jnp,
        )[0, :, 0],
        dtype=np.int32,
    )
    spec = ActorCriticSpec(
        flat_obs_dim=1,
        training_player_count=players_per_team,
        action_dim_per_player=14,
        total_action_dim=players_per_team * 14,
        hidden_dims=(),
        model_type="attention",
        token_player_count=player_count,
        token_dim=1,
        global_dim=0,
        action_head_mode="pointer_targeted",
    )
    pointer_token_targets = _build_pointer_slot_target_ids(spec)
    env_pointer_targets = np.asarray(static.pointer_pass_target_ids, dtype=np.int32)

    for viewer_team in (TEAM_A, TEAM_B):
        viewer_is_offense = viewer_team == offense_team
        selected_start = 0 if viewer_is_offense else players_per_team
        for roster_slot in range(players_per_team):
            token_index = selected_start + roster_slot
            player_id = int(canonical_ids[token_index])
            for pass_slot, target_token_index in enumerate(
                pointer_token_targets[token_index].tolist()
            ):
                actual_target = (
                    int(canonical_ids[target_token_index])
                    if target_token_index >= 0
                    else -1
                )
                assert actual_target == int(
                    env_pointer_targets[player_id, pass_slot]
                )
