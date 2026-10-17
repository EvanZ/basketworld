"""Replay a fixed JAX policy on an audited kernel, on two CPU cores.

Run once per revision, sequentially. This diagnostic uses previously audited
static NPZ inputs, so it does not reinterpret historical CLI defaults. Historical
kernel code is loaded from git, adapting only its two retired import dependencies.
Policy-only deterministic self-play matches native multi-possession evaluation
(selector disabled). Episode seeds and step RNG keys are explicit and shared.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import types


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True, help="Git ref or current")
    parser.add_argument("--static", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=16)
    parser.add_argument("--horizon", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2000000)
    args = parser.parse_args()
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    print("Loading CPU runtime", flush=True)
    import numpy as np
    import jax
    import jax.numpy as jnp
    from basketworld_jax.env import minimal as current
    from basketworld_jax.eval.native import _load_checkpoint_params
    from basketworld_jax.models import actor_critic_forward, apply_action_mask

    m = current
    if args.revision != "current":
        source = subprocess.check_output(
            ["git", "show", f"{args.revision}:basketworld_jax/env/minimal.py"],
            cwd=root,
            text=True,
        )
        tree = ast.parse(source)
        for node in tree.body:
            if (
                isinstance(node, ast.ImportFrom)
                and node.module == "basketworld.envs.basketworld_env_v2"
            ):
                node.module = "basketworld.domain"
        tree.body = [
            node
            for node in tree.body
            if not (
                isinstance(node, ast.ImportFrom) and node.module == "train.env_factory"
            )
        ]
        m = types.ModuleType("audit_historical_kernel")
        m.__file__ = str(root / "basketworld_jax/env/minimal.py")
        sys.modules[m.__name__] = m
        exec(compile(tree, f"<git:{args.revision}:minimal.py>", "exec"), m.__dict__)
    with np.load(args.static) as data:
        static = m.KernelStatic(
            **{k: jnp.asarray(data[k]) for k in m.KernelStatic._fields}
        )
    print("Loading checkpoint", flush=True)
    payload, params, spec = _load_checkpoint_params(args.checkpoint, jax=jax)
    print("Resetting episodes", flush=True)
    assert spec.multi_possession_features
    keys = jax.vmap(lambda i: jax.random.PRNGKey(i))(
        jnp.arange(args.seed, args.seed + args.episodes, dtype=jnp.uint32)
    )
    state = jax.jit(lambda ks: m.reset_batch_minimal(static, ks, jax, jnp))(keys)
    initial = {f"initial.{k}": np.asarray(v) for k, v in state._asdict().items()}
    player_ids = (static.offense_ids, static.defense_ids)
    obs_kwargs = dict(
        model_type=spec.model_type,
        rebound_win_prob_features=spec.rebound_win_prob_features,
        rebound_target_observation_features=spec.rebound_target_observation_features,
        multi_possession_features=True,
        check_clearance_features=spec.observation_schema_version >= 5,
    )

    @jax.jit
    def policy(st):
        masks = m.build_action_masks_batch(static, st, jnp)
        actions = []
        for flags, ids in zip(
            (
                m.build_training_role_flags_batch(static, st, jnp),
                m.build_opponent_role_flags_batch(static, st, jnp),
            ),
            player_ids,
        ):
            obs = m.build_policy_observation_batch_with_role_flag(
                static, st, flags, jnp, **obs_kwargs
            )
            context = m.build_policy_intent_context_batch_with_role_flag(
                static, st, flags, jnp
            )
            forward = actor_critic_forward(
                params, obs, spec, jnp, intent_context=context
            )
            actions.append(
                apply_action_mask(
                    forward["flat_policy_logits"], masks[:, ids, :], spec, jax, jnp
                )["deterministic_actions"]
            )
        return m.assemble_full_actions_jax(
            *actions, *player_ids, spec.token_player_count, jnp
        )

    @jax.jit
    def step(st, actions, ks):
        return m.step_batch_minimal(static, st, actions, ks, jax, jnp)

    trace = {}

    def append(name, value):
        trace.setdefault(name, []).append(np.asarray(value))

    events = (
        "possession_ended",
        "possession_end_reason",
        "turnover_reason",
        "check_pickup",
        "check_violation",
        "shot_attempt",
        "shot_success",
    )
    # Keep a fixed horizon and stop accounting for an episode after it ends.
    for tick in range(args.horizon):
        actions = policy(state)
        step_keys = jax.vmap(lambda k: jax.random.fold_in(k, tick + 1))(keys)
        out = step(state, actions, step_keys)
        append("actions", actions)
        append("active", ~state.episode_ended.astype(jnp.bool_))
        append("pending_inbounder", state.inbound_player)
        for k, value in out.state._asdict().items():
            append(f"state.{k}", value)
        for k in events:
            if hasattr(out, k):
                append(k, getattr(out, k))
        state = out.state
        if tick % 250 == 0:
            print(
                args.revision,
                "step",
                tick,
                "completed",
                int(np.asarray(state.episode_ended).sum()),
                flush=True,
            )
        if np.asarray(state.episode_ended).all():
            break
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output, **initial, **{k: np.stack(v) for k, v in trace.items()}
    )
    metadata = dict(
        revision=args.revision,
        checkpoint=str(args.checkpoint.resolve()),
        checkpoint_update=payload["update_index"],
        episodes=args.episodes,
        seed=args.seed,
        horizon=args.horizon,
        ticks=tick + 1,
        completed=int(np.asarray(state.episode_ended).sum()),
        selector="disabled; native multi-possession policy-only evaluation",
        policy_actions="deterministic",
        backend="cpu",
        cpu_count=2,
    )
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata), flush=True)


if __name__ == "__main__":
    main()
