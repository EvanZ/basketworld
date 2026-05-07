from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np

from basketworld_jax.train.types import RolloutOutput


@dataclass(frozen=True)
class IntentDiscriminatorSpec:
    encoder_type: str
    input_dim: int
    hidden_dim: int
    num_intents: int
    learning_rate: float
    batch_size: int
    updates_per_rollout: int
    beta_target: float
    warmup_updates: int | None
    ramp_updates: int | None
    warmup_steps: int
    ramp_steps: int
    bonus_clip: float
    eval_holdout_fraction: float
    max_obs_dim: int
    action_dim_per_player: int
    training_player_count: int
    token_player_count: int
    token_dim: int
    global_dim: int
    set_heads: int
    set_cls_tokens: int
    include_shot_clock: bool
    include_pressure_exposure: bool

    def asdict(self) -> dict[str, Any]:
        return asdict(self)


def build_intent_discriminator_spec(args, policy_spec) -> IntentDiscriminatorSpec:
    encoder_type = str(getattr(args, "intent_disc_encoder_type", "mlp_mean")).strip().lower()
    obs_dim = min(int(getattr(args, "intent_disc_max_obs_dim", 256)), int(policy_spec.flat_obs_dim))
    action_dim = int(policy_spec.training_player_count)
    event_dim = 11
    if encoder_type == "set_step":
        input_dim = (
            int(policy_spec.token_player_count)
            * (
                int(policy_spec.token_dim)
                + int(policy_spec.global_dim)
                + 1
            )
        )
    else:
        input_dim = int(obs_dim + action_dim + event_dim)
    return IntentDiscriminatorSpec(
        encoder_type=encoder_type,
        input_dim=int(input_dim),
        hidden_dim=int(getattr(args, "intent_disc_hidden_dim", 128)),
        num_intents=int(getattr(args, "num_intents", 8)),
        learning_rate=float(getattr(args, "intent_disc_lr", 3e-4)),
        batch_size=int(getattr(args, "intent_disc_batch_size", 256)),
        updates_per_rollout=int(getattr(args, "intent_disc_updates_per_rollout", 2)),
        beta_target=float(getattr(args, "intent_diversity_beta_target", 0.05)),
        warmup_updates=(
            None
            if getattr(args, "intent_diversity_warmup_updates", None) is None
            else int(getattr(args, "intent_diversity_warmup_updates"))
        ),
        ramp_updates=(
            None
            if getattr(args, "intent_diversity_ramp_updates", None) is None
            else int(getattr(args, "intent_diversity_ramp_updates"))
        ),
        warmup_steps=int(getattr(args, "intent_diversity_warmup_steps", 1_000_000)),
        ramp_steps=int(getattr(args, "intent_diversity_ramp_steps", 1_000_000)),
        bonus_clip=float(getattr(args, "intent_diversity_clip", 2.0)),
        eval_holdout_fraction=float(getattr(args, "intent_disc_eval_holdout_fraction", 0.25)),
        max_obs_dim=int(obs_dim),
        action_dim_per_player=int(policy_spec.action_dim_per_player),
        training_player_count=int(policy_spec.training_player_count),
        token_player_count=int(policy_spec.token_player_count),
        token_dim=int(policy_spec.token_dim),
        global_dim=int(policy_spec.global_dim),
        set_heads=int(max(1, getattr(args, "attention_num_heads", 4))),
        set_cls_tokens=1,
        include_shot_clock=bool(getattr(args, "intent_disc_include_shot_clock", True)),
        include_pressure_exposure=bool(getattr(args, "intent_disc_include_pressure_exposure", True)),
    )


def build_intent_discriminator_module(spec: IntentDiscriminatorSpec):
    from flax import linen as nn
    import jax.numpy as jnp

    class IntentDiscriminatorModule(nn.Module):
        @nn.compact
        def _set_step_forward(self, features):
            players = features["players"].astype(jnp.float32)
            globals_vec = features["globals"].astype(jnp.float32)
            role_flag = features["role_flag"].astype(jnp.float32)
            if role_flag.ndim == 1:
                role_flag = role_flag[:, None]
            globals_expanded = jnp.broadcast_to(
                globals_vec[:, None, :],
                (players.shape[0], players.shape[1], globals_vec.shape[-1]),
            )
            role_expanded = jnp.broadcast_to(
                role_flag[:, None, :],
                (players.shape[0], players.shape[1], role_flag.shape[-1]),
            )
            tokens = jnp.concatenate([players, globals_expanded, role_expanded], axis=-1)
            hidden = nn.Dense(int(spec.hidden_dim), name="set_token_mlp_0")(tokens)
            hidden = nn.relu(hidden)
            hidden = nn.Dense(int(spec.hidden_dim), name="set_token_mlp_1")(hidden)

            cls_count = max(0, int(spec.set_cls_tokens))
            if cls_count > 0:
                cls_tokens = self.param(
                    "set_cls_tokens",
                    nn.initializers.zeros_init(),
                    (cls_count, int(spec.hidden_dim)),
                )
                cls_batch = jnp.broadcast_to(
                    cls_tokens[None, :, :],
                    (hidden.shape[0], cls_count, int(spec.hidden_dim)),
                )
                hidden = jnp.concatenate([hidden, cls_batch], axis=1)

            num_heads = max(1, int(spec.set_heads))
            head_dim = int(spec.hidden_dim) // num_heads
            qkv = nn.Dense(3 * int(spec.hidden_dim), name="set_attention_qkv")(hidden)
            qkv = qkv.reshape(hidden.shape[0], hidden.shape[1], 3, num_heads, head_dim)
            query = qkv[:, :, 0]
            key = qkv[:, :, 1]
            value = qkv[:, :, 2]
            scale = jnp.asarray(head_dim, dtype=jnp.float32) ** -0.5
            scores = jnp.einsum("bthd,bshd->bhts", query, key) * scale
            weights = nn.softmax(scores, axis=-1)
            attended = jnp.einsum("bhts,bshd->bthd", weights, value)
            attended = attended.reshape(hidden.shape[0], hidden.shape[1], int(spec.hidden_dim))
            projected = nn.Dense(int(spec.hidden_dim), name="set_attention_out")(attended)
            hidden = nn.LayerNorm(name="set_attention_norm")(hidden + projected)
            ff = nn.Dense(int(spec.hidden_dim), name="set_ff_0")(hidden)
            ff = nn.relu(ff)
            ff = nn.Dense(int(spec.hidden_dim), name="set_ff_1")(ff)
            hidden = nn.LayerNorm(name="set_ff_norm")(hidden + ff)

            if cls_count > 0:
                embedding = jnp.mean(hidden[:, -cls_count:, :], axis=1)
            else:
                embedding = jnp.mean(hidden, axis=1)
            logits = nn.Dense(int(spec.num_intents), name="intent_head")(embedding)
            return {
                "embedding": embedding,
                "logits": logits,
            }

        @nn.compact
        def __call__(self, features):
            if str(spec.encoder_type) == "set_step":
                return self._set_step_forward(features)
            hidden = nn.Dense(int(spec.hidden_dim), name="hidden_0")(features.astype(jnp.float32))
            hidden = nn.relu(hidden)
            embedding = nn.Dense(int(spec.hidden_dim), name="embedding")(hidden)
            embedding = nn.relu(embedding)
            logits = nn.Dense(int(spec.num_intents), name="intent_head")(embedding)
            return {
                "embedding": embedding,
                "logits": logits,
            }

    return IntentDiscriminatorModule()


def init_intent_discriminator_params(jax, jnp, spec: IntentDiscriminatorSpec, *, seed: int):
    from flax.core import unfreeze

    module = build_intent_discriminator_module(spec)
    if str(spec.encoder_type) == "set_step":
        sample = {
            "players": jnp.zeros(
                (1, int(spec.token_player_count), int(spec.token_dim)),
                dtype=jnp.float32,
            ),
            "globals": jnp.zeros((1, int(spec.global_dim)), dtype=jnp.float32),
            "role_flag": jnp.zeros((1, 1), dtype=jnp.float32),
        }
    else:
        sample = jnp.zeros((1, int(spec.input_dim)), dtype=jnp.float32)
    variables = module.init(jax.random.PRNGKey(int(seed)), sample)
    return unfreeze(variables["params"])


def build_intent_step_features_from_rollout(rollout: RolloutOutput, spec: IntentDiscriminatorSpec, jnp):
    trajectory = rollout.trajectory
    if str(spec.encoder_type) == "set_step":
        flat_obs = trajectory.flat_obs.astype(jnp.float32)
        player_dim = int(spec.token_player_count) * int(spec.token_dim)
        global_start = player_dim
        global_end = global_start + int(spec.global_dim)
        players = flat_obs[..., :player_dim].reshape(
            flat_obs.shape[0],
            flat_obs.shape[1],
            int(spec.token_player_count),
            int(spec.token_dim),
        )
        globals_vec = flat_obs[..., global_start:global_end]
        if int(spec.global_dim) >= 1 and not bool(spec.include_shot_clock):
            globals_vec = globals_vec.at[..., 0].set(0.0)
        if int(spec.global_dim) >= 2 and not bool(spec.include_pressure_exposure):
            globals_vec = globals_vec.at[..., 1].set(0.0)
        role_flag = flat_obs[..., global_end : global_end + 1]
        labels = trajectory.policy_intent_index.astype(jnp.int32)
        active_mask = trajectory.policy_intent_gate.astype(jnp.float32) > 0.5
        return {
            "players": players,
            "globals": globals_vec,
            "role_flag": role_flag,
        }, labels, active_mask

    obs = trajectory.flat_obs[..., : int(spec.max_obs_dim)].astype(jnp.float32)
    action_den = jnp.asarray(max(1, int(spec.action_dim_per_player) - 1), dtype=jnp.float32)
    actions = trajectory.actions.astype(jnp.float32) / action_den
    events = jnp.stack(
        [
            trajectory.pass_attempts.astype(jnp.float32),
            trajectory.completed_passes.astype(jnp.float32),
            trajectory.assists.astype(jnp.float32),
            trajectory.turnovers.astype(jnp.float32),
            trajectory.shot_attempts.astype(jnp.float32),
            trajectory.shot_makes.astype(jnp.float32),
            trajectory.shot_dunks.astype(jnp.float32),
            trajectory.shot_twos.astype(jnp.float32),
            trajectory.shot_threes.astype(jnp.float32),
            trajectory.offense_score_delta.astype(jnp.float32),
            trajectory.defense_score_delta.astype(jnp.float32),
        ],
        axis=-1,
    )
    features = jnp.concatenate([obs, actions, events], axis=-1).astype(jnp.float32)
    labels = trajectory.policy_intent_index.astype(jnp.int32)
    active_mask = trajectory.policy_intent_gate.astype(jnp.float32) > 0.5
    return features, labels, active_mask


def build_intent_discriminator_update_runner(jax, jnp, spec: IntentDiscriminatorSpec):
    import optax

    module = build_intent_discriminator_module(spec)
    transform = optax.adam(float(spec.learning_rate))
    sample_count = int(spec.batch_size)
    updates_per_rollout = int(spec.updates_per_rollout)
    holdout_fraction = float(min(max(float(spec.eval_holdout_fraction), 0.0), 1.0))
    num_intents = int(spec.num_intents)

    def _flatten_features(features):
        if str(spec.encoder_type) != "set_step":
            return features.reshape((-1, int(spec.input_dim))).astype(jnp.float32)
        return {
            "players": features["players"].reshape(
                -1,
                int(spec.token_player_count),
                int(spec.token_dim),
            ).astype(jnp.float32),
            "globals": features["globals"].reshape(-1, int(spec.global_dim)).astype(jnp.float32),
            "role_flag": features["role_flag"].reshape(-1, 1).astype(jnp.float32),
        }

    def _feature_count(features) -> int:
        if str(spec.encoder_type) == "set_step":
            return int(features["players"].shape[0])
        return int(features.shape[0])

    def _take_features(features, indices):
        if str(spec.encoder_type) == "set_step":
            return {
                key: value[indices]
                for key, value in features.items()
            }
        return features[indices]

    def _forward(params, features):
        if str(spec.encoder_type) == "set_step":
            return module.apply({"params": params}, features)
        return module.apply({"params": params}, features.astype(jnp.float32))

    def _loss_fn(params, features, labels, weights):
        out = _forward(params, features)
        logits = out["logits"]
        labels = jnp.clip(labels.astype(jnp.int32), 0, num_intents - 1)
        losses = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
        denom = jnp.maximum(jnp.sum(weights), 1.0)
        loss = jnp.sum(losses * weights) / denom
        pred = jnp.argmax(logits, axis=-1).astype(jnp.int32)
        accuracy = jnp.sum((pred == labels).astype(jnp.float32) * weights) / denom
        probs = jax.nn.softmax(logits, axis=-1)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        entropy = -jnp.sum(probs * log_probs, axis=-1)
        entropy = jnp.sum(entropy * weights) / denom
        return loss, {
            "loss": loss,
            "accuracy": accuracy,
            "entropy": entropy,
            "active_count": jnp.sum(weights),
        }

    def _binary_auc_from_scores(scores, labels, weights, class_idx):
        active = weights.astype(jnp.float32)
        labels = jnp.clip(labels.astype(jnp.int32), 0, num_intents - 1)
        positives = ((labels == class_idx).astype(jnp.float32) * active).astype(jnp.float32)
        negatives = ((labels != class_idx).astype(jnp.float32) * active).astype(jnp.float32)
        n_pos = jnp.sum(positives)
        n_neg = jnp.sum(negatives)
        order = jnp.argsort(scores, axis=0)
        sorted_active = active[order]
        sorted_pos = positives[order]
        active_rank = jnp.cumsum(sorted_active)
        pos_rank_sum = jnp.sum(active_rank * sorted_pos)
        denom = jnp.maximum(n_pos * n_neg, 1.0)
        auc = (pos_rank_sum - (n_pos * (n_pos + 1.0) * 0.5)) / denom
        valid = (n_pos > 0.0) & (n_neg > 0.0)
        return jnp.where(valid, auc, 0.0), valid.astype(jnp.float32)

    def _macro_ovr_auc(logits, labels, weights):
        class_indices = jnp.arange(num_intents, dtype=jnp.int32)

        def _one_class(class_idx):
            return _binary_auc_from_scores(logits[:, class_idx], labels, weights, class_idx)

        aucs, valid = jax.vmap(_one_class)(class_indices)
        valid_count = jnp.sum(valid)
        macro_auc = jnp.sum(aucs * valid) / jnp.maximum(valid_count, 1.0)
        return macro_auc, valid_count

    def _metric_snapshot(params, features, labels, weights):
        out = _forward(params, features)
        logits = out["logits"]
        labels = jnp.clip(labels.astype(jnp.int32), 0, num_intents - 1)
        weights = weights.astype(jnp.float32)
        losses = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
        denom = jnp.maximum(jnp.sum(weights), 1.0)
        loss = jnp.sum(losses * weights) / denom
        pred = jnp.argmax(logits, axis=-1).astype(jnp.int32)
        top1 = jnp.sum((pred == labels).astype(jnp.float32) * weights) / denom
        probs = jax.nn.softmax(logits, axis=-1)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        entropy = -jnp.sum(probs * log_probs, axis=-1)
        entropy = jnp.sum(entropy * weights) / denom
        auc, auc_valid_count = _macro_ovr_auc(logits, labels, weights)
        label_counts = jnp.bincount(labels, weights=weights, length=num_intents)
        pred_counts = jnp.bincount(pred, weights=weights, length=num_intents)
        label_probs = label_counts / jnp.maximum(jnp.sum(label_counts), 1.0)
        pred_probs = pred_counts / jnp.maximum(jnp.sum(pred_counts), 1.0)
        return {
            "loss": loss,
            "top1": top1,
            "entropy": entropy,
            "active_count": jnp.sum(weights),
            "auc_ovr_macro": auc,
            "auc_valid_class_count": auc_valid_count,
            "label_counts": label_counts,
            "label_probs": label_probs,
            "pred_counts": pred_counts,
            "pred_probs": pred_probs,
        }

    def _take_batch(features, labels, weights, key):
        total_count = _feature_count(features)
        weight_sum = jnp.sum(weights)
        probs = jnp.where(
            weight_sum > 0.0,
            weights / jnp.maximum(weight_sum, 1.0),
            jnp.full((total_count,), 1.0 / float(total_count), dtype=jnp.float32),
        )
        indices = jax.random.choice(
            key,
            jnp.arange(total_count, dtype=jnp.int32),
            shape=(sample_count,),
            replace=True,
            p=probs,
        )
        return _take_features(features, indices), labels[indices], weights[indices]

    def _runner(params, opt_state, features, labels, active_mask, key):
        flat_features = _flatten_features(features)
        flat_labels = labels.reshape((-1,)).astype(jnp.int32)
        flat_weights = active_mask.reshape((-1,)).astype(jnp.float32)
        total_active = jnp.sum(flat_weights)
        split_key, train_key = jax.random.split(key)
        holdout_draw = jax.random.uniform(split_key, shape=flat_weights.shape, dtype=jnp.float32)
        raw_holdout_weights = jnp.where(
            (flat_weights > 0.0) & (holdout_draw < holdout_fraction),
            jnp.ones_like(flat_weights),
            jnp.zeros_like(flat_weights),
        )
        raw_train_weights = jnp.where(
            (flat_weights > 0.0) & (holdout_draw >= holdout_fraction),
            jnp.ones_like(flat_weights),
            jnp.zeros_like(flat_weights),
        )
        train_weights = jnp.where(
            jnp.sum(raw_train_weights) > 0.0,
            raw_train_weights,
            flat_weights,
        )
        eval_weights = jnp.where(
            jnp.sum(raw_holdout_weights) > 0.0,
            raw_holdout_weights,
            flat_weights,
        )

        def _update_step(carry, step_idx):
            step_params, step_opt_state, step_key = carry
            step_key = jax.random.fold_in(step_key, step_idx)
            mb_features, mb_labels, mb_weights = _take_batch(
                flat_features,
                flat_labels,
                train_weights,
                step_key,
            )
            (_, train_metrics), grads = jax.value_and_grad(_loss_fn, has_aux=True)(
                step_params,
                mb_features,
                mb_labels,
                mb_weights,
            )
            updates, next_opt_state = transform.update(grads, step_opt_state, step_params)
            next_params = optax.apply_updates(step_params, updates)
            return (next_params, next_opt_state, step_key), train_metrics

        (next_params, next_opt_state, _), train_metrics = jax.lax.scan(
            _update_step,
            (params, opt_state, train_key),
            jnp.arange(updates_per_rollout, dtype=jnp.int32),
        )
        full_metrics = _metric_snapshot(next_params, flat_features, flat_labels, flat_weights)
        trainbatch_metrics = _metric_snapshot(next_params, flat_features, flat_labels, train_weights)
        eval_metrics = _metric_snapshot(next_params, flat_features, flat_labels, eval_weights)
        out = _forward(next_params, flat_features)
        log_probs = jax.nn.log_softmax(out["logits"], axis=-1)
        clipped_labels = jnp.clip(flat_labels, 0, num_intents - 1)
        raw_bonus = (
            jnp.take_along_axis(log_probs, clipped_labels[:, None], axis=-1)[:, 0]
            + jnp.log(jnp.asarray(float(num_intents), dtype=jnp.float32))
        )
        raw_bonus = raw_bonus.reshape(labels.shape)
        metrics = {
            "intent_disc_loss": full_metrics["loss"],
            "intent_disc_top1_acc_trainbatch": trainbatch_metrics["top1"],
            "intent_disc_auc_ovr_macro_trainbatch": trainbatch_metrics["auc_ovr_macro"],
            "intent_disc_top1_acc_holdout": eval_metrics["top1"],
            "intent_disc_auc_ovr_macro_holdout": eval_metrics["auc_ovr_macro"],
            "intent_disc_entropy": full_metrics["entropy"],
            "intent_disc_active_count": full_metrics["active_count"],
            "intent_disc_trainbatch_size": trainbatch_metrics["active_count"],
            "intent_disc_holdout_size": eval_metrics["active_count"],
            "intent_disc_holdout_fraction_realized": (
                eval_metrics["active_count"] / jnp.maximum(total_active, 1.0)
            ),
            "intent_disc_auc_valid_class_count_trainbatch": trainbatch_metrics["auc_valid_class_count"],
            "intent_disc_auc_valid_class_count_holdout": eval_metrics["auc_valid_class_count"],
        }
        for intent_idx in range(num_intents):
            metrics[f"intent_disc_label_count_by_intent/{intent_idx}"] = full_metrics["label_counts"][intent_idx]
            metrics[f"intent_disc_label_prob_by_intent/{intent_idx}"] = full_metrics["label_probs"][intent_idx]
            metrics[f"intent_disc_pred_count_by_intent/{intent_idx}"] = full_metrics["pred_counts"][intent_idx]
            metrics[f"intent_disc_pred_prob_by_intent/{intent_idx}"] = full_metrics["pred_probs"][intent_idx]
        return next_params, next_opt_state, metrics, raw_bonus

    return jax.jit(_runner), transform


def init_bonus_stats() -> dict[str, float]:
    return {
        "count": 1.0e-6,
        "mean": 0.0,
        "var": 1.0,
    }


def update_bonus_stats(stats: dict[str, float], values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return dict(stats)
    old_count = float(stats.get("count", 1.0e-6))
    old_mean = float(stats.get("mean", 0.0))
    old_var = max(float(stats.get("var", 1.0)), 1.0e-12)
    batch_count = float(arr.size)
    batch_mean = float(np.mean(arr))
    batch_var = float(np.var(arr))
    delta = batch_mean - old_mean
    total_count = old_count + batch_count
    next_mean = old_mean + delta * (batch_count / max(total_count, 1.0e-12))
    m2 = (
        (old_var * old_count)
        + (batch_var * batch_count)
        + (delta * delta) * old_count * batch_count / max(total_count, 1.0e-12)
    )
    return {
        "count": float(total_count),
        "mean": float(next_mean),
        "var": float(max(m2 / max(total_count, 1.0e-12), 1.0e-12)),
    }


def compute_intent_beta(
    *,
    global_step: int,
    spec: IntentDiscriminatorSpec,
    update_index: int | None = None,
) -> float:
    if spec.warmup_updates is not None and update_index is not None:
        step = int(update_index)
        if step < int(spec.warmup_updates):
            return 0.0
        if spec.ramp_updates is not None and int(spec.ramp_updates) <= 0:
            return float(spec.beta_target)
        ramp = max(1, int(spec.ramp_updates if spec.ramp_updates is not None else 1))
        progress = min(1.0, max(0.0, (step - int(spec.warmup_updates)) / float(ramp)))
        return float(spec.beta_target) * float(progress)

    step = int(global_step)
    if step < int(spec.warmup_steps):
        return 0.0
    ramp = max(1, int(spec.ramp_steps))
    progress = min(1.0, max(0.0, (step - int(spec.warmup_steps)) / float(ramp)))
    return float(spec.beta_target) * float(progress)


def compute_normalized_intent_bonus(raw_bonus, active_mask, stats, *, beta: float, clip: float, jnp):
    mean = jnp.asarray(float(stats.get("mean", 0.0)), dtype=jnp.float32)
    std = jnp.sqrt(jnp.asarray(max(float(stats.get("var", 1.0)), 1.0e-12), dtype=jnp.float32))
    norm_bonus = (raw_bonus.astype(jnp.float32) - mean) / jnp.maximum(std, 1.0e-6)
    clipped = jnp.clip(norm_bonus, -float(clip), float(clip))
    return jnp.where(
        active_mask,
        jnp.asarray(float(beta), dtype=jnp.float32) * clipped,
        jnp.zeros_like(clipped, dtype=jnp.float32),
    )


def apply_intent_bonus_to_rollout(rollout: RolloutOutput, bonus, jnp) -> RolloutOutput:
    trajectory = rollout.trajectory
    updated_trajectory = trajectory._replace(
        rewards=trajectory.rewards.astype(jnp.float32) + bonus.astype(jnp.float32)
    )
    return rollout._replace(trajectory=updated_trajectory)


def _intent_discriminator_embeddings(params, features, spec: IntentDiscriminatorSpec, jax, jnp):
    module = build_intent_discriminator_module(spec)
    if str(spec.encoder_type) == "set_step":
        flat_features = {
            "players": features["players"].reshape(
                -1,
                int(spec.token_player_count),
                int(spec.token_dim),
            ).astype(jnp.float32),
            "globals": features["globals"].reshape(-1, int(spec.global_dim)).astype(jnp.float32),
            "role_flag": features["role_flag"].reshape(-1, 1).astype(jnp.float32),
        }
    else:
        flat_features = features.reshape((-1, int(spec.input_dim))).astype(jnp.float32)
    out = module.apply({"params": params}, flat_features)
    if str(spec.encoder_type) == "set_step":
        return out["embedding"].reshape(features["players"].shape[0], features["players"].shape[1], -1)
    return out["embedding"].reshape(features.shape[0], features.shape[1], -1)


def _sample_features_to_numpy(features, spec: IntentDiscriminatorSpec, jax) -> dict[str, np.ndarray]:
    if str(spec.encoder_type) != "set_step":
        feature_arr = np.asarray(jax.device_get(features), dtype=np.float32).reshape(-1, int(spec.input_dim))
        return {
            "features": feature_arr.astype(np.float32),
        }
    players = np.asarray(jax.device_get(features["players"]), dtype=np.float32).reshape(
        -1,
        int(spec.token_player_count),
        int(spec.token_dim),
    )
    globals_vec = np.asarray(jax.device_get(features["globals"]), dtype=np.float32).reshape(
        -1,
        int(spec.global_dim),
    )
    role_flag = np.asarray(jax.device_get(features["role_flag"]), dtype=np.float32).reshape(-1, 1)
    globals_expanded = np.broadcast_to(
        globals_vec[:, None, :],
        (players.shape[0], players.shape[1], globals_vec.shape[-1]),
    )
    role_expanded = np.broadcast_to(
        role_flag[:, None, :],
        (players.shape[0], players.shape[1], role_flag.shape[-1]),
    )
    token_features = np.concatenate([players, globals_expanded, role_expanded], axis=-1)
    return {
        "players": players.astype(np.float32),
        "globals": globals_vec.astype(np.float32),
        "role_flag": role_flag.astype(np.float32),
        "features": token_features.reshape(token_features.shape[0], -1).astype(np.float32),
    }


def build_intent_sample_dump(
    *,
    params,
    features,
    labels,
    active_mask,
    bonus,
    rollout: RolloutOutput,
    spec: IntentDiscriminatorSpec,
    jax,
    jnp,
    update_index: int,
    max_samples: int,
) -> dict[str, np.ndarray]:
    embeddings = _intent_discriminator_embeddings(params, features, spec, jax, jnp)
    feature_payload = _sample_features_to_numpy(features, spec, jax)
    features_np = feature_payload.pop("features")
    embeddings_np = np.asarray(jax.device_get(embeddings), dtype=np.float32).reshape(features_np.shape[0], -1)
    labels_np = np.asarray(jax.device_get(labels), dtype=np.int32).reshape(-1)
    active_np = np.asarray(jax.device_get(active_mask), dtype=bool).reshape(-1)
    bonus_np = np.asarray(jax.device_get(bonus), dtype=np.float32).reshape(-1)
    indices = np.flatnonzero(active_np)
    cap = max(0, int(max_samples))
    if cap > 0 and indices.size > cap:
        positions = np.linspace(0, indices.size - 1, cap).astype(np.int64)
        indices = indices[positions]
    trajectory = rollout.trajectory
    payload = {
        "update_index": np.full((indices.size,), int(update_index), dtype=np.int32),
        "source_current_policy": np.ones((indices.size,), dtype=np.int8),
        "intent_index": labels_np[indices].astype(np.int32),
        "features": features_np[indices].astype(np.float32),
        "embedding": embeddings_np[indices].astype(np.float32),
        "bonus": bonus_np[indices].astype(np.float32),
        "actions": np.asarray(jax.device_get(trajectory.actions), dtype=np.int32).reshape(
            -1,
            int(spec.training_player_count),
        )[indices],
        "pass_attempt": np.asarray(jax.device_get(trajectory.pass_attempts), dtype=np.int8).reshape(-1)[indices],
        "completed_pass": np.asarray(jax.device_get(trajectory.completed_passes), dtype=np.int8).reshape(-1)[indices],
        "assist": np.asarray(jax.device_get(trajectory.assists), dtype=np.int8).reshape(-1)[indices],
        "turnover": np.asarray(jax.device_get(trajectory.turnovers), dtype=np.int8).reshape(-1)[indices],
        "shot_attempt": np.asarray(jax.device_get(trajectory.shot_attempts), dtype=np.int8).reshape(-1)[indices],
        "shot_make": np.asarray(jax.device_get(trajectory.shot_makes), dtype=np.int8).reshape(-1)[indices],
        "shot_dunk": np.asarray(jax.device_get(trajectory.shot_dunks), dtype=np.int8).reshape(-1)[indices],
        "shot_two": np.asarray(jax.device_get(trajectory.shot_twos), dtype=np.int8).reshape(-1)[indices],
        "shot_three": np.asarray(jax.device_get(trajectory.shot_threes), dtype=np.int8).reshape(-1)[indices],
        "offense_score_delta": np.asarray(
            jax.device_get(trajectory.offense_score_delta),
            dtype=np.float32,
        ).reshape(-1)[indices],
        "defense_score_delta": np.asarray(
            jax.device_get(trajectory.defense_score_delta),
            dtype=np.float32,
        ).reshape(-1)[indices],
    }
    for key, value in feature_payload.items():
        payload[key] = value[indices]
    return payload
