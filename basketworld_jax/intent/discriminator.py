from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np

from basketworld_jax.train.types import RolloutOutput


def _is_set_encoder(encoder_type: str) -> bool:
    return str(encoder_type) == "set_step"


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
    dropout: float
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
        dropout=float(min(max(0.0, getattr(args, "intent_disc_dropout", 0.1)), 0.99)),
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
        def _set_step_forward(self, features, *, train: bool = False):
            dropout_rate = float(min(max(0.0, spec.dropout), 0.99))
            deterministic = (not bool(train)) or dropout_rate <= 0.0
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
            hidden = nn.Dropout(rate=dropout_rate, name="set_token_dropout")(
                hidden,
                deterministic=deterministic,
            )
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
            weights = nn.Dropout(rate=dropout_rate, name="set_attention_dropout")(
                weights,
                deterministic=deterministic,
            )
            attended = jnp.einsum("bhts,bshd->bthd", weights, value)
            attended = attended.reshape(hidden.shape[0], hidden.shape[1], int(spec.hidden_dim))
            projected = nn.Dense(int(spec.hidden_dim), name="set_attention_out")(attended)
            hidden = nn.LayerNorm(name="set_attention_norm")(hidden + projected)
            ff = nn.Dense(int(spec.hidden_dim), name="set_ff_0")(hidden)
            ff = nn.relu(ff)
            ff = nn.Dropout(rate=dropout_rate, name="set_ff_dropout")(
                ff,
                deterministic=deterministic,
            )
            hidden = nn.LayerNorm(name="set_ff_norm")(hidden + ff)

            if cls_count > 0:
                embedding = jnp.mean(hidden[:, -cls_count:, :], axis=1)
            else:
                embedding = jnp.mean(hidden, axis=1)
            head_input = nn.Dropout(rate=dropout_rate, name="intent_head_dropout")(
                embedding,
                deterministic=deterministic,
            )
            logits = nn.Dense(int(spec.num_intents), name="intent_head")(head_input)
            return {
                "embedding": embedding,
                "logits": logits,
            }

        @nn.compact
        def __call__(self, features, *, train: bool = False):
            dropout_rate = float(min(max(0.0, spec.dropout), 0.99))
            deterministic = (not bool(train)) or dropout_rate <= 0.0
            if _is_set_encoder(spec.encoder_type):
                return self._set_step_forward(features, train=train)
            hidden = nn.Dense(int(spec.hidden_dim), name="hidden_0")(features.astype(jnp.float32))
            hidden = nn.relu(hidden)
            hidden = nn.Dropout(rate=dropout_rate, name="hidden_dropout")(
                hidden,
                deterministic=deterministic,
            )
            embedding = nn.Dense(int(spec.hidden_dim), name="embedding")(hidden)
            embedding = nn.relu(embedding)
            head_input = nn.Dropout(rate=dropout_rate, name="intent_head_dropout")(
                embedding,
                deterministic=deterministic,
            )
            logits = nn.Dense(int(spec.num_intents), name="intent_head")(head_input)
            return {
                "embedding": embedding,
                "logits": logits,
            }

    return IntentDiscriminatorModule()


def init_intent_discriminator_params(jax, jnp, spec: IntentDiscriminatorSpec, *, seed: int):
    from flax.core import unfreeze

    module = build_intent_discriminator_module(spec)
    if _is_set_encoder(spec.encoder_type):
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


def build_intent_step_features_from_rollout(
    rollout: RolloutOutput,
    spec: IntentDiscriminatorSpec,
    jnp,
    *,
    training_mask=None,
):
    trajectory = rollout.trajectory
    if training_mask is None:
        training_mask = trajectory.active_mask.astype(jnp.float32)
    training_mask = training_mask.astype(jnp.float32)
    labels = trajectory.policy_intent_index.astype(jnp.int32)
    base_active_mask = (
        (trajectory.policy_intent_gate.astype(jnp.float32) > 0.5)
        & (training_mask > 0.5)
    )
    previous_active = jnp.concatenate(
        [jnp.zeros_like(base_active_mask[:1]), base_active_mask[:-1]],
        axis=0,
    )
    previous_labels = jnp.concatenate([labels[:1], labels[:-1]], axis=0)
    previous_roles = jnp.concatenate(
        [trajectory.training_role[:1], trajectory.training_role[:-1]],
        axis=0,
    )
    previous_possession_ended = jnp.concatenate(
        [
            jnp.zeros_like(trajectory.possession_ended[:1]),
            trajectory.possession_ended[:-1],
        ],
        axis=0,
    )
    segment_start = base_active_mask & (
        (~previous_active)
        | (labels != previous_labels)
        | (trajectory.training_role != previous_roles)
        | (trajectory.selector_applied.astype(jnp.bool_))
        | (previous_possession_ended.astype(jnp.bool_))
    )
    local_segment_id = jnp.cumsum(segment_start.astype(jnp.int32), axis=0)
    batch_size = int(labels.shape[1])
    time_steps = int(labels.shape[0])
    segment_offsets = (
        jnp.arange(batch_size, dtype=jnp.int32)[None, :]
        * jnp.asarray(time_steps + 1, dtype=jnp.int32)
    )
    segment_ids = jnp.where(
        base_active_mask,
        local_segment_id + segment_offsets,
        jnp.asarray(-1, dtype=jnp.int32),
    )
    intent_age = trajectory.intent_age.astype(jnp.int32)

    if _is_set_encoder(spec.encoder_type):
        # State-only DIAYN contract: classify the state produced by the action
        # conditioned on z_t.  The rollout stores pre-action observations, so
        # shift once to align label z_t with s_(t+1).  No selected action or
        # event/outcome feature is available to this encoder.
        flat_obs = jnp.concatenate(
            [
                trajectory.flat_obs[1:].astype(jnp.float32),
                rollout.final_flat_obs.astype(jnp.float32)[None, ...],
            ],
            axis=0,
        )
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
        features = {
            "players": players,
            "globals": globals_vec,
            "role_flag": role_flag,
        }
        post_state_is_same_offensive_context = (
            (role_flag[..., 0] > 0.0)
            & (~trajectory.dones.astype(jnp.bool_))
            & (~trajectory.possession_ended.astype(jnp.bool_))
        )
        active_mask = base_active_mask & post_state_is_same_offensive_context
        return features, labels, active_mask, segment_ids, intent_age

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
    return features, labels, base_active_mask, segment_ids, intent_age


def build_segment_grouped_holdout_weights(
    active_mask,
    segment_ids,
    key,
    *,
    holdout_fraction: float,
    jax,
    jnp,
):
    """Split discriminator samples without leaking one intent segment across sets."""
    flat_weights = active_mask.reshape((-1,)).astype(jnp.float32)
    flat_segment_ids = segment_ids.reshape((-1,)).astype(jnp.int32)
    safe_segment_ids = jnp.maximum(flat_segment_ids, 0)
    holdout_draw = jax.vmap(
        lambda segment_id: jax.random.uniform(
            jax.random.fold_in(key, segment_id),
            shape=(),
            dtype=jnp.float32,
        )
    )(safe_segment_ids)
    fraction = float(min(max(float(holdout_fraction), 0.0), 1.0))
    holdout_weights = jnp.where(
        (flat_weights > 0.0) & (holdout_draw < fraction),
        jnp.ones_like(flat_weights),
        jnp.zeros_like(flat_weights),
    )
    train_weights = jnp.where(
        (flat_weights > 0.0) & (holdout_draw >= fraction),
        jnp.ones_like(flat_weights),
        jnp.zeros_like(flat_weights),
    )
    if 0.0 < fraction < 1.0:
        active_rows = flat_weights > 0.0
        active_segment_ids = jnp.where(
            active_rows,
            flat_segment_ids,
            jnp.asarray(-1, dtype=jnp.int32),
        )
        highest_segment = jnp.max(active_segment_ids)
        has_multiple_segments = jnp.any(
            active_rows & (flat_segment_ids != highest_segment)
        )
        force_holdout = (jnp.sum(holdout_weights) <= 0.0) & has_multiple_segments
        forced_holdout_rows = force_holdout & active_rows & (
            flat_segment_ids == highest_segment
        )
        holdout_weights = jnp.where(
            forced_holdout_rows,
            jnp.ones_like(holdout_weights),
            holdout_weights,
        )
        train_weights = jnp.where(
            forced_holdout_rows,
            jnp.zeros_like(train_weights),
            train_weights,
        )

        lowest_segment = jnp.min(
            jnp.where(
                active_rows,
                flat_segment_ids,
                jnp.asarray(np.iinfo(np.int32).max, dtype=jnp.int32),
            )
        )
        force_train = (jnp.sum(train_weights) <= 0.0) & has_multiple_segments
        forced_train_rows = force_train & active_rows & (
            flat_segment_ids == lowest_segment
        )
        train_weights = jnp.where(
            forced_train_rows,
            jnp.ones_like(train_weights),
            train_weights,
        )
        holdout_weights = jnp.where(
            forced_train_rows,
            jnp.zeros_like(holdout_weights),
            holdout_weights,
        )
    return train_weights, holdout_weights


def build_intent_discriminator_update_runner(jax, jnp, spec: IntentDiscriminatorSpec):
    import optax
    from jax.scipy.stats import rankdata

    module = build_intent_discriminator_module(spec)
    transform = optax.adam(float(spec.learning_rate))
    sample_count = int(spec.batch_size)
    updates_per_rollout = int(spec.updates_per_rollout)
    holdout_fraction = float(min(max(float(spec.eval_holdout_fraction), 0.0), 1.0))
    num_intents = int(spec.num_intents)

    def _flatten_features(features):
        if not _is_set_encoder(spec.encoder_type):
            return features.reshape((-1, int(spec.input_dim))).astype(jnp.float32)
        return {
            key: value.reshape((-1,) + tuple(value.shape[2:])).astype(jnp.float32)
            for key, value in features.items()
        }

    def _feature_count(features) -> int:
        if _is_set_encoder(spec.encoder_type):
            return int(features["players"].shape[0])
        return int(features.shape[0])

    def _take_features(features, indices):
        if _is_set_encoder(spec.encoder_type):
            return {
                key: value[indices]
                for key, value in features.items()
            }
        return features[indices]

    def _forward(params, features, *, train: bool = False, rng=None):
        apply_kwargs = {"train": bool(train)}
        if bool(train) and float(spec.dropout) > 0.0 and rng is not None:
            apply_kwargs["rngs"] = {"dropout": rng}
        if _is_set_encoder(spec.encoder_type):
            return module.apply({"params": params}, features, **apply_kwargs)
        return module.apply({"params": params}, features.astype(jnp.float32), **apply_kwargs)

    def _loss_fn(params, features, labels, weights, rng):
        out = _forward(params, features, train=True, rng=rng)
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
        inactive_count = jnp.sum((active <= 0.0).astype(jnp.float32))
        # Average tied ranks and remove the rank offset introduced by masked
        # rows. This keeps a constant, uninformative classifier at AUC 0.5.
        masked_scores = jnp.where(active > 0.0, scores, -jnp.inf)
        active_rank = rankdata(masked_scores, method="average") - inactive_count
        pos_rank_sum = jnp.sum(active_rank * positives)
        denom = jnp.maximum(n_pos * n_neg, 1.0)
        auc = (pos_rank_sum - (n_pos * (n_pos + 1.0) * 0.5)) / denom
        valid = (n_pos > 0.0) & (n_neg > 0.0)
        return jnp.where(valid, auc, 0.0), valid.astype(jnp.float32)

    def _macro_ovr_auc(probs, labels, weights):
        class_indices = jnp.arange(num_intents, dtype=jnp.int32)

        def _one_class(class_idx):
            return _binary_auc_from_scores(probs[:, class_idx], labels, weights, class_idx)

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
        auc, auc_valid_count = _macro_ovr_auc(probs, labels, weights)
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

    def _runner(params, opt_state, features, labels, active_mask, segment_ids, intent_age, key):
        flat_features = _flatten_features(features)
        flat_labels = labels.reshape((-1,)).astype(jnp.int32)
        flat_weights = active_mask.reshape((-1,)).astype(jnp.float32)
        flat_segment_ids = segment_ids.reshape((-1,)).astype(jnp.int32)
        flat_intent_age = intent_age.reshape((-1,)).astype(jnp.int32)
        total_active = jnp.sum(flat_weights)
        split_key, train_key = jax.random.split(key)
        raw_train_weights, raw_holdout_weights = build_segment_grouped_holdout_weights(
            flat_weights,
            flat_segment_ids,
            split_key,
            holdout_fraction=holdout_fraction,
            jax=jax,
            jnp=jnp,
        )
        train_weights = raw_train_weights
        eval_weights = raw_holdout_weights

        def _update_step(carry, step_idx):
            step_params, step_opt_state, step_key = carry
            step_key = jax.random.fold_in(step_key, step_idx)
            sample_key, dropout_key = jax.random.split(step_key)
            mb_features, mb_labels, mb_weights = _take_batch(
                flat_features,
                flat_labels,
                train_weights,
                sample_key,
            )
            (_, train_metrics), grads = jax.value_and_grad(_loss_fn, has_aux=True)(
                step_params,
                mb_features,
                mb_labels,
                mb_weights,
                dropout_key,
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
        boundary_eval_metrics = _metric_snapshot(
            next_params,
            flat_features,
            flat_labels,
            eval_weights * (flat_intent_age == 0).astype(jnp.float32),
        )
        mature_eval_metrics = _metric_snapshot(
            next_params,
            flat_features,
            flat_labels,
            eval_weights * (flat_intent_age > 0).astype(jnp.float32),
        )
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
            "intent_disc_auc_ovr_macro_holdout_boundary": boundary_eval_metrics["auc_ovr_macro"],
            "intent_disc_auc_valid_class_count_holdout_boundary": boundary_eval_metrics["auc_valid_class_count"],
            "intent_disc_holdout_boundary_size": boundary_eval_metrics["active_count"],
            "intent_disc_auc_ovr_macro_holdout_mature": mature_eval_metrics["auc_ovr_macro"],
            "intent_disc_auc_valid_class_count_holdout_mature": mature_eval_metrics["auc_valid_class_count"],
            "intent_disc_holdout_mature_size": mature_eval_metrics["active_count"],
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


def build_intent_policy_sensitivity_runner(jax, jnp, policy_spec, *, sample_count: int):
    """Build a fixed-shape diagnostic comparing one policy under every intent."""
    from basketworld_jax.models import actor_critic_forward, apply_action_mask

    count = max(1, int(sample_count))
    num_intents = int(policy_spec.num_intents)
    player_count = int(policy_spec.training_player_count)
    pair_i, pair_j = np.triu_indices(num_intents, k=1)
    pair_i = jnp.asarray(pair_i, dtype=jnp.int32)
    pair_j = jnp.asarray(pair_j, dtype=jnp.int32)
    pair_count = int(len(pair_i))

    def _runner(params, flat_obs, action_mask, active_mask):
        flat_active = active_mask.reshape(-1).astype(jnp.bool_)
        valid_count = jnp.minimum(
            jnp.sum(flat_active.astype(jnp.int32)),
            jnp.asarray(count, dtype=jnp.int32),
        )
        indices = jnp.nonzero(flat_active, size=count, fill_value=0)[0]
        valid_rows = jnp.arange(count, dtype=jnp.int32) < valid_count
        sample_obs = flat_obs.reshape(-1, int(policy_spec.flat_obs_dim))[indices]
        sample_mask = action_mask.reshape(
            -1,
            player_count,
            int(policy_spec.action_dim_per_player),
        )[indices]

        if pair_count == 0:
            zero = jnp.asarray(0.0, dtype=jnp.float32)
            return {
                "intent_policy_sensitivity_sample_states": valid_count.astype(jnp.float32),
                "intent_policy_sensitivity_pairs": zero,
                "intent_policy_sensitivity_tv_mean": zero,
                "intent_policy_sensitivity_tv_p95": zero,
                "intent_policy_sensitivity_tv_max": zero,
                "intent_policy_sensitivity_argmax_disagreement": zero,
            }

        def _intent_probs(intent_index):
            out = actor_critic_forward(
                params,
                sample_obs,
                policy_spec,
                jnp,
                intent_context={
                    "intent_index": jnp.full(
                        (count,),
                        intent_index,
                        dtype=jnp.int32,
                    ),
                    "intent_gate": jnp.ones((count,), dtype=jnp.float32),
                },
            )
            return apply_action_mask(
                out["flat_policy_logits"],
                sample_mask,
                policy_spec,
                jax,
                jnp,
            )["probs"]

        probs = jax.vmap(_intent_probs)(jnp.arange(num_intents, dtype=jnp.int32))
        paired_tv = 0.5 * jnp.sum(
            jnp.abs(probs[pair_i] - probs[pair_j]),
            axis=-1,
        )
        paired_argmax_disagreement = (
            jnp.argmax(probs[pair_i], axis=-1)
            != jnp.argmax(probs[pair_j], axis=-1)
        ).astype(jnp.float32)
        valid_decisions = jnp.broadcast_to(
            valid_rows[None, :, None],
            (pair_count, count, player_count),
        )
        weights = valid_decisions.astype(jnp.float32)
        denominator = jnp.maximum(jnp.sum(weights), 1.0)
        tv_mean = jnp.sum(paired_tv * weights) / denominator
        disagreement_mean = jnp.sum(paired_argmax_disagreement * weights) / denominator
        valid_tv = jnp.where(valid_decisions, paired_tv, jnp.asarray(jnp.inf, dtype=jnp.float32))
        sorted_tv = jnp.sort(valid_tv.reshape(-1))
        percentile_index = jnp.maximum(
            0,
            jnp.ceil(0.95 * jnp.maximum(jnp.sum(weights), 1.0)).astype(jnp.int32) - 1,
        )
        tv_p95 = jnp.where(
            valid_count > 0,
            sorted_tv[jnp.minimum(percentile_index, sorted_tv.shape[0] - 1)],
            jnp.asarray(0.0, dtype=jnp.float32),
        )
        tv_max = jnp.max(jnp.where(valid_decisions, paired_tv, 0.0))
        return {
            "intent_policy_sensitivity_sample_states": valid_count.astype(jnp.float32),
            "intent_policy_sensitivity_pairs": jnp.asarray(pair_count, dtype=jnp.float32),
            "intent_policy_sensitivity_tv_mean": tv_mean,
            "intent_policy_sensitivity_tv_p95": tv_p95,
            "intent_policy_sensitivity_tv_max": tv_max,
            "intent_policy_sensitivity_argmax_disagreement": disagreement_mean,
        }

    return jax.jit(_runner)


def _intent_discriminator_embeddings(params, features, spec: IntentDiscriminatorSpec, jax, jnp):
    module = build_intent_discriminator_module(spec)
    if _is_set_encoder(spec.encoder_type):
        flat_features = {
            key: value.reshape((-1,) + tuple(value.shape[2:])).astype(jnp.float32)
            for key, value in features.items()
        }
    else:
        flat_features = features.reshape((-1, int(spec.input_dim))).astype(jnp.float32)
    out = module.apply({"params": params}, flat_features)
    if _is_set_encoder(spec.encoder_type):
        return out["embedding"].reshape(features["players"].shape[0], features["players"].shape[1], -1)
    return out["embedding"].reshape(features.shape[0], features.shape[1], -1)


def _sample_features_to_numpy(features, spec: IntentDiscriminatorSpec, jax) -> dict[str, np.ndarray]:
    if not _is_set_encoder(spec.encoder_type):
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
    payload = {
        "players": players.astype(np.float32),
        "globals": globals_vec.astype(np.float32),
    }
    globals_expanded = np.broadcast_to(
        globals_vec[:, None, :],
        (players.shape[0], players.shape[1], globals_vec.shape[-1]),
    )
    role_expanded = np.broadcast_to(
        role_flag[:, None, :],
        (players.shape[0], players.shape[1], role_flag.shape[-1]),
    )
    token_features = np.concatenate([players, globals_expanded, role_expanded], axis=-1)
    payload.update(
        {
            "role_flag": role_flag.astype(np.float32),
            "features": token_features.reshape(token_features.shape[0], -1).astype(np.float32),
        }
    )
    return payload


def build_intent_sample_dump(
    *,
    params,
    features,
    labels,
    active_mask,
    segment_ids,
    intent_age,
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
    all_actions = np.asarray(
        jax.device_get(trajectory.actions),
        dtype=np.int32,
    ).reshape(-1, int(spec.training_player_count))
    event_sources = {
        "pass_attempt_rate": trajectory.pass_attempts,
        "completed_pass_rate": trajectory.completed_passes,
        "turnover_rate": trajectory.turnovers,
        "shot_attempt_rate": trajectory.shot_attempts,
        "shot_make_rate": trajectory.shot_makes,
        "shot_two_rate": trajectory.shot_twos,
        "shot_three_rate": trajectory.shot_threes,
    }
    all_events = {
        name: np.asarray(jax.device_get(values), dtype=np.float32).reshape(-1)
        for name, values in event_sources.items()
    }
    payload = {
        "update_index": np.full((indices.size,), int(update_index), dtype=np.int32),
        "source_current_policy": np.ones((indices.size,), dtype=np.int8),
        "intent_index": labels_np[indices].astype(np.int32),
        "intent_segment_id": np.asarray(
            jax.device_get(segment_ids),
            dtype=np.int32,
        ).reshape(-1)[indices],
        "intent_age": np.asarray(
            jax.device_get(intent_age),
            dtype=np.int32,
        ).reshape(-1)[indices],
        "features": features_np[indices].astype(np.float32),
        "embedding": embeddings_np[indices].astype(np.float32),
        "bonus": bonus_np[indices].astype(np.float32),
        "actions": all_actions[indices],
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
    intent_counts = np.zeros((int(spec.num_intents),), dtype=np.int32)
    action_probs = np.zeros(
        (int(spec.num_intents), int(spec.action_dim_per_player)),
        dtype=np.float32,
    )
    event_rates = {
        name: np.zeros((int(spec.num_intents),), dtype=np.float32)
        for name in all_events
    }
    for intent_idx in range(int(spec.num_intents)):
        intent_mask = active_np & (labels_np == intent_idx)
        intent_counts[intent_idx] = int(np.sum(intent_mask))
        selected_actions = all_actions[intent_mask].reshape(-1)
        if selected_actions.size:
            action_probs[intent_idx] = np.bincount(
                selected_actions,
                minlength=int(spec.action_dim_per_player),
            )[: int(spec.action_dim_per_player)] / float(selected_actions.size)
        for name, values in all_events.items():
            if intent_counts[intent_idx] > 0:
                event_rates[name][intent_idx] = float(np.mean(values[intent_mask]))
    payload["summary_intent_active_count"] = intent_counts
    payload["summary_action_prob_by_intent"] = action_probs
    for name, values in event_rates.items():
        payload[f"summary_{name}_by_intent"] = values
    return payload
