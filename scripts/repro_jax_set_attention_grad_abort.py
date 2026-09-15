"""Reproduce a JAX 0.10.x CPU abort in jitted set-attention gradients."""

import jax
import jax.numpy as jnp
from flax import linen as nn

BATCH = 512
PLAYERS = 10
PLAYER_DIM = 18
GLOBAL_DIM = 7
HIDDEN = 128
HEADS = 4
CLASSES = 8


class SetAttentionClassifier(nn.Module):
    @nn.compact
    def __call__(self, features):
        players = features["players"]
        globals_vec = features["globals"]
        role_flag = features["role_flag"]

        globals_expanded = jnp.broadcast_to(
            globals_vec[:, None, :],
            (players.shape[0], players.shape[1], GLOBAL_DIM),
        )
        role_expanded = jnp.broadcast_to(
            role_flag[:, None, :],
            (players.shape[0], players.shape[1], 1),
        )
        tokens = jnp.concatenate([players, globals_expanded, role_expanded], axis=-1)
        hidden = nn.relu(nn.Dense(HIDDEN)(tokens))
        hidden = nn.Dense(HIDDEN)(hidden)

        cls_token = self.param("cls_token", nn.initializers.zeros_init(), (1, HIDDEN))
        cls_batch = jnp.broadcast_to(
            cls_token[None, :, :], (hidden.shape[0], 1, HIDDEN)
        )
        hidden = jnp.concatenate([hidden, cls_batch], axis=1)

        head_dim = HIDDEN // HEADS
        qkv = nn.Dense(3 * HIDDEN)(hidden)
        qkv = qkv.reshape(hidden.shape[0], hidden.shape[1], 3, HEADS, head_dim)
        query, key, value = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        scores = jnp.einsum("bthd,bshd->bhts", query, key) * (head_dim**-0.5)
        weights = jax.nn.softmax(scores, axis=-1)
        attended = jnp.einsum("bhts,bshd->bthd", weights, value)
        attended = attended.reshape(hidden.shape[0], hidden.shape[1], HIDDEN)

        projected = nn.Dense(HIDDEN)(attended)
        hidden = nn.LayerNorm()(hidden + projected)
        ff = nn.relu(nn.Dense(HIDDEN)(hidden))
        hidden = nn.LayerNorm()(hidden + ff)
        return nn.Dense(CLASSES)(hidden[:, -1, :])


features = {
    "players": jnp.zeros((BATCH, PLAYERS, PLAYER_DIM), dtype=jnp.float32),
    "globals": jnp.zeros((BATCH, GLOBAL_DIM), dtype=jnp.float32),
    "role_flag": jnp.zeros((BATCH, 1), dtype=jnp.float32),
}
labels = jnp.arange(BATCH, dtype=jnp.int32) % CLASSES
model = SetAttentionClassifier()
params = model.init(jax.random.PRNGKey(0), features)["params"]


def loss_fn(model_params):
    logits = model.apply({"params": model_params}, features)
    selected = jnp.take_along_axis(
        jax.nn.log_softmax(logits), labels[:, None], axis=-1
    )[:, 0]
    return -jnp.mean(selected)


print("jax:", jax.__version__, "backend:", jax.default_backend(), flush=True)
print("non-jit loss:", float(jax.value_and_grad(loss_fn)(params)[0]), flush=True)
print("starting jitted value_and_grad", flush=True)
loss, grads = jax.jit(jax.value_and_grad(loss_fn))(params)
jax.tree.map(lambda x: x.block_until_ready(), grads)
print("jitted loss:", float(loss), flush=True)
