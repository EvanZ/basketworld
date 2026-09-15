# GitHub Issue: CPU jitted `value_and_grad` abort in JAX 0.10.x

## Suggested title

`[CPU] jax.jit(jax.value_and_grad) aborts with "(0 <= -10) is statically false" for a Flax set-attention model in JAX 0.10.x`

## Description

On CPU, JAX 0.10.0 and 0.10.1 abort the Python process while compiling/executing `jax.jit(jax.value_and_grad(loss_fn))` for the small Flax set-attention classifier in the reproducer below.

The same model and loss behave as follows:

- The forward pass succeeds.
- Non-jitted `jax.value_and_grad(loss_fn)` succeeds.
- Jitted `jax.value_and_grad(loss_fn)` aborts with exit code 134 and only prints:

```text
(0 <= -10) is statically false.
Aborted (core dumped)
```

The same script succeeds with JAX/JAXLIB 0.9.2, so this appears to be a JAX 0.10.x CPU compilation regression. The failure is a process abort rather than a catchable Python exception.

## Minimal reproducer

Install the affected versions:

```bash
python -m venv /tmp/jax-repro
/tmp/jax-repro/bin/pip install jax==0.10.1 jaxlib==0.10.1 flax==0.12.6
/tmp/jax-repro/bin/python repro_jax_set_attention_grad_abort.py
```

Reproducer:

```python
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


def loss_fn(params):
    logits = model.apply({"params": params}, features)
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
```

## Actual behavior

JAX/JAXLIB 0.10.1:

```text
jax: 0.10.1 backend: cpu
non-jit loss: 2.0794413089752197
starting jitted value_and_grad
(0 <= -10) is statically false.
Aborted (core dumped)
```

JAX/JAXLIB 0.10.0 fails in the same way.

## Expected behavior

The jitted gradient should compile and return the same loss as the eager gradient, without aborting the process.

For comparison, JAX/JAXLIB 0.9.2 produces:

```text
jax: 0.9.2 backend: cpu
non-jit loss: 2.0794413089752197
starting jitted value_and_grad
jitted loss: 2.0794413089752197
```

## System information

```text
OS: Ubuntu 24.04, Linux 6.17.0-35-generic, x86_64
CPU: AMD Ryzen 9 9955HX 16-Core Processor
Python: 3.12.3
Backend: CPU
Devices: [CpuDevice(id=0)]

Affected test environment:
jax: 0.10.1
jaxlib: 0.10.1
flax: 0.12.6
numpy: 2.5.1
scipy: 1.18.0

Also reproduced in the application environment with:
jax: 0.10.0
jaxlib: 0.10.0
flax: 0.12.6
numpy: 2.3.4
scipy: 1.16.3

Known working comparison:
jax: 0.9.2
jaxlib: 0.9.2
flax: 0.12.6
```

## Additional observations

- CPU is the only backend tested.
- Removing dropout does not avoid the failure.
- The raw attention gradient in isolation succeeds; the abort requires the complete Flax graph shown above.
- In the original training application, the abort occurred on the first invocation of this jitted discriminator update after a 100-update warmup. The delayed failure was therefore caused by deferred execution of the affected function, not by update 100 itself.

