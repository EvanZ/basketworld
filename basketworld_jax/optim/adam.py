from __future__ import annotations


def build_adam_transform(optax, *, learning_rate, grad_clip_norm: float | None = None):
    transforms = []
    if grad_clip_norm is not None:
        transforms.append(optax.clip_by_global_norm(float(grad_clip_norm)))
    transforms.append(optax.adam(float(learning_rate)))
    return optax.chain(*transforms)


def init_optimizer_state(transform, params):
    return transform.init(params)


def optimizer_update(params, grads, opt_state, *, transform, optax):
    updates, new_opt_state = transform.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state


def global_norm(tree, optax):
    return optax.tree.norm(tree)

