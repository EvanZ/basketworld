from __future__ import annotations

from typing import Any


class EnvView:
    """Read-only wrapper-safe env attribute view for backend serialization code."""

    def __init__(self, env: Any) -> None:
        object.__setattr__(self, "_env", env)

    def __getattr__(self, name: str) -> Any:
        env = object.__getattribute__(self, "_env")
        return get_env_attr(env, name)


def get_env_attr(env: Any, name: str, default: Any = None) -> Any:
    """Read env/wrapper attributes without relying on Gym wrapper forwarding.

    Gymnasium warns on direct wrapper attribute access such as `env.foo` when `foo`
    lives on another wrapper or the base env. Prefer `get_wrapper_attr()` when
    available, otherwise fall back to `env.unwrapped`.
    """

    if env is None:
        return default

    getter = getattr(env, "get_wrapper_attr", None)
    if callable(getter):
        try:
            return getter(str(name))
        except Exception:
            pass

    base_env = getattr(env, "unwrapped", env)
    return getattr(base_env, str(name), default)


def env_view(env: Any) -> EnvView:
    return EnvView(env)
