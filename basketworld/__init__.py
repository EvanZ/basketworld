
__all__ = ["Basketball3v3Env", "HexagonBasketballEnv"]


def __getattr__(name: str):
    """Keep legacy environment exports lazy so neutral modules stay neutral."""
    if name == "Basketball3v3Env":
        from .envs.basketworld_3x3_env import Basketball3v3Env

        return Basketball3v3Env
    if name == "HexagonBasketballEnv":
        from .envs.basketworld_env_v2 import HexagonBasketballEnv

        return HexagonBasketballEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
