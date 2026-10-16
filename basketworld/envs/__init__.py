__all__ = ["Basketball3v3Env", "HexagonBasketballEnv"]


def __getattr__(name: str):
    if name == "Basketball3v3Env":
        from .basketworld_3x3_env import Basketball3v3Env

        return Basketball3v3Env
    if name == "HexagonBasketballEnv":
        from .basketworld_env_v2 import HexagonBasketballEnv

        return HexagonBasketballEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
