#!/usr/bin/env bash
set -euo pipefail

python - <<'PY'
import basketworld
from basketworld.utils.wrappers import SetObservationWrapper

env = SetObservationWrapper(basketworld.HexagonBasketballEnv(players=3))
obs, _ = env.reset(seed=0)

print("keys:", sorted(obs.keys()))
print("players shape:", obs["players"].shape)
print("globals shape:", obs["globals"].shape)
print("players:\n", obs["players"])
print("globals:", obs["globals"])
PY
