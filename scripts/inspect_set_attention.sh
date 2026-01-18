#!/usr/bin/env bash
set -euo pipefail

python - <<'PY'
import numpy as np
import torch as th
import basketworld

from basketworld.utils.wrappers import SetObservationWrapper
from basketworld.policies.set_attention_policy import SetAttentionDualCriticPolicy

env = SetObservationWrapper(basketworld.HexagonBasketballEnv(players=3))
obs, _ = env.reset(seed=0)

print("keys:", sorted(obs.keys()))
print("players:\n", obs["players"])
print("globals:", obs["globals"])

policy = SetAttentionDualCriticPolicy(
    env.observation_space,
    env.action_space,
    lr_schedule=lambda _: 0.0003,
)
extractor = policy.features_extractor

players = th.as_tensor(obs["players"]).unsqueeze(0)
globals_vec = th.as_tensor(obs["globals"]).unsqueeze(0)
g = globals_vec.unsqueeze(1).expand(-1, players.size(1), -1)
tokens = th.cat([players, g], dim=-1)
emb = extractor.token_mlp(tokens)
cls = extractor.cls_tokens.unsqueeze(0).expand(emb.size(0), -1, -1)
emb_with_cls = th.cat([emb, cls], dim=1)
attn_out, attn_weights = extractor.attn(
    emb_with_cls,
    emb_with_cls,
    emb_with_cls,
    need_weights=True,
    average_attn_weights=False,
)

print("attn_out shape:", tuple(attn_out.shape))
print("attn_weights shape:", tuple(attn_weights.shape))

ball_holder = int(env.unwrapped.ball_holder)
off_cls_idx = env.unwrapped.n_players
def_cls_idx = env.unwrapped.n_players + 1

print("ball_holder:", ball_holder)
def _print_weights(label, token_idx):
    if attn_weights.dim() == 3:
        # (B, tgt_len, src_len)
        weights = attn_weights[0, token_idx].detach().cpu().numpy()
        print(f"{label} weights:", weights)
        return
    if attn_weights.dim() == 4:
        # (B, heads, tgt_len, src_len)
        per_head = attn_weights[0, :, token_idx, :].detach().cpu().numpy()
        mean = per_head.mean(axis=0)
        print(f"{label} weights (per-head):\n", per_head)
        print(f"{label} weights (mean):", mean)
        return
    print(f"{label} weights: unexpected shape", tuple(attn_weights.shape))

_print_weights("ball_holder", ball_holder)
_print_weights("offense CLS", off_cls_idx)
_print_weights("defense CLS", def_cls_idx)

def value_for_role(role_value: float) -> float:
    obs_role = dict(obs)
    obs_role["role_flag"] = np.array([role_value], dtype=np.float32)
    obs_tensor, _ = policy.obs_to_tensor(obs_role)
    with th.no_grad():
        return float(policy.predict_values(obs_tensor).item())

print("value_offense:", value_for_role(1.0))
print("value_defense:", value_for_role(-1.0))
PY
