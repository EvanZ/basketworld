#!/usr/bin/env bash
set -euo pipefail

python - <<'PY'
import numpy as np
import torch as th
import basketworld

from basketworld.utils.wrappers import SetObservationWrapper
from basketworld.policies.set_attention_policy import SetAttentionDualCriticPolicy

rng = np.random.default_rng(0)

env = SetObservationWrapper(basketworld.HexagonBasketballEnv(players=3))
obs, _ = env.reset(seed=0)

n_players = obs["players"].shape[0]
n_actions = int(env.action_space.nvec[0])

policy = SetAttentionDualCriticPolicy(
    env.observation_space,
    env.action_space,
    lr_schedule=lambda _: 0.0003,
)
policy.eval()


def get_logits_and_value(obs_dict):
    obs_tensor, _ = policy.obs_to_tensor(obs_dict)
    features = policy.extract_features(obs_tensor)
    latent_pi, latent_vf = policy.mlp_extractor(features)
    logits = policy._get_action_logits(latent_pi).reshape(1, n_players, n_actions)
    values = policy._get_value_from_latent(latent_vf, obs_tensor["role_flag"])
    return logits.detach().cpu().numpy()[0], float(values.detach().cpu().numpy().squeeze())


def topk(logits, k=3):
    idx = np.argsort(-logits)[:k]
    return [(int(i), float(logits[i])) for i in idx]


orig_logits, orig_value = get_logits_and_value(obs)

perm = rng.permutation(n_players)
inv_perm = np.argsort(perm)
perm_obs = dict(obs)
perm_obs["players"] = obs["players"][perm]
perm_obs["action_mask"] = obs["action_mask"][perm]

perm_logits, perm_value = get_logits_and_value(perm_obs)
perm_logits_reordered = perm_logits[inv_perm]

diff = np.abs(orig_logits - perm_logits_reordered)

print("Permutation:", perm.tolist())
print("Max abs diff (logits):", float(diff.max()))
print("Mean abs diff (logits):", float(diff.mean()))
print("Value orig:", orig_value)
print("Value perm:", perm_value)
print("")

for p in range(n_players):
    print(f"P{p} top3 orig:     {topk(orig_logits[p])}")
    print(f"P{p} top3 permuted: {topk(perm_logits_reordered[p])}")
    print("")
PY
