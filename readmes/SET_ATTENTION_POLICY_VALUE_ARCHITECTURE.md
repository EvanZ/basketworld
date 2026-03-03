# Set-Attention Policy/Value Architecture (Mermaid)

This is the current architecture for `SetAttentionDualCriticPolicy`, including both actor branches (`directional` and `pointer_targeted`) and the dual-critic value path.

```mermaid
flowchart TD
    Obs["Observation Dict<br/>players, globals, role_flag"]
    Ext["SetAttentionExtractor<br/>broadcast globals -> token_mlp -> append CLS -> self-attention + LN<br/>(B, (P+C)*D)"]
    Split["_split_tokens<br/>(B, P+C, D)"]

    Obs --> Ext --> Split

    subgraph Actor["Actor / Policy Path"]
        PiTok["_extract_pi_player_tokens<br/>optional token_head_mlp_pi"]
        Pick["_select_action_player_indices + gather"]
        Mode{"pass_mode"}

        DirLogits["_get_all_directional_logits<br/>action_head or offense/defense action heads"]
        DirBias["_apply_pass_bias"]
        DirDist["SB3 MultiCategorical<br/>proba_distribution(action_logits)"]

        PtrQK["_get_pointer_qk<br/>pointer_query_head + pointer_key_head"]
        PtrSlots["_get_pointer_pass_slot_logits<br/>QK scores + pointer_slot_target_ids + legality masking"]
        PtrType["_get_pointer_action_type_logits<br/>pointer_action_type_head"]
        PtrBias["_apply_pointer_pass_bias<br/>PASS type bias/floor"]
        PtrDist["PointerTargetedMultiCategoricalDistribution<br/>log pi = log pi_type + 1[PASS]*log pi_slot"]

        PiTok --> Pick --> Mode

        Mode -->|directional| DirLogits --> DirBias --> DirDist

        Mode -->|pointer_targeted| PtrQK --> PtrSlots
        Mode -->|pointer_targeted| PtrType --> PtrBias --> PtrDist
        PtrSlots --> PtrDist
    end

    subgraph Value["Value / Critic Path (same in both pass modes)"]
        VfTok["optional token_head_mlp_vf"]
        Cls["Select CLS tokens<br/>offense=tokens[:, P, :], defense=tokens[:, P+1, :]"]
        Heads["value_net_offense + value_net_defense"]
        Gate["Role gate by role_flag"]
        VOut["V(s) (B,1)"]

        VfTok --> Cls --> Heads --> Gate --> VOut
    end

    Split --> PiTok
    Split --> VfTok

    DirDist --> AOut["Sampled/deterministic action IDs<br/>(legacy MultiDiscrete action space)"]
    PtrDist --> AOut
```

## Notes

- Pointer targeting adds actor-side heads and factorized action distribution only.
- Value estimation still uses the same two critics: `value_net_offense` and `value_net_defense`.
- Both actor branches emit actions in the same legacy action ID space so SB3 rollout buffers remain compatible.

## Code Anchors

- `basketworld/policies/set_attention_policy.py`
  - `SetAttentionExtractor.forward`
  - `_get_action_dist_from_latent`
  - `_get_pointer_pass_slot_logits`
  - `_get_value_from_latent`
