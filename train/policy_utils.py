import os
import re
import tempfile
from typing import Callable, Optional, List

import mlflow
import tempfile
import torch
from stable_baselines3 import PPO
from basketworld.utils.policies import PassBiasMultiInputPolicy, PassBiasDualCriticPolicy
from basketworld.policies import SetAttentionDualCriticPolicy, SetAttentionExtractor


def get_random_policy_from_artifacts(
    client,
    run_id,
    model_prefix,
    tmpdir,
    K: int = 10,
    beta: float = 0.8,
    uniform_eps: float = 0.0,
    sample_geometric_fn: Callable[[List[int], float], int] | None = None,
):
    """Sample an opponent checkpoint using a geometric decay over recent K snapshots."""
    artifact_path = "models"
    all_artifacts = client.list_artifacts(run_id, artifact_path)

    # Extract paths for prefix (e.g., unified)
    pattern = re.compile(rf"{model_prefix}_(?:alternation|iter)_(\d+)\.zip$")

    def sort_key(p):
        # Accept models/{prefix}_alternation_{idx}.zip or models/{prefix}_iter_{idx}.zip
        m = pattern.search(p.path)
        if m:
            return int(m.group(1))
        return -1

    filtered = [p for p in all_artifacts if pattern.search(p.path)]
    filtered = sorted(filtered, key=sort_key)

    if not filtered:
        raise ValueError(f"No artifacts found for prefix '{model_prefix}'")

    # Use geometric decay over last K checkpoints
    recent_pols = filtered[-K:]
    # Reservoir of all policies if we want a uniform exploration sample
    team_policies = filtered

    import random
    if sample_geometric_fn is None:
        try:
            from train_utils import sample_geometric
        except ImportError:
            from train.train_utils import sample_geometric
        sample_geometric_fn = sample_geometric

    if random.random() < uniform_eps and len(team_policies) > len(recent_pols):
        chosen = random.choice(team_policies)
    else:
        idxs = list(range(len(recent_pols)))
        chosen_idx = sample_geometric_fn(idxs, beta)
        chosen = recent_pols[chosen_idx]

    local_path = client.download_artifacts(run_id, chosen.path, tmpdir)
    return local_path


def get_opponent_policy_pool_for_envs(
    client,
    run_id: str,
    team_prefix: str,
    tmpdir: str,
    num_envs: int,
    K: int,
    beta: float,
    uniform_eps: float,
    per_env_sampling: bool,
) -> list[str]:
    """Return a list of policy paths (length 1 or num_envs) for opponent sampling."""
    artifact_path = "models"
    all_artifacts = client.list_artifacts(run_id, artifact_path)

    pattern = re.compile(rf"{team_prefix}_(?:alternation|iter)_(\d+)\.zip$")

    def sort_key(p):
        m = pattern.search(p.path)
        if m:
            return int(m.group(1))
        return -1

    filtered = [p for p in all_artifacts if pattern.search(p.path)]
    filtered = sorted(filtered, key=sort_key)
    if not filtered:
        return []

    import random
    try:
        from train_utils import sample_geometric
    except ImportError:
        from train.train_utils import sample_geometric

    if per_env_sampling:
        paths = []
        recent_pols = filtered[-K:]
        team_policies = filtered
        for _ in range(num_envs):
            if random.random() < uniform_eps and len(team_policies) > len(recent_pols):
                chosen = random.choice(team_policies)
            else:
                idxs = list(range(len(recent_pols)))
                chosen_idx = sample_geometric(idxs, beta)
                chosen = recent_pols[chosen_idx]
            local_path = client.download_artifacts(run_id, chosen.path, tmpdir)
            paths.append(local_path)
        return paths
    else:
        chosen = filtered[-1]
        local_path = client.download_artifacts(run_id, chosen.path, tmpdir)
        return [local_path]


def get_latest_policy_path(client, run_id: str, team_prefix: str) -> Optional[str]:
    """Return latest policy artifact path for team prefix."""
    artifacts = client.list_artifacts(run_id, "models")

    pattern = re.compile(rf"{team_prefix}_(?:alternation|iter)_(\d+)\.zip$")

    def sort_key(p):
        m = pattern.search(p.path)
        if m:
            return int(m.group(1))
        return -1

    filtered = [p for p in artifacts if pattern.search(p.path)]
    if not filtered:
        return None
    filtered = sorted(filtered, key=sort_key)
    latest = filtered[-1]
    return latest.path


def get_latest_unified_policy_path(client, run_id: str) -> Optional[str]:
    """Return latest unified policy artifact path."""
    return get_latest_policy_path(client, run_id, "unified")


def get_max_alternation_index(client, run_id: str) -> int:
    """Return max alternation index from saved models in an MLflow run."""
    artifacts = client.list_artifacts(run_id, "models")
    idxs = []
    pattern = re.compile(r"_(?:alternation|iter)_(\d+)\.zip$")
    for f in artifacts:
        m = pattern.search(f.path)
        if m:
            idxs.append(int(m.group(1)))
    return max(idxs) if idxs else 0


def transfer_critic_weights(args, unified_policy) -> None:
    """Transfer dual critic weights from a prior run into the current unified_policy."""
    if args.init_critic_from_run is None:
        mlflow.log_param("critic_transfer_enabled", False)
        return

    print(f"\n{'='*80}")
    print(f"[Critic Transfer] Loading critic weights from run: {args.init_critic_from_run}")
    print(f"{'='*80}")

    try:
        source_run = mlflow.get_run(args.init_critic_from_run)
        client = mlflow.tracking.MlflowClient()
        with tempfile.TemporaryDirectory() as tmpd:
            max_alt_idx = get_max_alternation_index(client, args.init_critic_from_run)
            if max_alt_idx == 0:
                artifact_path = "unified_policy_final"
                print(f"[Critic Transfer] No alternation models found, trying unified_policy_final...")
            else:
                artifact_path = f"models/unified_policy_alt_{max_alt_idx}.zip"
                print(f"[Critic Transfer] Found latest alternation: {max_alt_idx}")

            print(f"[Critic Transfer] Downloading artifacts from run {args.init_critic_from_run}...")
            print(f"[Critic Transfer] Artifact path: {artifact_path}")
            local_path = client.download_artifacts(
                args.init_critic_from_run, artifact_path, tmpd
            )

            print(f"[Critic Transfer] Loading source model from: {local_path}")
            custom_objects = {
                "PassBiasMultiInputPolicy": PassBiasMultiInputPolicy,
                "PassBiasDualCriticPolicy": PassBiasDualCriticPolicy,
                "SetAttentionDualCriticPolicy": SetAttentionDualCriticPolicy,
                "SetAttentionExtractor": SetAttentionExtractor,
            }
            source_policy = PPO.load(local_path, custom_objects=custom_objects)

            source_use_dual_critic = source_run.data.params.get("use_dual_critic", "false").lower() == "true"
            if not source_use_dual_critic:
                raise ValueError(
                    f"Source run {args.init_critic_from_run} does not use dual critic architecture."
                )

            source_net_arch_used = source_run.data.params.get("net_arch_used", "unknown")
            target_net_arch_used = str(unified_policy.policy.net_arch)

            print(f"[Critic Transfer] Source net_arch: {source_net_arch_used}")
            print(f"[Critic Transfer] Target net_arch: {target_net_arch_used}")

            def extract_vf_arch(net_arch_str):
                try:
                    import ast
                    net_arch = ast.literal_eval(net_arch_str)
                    if isinstance(net_arch, list) and len(net_arch) > 0:
                        if isinstance(net_arch[0], dict):
                            return net_arch[0].get("vf", net_arch)
                    return net_arch
                except Exception:
                    return net_arch_str

            source_vf_arch = extract_vf_arch(source_net_arch_used)
            target_vf_arch = extract_vf_arch(target_net_arch_used)

            if str(source_vf_arch) != str(target_vf_arch):
                print(f"[Critic Transfer] WARNING: Value network architectures differ!")
                print(f"  Source vf: {source_vf_arch}")
                print(f"  Target vf: {target_vf_arch}")
                print(f"[Critic Transfer] Attempting transfer anyway - weights will be copied where dimensions match.")

            source_policy_net = source_policy.policy
            target_policy_net = unified_policy.policy

            if not (hasattr(source_policy_net, "value_net_offense") and hasattr(source_policy_net, "value_net_defense")):
                raise ValueError("Source policy does not have value_net_offense/value_net_defense attributes")
            if not (hasattr(target_policy_net, "value_net_offense") and hasattr(target_policy_net, "value_net_defense")):
                raise ValueError("Target policy does not have value_net_offense/value_net_defense attributes")

            target_off_before = list(target_policy_net.value_net_offense.parameters())[0].data.clone()
            target_def_before = list(target_policy_net.value_net_defense.parameters())[0].data.clone()
            source_off_weights = list(source_policy_net.value_net_offense.parameters())[0].data.clone()
            source_def_weights = list(source_policy_net.value_net_defense.parameters())[0].data.clone()

            print(f"[Critic Transfer] Pre-transfer target offense critic weight sample: {target_off_before.flatten()[:5].tolist()}")
            print(f"[Critic Transfer] Source offense critic weight sample: {source_off_weights.flatten()[:5].tolist()}")

            print(f"[Critic Transfer] Checking mlp_extractor.value_net structure...")
            target_vf_first_before = None
            if hasattr(source_policy_net.mlp_extractor, "value_net"):
                print(f"[Critic Transfer]   Source value_net: {source_policy_net.mlp_extractor.value_net}")
                print(f"[Critic Transfer]   Target value_net: {target_policy_net.mlp_extractor.value_net}")
                source_vf_params = sum(p.numel() for p in source_policy_net.mlp_extractor.value_net.parameters())
                target_vf_params = sum(p.numel() for p in target_policy_net.mlp_extractor.value_net.parameters())
                print(f"[Critic Transfer]   Source value_net params: {source_vf_params}")
                print(f"[Critic Transfer]   Target value_net params: {target_vf_params}")

                source_vf_first = list(source_policy_net.mlp_extractor.value_net.parameters())[0].data.flatten()[:5].clone()
                target_vf_first_before = list(target_policy_net.mlp_extractor.value_net.parameters())[0].data.flatten()[:5].clone()
                print(f"[Critic Transfer]   Source value_net first layer sample: {source_vf_first.tolist()}")
                print(f"[Critic Transfer]   Target value_net first layer (before): {target_vf_first_before.tolist()}")

            print("[Critic Transfer] Transferring value feature extractor (mlp_extractor.value_net)...")
            if hasattr(source_policy_net.mlp_extractor, "value_net"):
                target_policy_net.mlp_extractor.value_net.load_state_dict(
                    source_policy_net.mlp_extractor.value_net.state_dict()
                )
                print("[Critic Transfer]   ✓ Value feature extractor transferred")
            else:
                print("[Critic Transfer]   ⚠️  No separate value_net in mlp_extractor (shared features)")

            print("[Critic Transfer] Transferring offense critic value head...")
            target_policy_net.value_net_offense.load_state_dict(
                source_policy_net.value_net_offense.state_dict()
            )
            print("[Critic Transfer] Transferring defense critic value head...")
            target_policy_net.value_net_defense.load_state_dict(
                source_policy_net.value_net_defense.state_dict()
            )

            target_off_after = list(target_policy_net.value_net_offense.parameters())[0].data
            target_def_after = list(target_policy_net.value_net_defense.parameters())[0].data

            print(f"[Critic Transfer] Post-transfer target offense critic weight sample: {target_off_after.flatten()[:5].tolist()}")

            off_match = torch.allclose(target_off_after, source_off_weights, rtol=1e-5, atol=1e-7)
            def_match = torch.allclose(target_def_after, source_def_weights, rtol=1e-5, atol=1e-7)
            off_changed = not torch.allclose(target_off_after, target_off_before, rtol=1e-5, atol=1e-7)
            def_changed = not torch.allclose(target_def_after, target_def_before, rtol=1e-5, atol=1e-7)

            vf_match = False
            vf_changed = False
            if hasattr(source_policy_net.mlp_extractor, "value_net") and target_vf_first_before is not None:
                target_vf_first_after = list(target_policy_net.mlp_extractor.value_net.parameters())[0].data.flatten()[:5].clone()
                print(f"[Critic Transfer]   Target value_net first layer (after): {target_vf_first_after.tolist()}")
                vf_match = all(
                    torch.allclose(tp.data, sp.data, rtol=1e-5, atol=1e-7)
                    for tp, sp in zip(
                        target_policy_net.mlp_extractor.value_net.parameters(),
                        source_policy_net.mlp_extractor.value_net.parameters(),
                    )
                )
                vf_changed = not torch.allclose(target_vf_first_after, target_vf_first_before, rtol=1e-5, atol=1e-7)
                print(f"[Critic Transfer] Value feature extractor matches source: {vf_match}")
                print(f"[Critic Transfer] Value feature extractor changed from initial: {vf_changed}")
            elif hasattr(source_policy_net.mlp_extractor, "value_net"):
                print(f"[Critic Transfer] ⚠️  Could not verify value feature extractor (before sample was None)")

            all_verified = off_match and def_match and off_changed and def_changed
            if hasattr(source_policy_net.mlp_extractor, "value_net"):
                all_verified = all_verified and vf_match and vf_changed

            if not all_verified:
                print(f"[Critic Transfer] ⚠️  WARNING: Weight transfer verification failed!")
                if not (vf_match and vf_changed):
                    print(f"[Critic Transfer]     ❌ Value feature extractor transfer failed!")
                if not (off_match and off_changed):
                    print(f"[Critic Transfer]     ❌ Offense head transfer failed!")
                if not (def_match and def_changed):
                    print(f"[Critic Transfer]     ❌ Defense head transfer failed!")
            else:
                print(f"[Critic Transfer] ✓ All weight transfers verified successfully")

            print(f"[Critic Transfer] ✓ Successfully transferred critic weights from run: {args.init_critic_from_run}")
            print(f"[Critic Transfer] Actor network remains randomly initialized for fresh policy learning.")
            print(f"{'='*80}\n")

            mlflow.log_param("critic_transfer_source_run", args.init_critic_from_run)
            mlflow.log_param("critic_transfer_enabled", True)
            mlflow.log_param("critic_transfer_offense_head_verified", off_match and off_changed)
            mlflow.log_param("critic_transfer_defense_head_verified", def_match and def_changed)
            mlflow.log_param("critic_transfer_value_extractor_verified", vf_match and vf_changed)
            mlflow.log_param("critic_transfer_all_verified", all_verified)

    except Exception as e:
        print(f"[Critic Transfer] ERROR: Failed to transfer critic weights: {e}")
        print(f"[Critic Transfer] Continuing with randomly initialized critics.")
        print(f"{'='*80}\n")
        mlflow.log_param("critic_transfer_enabled", False)
        mlflow.log_param("critic_transfer_error", str(e))
