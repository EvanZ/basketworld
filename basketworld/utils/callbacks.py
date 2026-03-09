import time
import os
import csv
import tempfile
import random
from typing import Optional

import numpy as np
import mlflow
import torch
import torch.nn.functional as F
from stable_baselines3.common.callbacks import BaseCallback

from basketworld.utils.intent_discovery import (
    IntentDiscriminator,
    IntentEpisodeBuffer,
    IntentTransition,
    RunningMeanStd,
    compute_episode_embeddings,
    extract_action_features_for_env,
    flatten_observation_for_env,
)


class TimingCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.rollout_start = None
        self.update_start = None
        self.rollout_times = []
        self.update_times = []

    # ---------- roll-out ----------
    def _on_rollout_start(self) -> None:
        """Called at the start of the rollout phase."""
        self.rollout_start = time.perf_counter()

    def _on_rollout_end(self) -> None:
        """Called at the end of the rollout phase."""
        dt = time.perf_counter() - self.rollout_start
        self.rollout_times.append(dt)
        self.logger.record("perf/rollout_sec", dt)

    # ---------- gradient update ----------
    def _on_training_start(self) -> None:
        """Called at the start of the gradient update phase."""
        # Note: This is a bit of a misnomer in SB3.
        # _on_training_start is called once before the whole training loop.
        # We will use _on_step to mark the start of the update phase, as it's called right before.
        # However, for simplicity and to align with the provided snippet, we will use this hook,
        # acknowledging it measures the total training time per `learn` call, not per update cycle.
        # A more precise hook would be to wrap the train method itself.
        # For this implementation, we will place the start timer before the update phase.
        # The user's code uses _on_training_start, but that's called only once.
        # The _on_step is called multiple times, but before the rollout.
        # Looking at SB3 docs, _on_rollout_end is followed by the update. So _on_rollout_end
        # can be used to time the start of the update, and _on_rollout_start for the end of the update.
        # This seems counter-intuitive.

        # Let's re-read the user request.
        # `_on_training_start` is called before `model.learn`.
        # `_on_rollout_start` is called before collecting rollouts.
        # `_on_rollout_end` is called after collecting rollouts.
        # `_on_step` is called at each step inside the rollout collection.
        # `_on_training_end` is called after `model.learn`.

        # The user's code uses `_on_training_start` and `_on_training_end`.
        # This seems to be incorrect if `learn` is called only once.
        # `_on_training_start` runs once at the beginning of `learn`.
        # `_on_training_end` runs once at the end of `learn`.
        # This would only produce one measurement for the entire training run.

        # The goal is to measure each update cycle.
        # An update cycle in PPO is: collect rollouts -> update policy.
        # So: `_on_rollout_start` -> ... collect ... -> `_on_rollout_end` -> ... update ... -> `_on_rollout_start`

        # The time between `_on_rollout_end` and the next `_on_rollout_start` is the update time.
        # This seems like a more robust way to measure it.

        self.update_start = time.perf_counter()

    def _on_step(self) -> bool:
        # This is called within the rollout collection, so we don't need it for timing the update phase.
        return True

    def _on_training_end(self) -> None:
        """Called at the end of the gradient update phase."""
        # This is called after the training loop, so we can't use it to time each update.
        # Let's adjust the logic slightly to correctly capture per-update-cycle times.
        # We will time the update phase from the end of one rollout to the start of the next.

        # The provided code seems to have a slight flaw in its logic for timing updates with SB3's hooks.
        # `_on_training_start` and `_on_training_end` are only called once for the entire `.learn()` process.
        # To measure each update cycle, we should time the period *between* rollout collections.
        pass


# A more accurate implementation based on SB3 hook execution order
class RolloutUpdateTimingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.rollout_time = 0
        self.update_time = 0
        self.rollout_start_time = 0
        self.update_start_time = 0

        self.rollout_times = []
        self.update_times = []

    def _on_rollout_start(self) -> None:
        """Hook called before the rollout collection starts."""
        # This marks the end of the previous update phase.
        if self.update_start_time != 0:
            self.update_time = time.perf_counter() - self.update_start_time
            self.update_times.append(self.update_time)
            self.logger.record("perf/update_sec", self.update_time)

        self.rollout_start_time = time.perf_counter()

    def _on_rollout_end(self) -> None:
        """Hook called after the rollout collection finishes."""
        # This marks the end of the rollout phase and the start of the update phase.
        self.rollout_time = time.perf_counter() - self.rollout_start_time
        self.rollout_times.append(self.rollout_time)
        self.logger.record("perf/rollout_sec", self.rollout_time)

        self.update_start_time = time.perf_counter()

    def _on_step(self) -> bool:
        return True


# --- MLflow Metrics Callback ---


class MLflowCallback(BaseCallback):
    """
    Log episode-aggregated metrics to MLflow at a fixed frequency using SB3 ep_info_buffer.
    """

    def __init__(self, team_name: str, log_freq: int = 2048, verbose=0):
        super(MLflowCallback, self).__init__(verbose)
        self.team_name = team_name
        self.log_freq = log_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.log_freq == 0:
            if self.model.ep_info_buffer:
                # Global step from SB3
                global_step = self.model.num_timesteps

                ep_rew_mean = np.mean(
                    [ep.get("r", 0.0) for ep in self.model.ep_info_buffer]
                )
                ep_len_mean = np.mean(
                    [ep.get("l", 0.0) for ep in self.model.ep_info_buffer]
                )

                def mean_key(key: str, default: float = 0.0):
                    vals = [
                        ep.get(key) for ep in self.model.ep_info_buffer if key in ep
                    ]
                    return float(np.mean(vals)) if vals else default

                shot_dunk_pct = mean_key("shot_dunk")
                shot_2pt_pct = mean_key("shot_2pt")
                shot_3pt_pct = mean_key("shot_3pt")
                asst_dunk_pct = mean_key("assisted_dunk")
                asst_2pt_pct = mean_key("assisted_2pt")
                asst_3pt_pct = mean_key("assisted_3pt")
                pot_asst_dunk_pct = mean_key("potential_assisted_dunk")
                pot_asst_2pt_pct = mean_key("potential_assisted_2pt")
                pot_asst_3pt_pct = mean_key("potential_assisted_3pt")
                pot_asst_total = mean_key("potential_assists")
                passes_avg = mean_key("passes")
                turnover_pct = mean_key("turnover")
                turnover_pass_oob_pct = mean_key("turnover_pass_oob")
                turnover_intercepted_pct = mean_key("turnover_intercepted")
                turnover_pressure_pct = mean_key("turnover_pressure")
                turnover_offensive_lane_pct = mean_key("turnover_offensive_lane")
                defensive_lane_violation_pct = mean_key("defensive_lane_violation")
                rejected_move_occupied = mean_key("move_rejected_occupied")

                def mean_ppp(default: float = 0.0):
                    numer = []
                    for ep in self.model.ep_info_buffer:
                        m2 = float(ep.get("made_2pt", 0.0))
                        m3 = float(ep.get("made_3pt", 0.0))
                        md = float(ep.get("made_dunk", 0.0))
                        # Defensive 3-second violation awards offense 1 point.
                        lane_violation_pts = float(ep.get("defensive_lane_violation", 0.0))
                        att = float(ep.get("attempts", 0.0))
                        tov = float(ep.get("turnover", 0.0))
                        n = (2.0 * m2) + (3.0 * m3) + (2.0 * md) + lane_violation_pts
                        d = max(1.0, att + tov + lane_violation_pts)
                        numer.append(n / d)
                    return float(np.mean(numer)) if numer else default

                ppp_avg = mean_ppp()

                mlflow.log_metric(
                    f"{self.team_name} Mean Episode Reward",
                    ep_rew_mean,
                    step=global_step,
                )
                mlflow.log_metric(
                    f"{self.team_name} Mean Episode Length",
                    ep_len_mean,
                    step=global_step,
                )

                # Entropy coefficient (supports constant or schedule)
                try:
                    total_ts = getattr(self.model, "_total_timesteps", None)
                    if callable(getattr(self.model, "ent_coef", None)):
                        progress_remaining = 1.0
                        if total_ts and total_ts > 0:
                            progress_remaining = max(
                                0.0,
                                min(
                                    1.0,
                                    1.0 - (self.model.num_timesteps / float(total_ts)),
                                ),
                            )
                        current_ent_coef = float(
                            self.model.ent_coef(progress_remaining)
                        )
                    else:
                        current_ent_coef = float(getattr(self.model, "ent_coef", 0.0))
                    mlflow.log_metric(
                        f"{self.team_name} Entropy Coef",
                        current_ent_coef,
                        step=global_step,
                    )
                except Exception:
                    pass

                mlflow.log_metric(
                    f"{self.team_name} ShotPct Dunk", shot_dunk_pct, step=global_step
                )
                mlflow.log_metric(
                    f"{self.team_name} ShotPct 2PT", shot_2pt_pct, step=global_step
                )
                mlflow.log_metric(
                    f"{self.team_name} ShotPct 3PT", shot_3pt_pct, step=global_step
                )
                mlflow.log_metric(
                    f"{self.team_name} Assist ShotPct Dunk",
                    asst_dunk_pct,
                    step=global_step,
                )
                mlflow.log_metric(
                    f"{self.team_name} Assist ShotPct 2PT",
                    asst_2pt_pct,
                    step=global_step,
                )
                mlflow.log_metric(
                    f"{self.team_name} Assist ShotPct 3PT",
                    asst_3pt_pct,
                    step=global_step,
                )
                mlflow.log_metric(
                    f"{self.team_name} Potential Assist Dunk",
                    pot_asst_dunk_pct,
                    step=global_step,
                )
                mlflow.log_metric(
                    f"{self.team_name} Potential Assist 2PT",
                    pot_asst_2pt_pct,
                    step=global_step,
                )
                mlflow.log_metric(
                    f"{self.team_name} Potential Assist 3PT",
                    pot_asst_3pt_pct,
                    step=global_step,
                )
                mlflow.log_metric(
                    f"{self.team_name} Potential Assists",
                    pot_asst_total,
                    step=global_step,
                )
                mlflow.log_metric(
                    f"{self.team_name} Passes / Episode", passes_avg, step=global_step
                )
                mlflow.log_metric(
                    f"{self.team_name} TurnoverPct", turnover_pct, step=global_step
                )
                mlflow.log_metric(
                    f"{self.team_name} Turnover Pass OOB",
                    turnover_pass_oob_pct,
                    step=global_step,
                )
                mlflow.log_metric(
                    f"{self.team_name} Turnover Intercepted",
                    turnover_intercepted_pct,
                    step=global_step,
                )
                mlflow.log_metric(
                    f"{self.team_name} Turnover Pressure",
                    turnover_pressure_pct,
                    step=global_step,
                )
                mlflow.log_metric(
                    f"{self.team_name} 3-Second Violation",
                    turnover_offensive_lane_pct,
                    step=global_step,
                )
                mlflow.log_metric(
                    f"{self.team_name} Illegal Defense Violation",
                    defensive_lane_violation_pct,
                    step=global_step,
                )
                mlflow.log_metric(
                    f"{self.team_name} Rejected Move Occupied",
                    rejected_move_occupied,
                    step=global_step,
                )
                mlflow.log_metric(f"{self.team_name} PPP", ppp_avg, step=global_step)
                # Log current pass logit bias if custom policy exposes it
                try:
                    pass_bias = float(
                        getattr(self.model.policy, "pass_logit_bias", 0.0)
                    )
                    mlflow.log_metric(
                        f"{self.team_name} Pass Logit Bias", pass_bias, step=global_step
                    )
                except Exception:
                    pass
                # --- Phi diagnostics: averages per episode in buffer (if present) ---
                try:
                    phi_beta_avg = mean_key("phi_beta")
                    phi_prev_avg = mean_key("phi_prev")
                    phi_next_avg = mean_key("phi_next")
                    mlflow.log_metric(
                        f"{self.team_name} Phi Beta", phi_beta_avg, step=global_step
                    )
                    mlflow.log_metric(
                        f"{self.team_name} Phi Prev", phi_prev_avg, step=global_step
                    )
                    mlflow.log_metric(
                        f"{self.team_name} Phi Next", phi_next_avg, step=global_step
                    )
                except Exception:
                    pass
                # (Reverted) no ground-truth auditing metrics
        return True


class GradNormCallback(BaseCallback):
    """Log gradient norms at rollout start (after the previous update)."""

    def __init__(self, log_freq_rollouts: int = 1, verbose=0):
        super().__init__(verbose)
        self.log_freq_rollouts = max(1, int(log_freq_rollouts))
        self._rollouts = 0

    @staticmethod
    def _grad_norm(module) -> Optional[float]:
        if module is None:
            return None
        total = 0.0
        has_grad = False
        for param in module.parameters(recurse=True):
            if param.grad is None:
                continue
            grad_norm = float(param.grad.data.norm(2).item())
            total += grad_norm * grad_norm
            has_grad = True
        return (total ** 0.5) if has_grad else None

    def _on_rollout_start(self) -> None:
        self._rollouts += 1
        if self._rollouts % self.log_freq_rollouts != 0:
            return

        policy = getattr(self.model, "policy", None)
        if policy is None:
            return

        step = int(getattr(self.model, "num_timesteps", 0))
        metrics = {}

        features = getattr(policy, "features_extractor", None)
        if features is not None:
            metrics["GradNorm/token_mlp"] = self._grad_norm(
                getattr(features, "token_mlp", None)
            )
            metrics["GradNorm/attn"] = self._grad_norm(
                getattr(features, "attn", None)
            )

        metrics["GradNorm/action_head"] = self._grad_norm(
            getattr(policy, "action_head", None)
        )
        metrics["GradNorm/action_head_offense"] = self._grad_norm(
            getattr(policy, "action_head_offense", None)
        )
        metrics["GradNorm/action_head_defense"] = self._grad_norm(
            getattr(policy, "action_head_defense", None)
        )
        metrics["GradNorm/value_head_offense"] = self._grad_norm(
            getattr(policy, "value_net_offense", None)
        )
        metrics["GradNorm/value_head_defense"] = self._grad_norm(
            getattr(policy, "value_net_defense", None)
        )
        metrics["GradNorm/token_head_mlp_pi"] = self._grad_norm(
            getattr(policy, "token_head_mlp_pi", None)
        )
        metrics["GradNorm/token_head_mlp_vf"] = self._grad_norm(
            getattr(policy, "token_head_mlp_vf", None)
        )

        for key, value in metrics.items():
            if value is None:
                continue
            try:
                mlflow.log_metric(key, value, step=step)
            except Exception:
                pass

    def _on_step(self) -> bool:
        return True


class AccumulativeMetricsCallback(BaseCallback):
    """Log episode metrics separated by training team (Offense/Defense).
    
    For mixed training environments, automatically groups episodes by training_team
    and logs separate metrics for each team.
    """

    def __init__(self, log_freq_rollouts: int = 1, verbose=0):
        super().__init__(verbose)
        self.log_freq_rollouts = max(1, int(log_freq_rollouts))
        self._rollouts = 0
        self._seen_episode_ids = set()

    def _extract_episode_metrics(self, ep: dict) -> dict:
        """Extract only the metrics we need from an episode dict."""
        return {
            "r": float(ep.get("r", 0.0)),
            "l": float(ep.get("l", 0.0)),
            "shot_dunk": float(ep.get("shot_dunk", 0.0)),
            "shot_2pt": float(ep.get("shot_2pt", 0.0)),
            "shot_3pt": float(ep.get("shot_3pt", 0.0)),
            "assisted_dunk": float(ep.get("assisted_dunk", 0.0)),
            "assisted_2pt": float(ep.get("assisted_2pt", 0.0)),
            "assisted_3pt": float(ep.get("assisted_3pt", 0.0)),
            "potential_assisted_dunk": float(ep.get("potential_assisted_dunk", 0.0)),
            "potential_assisted_2pt": float(ep.get("potential_assisted_2pt", 0.0)),
            "potential_assisted_3pt": float(ep.get("potential_assisted_3pt", 0.0)),
            "potential_assists": float(ep.get("potential_assists", 0.0)),
            "passes": float(ep.get("passes", 0.0)),
            "turnover": float(ep.get("turnover", 0.0)),
            "turnover_pass_oob": float(ep.get("turnover_pass_oob", 0.0)),
            "turnover_intercepted": float(ep.get("turnover_intercepted", 0.0)),
            "turnover_pressure": float(ep.get("turnover_pressure", 0.0)),
            "turnover_offensive_lane": float(ep.get("turnover_offensive_lane", 0.0)),
            "turnover_move_oob": float(ep.get("turnover_move_oob", 0.0)),
            "turnover_other": float(ep.get("turnover_other", 0.0)),
            "defensive_lane_violation": float(ep.get("defensive_lane_violation", 0.0)),
            "move_rejected_occupied": float(ep.get("move_rejected_occupied", 0.0)),
            "made_dunk": float(ep.get("made_dunk", 0.0)),
            "made_2pt": float(ep.get("made_2pt", 0.0)),
            "made_3pt": float(ep.get("made_3pt", 0.0)),
            "attempts": float(ep.get("attempts", 0.0)),
            "expected_points": float(ep.get("expected_points", 0.0)),
            "pressure_exposure": float(ep.get("pressure_exposure", 0.0)),
            "phi_beta": float(ep.get("phi_beta", 0.0)),
            "phi_prev": float(ep.get("phi_prev", 0.0)),
            "phi_next": float(ep.get("phi_next", 0.0)),
            "pointer_pass_attempts": float(ep.get("pointer_pass_attempts", 0.0)),
            "pointer_pass_intent_match_rate": float(
                ep.get("pointer_pass_intent_match_rate", 0.0)
            ),
            "pointer_pass_target_entropy": float(
                ep.get("pointer_pass_target_entropy", 0.0)
            ),
            "pointer_pass_target_entropy_norm": float(
                ep.get("pointer_pass_target_entropy_norm", 0.0)
            ),
            "pointer_pass_target_kl_uniform": float(
                ep.get("pointer_pass_target_kl_uniform", 0.0)
            ),
        }

    def reset(self):
        """Reset the episode cache to start fresh. Call this between training phases."""
        self.episode_cache = []
        self._seen_episode_ids = set()

    def _on_rollout_end(self) -> None:
        """Aggregate and log metrics by team."""
        self._rollouts += 1
        if self._rollouts % self.log_freq_rollouts != 0:
            return
        global_step = self.model.num_timesteps
        
        # Log entropy coefficient (model-level, not per-episode)
        try:
            ent_coef = getattr(self.model, "ent_coef", None)
            if ent_coef is not None:
                current_ent_coef = float(ent_coef)
                mlflow.log_metric("Entropy Coef", current_ent_coef, step=global_step)
        except Exception:
            pass
        
        if not self.model.ep_info_buffer:
            return

        # Group episodes by training_team
        episodes_by_team = {}
        for ep in self.model.ep_info_buffer:
            ep_id = id(ep)
            if ep_id not in self._seen_episode_ids:
                self._seen_episode_ids.add(ep_id)
                team_name = ep.get("training_team", "Unknown")
                if team_name not in episodes_by_team:
                    episodes_by_team[team_name] = []
                episodes_by_team[team_name].append(ep)
        if len(self._seen_episode_ids) > 200_000:
            self._seen_episode_ids.clear()
        
        # Log metrics for each team
        global_step = self.model.num_timesteps
        for team_name, episodes in episodes_by_team.items():
            self._log_team_metrics(team_name, episodes, global_step)

    def _on_step(self) -> bool:
        return True
    
    def _log_team_metrics(self, team_name: str, episodes: list, global_step: int):
        """Log metrics for a specific team."""
        if not episodes:
            return
        
        # Format team name for display (e.g., "OFFENSE" -> "Offense")
        team_label = team_name.capitalize() if isinstance(team_name, str) else team_name
        
        # Basic metrics
        ep_rewards = [ep.get("r", 0.0) for ep in episodes]
        ep_lengths = [ep.get("l", 0.0) for ep in episodes]
        
        ep_rew_mean = float(np.mean(ep_rewards)) if ep_rewards else 0.0
        ep_len_mean = float(np.mean(ep_lengths)) if ep_lengths else 0.0
        
        mlflow.log_metric(f"{team_label} Mean Episode Reward", ep_rew_mean, step=global_step)
        mlflow.log_metric(f"{team_label} Mean Episode Length", ep_len_mean, step=global_step)
        
        # Helper function to compute mean of a metric
        def mean_key(key: str, default: float = 0.0):
            vals = [float(ep.get(key, default)) for ep in episodes]
            return float(np.mean(vals)) if vals else default
        
        # Shot metrics
        mlflow.log_metric(f"{team_label} ShotPct Dunk", mean_key("shot_dunk"), step=global_step)
        mlflow.log_metric(f"{team_label} ShotPct 2PT", mean_key("shot_2pt"), step=global_step)
        mlflow.log_metric(f"{team_label} ShotPct 3PT", mean_key("shot_3pt"), step=global_step)
        
        # Assist metrics
        mlflow.log_metric(f"{team_label} Assist ShotPct Dunk", mean_key("assisted_dunk"), step=global_step)
        mlflow.log_metric(f"{team_label} Assist ShotPct 2PT", mean_key("assisted_2pt"), step=global_step)
        mlflow.log_metric(f"{team_label} Assist ShotPct 3PT", mean_key("assisted_3pt"), step=global_step)
        mlflow.log_metric(f"{team_label} Potential Assist Dunk", mean_key("potential_assisted_dunk"), step=global_step)
        mlflow.log_metric(f"{team_label} Potential Assist 2PT", mean_key("potential_assisted_2pt"), step=global_step)
        mlflow.log_metric(f"{team_label} Potential Assist 3PT", mean_key("potential_assisted_3pt"), step=global_step)
        mlflow.log_metric(f"{team_label} Potential Assists", mean_key("potential_assists"), step=global_step)
        
        # Other metrics
        mlflow.log_metric(f"{team_label} Passes / Episode", mean_key("passes"), step=global_step)
        mlflow.log_metric(f"{team_label} Pressure Exposure", mean_key("pressure_exposure"), step=global_step)
        mlflow.log_metric(f"{team_label} TurnoverPct", mean_key("turnover"), step=global_step)
        mlflow.log_metric(f"{team_label} Turnover Pass OOB", mean_key("turnover_pass_oob"), step=global_step)
        mlflow.log_metric(f"{team_label} Turnover Intercepted", mean_key("turnover_intercepted"), step=global_step)
        mlflow.log_metric(f"{team_label} Turnover Pressure", mean_key("turnover_pressure"), step=global_step)
        mlflow.log_metric(f"{team_label} 3-Second Violation", mean_key("turnover_offensive_lane"), step=global_step)
        mlflow.log_metric(f"{team_label} Turnover Move OOB", mean_key("turnover_move_oob"), step=global_step)
        mlflow.log_metric(f"{team_label} Turnover Other", mean_key("turnover_other"), step=global_step)
        mlflow.log_metric(f"{team_label} Illegal Defense Violation", mean_key("defensive_lane_violation"), step=global_step)
        mlflow.log_metric(f"{team_label} Rejected Move Occupied", mean_key("move_rejected_occupied"), step=global_step)
        legal_key = "legal_actions_offense" if str(team_name).lower() == "offense" else "legal_actions_defense"
        mlflow.log_metric(f"{team_label} Legal Actions", mean_key(legal_key), step=global_step)
        
        # PPP (Points Per Possession)
        ppp_values = []
        expected_ppp_values = []
        for ep in episodes:
            m2 = float(ep.get("made_2pt", 0.0))
            m3 = float(ep.get("made_3pt", 0.0))
            md = float(ep.get("made_dunk", 0.0))
            # Defensive 3-second violation awards offense 1 point.
            lane_violation_pts = float(ep.get("defensive_lane_violation", 0.0))
            att = float(ep.get("attempts", 0.0))
            tov = float(ep.get("turnover", 0.0))
            points = (2.0 * m2) + (3.0 * m3) + (2.0 * md) + lane_violation_pts
            possessions = max(1.0, att + tov + lane_violation_pts)
            ppp_values.append(points / possessions)
            expected_points = float(ep.get("expected_points", 0.0)) + lane_violation_pts
            expected_ppp_values.append(expected_points / possessions)
        ppp_mean = float(np.mean(ppp_values)) if ppp_values else 0.0
        expected_ppp_mean = (
            float(np.mean(expected_ppp_values)) if expected_ppp_values else 0.0
        )
        mlflow.log_metric(f"{team_label} PPP", ppp_mean, step=global_step)
        mlflow.log_metric(
            f"{team_label} Expected PPP", expected_ppp_mean, step=global_step
        )
        mlflow.log_metric(
            f"{team_label} Expected Points / Episode",
            mean_key("expected_points"),
            step=global_step,
        )
        
        # Phi diagnostics
        try:
            mlflow.log_metric(f"{team_label} Phi Beta", mean_key("phi_beta"), step=global_step)
            mlflow.log_metric(f"{team_label} Phi Prev", mean_key("phi_prev"), step=global_step)
            mlflow.log_metric(f"{team_label} Phi Next", mean_key("phi_next"), step=global_step)
        except Exception:
            pass
        # Pointer-targeted pass diagnostics (0.0 for directional runs)
        try:
            mlflow.log_metric(
                f"{team_label} Pass Attempts Pointer",
                mean_key("pointer_pass_attempts"),
                step=global_step,
            )
            mlflow.log_metric(
                f"{team_label} Pass IntentVsOutcomeMatch",
                mean_key("pointer_pass_intent_match_rate"),
                step=global_step,
            )
            mlflow.log_metric(
                f"{team_label} Pass Target Entropy",
                mean_key("pointer_pass_target_entropy"),
                step=global_step,
            )
            mlflow.log_metric(
                f"{team_label} Pass Target Entropy Norm",
                mean_key("pointer_pass_target_entropy_norm"),
                step=global_step,
            )
            mlflow.log_metric(
                f"{team_label} Pass Target KL Uniform",
                mean_key("pointer_pass_target_kl_uniform"),
                step=global_step,
            )
        except Exception:
            pass



class IntentDiversityCallback(BaseCallback):
    """DIAYN-style diversity shaping for latent intent episodes.

    This callback is fully gated by `enabled`; when disabled it is a no-op.
    """

    def __init__(
        self,
        enabled: bool = False,
        num_intents: int = 8,
        beta_target: float = 0.05,
        warmup_steps: int = 1_000_000,
        ramp_steps: int = 1_000_000,
        bonus_clip: float = 2.0,
        disc_lr: float = 3e-4,
        disc_batch_size: int = 256,
        disc_updates_per_rollout: int = 2,
        disc_hidden_dim: int = 128,
        disc_dropout: float = 0.1,
        max_obs_dim: int = 256,
        max_action_dim: int = 16,
    ) -> None:
        super().__init__()
        self.enabled = bool(enabled)
        self.num_intents = max(1, int(num_intents))
        self.beta_target = float(max(0.0, beta_target))
        self.warmup_steps = int(max(0, warmup_steps))
        self.ramp_steps = int(max(1, ramp_steps))
        self.bonus_clip = float(max(0.0, bonus_clip))
        self.disc_lr = float(max(1e-8, disc_lr))
        self.disc_batch_size = int(max(1, disc_batch_size))
        self.disc_updates_per_rollout = int(max(1, disc_updates_per_rollout))
        self.disc_hidden_dim = int(max(16, disc_hidden_dim))
        self.disc_dropout = float(max(0.0, disc_dropout))
        self.max_obs_dim = int(max(8, max_obs_dim))
        self.max_action_dim = int(max(1, max_action_dim))

        self._rollout_step_idx = 0
        self._episodes = IntentEpisodeBuffer()
        self._bonus_stats = RunningMeanStd()
        self._disc: Optional[IntentDiscriminator] = None
        self._disc_opt: Optional[torch.optim.Optimizer] = None
        self._device = torch.device("cpu")
        self._warned_recompute = False
        self._last_disc_loss = 0.0
        self._last_disc_acc = 0.0
        self._last_disc_auc: Optional[float] = None
        self._defense_unknown_baseline_ppp: Optional[float] = None

    def _on_training_start(self) -> None:
        try:
            policy = getattr(self.model, "policy", None)
            if policy is not None and hasattr(policy, "device"):
                self._device = torch.device(policy.device)
        except Exception:
            self._device = torch.device("cpu")

    def _on_rollout_start(self) -> None:
        self._rollout_step_idx = 0

    def _beta_current(self) -> float:
        t = int(getattr(self.model, "num_timesteps", 0))
        if t < self.warmup_steps:
            return 0.0
        if self.ramp_steps <= 0:
            return float(self.beta_target)
        progress = min(1.0, max(0.0, (t - self.warmup_steps) / float(self.ramp_steps)))
        return float(self.beta_target * progress)

    @staticmethod
    def _extract_scalar_from_obs(obs_payload, key: str, env_idx: int, default: float = 0.0) -> float:
        try:
            if not isinstance(obs_payload, dict):
                return float(default)
            if key not in obs_payload:
                return float(default)
            arr = np.asarray(obs_payload[key], dtype=np.float32)
            if arr.ndim == 0:
                return float(arr)
            if arr.shape[0] <= int(env_idx):
                return float(default)
            return float(arr[int(env_idx)].reshape(-1)[0])
        except Exception:
            return float(default)

    def _build_transition(
        self, obs_payload, actions_payload, info: dict, env_idx: int
    ) -> IntentTransition:
        obs_feat = flatten_observation_for_env(obs_payload, env_idx)
        act_feat = extract_action_features_for_env(actions_payload, env_idx)
        feat = np.zeros(self.max_obs_dim + self.max_action_dim, dtype=np.float32)
        obs_take = min(self.max_obs_dim, obs_feat.shape[0])
        if obs_take > 0:
            feat[:obs_take] = obs_feat[:obs_take]
        act_take = min(self.max_action_dim, act_feat.shape[0])
        if act_take > 0:
            start = self.max_obs_dim
            feat[start : start + act_take] = act_feat[:act_take]

        role_flag = self._extract_scalar_from_obs(obs_payload, "role_flag", env_idx, default=0.0)
        intent_active = self._extract_scalar_from_obs(
            obs_payload,
            "intent_active",
            env_idx,
            default=float(info.get("intent_active", 0.0)),
        )
        intent_index = self._extract_scalar_from_obs(
            obs_payload,
            "intent_index",
            env_idx,
            default=float(info.get("intent_index", 0.0)),
        )

        return IntentTransition(
            feature=feat,
            buffer_step_idx=int(self._rollout_step_idx),
            env_idx=int(env_idx),
            role_flag=float(role_flag),
            intent_active=bool(intent_active > 0.5),
            intent_index=int(max(0, min(self.num_intents - 1, int(intent_index)))),
        )

    def _on_step(self) -> bool:
        if not self.enabled:
            return True
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        actions = self.locals.get("actions", None)
        obs_payload = self.locals.get("new_obs", None)
        if obs_payload is None:
            obs_payload = self.locals.get("obs", None)

        n_envs = len(infos) if isinstance(infos, (list, tuple)) else 0
        for env_idx in range(n_envs):
            info = infos[env_idx] or {}
            tr = self._build_transition(obs_payload, actions, info, env_idx)
            self._episodes.append(env_idx, tr)
            if env_idx < len(dones) and bool(dones[env_idx]):
                self._episodes.close_episode(env_idx)

        self._rollout_step_idx += 1
        return True

    def _maybe_build_discriminator(self, input_dim: int) -> None:
        if self._disc is not None:
            return
        self._disc = IntentDiscriminator(
            input_dim=input_dim,
            hidden_dim=self.disc_hidden_dim,
            num_intents=self.num_intents,
            dropout=self.disc_dropout,
        ).to(self._device)
        self._disc_opt = torch.optim.Adam(self._disc.parameters(), lr=self.disc_lr)

    def _train_discriminator(self, x_np: np.ndarray, y_np: np.ndarray) -> tuple[float, float]:
        if x_np.shape[0] == 0:
            return 0.0, 0.0
        self._maybe_build_discriminator(x_np.shape[1])
        assert self._disc is not None
        assert self._disc_opt is not None

        x = torch.as_tensor(x_np, dtype=torch.float32, device=self._device)
        y = torch.as_tensor(y_np, dtype=torch.long, device=self._device)

        self._disc.train()
        last_loss = 0.0
        last_acc = 0.0
        n = int(x.shape[0])
        for _ in range(self.disc_updates_per_rollout):
            if n <= self.disc_batch_size:
                idx = torch.arange(n, device=self._device)
            else:
                idx = torch.randint(0, n, (self.disc_batch_size,), device=self._device)
            xb = x[idx]
            yb = y[idx]
            logits = self._disc(xb)
            loss = F.cross_entropy(logits, yb)
            self._disc_opt.zero_grad(set_to_none=True)
            loss.backward()
            self._disc_opt.step()

            with torch.no_grad():
                pred = torch.argmax(logits, dim=-1)
                acc = (pred == yb).float().mean()
                last_loss = float(loss.detach().cpu().item())
                last_acc = float(acc.detach().cpu().item())
        return last_loss, last_acc

    def _compute_episode_bonus(self, x_np: np.ndarray, y_np: np.ndarray) -> np.ndarray:
        if x_np.shape[0] == 0:
            return np.zeros((0,), dtype=np.float32)
        assert self._disc is not None
        x = torch.as_tensor(x_np, dtype=torch.float32, device=self._device)
        y = torch.as_tensor(y_np, dtype=torch.long, device=self._device)
        self._disc.eval()
        with torch.no_grad():
            logits = self._disc(x)
            log_probs = F.log_softmax(logits, dim=-1)
            chosen = log_probs[torch.arange(log_probs.shape[0], device=self._device), y]
            bonus = chosen + float(np.log(float(self.num_intents)))
        return bonus.detach().cpu().numpy().astype(np.float32, copy=False)

    @staticmethod
    def _binary_auc_from_scores(y_true: np.ndarray, y_score: np.ndarray) -> Optional[float]:
        """Compute binary ROC-AUC from labels and scores via rank statistic."""
        y_true_arr = np.asarray(y_true, dtype=np.int64).reshape(-1)
        y_score_arr = np.asarray(y_score, dtype=np.float64).reshape(-1)
        if y_true_arr.size == 0 or y_true_arr.size != y_score_arr.size:
            return None

        pos = (y_true_arr > 0).astype(np.int64)
        n_pos = int(np.sum(pos))
        n_neg = int(pos.size - n_pos)
        if n_pos == 0 or n_neg == 0:
            return None

        order = np.argsort(y_score_arr, kind="mergesort")
        sorted_scores = y_score_arr[order]
        ranks = np.empty_like(sorted_scores, dtype=np.float64)
        i = 0
        n = sorted_scores.size
        while i < n:
            j = i + 1
            while j < n and sorted_scores[j] == sorted_scores[i]:
                j += 1
            avg_rank = 0.5 * ((i + 1) + j)  # 1-indexed average rank
            ranks[i:j] = avg_rank
            i = j

        rank_by_original_index = np.empty_like(ranks)
        rank_by_original_index[order] = ranks
        sum_pos_ranks = float(np.sum(rank_by_original_index[pos == 1]))
        auc = (sum_pos_ranks - (n_pos * (n_pos + 1) / 2.0)) / float(n_pos * n_neg)
        return float(np.clip(auc, 0.0, 1.0))

    @classmethod
    def _multiclass_auc_ovr_macro(
        cls, logits_np: np.ndarray, y_np: np.ndarray, num_classes: int
    ) -> Optional[float]:
        """Compute macro OVR ROC-AUC for multiclass logits."""
        logits = np.asarray(logits_np, dtype=np.float64)
        y_true = np.asarray(y_np, dtype=np.int64).reshape(-1)
        if logits.ndim != 2 or logits.shape[0] == 0 or logits.shape[0] != y_true.size:
            return None
        if num_classes <= 1:
            return None

        logits = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        denom = np.clip(np.sum(exp_logits, axis=1, keepdims=True), 1e-12, None)
        probs = exp_logits / denom

        aucs: list[float] = []
        k = min(int(num_classes), int(probs.shape[1]))
        for class_idx in range(k):
            y_bin = (y_true == class_idx).astype(np.int64)
            auc = cls._binary_auc_from_scores(y_bin, probs[:, class_idx])
            if auc is not None:
                aucs.append(float(auc))
        if not aucs:
            return None
        return float(np.mean(aucs))

    def _compute_disc_auc(self, x_np: np.ndarray, y_np: np.ndarray) -> Optional[float]:
        if x_np.shape[0] == 0 or self._disc is None:
            return None
        x = torch.as_tensor(x_np, dtype=torch.float32, device=self._device)
        self._disc.eval()
        with torch.no_grad():
            logits = self._disc(x)
        logits_np = logits.detach().cpu().numpy()
        return self._multiclass_auc_ovr_macro(
            logits_np,
            y_np,
            num_classes=self.num_intents,
        )

    def _recompute_returns_and_advantage(self) -> None:
        try:
            rollout_buffer = self.model.rollout_buffer
            policy = self.model.policy
            with torch.no_grad():
                obs_tensor, _ = policy.obs_to_tensor(self.model._last_obs)
                last_values = policy.predict_values(obs_tensor)
            rollout_buffer.compute_returns_and_advantage(
                last_values=last_values,
                dones=self.model._last_episode_starts,
            )
        except Exception as e:
            if not self._warned_recompute:
                print(
                    f"[IntentDiversityCallback] Warning: could not recompute returns/advantages: {e}"
                )
                self._warned_recompute = True

    @staticmethod
    def _usage_entropy(y_np: np.ndarray, num_intents: int) -> tuple[float, float]:
        if y_np.size == 0:
            return 0.0, 0.0
        counts = np.bincount(y_np, minlength=max(1, num_intents)).astype(np.float64)
        probs = counts / max(1.0, float(np.sum(counts)))
        probs = np.clip(probs, 1e-12, 1.0)
        entropy = float(-np.sum(probs * np.log(probs)))
        min_prob = float(np.min(probs))
        return entropy, min_prob

    @staticmethod
    def _episode_ppp(ep: dict) -> float:
        m2 = float(ep.get("made_2pt", 0.0))
        m3 = float(ep.get("made_3pt", 0.0))
        md = float(ep.get("made_dunk", 0.0))
        lane_violation_pts = float(ep.get("defensive_lane_violation", 0.0))
        att = float(ep.get("attempts", 0.0))
        tov = float(ep.get("turnover", 0.0))
        points = (2.0 * m2) + (3.0 * m3) + (2.0 * md) + lane_violation_pts
        possessions = max(1.0, att + tov + lane_violation_pts)
        return float(points / possessions)

    @staticmethod
    def _episode_shot_three_share(ep: dict) -> float:
        s2 = float(ep.get("shot_2pt", 0.0))
        s3 = float(ep.get("shot_3pt", 0.0))
        sd = float(ep.get("shot_dunk", 0.0))
        total = s2 + s3 + sd
        if total <= 0.0:
            return 0.0
        return float(s3 / total)

    def _log_intent_behavior_metrics(self, global_step: int) -> None:
        episodes = list(getattr(self.model, "ep_info_buffer", []) or [])
        if not episodes:
            return

        grouped: dict[int, dict[str, list[float]]] = {}
        for ep in episodes:
            try:
                team = str(ep.get("training_team", "")).strip().lower()
                if team != "offense":
                    continue
                if float(ep.get("intent_active", 0.0)) <= 0.5:
                    continue
                z = int(float(ep.get("intent_index", 0.0)))
            except Exception:
                continue
            z = max(0, min(self.num_intents - 1, z))
            slot = grouped.setdefault(z, {"ppp": [], "passes": [], "shot_three_share": []})
            slot["ppp"].append(self._episode_ppp(ep))
            slot["passes"].append(float(ep.get("passes", 0.0)))
            slot["shot_three_share"].append(self._episode_shot_three_share(ep))

        if not grouped:
            return

        for z, vals in grouped.items():
            if vals["ppp"]:
                mlflow.log_metric(
                    f"intent/ppp_by_intent/{z}",
                    float(np.mean(vals["ppp"])),
                    step=global_step,
                )
            if vals["passes"]:
                mlflow.log_metric(
                    f"intent/pass_rate_by_intent/{z}",
                    float(np.mean(vals["passes"])),
                    step=global_step,
                )
            if vals["shot_three_share"]:
                mlflow.log_metric(
                    f"intent/shot_dist_by_intent/{z}",
                    float(np.mean(vals["shot_three_share"])),
                    step=global_step,
                )

    def _log_defense_unknown_intent_proxy(self, global_step: int) -> None:
        episodes = list(getattr(self.model, "ep_info_buffer", []) or [])
        if not episodes:
            return
        values: list[float] = []
        for ep in episodes:
            try:
                team = str(ep.get("training_team", "")).strip().lower()
                if team != "defense":
                    continue
                visible = float(ep.get("intent_visible_training_obs", 0.0))
                if visible > 0.5:
                    continue
                values.append(self._episode_ppp(ep))
            except Exception:
                continue
        if not values:
            return
        current = float(np.mean(values))
        if self._defense_unknown_baseline_ppp is None:
            self._defense_unknown_baseline_ppp = current
        delta = float(current - float(self._defense_unknown_baseline_ppp))
        mlflow.log_metric("intent/defense_unknown_intent_ppp", current, step=global_step)
        mlflow.log_metric(
            "intent/defense_unknown_intent_delta_vs_baseline",
            delta,
            step=global_step,
        )

    def _on_rollout_end(self) -> None:
        if not self.enabled:
            return

        global_step = int(getattr(self.model, "num_timesteps", 0))
        try:
            self._log_intent_behavior_metrics(global_step)
            self._log_defense_unknown_intent_proxy(global_step)
        except Exception:
            pass

        episodes = self._episodes.pop_completed(
            filter_fn=lambda ep: ep.role_is_offense and ep.intent_active and ep.length > 0
        )
        if not episodes:
            return
        beta = self._beta_current()
        if beta <= 0.0:
            return

        x_np, y_np = compute_episode_embeddings(
            episodes,
            max_obs_dim=self.max_obs_dim,
            max_action_dim=self.max_action_dim,
        )
        if x_np.shape[0] == 0:
            return

        self._last_disc_loss, self._last_disc_acc = self._train_discriminator(x_np, y_np)
        self._last_disc_auc = self._compute_disc_auc(x_np, y_np)
        raw_bonus = self._compute_episode_bonus(x_np, y_np)
        if raw_bonus.size == 0:
            return

        self._bonus_stats.update(raw_bonus)
        norm_bonus = (raw_bonus - float(self._bonus_stats.mean)) / float(self._bonus_stats.std)
        clipped_bonus = np.clip(norm_bonus, -self.bonus_clip, self.bonus_clip)

        rb = self.model.rollout_buffer
        for ep_idx, episode in enumerate(episodes):
            if ep_idx >= clipped_bonus.shape[0]:
                break
            per_step = float(beta * clipped_bonus[ep_idx] / max(1, episode.length))
            for t_idx, env_idx in episode.buffer_indices:
                if (
                    0 <= int(t_idx) < rb.rewards.shape[0]
                    and 0 <= int(env_idx) < rb.rewards.shape[1]
                ):
                    rb.rewards[int(t_idx), int(env_idx)] += per_step

        self._recompute_returns_and_advantage()

        try:
            mlflow.log_metric("intent/disc_loss", float(self._last_disc_loss), step=global_step)
            mlflow.log_metric("intent/disc_top1_acc", float(self._last_disc_acc), step=global_step)
            if self._last_disc_auc is not None:
                mlflow.log_metric(
                    "intent/disc_auc_ovr_macro",
                    float(self._last_disc_auc),
                    step=global_step,
                )
            mlflow.log_metric("intent/bonus_raw_mean", float(np.mean(raw_bonus)), step=global_step)
            mlflow.log_metric(
                "intent/bonus_norm_mean", float(np.mean(norm_bonus)), step=global_step
            )
            mlflow.log_metric(
                "intent/bonus_norm_std", float(np.std(norm_bonus)), step=global_step
            )
            mlflow.log_metric("intent/beta_current", float(beta), step=global_step)
            usage_entropy, min_prob = self._usage_entropy(y_np, self.num_intents)
            mlflow.log_metric("intent/usage_entropy", float(usage_entropy), step=global_step)
            mlflow.log_metric("intent/usage_min_prob", float(min_prob), step=global_step)
        except Exception:
            pass


# --- Entropy Schedules ---


class EntropyScheduleCallback(BaseCallback):
    """Linear decay of entropy coefficient across the run."""

    def __init__(
        self,
        start: float,
        end: float,
        total_planned_timesteps: int,
        timestep_offset: int = 0,
    ):
        super().__init__()
        self.start = float(start)
        self.end = float(end)
        self.total = int(max(1, total_planned_timesteps))
        self.timestep_offset = int(timestep_offset)

    def _on_step(self) -> bool:
        try:
            t = int(getattr(self.model, "num_timesteps", 0)) - self.timestep_offset
            progress = min(1.0, max(0.0, t / float(self.total)))
            current = self.end + (self.start - self.end) * (1.0 - progress)
            self.model.ent_coef = float(current)
        except Exception:
            pass
        return True


class EntropyExpScheduleCallback(BaseCallback):
    """Log-linear decay with optional short bump at segment starts."""

    def __init__(
        self,
        start: float,
        end: float,
        total_planned_timesteps: int,
        bump_updates: int = 0,
        bump_multiplier: float = 1.0,
        timestep_offset: int = 0,
    ):
        super().__init__()
        self.start = float(max(1e-12, start))
        self.end = float(max(1e-12, end))
        self.total = int(max(1, total_planned_timesteps))
        self.bump_updates = int(max(0, bump_updates))
        self.bump_multiplier = float(max(1.0, bump_multiplier))
        self.timestep_offset = int(timestep_offset)
        self._bump_updates_remaining = 0

    def start_new_alternation(self):
        self._bump_updates_remaining = self.bump_updates

    def _scheduled_value(self, t: int) -> float:
        progress = min(1.0, max(0.0, t / float(self.total)))
        ratio = self.start / self.end
        current = self.end * (ratio ** (1.0 - progress))
        return float(current)

    def _apply_current(self):
        try:
            t = int(getattr(self.model, "num_timesteps", 0)) - self.timestep_offset
            current = self._scheduled_value(t)
            if self._bump_updates_remaining > 0:
                current = float(current * self.bump_multiplier)
            self.model.ent_coef = float(current)
        except Exception:
            pass

    def _on_training_start(self) -> None:
        self._apply_current()

    def _on_rollout_start(self) -> None:
        self._apply_current()

    def _on_rollout_end(self) -> None:
        if self._bump_updates_remaining > 0:
            self._bump_updates_remaining -= 1
        self._apply_current()

    def _on_step(self) -> bool:
        # Throttled: avoid per-step cross-process RPC
        return True


# --- Potential-Based Reward Shaping (Phi) Schedules ---


class PotentialBetaExpScheduleCallback(BaseCallback):
    """Exponential decay for phi_beta across the planned training timesteps.

    This callback updates each VecEnv worker's underlying env attribute `phi_beta`
    so that potential-based shaping can be annealed toward zero during training.
    Optionally supports a short multiplicative bump at the start of each
    alternation segment (mirroring the entropy scheduler API).
    """

    def __init__(
        self,
        start: float,
        end: float,
        total_planned_timesteps: int,
        bump_updates: int = 0,
        bump_multiplier: float = 1.0,
        timestep_offset: int = 0,
    ):
        super().__init__()
        self.start = float(max(0.0, start))
        self.end = float(max(0.0, end))
        self.total = int(max(1, total_planned_timesteps))
        self.bump_updates = int(max(0, bump_updates))
        self.bump_multiplier = float(max(1.0, bump_multiplier))
        self.timestep_offset = int(timestep_offset)
        self._bump_updates_remaining = 0

    def start_new_alternation(self):
        self._bump_updates_remaining = self.bump_updates

    def _scheduled_value(self, t: int) -> float:
        progress = min(1.0, max(0.0, t / float(self.total)))
        ratio = (
            (self.start / max(1e-12, self.end))
            if self.end > 0
            else (self.start / 1e-12)
        )
        current = self.end * (ratio ** (1.0 - progress))
        return float(max(0.0, current))

    def _apply_current(self):
        try:
            t = int(getattr(self.model, "num_timesteps", 0)) - self.timestep_offset
            current = self._scheduled_value(t)
            if self._bump_updates_remaining > 0:
                current = float(current * self.bump_multiplier)
            # Update all env workers' phi_beta (prefer env_method to hit unwrapped env)
            vecenv = self.model.get_env()
            try:
                if vecenv is not None and hasattr(vecenv, "env_method"):
                    vecenv.env_method("set_phi_beta", float(current))
                elif vecenv is not None and hasattr(vecenv, "set_attr"):
                    vecenv.set_attr("phi_beta", float(current))
            except Exception:
                pass
        except Exception:
            pass

    def _on_training_start(self) -> None:
        self._apply_current()

    def _on_rollout_start(self) -> None:
        self._apply_current()

    def _on_rollout_end(self) -> None:
        if self._bump_updates_remaining > 0:
            self._bump_updates_remaining -= 1
        self._apply_current()

    def _on_step(self) -> bool:
        self._apply_current()
        return True


class PassLogitBiasExpScheduleCallback(BaseCallback):
    """Exponential decay schedule for additive pass-logit bias.

    Updates the current policy via env/policy hooks so pass actions keep
    non-trivial probability mass early in training.
    """

    def __init__(
        self,
        start: float,
        end: float,
        total_planned_timesteps: int,
        log_freq_rollouts: int = 1,
        timestep_offset: int = 0,
    ):
        super().__init__()
        self.start = float(start)
        self.end = float(end)
        self.total = int(max(1, total_planned_timesteps))
        self.log_freq_rollouts = max(1, int(log_freq_rollouts))
        self._rollouts = 0
        self.timestep_offset = int(timestep_offset)

    def _scheduled_value(self, t: int) -> float:
        progress = min(1.0, max(0.0, t / float(self.total)))
        # Handle end==0 by decaying toward a tiny epsilon instead of hard 0,
        # so we can observe the decay and avoid collapsing to zero immediately.
        if self.end <= 0.0:
            eps = 1e-6
            current = (
                float(self.start) * (eps / max(1e-12, float(self.start))) ** progress
            )
        else:
            ratio = float(self.start) / float(self.end)
            current = float(self.end) * (ratio ** (1.0 - progress))
        return float(current)

    def _apply_current(self):
        try:
            t = int(getattr(self.model, "num_timesteps", 0)) - self.timestep_offset
            current = self._scheduled_value(t)
            # Update policy hook if available
            try:
                # Policy may expose set_pass_logit_bias
                if hasattr(self.model.policy, "set_pass_logit_bias"):
                    self.model.policy.set_pass_logit_bias(float(current))
            except Exception:
                pass
        except Exception:
            pass
        return None

    def _log_current(self):
        try:
            global_step = int(getattr(self.model, "num_timesteps", 0))
            t = global_step - self.timestep_offset
            current = self._scheduled_value(t)
            mlflow.log_metric("Pass Logit Bias", float(current), step=global_step)
        except Exception:
            pass

    def _on_training_start(self) -> None:
        self._apply_current()
        self._log_current()

    def _on_rollout_start(self) -> None:
        self._apply_current()

    def _on_rollout_end(self) -> None:
        self._apply_current()
        self._rollouts += 1
        if self._rollouts % self.log_freq_rollouts == 0:
            self._log_current()

    def _on_step(self) -> bool:
        # Throttled: avoid per-step update
        return True


class PassCurriculumExpScheduleCallback(BaseCallback):
    """Schedule passing arc (deg) and OOB turnover probability.

    - arc(t): exponential from arc_start -> arc_end over total timesteps
    - oob_prob(t): exponential from p_start -> p_end over total timesteps
    Applies via VecEnv.env_method to avoid wrapper forwarding warnings.
    Logs both to MLflow.
    """

    def __init__(
        self,
        arc_start: float,
        arc_end: float,
        oob_start: float,
        oob_end: float,
        total_planned_timesteps: int,
        arc_power: float = 2.0,
        oob_power: float = 2.0,
        log_freq_rollouts: int = 1,
        timestep_offset: int = 0,
    ):
        super().__init__()
        self.arc_start = float(arc_start)
        self.arc_end = float(arc_end)
        self.oob_start = float(oob_start)
        self.oob_end = float(oob_end)
        self.total = int(max(1, total_planned_timesteps))
        self.arc_power = float(arc_power)
        self.oob_power = float(oob_power)
        self.log_freq_rollouts = max(1, int(log_freq_rollouts))
        self._rollouts = 0
        self.timestep_offset = int(timestep_offset)

    def _exp_sched(self, start: float, end: float, t: int, power: float) -> float:
        progress = min(1.0, max(0.0, t / float(self.total)))
        # epsilon handling for exact zero targets
        if end == 0.0:
            eps = 1e-6
            return float(start) * (eps / max(1e-12, float(start))) ** progress
        ratio = float(start) / float(end)
        # Apply power to (1-progress) to make decay steeper initially
        # power > 1 gives steeper initial decay, power = 1 is linear progress
        return float(end) * (ratio ** ((1.0 - progress) ** power))

    def _apply_current(self, log_to_mlflow: bool = True):
        try:
            t = int(getattr(self.model, "num_timesteps", 0)) - self.timestep_offset
            arc = self._exp_sched(self.arc_start, self.arc_end, t, self.arc_power)
            oob = self._exp_sched(self.oob_start, self.oob_end, t, self.oob_power)
            vecenv = self.model.get_env()
            try:
                if vecenv is not None and hasattr(vecenv, "env_method"):
                    vecenv.env_method("set_pass_arc_degrees", float(arc))
                    vecenv.env_method("set_pass_oob_turnover_prob", float(oob))
            except Exception:
                pass
            if log_to_mlflow:
                # Log to MLflow
                try:
                    # Use model.num_timesteps (with offset) for the metric step
                    mlflow.log_metric(
                        "Passing Arc Degrees", float(arc), step=t + self.timestep_offset
                    )
                    # Configured curriculum knob (not observed turnover frequency).
                    mlflow.log_metric(
                        "Pass OOB Turnover Prob (Config)",
                        float(oob),
                        step=t + self.timestep_offset,
                    )
                except Exception:
                    pass
        except Exception:
            pass

    def _on_training_start(self) -> None:
        self._apply_current(log_to_mlflow=True)

    def _on_rollout_start(self) -> None:
        self._apply_current(log_to_mlflow=False)

    def _on_rollout_end(self) -> None:
        self._rollouts += 1
        self._apply_current(
            log_to_mlflow=(self._rollouts % self.log_freq_rollouts == 0)
        )

    def _on_step(self) -> bool:
        # Throttled: avoid per-step update/logging
        return True


class IntentRobustnessScheduleCallback(BaseCallback):
    """Schedule intent null/visibility probabilities for robustness curriculum.

    - null_prob(t): linear from null_start -> null_end
    - visible_prob(t): linear from visible_start -> visible_end
    Applies via VecEnv.env_method on wrapped environments.
    """

    def __init__(
        self,
        null_start: float,
        null_end: float,
        visible_start: float,
        visible_end: float,
        total_planned_timesteps: int,
        log_freq_rollouts: int = 1,
        timestep_offset: int = 0,
    ):
        super().__init__()
        self.null_start = float(max(0.0, min(1.0, null_start)))
        self.null_end = float(max(0.0, min(1.0, null_end)))
        self.visible_start = float(max(0.0, min(1.0, visible_start)))
        self.visible_end = float(max(0.0, min(1.0, visible_end)))
        self.total = int(max(1, total_planned_timesteps))
        self.log_freq_rollouts = max(1, int(log_freq_rollouts))
        self.timestep_offset = int(timestep_offset)
        self._rollouts = 0

    def _linear(self, start: float, end: float, t: int) -> float:
        progress = min(1.0, max(0.0, t / float(self.total)))
        return float(start + (end - start) * progress)

    def _apply_current(self, log_to_mlflow: bool = True) -> None:
        try:
            t = int(getattr(self.model, "num_timesteps", 0)) - self.timestep_offset
            null_prob = self._linear(self.null_start, self.null_end, t)
            visible_prob = self._linear(self.visible_start, self.visible_end, t)
            vecenv = self.model.get_env()
            try:
                if vecenv is not None and hasattr(vecenv, "env_method"):
                    vecenv.env_method("set_intent_null_prob", float(null_prob))
                    vecenv.env_method(
                        "set_intent_visible_to_defense_prob", float(visible_prob)
                    )
            except Exception:
                pass
            if log_to_mlflow:
                step = int(t + self.timestep_offset)
                try:
                    mlflow.log_metric("intent/null_prob_config", float(null_prob), step=step)
                    mlflow.log_metric(
                        "intent/visible_to_defense_prob_config",
                        float(visible_prob),
                        step=step,
                    )
                except Exception:
                    pass
        except Exception:
            pass

    def _on_training_start(self) -> None:
        self._apply_current(log_to_mlflow=True)

    def _on_rollout_start(self) -> None:
        self._apply_current(log_to_mlflow=False)

    def _on_rollout_end(self) -> None:
        self._rollouts += 1
        self._apply_current(
            log_to_mlflow=(self._rollouts % self.log_freq_rollouts == 0)
        )

    def _on_step(self) -> bool:
        return True


# --- Episode Sampling CSV Logger ---


class EpisodeSampleLogger(BaseCallback):
    """Sample a small fraction of ended episodes during rollout and log to CSV."""

    def __init__(self, team_name: str, alternation_id: int, sample_prob: float = 1e-4) -> None:
        super().__init__()
        self.team_name = team_name
        self._team_key = str(team_name).strip().lower()
        self.alternation_id = int(alternation_id)
        self.sample_prob = float(sample_prob)
        self.update_index = 0
        self._records: list[dict] = []

    def _on_rollout_start(self) -> None:
        self._records = []

    def _on_step(self) -> bool:
        try:
            infos = self.locals.get("infos", [])
            dones = self.locals.get("dones", [])
            n = len(infos) if isinstance(infos, (list, tuple)) else 0
            for env_idx in range(n):
                if env_idx < len(dones) and dones[env_idx]:
                    info = infos[env_idx] or {}
                    info_team = str(info.get("training_team", "")).strip().lower()
                    if info_team and info_team != self._team_key:
                        continue
                    if random.random() < self.sample_prob:
                        shot_type = (
                            "dunk"
                            if info.get("shot_dunk", 0.0)
                            else (
                                "3pt"
                                if info.get("shot_3pt", 0.0)
                                else ("2pt" if info.get("shot_2pt", 0.0) else "none")
                            )
                        )
                        made = (
                            1.0
                            if (
                                info.get("made_dunk", 0.0)
                                or info.get("made_2pt", 0.0)
                                or info.get("made_3pt", 0.0)
                            )
                            else 0.0
                        )
                        turnover = float(info.get("turnover", 0.0))
                        turnover_pass_oob = float(info.get("turnover_pass_oob", 0.0))
                        turnover_intercepted = float(
                            info.get("turnover_intercepted", 0.0)
                        )
                        turnover_pressure = float(info.get("turnover_pressure", 0.0))
                        turnover_move_oob = float(info.get("turnover_move_oob", 0.0))
                        turnover_other = float(info.get("turnover_other", 0.0))
                        row = {
                            "team": self.team_name,
                            "alternation": self.alternation_id,
                            "update": self.update_index,
                            "env_index": env_idx,
                            "shot_clock": info.get("shot_clock", None),
                            "shot_type": shot_type,
                            "made": made,
                            "turnover": turnover,
                            "turnover_pass_oob": turnover_pass_oob,
                            "turnover_intercepted": turnover_intercepted,
                            "turnover_pressure": turnover_pressure,
                            "turnover_move_oob": turnover_move_oob,
                            "turnover_other": turnover_other,
                            "passes": float(info.get("passes", 0.0)),
                            "potential_assisted_dunk": float(info.get("potential_assisted_dunk", 0.0)),
                            "potential_assisted_2pt": float(info.get("potential_assisted_2pt", 0.0)),
                            "potential_assisted_3pt": float(info.get("potential_assisted_3pt", 0.0)),
                            "potential_assists": float(info.get("potential_assists", 0.0)),
                            "expected_points": float(info.get("expected_points", 0.0)),
                            "pressure_exposure": float(info.get("pressure_exposure", 0.0)),
                            "assisted_dunk": float(info.get("assisted_dunk", 0.0)),
                            "assisted_2pt": float(info.get("assisted_2pt", 0.0)),
                            "assisted_3pt": float(info.get("assisted_3pt", 0.0)),
                            # pressure-adjusted FG% (from env shot probability after pressure)
                            "shooter_fg_pct": float(info.get("shooter_fg_pct", -1.0)),
                            "gt_is_three": float(info.get("gt_is_three", 0.0)),
                            "gt_is_dunk": float(info.get("gt_is_dunk", 0.0)),
                            "gt_points": float(info.get("gt_points", 0.0)),
                            "gt_shooter_off": float(info.get("gt_shooter_off", 0.0)),
                            "gt_shooter_q": float(info.get("gt_shooter_q", 0.0)),
                            "gt_shooter_r": float(info.get("gt_shooter_r", 0.0)),
                            "gt_distance": float(info.get("gt_distance", 0.0)),
                            "basket_q": float(info.get("basket_q", 0.0)),
                            "basket_r": float(info.get("basket_r", 0.0)),
                            # phi diagnostics if present
                            "phi_beta": float(info.get("phi_beta", -1.0)),
                            "phi_prev": float(info.get("phi_prev", -1.0)),
                            "phi_next": float(info.get("phi_next", -1.0)),
                        }
                        self._records.append(row)
        except Exception:
            pass
        return True

    def _on_rollout_end(self) -> None:
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                csv_path = os.path.join(tmpdir, "episodes.csv")
                fieldnames = [
                    "team",
                    "alternation",
                    "update",
                    "env_index",
                    "shot_clock",
                    "shot_type",
                    "made",
                    "turnover",
                    "turnover_pass_oob",
                    "turnover_intercepted",
                    "turnover_pressure",
                    "turnover_move_oob",
                    "turnover_other",
                    "passes",
                    "potential_assisted_dunk",
                    "potential_assisted_2pt",
                    "potential_assisted_3pt",
                    "potential_assists",
                    "expected_points",
                    "pressure_exposure",
                    "assisted_dunk",
                    "assisted_2pt",
                    "assisted_3pt",
                    "shooter_fg_pct",
                    # minimal audit
                    "gt_is_three",
                    "gt_is_dunk",
                    "gt_points",
                    "gt_shooter_off",
                    "gt_shooter_q",
                    "gt_shooter_r",
                    "gt_distance",
                    "basket_q",
                    "basket_r",
                    # phi
                    "phi_beta",
                    "phi_prev",
                    "phi_next",
                ]
                with open(csv_path, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    if self._records:
                        for r in self._records:
                            writer.writerow(r)
                artifact_path = f"shot_logs/alternation_{self.alternation_id}/{self.team_name}/update_{self.update_index}"
                mlflow.log_artifact(csv_path, artifact_path=artifact_path)
        except Exception:
            pass
        finally:
            self.update_index += 1
            self._records = []
