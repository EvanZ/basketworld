import time
import os
import csv
import tempfile
import random
import numpy as np
import mlflow
from stable_baselines3.common.callbacks import BaseCallback


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
                passes_avg = mean_key("passes")
                turnover_pct = mean_key("turnover")

                def mean_ppp(default: float = 0.0):
                    numer = []
                    for ep in self.model.ep_info_buffer:
                        m2 = float(ep.get("made_2pt", 0.0))
                        m3 = float(ep.get("made_3pt", 0.0))
                        md = float(ep.get("made_dunk", 0.0))
                        att = float(ep.get("attempts", 0.0))
                        tov = float(ep.get("turnover", 0.0))
                        n = (2.0 * m2) + (3.0 * m3) + (2.0 * md)
                        d = max(1.0, att + tov)
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
                    f"{self.team_name} Passes / Episode", passes_avg, step=global_step
                )
                mlflow.log_metric(
                    f"{self.team_name} TurnoverPct", turnover_pct, step=global_step
                )
                mlflow.log_metric(f"{self.team_name} PPP", ppp_avg, step=global_step)
                # (Reverted) no ground-truth auditing metrics
        return True


# --- Entropy Schedules ---


class EntropyScheduleCallback(BaseCallback):
    """Linear decay of entropy coefficient across the run."""

    def __init__(self, start: float, end: float, total_planned_timesteps: int):
        super().__init__()
        self.start = float(start)
        self.end = float(end)
        self.total = int(max(1, total_planned_timesteps))

    def _on_step(self) -> bool:
        try:
            t = int(getattr(self.model, "num_timesteps", 0))
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
    ):
        super().__init__()
        self.start = float(max(1e-12, start))
        self.end = float(max(1e-12, end))
        self.total = int(max(1, total_planned_timesteps))
        self.bump_updates = int(max(0, bump_updates))
        self.bump_multiplier = float(max(1.0, bump_multiplier))
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
            t = int(getattr(self.model, "num_timesteps", 0))
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
        self._apply_current()
        return True


# --- Episode Sampling CSV Logger ---


class EpisodeSampleLogger(BaseCallback):
    """Sample a small fraction of ended episodes during rollout and log to CSV."""

    def __init__(self, team_name: str, alternation_id: int, sample_prob: float = 1e-4):
        super().__init__()
        self.team_name = team_name
        self.alternation_id = int(alternation_id)
        self.sample_prob = float(sample_prob)
        self.update_index = 0
        self._records = []

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
                        row = {
                            "team": self.team_name,
                            "alternation": self.alternation_id,
                            "update": self.update_index,
                            "env_index": env_idx,
                            "shot_clock": info.get("shot_clock", None),
                            "shot_type": shot_type,
                            "made": made,
                            "turnover": turnover,
                            "passes": float(info.get("passes", 0.0)),
                            "assisted_dunk": float(info.get("assisted_dunk", 0.0)),
                            "assisted_2pt": float(info.get("assisted_2pt", 0.0)),
                            "assisted_3pt": float(info.get("assisted_3pt", 0.0)),
                            "gt_is_three": float(info.get("gt_is_three", 0.0)),
                            "gt_is_dunk": float(info.get("gt_is_dunk", 0.0)),
                            "gt_points": float(info.get("gt_points", 0.0)),
                            "gt_shooter_off": float(info.get("gt_shooter_off", 0.0)),
                            "gt_shooter_q": float(info.get("gt_shooter_q", 0.0)),
                            "gt_shooter_r": float(info.get("gt_shooter_r", 0.0)),
                            "gt_distance": float(info.get("gt_distance", 0.0)),
                            "basket_q": float(info.get("basket_q", 0.0)),
                            "basket_r": float(info.get("basket_r", 0.0)),
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
                    "passes",
                    "assisted_dunk",
                    "assisted_2pt",
                    "assisted_3pt",
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
