"""Compact block, rim-contest, and layup metrics with matched completed populations."""

from typing import Any, NamedTuple

import numpy as np

from basketworld_jax.shot_defense_metrics import SHOT_DEFENSE_COUNTERS


TRAIN_BLOCK_METRICS = frozenset(
    f"{side}_{name}"
    for side in ("learner", "opponent")
    for name in ("block_pct", "blocks_per_defensive_possession", "blocks_per_episode")
)
DEPLOY_BLOCK_METRICS = frozenset(
    ("block_pct", "blocks_per_possession", "blocks_per_episode")
)
TRAIN_RIM_CONTEST_METRICS = frozenset(
    f"{side}_{name}" for side in ("learner", "opponent")
    for name in ("rim_contest_mean_reduction_pp", "rim_contested_shots_per_episode", "rim_contested_shot_pct")
)
DEPLOY_RIM_CONTEST_METRICS = frozenset(
    ("rim_contest_mean_reduction_pp", "rim_contested_shots_per_episode", "rim_contested_shot_pct")
)
TRAIN_LAYUP_METRICS = frozenset(
    f"{side}_{name}" for side in ("learner", "opponent")
    for name in ("layup_attempts_per_episode", "layup_makes_per_episode", "layup_fg_pct")
)
DEPLOY_LAYUP_METRICS = frozenset(
    ("layup_attempts_per_episode", "layup_makes_per_episode", "layup_fg_pct")
)
LEGACY_BLOCK_METRICS = frozenset(
    ("blocks", "learner_blocks", "opponent_blocks", "blocks_per_completed_possession")
)


def filter_block_logging(metrics, *, prefix):
    """Only approved compact block, rim-contest, and layup series, in all profiles."""
    allowed = (TRAIN_BLOCK_METRICS | TRAIN_RIM_CONTEST_METRICS | TRAIN_LAYUP_METRICS if prefix == "jax/train" else
               DEPLOY_BLOCK_METRICS | DEPLOY_RIM_CONTEST_METRICS | DEPLOY_LAYUP_METRICS if prefix == "jax/eval_deploy" else frozenset())
    block_names = (TRAIN_BLOCK_METRICS | DEPLOY_BLOCK_METRICS | LEGACY_BLOCK_METRICS
                   | TRAIN_RIM_CONTEST_METRICS | DEPLOY_RIM_CONTEST_METRICS
                   | TRAIN_LAYUP_METRICS | DEPLOY_LAYUP_METRICS)
    return {key: value for key, value in metrics.items()
            if key in allowed or (key not in block_names and "shot_defense_" not in key)}


class BlockMetricState(NamedTuple):
    possession_blocks: Any
    possession_finishes: Any
    game_blocks: Any
    possession_known: Any
    game_known: Any
    possession_contest_drop: Any
    possession_contest_attempts: Any
    game_contest_attempts: Any
    game_finishes: Any
    game_layup_attempts: Any
    game_layup_makes: Any
    # Columns 0-8 per defending side: possession blocks, finishes faced, defensive
    # possessions, completed-game blocks, completed games, contest probability
    # drop, contested finish attempts, completed-game contested finishes,
    # completed-game finishes faced. Columns 9-10 are completed-game layup
    # attempts and makes per SHOOTING side; column 4 is shared game count.
    totals: Any


def empty_block_metric_state(batch_size, xp=np):
    return BlockMetricState(
        xp.zeros((batch_size, 2), xp.float32),
        xp.zeros((batch_size, 2), xp.float32),
        xp.zeros((batch_size, 2), xp.float32),
        xp.zeros((batch_size,), dtype=bool),
        xp.zeros((batch_size,), dtype=bool),
        xp.zeros((batch_size, 2), xp.float32),
        xp.zeros((batch_size, 2), xp.float32),
        xp.zeros((batch_size, 2), xp.float32),
        xp.zeros((batch_size, 2), xp.float32),
        xp.zeros((batch_size, 2), xp.float32),
        xp.zeros((batch_size, 2), xp.float32),
        xp.zeros((2, 11), xp.float32),
    )


def advance_block_metrics(state, diagnostics, *, active, episode_start,
                          possession_ended, done, truncated, training_role, xp=np):
    """Vectorized over lanes; identical accounting in host training and JAX deploy."""
    active = xp.asarray(active, dtype=bool)
    start = active & xp.asarray(episode_start, dtype=bool)
    end = active & xp.asarray(done, dtype=bool)
    truncated = xp.asarray(truncated, dtype=bool)
    possession_end = active & xp.asarray(possession_ended, dtype=bool)
    blocked = diagnostics[..., ::-1, SHOT_DEFENSE_COUNTERS.index("blocked")]
    finishes = (diagnostics[..., ::-1, SHOT_DEFENSE_COUNTERS.index("layup_attempts")]
                + diagnostics[..., ::-1, SHOT_DEFENSE_COUNTERS.index("dunk_attempts")])
    blocked = xp.where(active[:, None], blocked, 0.0)
    finishes = xp.where(active[:, None], finishes, 0.0)
    pb = xp.where(start[:, None], 0.0, state.possession_blocks) + blocked
    pf = xp.where(start[:, None], 0.0, state.possession_finishes) + finishes
    gb = xp.where(start[:, None], 0.0, state.game_blocks) + blocked
    rim_attempts = diagnostics[..., ::-1, SHOT_DEFENSE_COUNTERS.index("rim_opportunities")]
    gc = xp.where(start[:, None], 0.0, state.game_contest_attempts) + xp.where(active[:, None], rim_attempts, 0.0)
    gf = xp.where(start[:, None], 0.0, state.game_finishes) + finishes
    # Layups are credited to the shooting side, unlike blocks/contests.
    layup_attempts = diagnostics[..., SHOT_DEFENSE_COUNTERS.index("layup_attempts")]
    layup_makes = diagnostics[..., SHOT_DEFENSE_COUNTERS.index("layup_makes")]
    gla = xp.where(start[:, None], 0.0, state.game_layup_attempts) + xp.where(active[:, None], layup_attempts, 0.0)
    glm = xp.where(start[:, None], 0.0, state.game_layup_makes) + xp.where(active[:, None], layup_makes, 0.0)
    drop_index = SHOT_DEFENSE_COUNTERS.index("rim_contest_probability_drop_sum")
    if diagnostics.shape[-1] > drop_index:
        drop = diagnostics[..., ::-1, drop_index]
        contested = diagnostics[..., ::-1, SHOT_DEFENSE_COUNTERS.index("rim_opportunities")]
    else:  # Older arrays cannot supply the probability drop; do not invent zero rates.
        drop = contested = xp.zeros_like(blocked)
    pd = xp.where(start[:, None], 0.0, state.possession_contest_drop) + xp.where(active[:, None], drop, 0.0)
    pc = xp.where(start[:, None], 0.0, state.possession_contest_attempts) + xp.where(active[:, None], contested, 0.0)
    pk = state.possession_known | start
    gk = state.game_known | start
    completed_possession = possession_end & ~truncated & pk
    completed_game = end & ~truncated & gk
    defending = xp.stack((training_role < 0, training_role > 0), axis=-1)
    delta = xp.stack((
        xp.where(completed_possession[:, None], pb, 0.0),
        xp.where(completed_possession[:, None], pf, 0.0),
        (completed_possession[:, None] & defending).astype(xp.float32),
        xp.where(completed_game[:, None], gb, 0.0),
        xp.broadcast_to(completed_game[:, None], gb.shape).astype(xp.float32),
        xp.where(completed_possession[:, None], pd, 0.0),
        xp.where(completed_possession[:, None], pc, 0.0),
        xp.where(completed_game[:, None], gc, 0.0),
        xp.where(completed_game[:, None], gf, 0.0),
        xp.where(completed_game[:, None], gla, 0.0),
        xp.where(completed_game[:, None], glm, 0.0),
    ), axis=-1).sum(axis=0)
    reset_possession = possession_end | end
    return BlockMetricState(
        xp.where(reset_possession[:, None], 0.0, pb),
        xp.where(reset_possession[:, None], 0.0, pf),
        xp.where(end[:, None], 0.0, gb),
        xp.where(end, False, pk | possession_end),
        xp.where(end, False, gk),
        xp.where(reset_possession[:, None], 0.0, pd),
        xp.where(reset_possession[:, None], 0.0, pc),
        xp.where(end[:, None], 0.0, gc),
        xp.where(end[:, None], 0.0, gf),
        xp.where(end[:, None], 0.0, gla),
        xp.where(end[:, None], 0.0, glm),
        state.totals + delta,
    )


def summarize_block_metrics(totals, *, combined=False):
    counts = np.asarray(totals, dtype=np.float64)
    rows = [("", counts.sum(axis=0))] if combined else list(zip(("learner_", "opponent_"), counts))
    output = {}
    for prefix, row in rows:
        possession_blocks, finishes, possessions, game_blocks, games = row[:5]
        if combined:
            games /= 2.0  # One game, not one game per team.
        if finishes > 0:
            output[prefix + "block_pct"] = float(100.0 * possession_blocks / finishes)
        if possessions > 0:
            output[prefix + ("blocks_per_possession" if combined else "blocks_per_defensive_possession")] = float(possession_blocks / possessions)
        if games > 0:
            output[prefix + "blocks_per_episode"] = float(game_blocks / games)
        if len(row) > 5 and row[6] > 0:
            output[prefix + "rim_contest_mean_reduction_pp"] = float(100.0 * row[5] / row[6])
        if len(row) >= 9:
            # row[4] counts team-games when combined: two per completed game.
            if row[4] > 0:
                output[prefix + "rim_contested_shots_per_episode"] = float(row[7] / row[4])
            if row[8] > 0:
                output[prefix + "rim_contested_shot_pct"] = float(100.0 * row[7] / row[8])
        if len(row) >= 11:
            if row[4] > 0:
                output[prefix + "layup_attempts_per_episode"] = float(row[9] / row[4])
                output[prefix + "layup_makes_per_episode"] = float(row[10] / row[4])
            if row[9] > 0:
                output[prefix + "layup_fg_pct"] = float(100.0 * row[10] / row[9])
    return output


class TrainingBlockMetrics:
    """Carry unfinished games across updates; keep only fixed-size lane counters."""

    def __init__(self):
        self.state = None

    def consume(self, trajectory):
        if trajectory.shot_defense_diagnostics is None:
            return
        diagnostics = np.asarray(trajectory.shot_defense_diagnostics)
        if self.state is None:
            self.state = empty_block_metric_state(diagnostics.shape[1])
        fields = {
            "active": np.asarray(trajectory.active_mask),
            "episode_start": np.asarray(trajectory.episode_start),
            "possession_ended": np.asarray(trajectory.possession_ended),
            "done": np.asarray(trajectory.dones),
            "truncated": np.asarray(trajectory.truncated),
            "training_role": np.asarray(trajectory.training_role),
        }
        for step in range(diagnostics.shape[0]):
            self.state = advance_block_metrics(
                self.state, diagnostics[step], **{k: v[step] for k, v in fields.items()})

    def flush(self):
        if self.state is None:
            return {}
        output = summarize_block_metrics(self.state.totals)
        self.state = self.state._replace(totals=np.zeros_like(self.state.totals))
        return output
