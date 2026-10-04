#!/usr/bin/env bash

if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-$ROOT/.env/bin/python}"
NUM_UPDATES="${NUM_UPDATES:-30000}"
if [[ -v HISTORICAL_EVAL_UPDATES ]]; then
  HISTORICAL_EVAL_UPDATES="$HISTORICAL_EVAL_UPDATES"
else
  HISTORICAL_EVAL_UPDATES="100,500,2500,5000,10000,20000,30000"
fi
HISTORICAL_EVAL_EPISODES="${HISTORICAL_EVAL_EPISODES:-200}"
HISTORICAL_EVAL_HORIZON="${HISTORICAL_EVAL_HORIZON:-2048}"
MULTI_POSSESSION_USE_INBOUNDS="${MULTI_POSSESSION_USE_INBOUNDS:-true}"
MADE_BASKET_RESTART_MODE="${MADE_BASKET_RESTART_MODE:-check}"
CHECK_SETUP_STEPS="${CHECK_SETUP_STEPS:-0}"
MULTI_POSSESSION_LIMIT_START="${MULTI_POSSESSION_LIMIT_START:-1}"
MULTI_POSSESSION_LIMIT_END="${MULTI_POSSESSION_LIMIT_END:-25}"
MULTI_POSSESSION_LIMIT_RAMP_UPDATES="${MULTI_POSSESSION_LIMIT_RAMP_UPDATES:-5000}"
MULTI_POSSESSION_OVERTIME_ROUND_CAP="${MULTI_POSSESSION_OVERTIME_ROUND_CAP:-0}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

# Additive milestone evaluation defaults to the 30K schedule below. For example:
# HISTORICAL_EVAL_UPDATES=100,2000,5000 ./scripts/run_jax_5v5_halfcourt_multi_possession.sh
# Set HISTORICAL_EVAL_UPDATES= explicitly to disable it for a short smoke run.
HISTORICAL_EVAL_ARGS=()
if [ -n "$HISTORICAL_EVAL_UPDATES" ]; then
  HISTORICAL_EVAL_ARGS=(
    --historical-eval-updates "$HISTORICAL_EVAL_UPDATES"
    --historical-eval-episodes "$HISTORICAL_EVAL_EPISODES"
    --historical-eval-horizon "$HISTORICAL_EVAL_HORIZON"
  )
fi

# This legacy flag still controls non-made-basket dead-ball restarts. Select
# baseline_inbound, check, or direct_handoff for made baskets with
# MADE_BASKET_RESTART_MODE (the experiment defaults to check).
case "${MULTI_POSSESSION_USE_INBOUNDS,,}" in
  1|true|yes|y|on)
    INBOUNDS_MODE_ARGS=(--multi-possession-use-inbounds)
    ;;
  0|false|no|n|off)
    INBOUNDS_MODE_ARGS=(--no-multi-possession-use-inbounds)
    ;;
  *)
    echo "MULTI_POSSESSION_USE_INBOUNDS must be true or false." >&2
    exit 2
    ;;
esac

# Fresh multi-possession run: no continuation checkpoint, pretrained policy,
# or historical opponent pool. Each of the two 512-row fixed-team cohorts
# contributes to one shared policy, for 1,024 total environments per update.
# The possession quota is per team. Training begins with short games and
# linearly grows to the final quota; completed games keep the quota they had at
# reset. A zero overtime cap follows each episode's active possession limit.
# Play discovery stays state-only: classify aligned post-action states, never action/event labels.
exec "$PYTHON_BIN" -m basketworld_jax.train.main \
  --run-train-loop \
  --enable-multi-possession \
  --multi-possession-limit "$MULTI_POSSESSION_LIMIT_END" \
  --multi-possession-limit-start "$MULTI_POSSESSION_LIMIT_START" \
  --multi-possession-limit-end "$MULTI_POSSESSION_LIMIT_END" \
  --multi-possession-limit-ramp-updates "$MULTI_POSSESSION_LIMIT_RAMP_UPDATES" \
  --multi-possession-overtime-round-cap "$MULTI_POSSESSION_OVERTIME_ROUND_CAP" \
  --multi-possession-reward-mode scoring_events \
  --score-potential-scale 0.0 \
  --game-winner-reward 5.0 \
  "${INBOUNDS_MODE_ARGS[@]}" \
  --made-basket-restart-mode "$MADE_BASKET_RESTART_MODE" \
  --inbound-deadline-steps 5 \
  --check-deadline-steps 5 \
  --check-setup-steps "$CHECK_SETUP_STEPS" \
  --players 5 \
  --court-rows 9 \
  --court-cols 8 \
  --shot-clock 24 \
  --min-shot-clock 24 \
  --layup-pct 0.60 \
  --layup-std 0.05 \
  --three-pt-pct 0.37 \
  --three-pt-std 0.05 \
  --dunk-pct 0.60 \
  --dunk-std 0.30 \
  --three-point-distance 4.25 \
  --three-point-short-distance 3 \
  --illegal-defense-enabled true \
  --offensive-three-seconds true \
  --three-second-lane-width 1 \
  --three-second-lane-height 3 \
  --three-second-max-steps 3 \
  --defender-guard-distance 1 \
  --shot-pressure-enabled true \
  --shot-pressure-max 0.25 \
  --shot-pressure-lambda 1.0 \
  --shot-pressure-arc-degrees 300 \
  --defender-pressure-distance 1 \
  --defender-pressure-turnover-chance 0.025 \
  --defender-pressure-decay-lambda 2 \
  --base-steal-rate 0.2 \
  --steal-perp-decay 1.5 \
  --steal-distance-factor 0.5 \
  --steal-position-weight-min 0.8 \
  --pass-interception-model reaction \
  --pass-speed 5 \
  --defender-reaction-time 0.15 \
  --defender-speed 1.1 \
  --defender-reach-radius 0.5 \
  --reaction-softness 0.55 \
  --base-passer-risk 0.02 \
  --passer-pressure-decay 1.0 \
  --base-receiver-risk 0.5 \
  --receiver-alignment-min 0.35 \
  --receiver-alignment-width 2.0 \
  --max-receiver-hazard 0.85 \
  --lane-weight 0.0 \
  --assist-window 3 \
  --mask-occupied-moves false \
  --enable-pass-gating true \
  --enable-rebounds \
  --rebound-table-model-dir "$ROOT"/analytics/rebound_physics/outputs/dataset_9x8/fitted_catch_model \
  --rebound-target-temperature 1.0 \
  --rebound-target-uniform-mix 0.0 \
  --rebound-winner-distance-weight 1.0 \
  --rebound-basket-position-weight 1.0 \
  --rebound-winner-temperature 1.0 \
  --rebound-skill-std 1 \
  --rebound-skill-sampling-mode gaussian \
  --rebound-skill-high 1.0 \
  --rebound-skill-low -0.25 \
  --rebound-skill-weight 1 \
  --rebound-contest-mode local_contest \
  --rebound-contest-radius 2 \
  --offensive-rebound-shot-clock-reset 14 \
  --kernel-batch-size 512 \
  --rollout-horizon 64 \
  --num-updates "$NUM_UPDATES" \
  --policy-update-epochs 1 \
  --ppo-minibatches 16 \
  --gamma 1.0 \
  --gae-lambda 0.95 \
  --vf-coef 0.75 \
  --learning-rate 3e-4 \
  --policy-model attention \
  --action-head-mode pointer_targeted \
  --intent-embedding-enabled \
  --intent-embedding-dim 16 \
  --intent-conditioning-scale 5.0 \
  --enable-intent-learning true \
  --enable-defense-intent-learning false \
  --defense-intent-null-prob 1.0 \
  --num-intents 8 \
  --intent-commitment-steps 8 \
  --intent-null-prob 0.0 \
  --intent-visible-to-defense-prob 0.0 \
  --intent-obs-mode private_offense \
  --intent-selector-enabled true \
  --intent-selector-hidden-dim 64 \
  --intent-selector-learning-rate 1e-4 \
  --intent-selector-alpha-start 0.0 \
  --intent-selector-alpha-end 1.0 \
  --intent-selector-alpha-warmup-updates 2500 \
  --intent-selector-alpha-ramp-updates 2500 \
  --intent-selector-eps-start 0.5 \
  --intent-selector-eps-end 0.15 \
  --intent-selector-eps-warmup-updates 2500 \
  --intent-selector-eps-ramp-updates 2500 \
  --intent-selector-value-coef 0.5 \
  --intent-selector-entropy-coef 0.03 \
  --intent-selector-usage-reg-coef 0.05 \
  --intent-selector-train-every-rollouts 8 \
  --intent-selector-max-samples-per-update 1024 \
  --intent-selector-multiselect-enabled true \
  --intent-selector-min-play-steps 6 \
  --intent-diversity-enabled true \
  --intent-diversity-beta-target 0.05 \
  --intent-diversity-warmup-updates 2500 \
  --intent-diversity-ramp-updates 2500 \
  --intent-diversity-clip 2.0 \
  --intent-disc-encoder-type set_step \
  --intent-disc-hidden-dim 128 \
  --intent-disc-dropout 0.1 \
  --intent-disc-batch-size 512 \
  --intent-disc-updates-per-rollout 4 \
  --intent-disc-eval-holdout-fraction 0.10 \
  --intent-disc-current-policy-only true \
  --intent-disc-include-shot-clock false \
  --intent-disc-include-pressure-exposure false \
  --intent-policy-sensitivity-enabled true \
  --intent-policy-sensitivity-sample-states 64 \
  --intent-policy-sensitivity-log-every-rollouts 10 \
  --disc-eval-batch-output true \
  --intent-sample-dump-size 4096 \
  --task-reward-scale-start 0.1 \
  --task-reward-scale-end 1.0 \
  --task-reward-scale-warmup-updates 0 \
  --task-reward-scale-ramp-updates 2000 \
  --enable-phi-shaping false \
  --opponent-pool-size 30 \
  --opponent-pool-beta 0.9 \
  --opponent-pool-exploration 0.30 \
  --opponent-deterministic-episode-prob-start 0.20 \
  --opponent-deterministic-episode-prob-end 0.80 \
  --opponent-deterministic-episode-prob-ramp-updates 15000 \
  --ent-coef-start 1.0 \
  --ent-coef-end 0.03 \
  --ent-schedule exp \
  --checkpoint-every-updates 250 \
  --log-every-updates 10 \
  --eval-every-updates 0 \
  --eval-deploy-every-updates 250 \
  --eval-deploy-batches 2 \
  --eval-deploy-horizon 2048 \
  "${HISTORICAL_EVAL_ARGS[@]}" \
  --mlflow-experiment-name halfcourt_multi_possessions \
  --log-mlflow \
  "$@"
