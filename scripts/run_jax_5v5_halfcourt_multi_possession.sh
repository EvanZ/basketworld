#!/usr/bin/env bash

if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-$ROOT/.env/bin/python}"
NUM_UPDATES="${NUM_UPDATES:-500}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

# Fresh multi-possession run: no continuation checkpoint, pretrained policy,
# or historical opponent pool. Each of the two 512-row fixed-team cohorts
# contributes to one shared policy, for 1,024 total environments per update.
exec "$PYTHON_BIN" -m basketworld_jax.train.main \
  --run-train-loop \
  --enable-multi-possession \
  --multi-possession-limit 25 \
  --multi-possession-reward-mode win_loss \
  --score-potential-scale 1.0 \
  --inbound-deadline-steps 5 \
  --players 5 \
  --court-rows 9 \
  --court-cols 8 \
  --shot-clock 24 \
  --min-shot-clock 14 \
  --three-point-distance 4.25 \
  --three-point-short-distance 3 \
  --spawn-distance 5 \
  --max-spawn-distance 7 \
  --defender-spawn-distance 3 \
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
  --enable-phi-shaping true \
  --reward-shaping-gamma 1.0 \
  --phi-beta-start 0.0 \
  --phi-beta-end 0.25 \
  --phi-beta-warmup-updates 0 \
  --phi-beta-ramp-updates 500 \
  --phi-blend-weight 0.0 \
  --opponent-pool-size 10 \
  --opponent-pool-beta 0.7 \
  --opponent-pool-exploration 0.30 \
  --opponent-deterministic-episode-prob-start 0.20 \
  --opponent-deterministic-episode-prob-end 0.80 \
  --opponent-deterministic-episode-prob-ramp-updates 500 \
  --checkpoint-every-updates 25 \
  --log-every-updates 10 \
  --eval-every-updates 0 \
  --eval-deploy-every-updates 100 \
  --eval-deploy-batches 20 \
  --eval-deploy-horizon 256 \
  --log-mlflow
