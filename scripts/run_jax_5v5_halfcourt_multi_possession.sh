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
  --intent-selector-alpha-warmup-updates 100 \
  --intent-selector-alpha-ramp-updates 400 \
  --intent-selector-eps-start 0.5 \
  --intent-selector-eps-end 0.15 \
  --intent-selector-eps-warmup-updates 100 \
  --intent-selector-eps-ramp-updates 400 \
  --intent-selector-value-coef 0.5 \
  --intent-selector-entropy-coef 0.03 \
  --intent-selector-usage-reg-coef 0.05 \
  --intent-selector-train-every-rollouts 8 \
  --intent-selector-max-samples-per-update 1024 \
  --intent-selector-multiselect-enabled true \
  --intent-selector-min-play-steps 6 \
  --intent-diversity-enabled true \
  --intent-diversity-beta-target 0.05 \
  --intent-diversity-warmup-updates 100 \
  --intent-diversity-ramp-updates 400 \
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
  --disc-eval-batch-output true \
  --intent-sample-dump-size 4096 \
  --task-reward-scale-start 0.1 \
  --task-reward-scale-end 1.0 \
  --task-reward-scale-warmup-updates 0 \
  --task-reward-scale-ramp-updates 500 \
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
  --eval-deploy-batches 4 \
  --eval-deploy-horizon 1024 \
  --mlflow-experiment-name halfcourt_multi_possessions \
  --log-mlflow
