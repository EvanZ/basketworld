# ============================================================
# Basketworld Environment Template
# ============================================================
# Backend:
#   Copy backend entries to: app/backend/.env.app
# Frontend:
#   Copy only the "Frontend (Vite)" section to: app/frontend/.env
#
# Notes:
# - Backend loader checks (in order): .env.app, .env.local, .env (files only).
# - `.env` directories used as virtualenvs are ignored automatically.
# - Blank BW_PLAYABLE_* entries mean that option is unavailable in playable UI.
# - Explicit BW_PLAYABLE_<N>_<DIFF>_* values override BW_PLAYABLE_POLICY_MATRIX_JSON.

# ============================================================
# Backend: App mode + CORS
# ============================================================
BW_PUBLIC_MODE=false
BACKEND_CORS_ALLOW_ORIGINS=http://localhost:5173,http://127.0.0.1:5173

# ============================================================
# Backend: MLflow
# ============================================================
# Local MLflow example:
MLFLOW_TRACKING_URI=http://127.0.0.1:5000

# Optional S3 artifact configuration:
MLFLOW_ARTIFACT_ROOT=
MLFLOW_S3_ENDPOINT_URL=
MLFLOW_AWS_ACCESS_KEY_ID=
MLFLOW_AWS_SECRET_ACCESS_KEY=
MLFLOW_AWS_DEFAULT_REGION=us-east-1

# Optional fallback AWS env vars (if not using MLFLOW_AWS_*):
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_DEFAULT_REGION=us-east-1

# ============================================================
# Backend: Playable policy matrix
# ============================================================
# Option A: JSON matrix (leave blank if using explicit entries below)
# Example format:
# {"1":{"easy":{"run_id":"<run_id>","checkpoint_index":101}},"3":{"hard":{"run_id":"<run_id>","checkpoint_index":450}}}
BW_PLAYABLE_POLICY_MATRIX_JSON=

# Option B: Explicit entries (blank = unavailable)
# 1v1
BW_PLAYABLE_1_EASY_RUN_ID=
BW_PLAYABLE_1_EASY_CHECKPOINT=
BW_PLAYABLE_1_MEDIUM_RUN_ID=
BW_PLAYABLE_1_MEDIUM_CHECKPOINT=
BW_PLAYABLE_1_HARD_RUN_ID=
BW_PLAYABLE_1_HARD_CHECKPOINT=

# 2v2
BW_PLAYABLE_2_EASY_RUN_ID=
BW_PLAYABLE_2_EASY_CHECKPOINT=
BW_PLAYABLE_2_MEDIUM_RUN_ID=
BW_PLAYABLE_2_MEDIUM_CHECKPOINT=
BW_PLAYABLE_2_HARD_RUN_ID=
BW_PLAYABLE_2_HARD_CHECKPOINT=

# 3v3
BW_PLAYABLE_3_EASY_RUN_ID=31a77df666594f2bb06cc9bf6a0a1adc
BW_PLAYABLE_3_EASY_CHECKPOINT=50
BW_PLAYABLE_3_MEDIUM_RUN_ID=31a77df666594f2bb06cc9bf6a0a1adc
BW_PLAYABLE_3_MEDIUM_CHECKPOINT=100
BW_PLAYABLE_3_HARD_RUN_ID=31a77df666594f2bb06cc9bf6a0a1adc
BW_PLAYABLE_3_HARD_CHECKPOINT=200

# 4v4
BW_PLAYABLE_4_EASY_RUN_ID=
BW_PLAYABLE_4_EASY_CHECKPOINT=
BW_PLAYABLE_4_MEDIUM_RUN_ID=
BW_PLAYABLE_4_MEDIUM_CHECKPOINT=
BW_PLAYABLE_4_HARD_RUN_ID=
BW_PLAYABLE_4_HARD_CHECKPOINT=

# 5v5
BW_PLAYABLE_5_EASY_RUN_ID=
BW_PLAYABLE_5_EASY_CHECKPOINT=
BW_PLAYABLE_5_MEDIUM_RUN_ID=
BW_PLAYABLE_5_MEDIUM_CHECKPOINT=
BW_PLAYABLE_5_HARD_RUN_ID=
BW_PLAYABLE_5_HARD_CHECKPOINT=