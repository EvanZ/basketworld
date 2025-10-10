# BasketWorld

BasketWorld is a grid-based, Gym-compatible simulation of  half-court basketball. It is designed for reinforcement learning research into emergent coordination, strategy, and multi-agent decision-making using a shared policy framework.

---

## 🧠 Core Concepts

- **Grid-Based Court**: A hexagonally tiled half-court with discrete agent movements and a hoop on one side.
- **Configurable Teams**: Play 2-on-2, 3-on-3, 5-on-5 — simply pass `--players <N>` (default 3). One shared policy controls all agents on a team.
- **Simultaneous Actions**: All agents act at each timestep.
- **Role-Conditioned Learning**: Observations and rewards are tailored to each agent's role (offense/defense).
- **Gym-Compatible**: Standard `reset()`, `step()`, and `render()` APIs.

---

## 📂 Project Structure

```
basketworld/
├── basketworld/
│   ├── envs/               # Environment and wrappers
│   ├── models/             # Shared PyTorch policy networks
│   ├── sim/                # Core simulation logic (court, rules, game state)
│   └── utils/              # Rendering, reward helpers, etc.
├── app/
│   ├── backend/            # FastAPI server powering the interactive demo
│   └── frontend/           # Vue 3 + Vite single-page application
├── train/                  # PPO training scripts and configs
├── tests/                  # Unit tests
├── assets/                 # Logos and visual assets
├── notebooks/              # Exploration and analysis notebooks
├── scripts/                # Dataset or rollout tools
├── README.md
├── setup.py
└── requirements.txt
```

---

## 🚀 Getting Started

### 1. Python backend / RL code

```bash
# 1 — Install Python deps
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2 — (Recommended) start the MLflow tracking server
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000 &

# 3 — Kick off training (vectorised, self-play PPO)
python train/train.py \
  --grid-size 12 \
  --players 3 \
  --alternations 10 \
  --steps-per-alternation 20000 \
  --num-envs 8            # parallel envs to speed up rollouts

# 4 — Watch progress in http://localhost:5000 and in the console.
```

The training script automatically:

* Alternates between offense & defense learning phases
* Logs metrics and model checkpoints to MLflow (`./mlruns/`)
* Utilises **vectorised environments** via `--num-envs` (defaults to 8) for faster PPO rollouts.

---

### 2. Interactive Web-App (FastAPI + Vue)

```bash
# 1 — Backend (FastAPI)
uvicorn app.backend.main:app --host 0.0.0.0 --port 8080 --reload

# 2 — Frontend (Vue 3 + Vite)
cd app/frontend
npm install   # first time only

# Configure the backend URL (defaults to localhost:8080 if unset)
echo "VITE_API_BASE_URL=http://localhost:8080" > .env

npm run dev   # opens http://localhost:5173
```

In the web UI enter an **MLflow run_id** from the training you just executed.  The app downloads the latest offense/defense models from that run and lets you play as either team, while visualising policy probabilities and action values.

---

## ⚙️  CLI & Environment Variables

| Component | Default | How to change |
|-----------|---------|---------------|
| FastAPI port | 8080 | `uvicorn ... --port <PORT>` |
| Frontend API URL | `VITE_API_BASE_URL` env | set in `.env` or export before `npm run dev` |
| MLflow UI port | 5000 | `mlflow ui --port <PORT>` |
| Parallel envs | 8 | `--num-envs` flag to `train.py` |

---

## ☁️  Remote Storage (S3)

MLflow can store experiments and artifacts in your personal S3 bucket instead of local storage.

**Quick Setup (Project-Specific Credentials - No Conflicts!):**

```bash
# 1. Install boto3
pip install boto3

# 2. Create .env file with project-specific credentials (won't affect other projects)
cat > .env << 'EOF'
MLFLOW_ARTIFACT_ROOT=s3://your-bucket/mlflow-artifacts
MLFLOW_AWS_ACCESS_KEY_ID=your-mlflow-key
MLFLOW_AWS_SECRET_ACCESS_KEY=your-mlflow-secret
MLFLOW_AWS_DEFAULT_REGION=us-east-1
EOF

# 3. Start MLflow server with S3 backend
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root s3://your-bucket/mlflow-artifacts \
  --port 5000

# 4. Run training - automatically uses project-specific S3 credentials!
python train/train.py --mlflow-experiment-name my-experiment
```

**Key Features:**
- ✅ `.env` file automatically loaded - no manual sourcing
- ✅ Project-specific `MLFLOW_AWS_*` credentials won't conflict with other AWS projects
- ✅ All scripts automatically detect and use S3 when configured
- ✅ Secure - `.env` is already in `.gitignore`

📖 **Documentation**: [S3 Setup](docs/mlflow_s3_setup.md) | [Quick Start](docs/mlflow_s3_quickstart.md) | [Project Credentials](docs/mlflow_project_credentials.md)

---

## 🤖 Reinforcement Learning Ready

BasketWorld is designed to work with single-policy RL algorithms like PPO, A2C, or DQN by:
- Exposing individual agent observations in a consistent format
- Using role-specific reward shaping
- Providing full control over environment dynamics and rendering

---

## 🏀 Use Cases
- Emergent passing and team play
- Defensive strategy learning
- Curriculum training from 1v1 to 5v5
- Multi-agent transfer learning
- Simulation-based basketball strategy research

---

## 📜 License
MIT License

---

## 🧩 Credits
BasketWorld is inspired by classic RL environments like [GridWorld](https://gymnasium.farama.org/environments/toy_text/frozen_lake/) and adapted for multi-agent, role-based learning in sports simulation.

> For ideas, bugs, or contributions — open an issue or pull request!
