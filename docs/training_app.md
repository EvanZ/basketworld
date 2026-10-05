# BasketWorld training application

The training application is a standalone local control plane. It is separate from the gameplay/self-play application and uses its own FastAPI backend, Vue/Vite frontend, ports, and DuckDB database.

## Install

```bash
.env/bin/pip install -r app/training_backend/requirements.txt
npm --prefix app/training_frontend install
```

## Run

Development, with separate Vite and FastAPI processes:

```bash
scripts/start_training_app.sh
```

- Frontend: <http://localhost:5174>
- API documentation: <http://localhost:8090/docs>
- DuckDB database: `var/training_app/training.duckdb`
- Per-run logs, control files, status, and local checkpoints: `var/training_app/runs/<run-id>/`

Production-style local serving, with FastAPI serving the built frontend:

```bash
scripts/start_training_app_prod.sh
```

The combined production-style application is available at <http://localhost:8090>.

Closing the browser or stopping the frontend does not stop a training worker. The backend launches each worker in a new process session and records its PID plus Linux process-start token to guard against PID reuse.

## v0 lifecycle

- **Checkpoint** writes a request file that the JAX trainer consumes after its current PPO update.
- **Pause** finishes the current update, writes a complete checkpoint, acknowledges the request, and exits normally.
- **Resume** launches the same absolute-update target from the local `latest` checkpoint without resetting environment or intent-discriminator state, and reopens the original MLflow run so metrics and artifacts continue in one history.
- **Stop** sends SIGTERM to the verified worker process group. It is distinct from pause and does not promise a fresh checkpoint.

DuckDB stores control-plane state only. MLflow remains the source of truth for training metrics, parameters, and uploaded artifacts.

Every launch explicitly records an MLflow tracking URI and experiment name.
The form defaults to `http://localhost:5000` and
`halfcourt_multi_possessions`; both values can be changed before launch and
are persisted with the run configuration and resolved worker environment.

## Configuration and cloning

The launch form reads its full configuration schema from the JAX trainer's
argparse definitions and applies an application-owned halfcourt preset. The
**All training configs** section shows the application-preset value, parser
default, value type, choices, help annotation, and whether each setting is
editable, internal, or frozen. Any setting can be pinned to the top of the form.
Pinned choices persist in the browser. Frozen and internal settings are hidden
from the full browser by default and can be displayed with independent filters;
an explicitly pinned setting remains visible regardless of those filters.
The application launches the Python trainer directly; it does not execute the
command-line training shell script.

`entropy_decay_updates` optionally decouples the entropy schedule from the
overall run length. When it is null, entropy reaches `ent_coef_end` on the last
configured update, preserving the original behavior. When it is set, entropy
reaches the end coefficient at that update and remains there for the rest of
the run.

The default `mlflow_metric_profile=core` logs a compact set of aggregate PPO,
episode, spatial, intent, selector, opponent-pool, and deploy-evaluation metrics.
It intentionally omits per-intent, per-action, raw-count, and duplicate metric
families so MLflow does not create hundreds of charts or exhaust browser local
storage. Set the profile to `full` when a diagnostic run needs every scalar;
the profile never changes training logic or the JSON/checkpoint artifacts.

The application schema excludes legacy SB3/shared-parser arguments that the JAX
trainer does not consume. For example, use `rollout_horizon` rather than
`n_steps`, `policy_update_epochs` rather than `n_epochs`, `kernel_batch_size`
rather than `num_envs`, and the JAX evaluation cadence fields rather than
`eval_freq`. This prevents a launch from accepting a setting that has no effect.

Use **Copy as new run** on an application run to duplicate its launch config.
To copy an MLflow run, paste its run ID in **Copy an MLflow run**. Runs created
after this feature include `metadata/resolved_training_config.json` and can be
reconstructed exactly. Older runs are reconstructed from their logged MLflow
parameters; the UI reports how many fields were recovered so a partial legacy
import is never presented as exact.
