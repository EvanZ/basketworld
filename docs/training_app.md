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

To restart this alongside MLflow and the gameplay development application, use:

```bash
scripts/restart_dev_services.sh
```

That script owns the trainer API and Vite process groups, but it deliberately
does not stop detached training workers.

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

For a nonzero check-clearance radius, checks use one deterministic safe anchor
by default. **Randomize check location** is an opt-in that samples each check
uniformly from safe outside-arc pickup tiles; either setting retains the same
clearance mechanics and policy observation schema.

**Check-clearance warmup updates** defaults to 500 for new runs. During those
updates, the policy sees the clearance-context fields but plays the zero-radius
check rule, allowing it to learn collection first. On the next update the
configured clearance activates and every training environment resets once;
evaluation always uses the configured final clearance rule.

## v0 lifecycle

- **Checkpoint** writes a request file that the JAX trainer consumes after its current PPO update.
- **Pause** finishes the current update, writes a complete checkpoint, acknowledges the request, and exits normally.
- **Resume** launches the same absolute-update target from the local `latest` checkpoint without resetting environment or intent-discriminator state, and reopens the original MLflow run so metrics and artifacts continue in one history. Before launch, the app validates checkpoint schema, policy/action/observation metadata, and resolved training settings.
- **Continue** extends a completed run from its latest checkpoint in the same MLflow run. The update target changes, but checkpoint-compatible training settings do not.
- **Stop** sends SIGTERM to the verified worker process group. It is distinct from pause and does not promise a fresh checkpoint. A stopped run can be deliberately recovered from its last validated checkpoint.

Every durable checkpoint records its Git commit, tracked-worktree cleanliness, and (when dirty) a SHA-256 fingerprint of the tracked diff. The trainer app compares that record with the current checkout before Resume, Recover, or Continue. An exact match proceeds normally; a mismatch, unavailable Git metadata, or a legacy checkpoint requires explicit confirmation. This provenance is diagnostic only—it does not guarantee dependency or runtime compatibility.

For check-based made-basket restarts, the trainer configuration exposes **Pre-check setup steps**, **Check pickup seconds**, and **Check-clearance rings**. New configurations use five seconds for setup and five seconds for pickup. A nonzero clearance samples one eligible pickup tile outside the clearance arc; the protected disk, plus one inbounds ring, must fit on the court. During setup both teams must clear it. An offense-only violation awards the defense one point and ends possession; a defense-only violation awards the offense one point and restarts the same possession with a defensive technical. The first two double violations repeat setup at the same tile, and the third assigns one of those single-team penalties with a seeded 50/50 draw. Nonzero-clearance policies use a new observation schema and should train from scratch.

Run detail shows the full Git commit and worktree state captured for the most
recent worker launch. For a resumed or continued run, these fields describe the
latest launch. Runs created before launch provenance was recorded show the
values as unavailable.

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

The shot-contest controls are pinned in the launch form by default:
`shot_contest_skill_mean`, `shot_contest_skill_std`, and
`shot_contest_skill_effect_weight`. They map directly to the trainer's
`--shot-contest-skill-mean`, `--shot-contest-skill-std`, and
`--shot-contest-skill-effect-weight` arguments and are included explicitly in
the resolved command for new runs.

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
When the copied run's update target changes, the form drops historical
evaluation milestones beyond the new target. If the source schedule evaluated
at its final update, that final milestone moves to the new target.
To copy an MLflow run, paste its run ID in **Copy an MLflow run**. Runs created
after this feature include `metadata/resolved_training_config.json` and can be
reconstructed exactly. Older runs are reconstructed from their logged MLflow
parameters; the UI reports how many fields were recovered so a partial legacy
import is never presented as exact.
