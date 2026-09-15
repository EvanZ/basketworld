# Getting started

## Serve the documentation locally

From the repository root, install the isolated documentation dependency and
start the live-reloading development server:

```bash
python -m pip install -r requirements-docs.txt
mkdocs serve
```

Open <http://127.0.0.1:8000/>. MkDocs watches `mkdocs.yml` and
`docs/site/`, rebuilding pages as they change.

To validate the entire site without starting a server:

```bash
mkdocs build --strict
```

The generated `site/` directory is disposable build output.

## Install BasketWorld

Training requires the project dependencies in addition to the documentation
theme. A conventional local setup is:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` installs BasketWorld in editable mode through `-e .`.
JAX device selection follows the JAX installation in the active environment;
confirm the available device before a long run:

```bash
python -c "import jax; print(jax.devices())"
```

## Trainer modes

The JAX entry point is:

```bash
python -m basketworld_jax.train.main
```

Without `--run-train-loop`, this runs the trainer scaffold: it constructs the
environment and policy, warms and benchmarks forward passes, collects compiled
rollouts, and benchmarks PPO updates. It is a performance/development mode,
not a long training job.

Add `--run-train-loop` to execute the multi-update offense-and-defense PPO
loop.

### Small attention-policy smoke run

This command deliberately uses tiny batches and disables periodic evaluation.
It validates the main training path without representing a useful experiment:

```bash
python -m basketworld_jax.train.main \
  --run-train-loop \
  --num-updates 1 \
  --kernel-batch-size 8 \
  --rollout-horizon 8 \
  --eval-every-updates 0 \
  --no-progress \
  --policy-model attention \
  --action-head-mode pointer_targeted
```

The attention and pointer-targeted flags must be used together for the pointer
head. The trainer validates incompatible combinations before compiling.

!!! warning "Compilation cost"

    The first invocation for a new combination of array shapes and static
    configuration includes JAX compilation. A one-update smoke test can
    therefore spend more time compiling than training.

## Next steps

- Read [Actions and observations](environment/actions-observations.md) before
  changing a policy input.
- Read [RL training](jax/training.md) before changing batch sizes, rollout
  horizons, or PPO minibatches.
- Read [Training operations](jax/operations.md) before enabling MLflow,
  opponent pools, checkpoint publication, or continuation.
- Treat `parse_args()` and `validate_train_args()` in
  `basketworld_jax/train/main.py` as the canonical CLI definition.
