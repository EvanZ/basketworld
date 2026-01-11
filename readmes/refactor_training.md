# Training Refactor Plan

## Goals
- Reduce `train/train.py` size and improve readability.
- Isolate responsibilities (CLI/config, env factories, policies, callbacks, profiling, eval).
- Preserve current CLI surface and behavior; no flag changes during extraction.

## Module Layout (current)
- `train/config.py`: argparse setup, `get_args()`.
- `train/env_factory.py`: env construction/wrappers, vector env builders, policy init env.
- `train/policy_utils.py`: opponent sampling helpers, latest model helpers, critic transfer utility.
- `train/callbacks.py`: builders for timing/entropy/beta/pass-bias/curriculum, mixed logger, opponent mapping log.
- `train/profiling.py`: VecEnv profiling aggregation/logging helper.
- `train/eval.py`: evaluation env setup and GIF logging loop.
- `train/train_utils.py`: device selection, schedules (linear, total timesteps), SPA schedule resolver, phi_beta schedule resolver, geometric sampling.
- `train/train.py`: orchestration; calls into the modules above.

## Changes Made
1) Extracted helpers to `train/train_utils.py` (device, schedules, SPA/phi_beta resolvers, geometric sampling).
2) Moved CLI to `train/config.py`.
3) Moved env builders/wrappers to `train/env_factory.py` (incl. policy init env).
4) Moved opponent/policy helpers and critic transfer into `train/policy_utils.py`.
5) Moved callback assembly/logging helpers into `train/callbacks.py`.
6) Added profiling aggregation helper in `train/profiling.py`; `train.py` calls it.
7) Moved evaluation loop to `train/eval.py`; `train.py` calls `run_evaluation`.
8) Simplified `train/train.py` to orchestrate using the extracted modules.

## Constraints & Testing
- CLI flags unchanged; behavior preserved.
- Checked with `python3 -m py_compile` after refactors.
