# League training verification (#87)

Verified 2026-11-28 for the implementation committed with this report.

## Acceptance evidence

- Trainer selection, launch, copy, resume, and extension: backend league tests
  exercise pinned versions, exact snapshots, later draft edits, archived versions,
  and rejection of missing artifacts or changed resume settings.
- Validation: backend tests cover team sizes, rule compatibility, checksums,
  invalid probabilities, and explicit source selection. Shared snapshot validation
  also resolves and validates the saved skill schema before compilation.
- Sampling: compiled JAX tests cover 3v3, 4v4, and 5v5; 0%/100% endpoints;
  empirical independent 75% source draws over 8,192 games; uniform team selection;
  identical-team matchups; exact saved skills; and fresh generation parameters.
- Stable ownership and resume: compiled tests exercise a possession handoff,
  partial batch reset, serialization/restoration, and identical subsequent roster
  assignments with different environment reset keys. Roster RNG is stored in
  kernel state and never consumes the opponent-policy RNG.
- Evaluation: fresh-only evaluation resets are tested. Historical evaluation
  tests verify matched seeds across league checkpoint comparisons while retaining
  legacy random-run seeding.
- Diagnostics: only the four approved source/coverage metrics are added. Tests
  cover interval reset, MLflow filtering, catalog availability, and normal picker
  grouping. A read-only live browser check confirmed all four names appear with
  the scoring-diagnostics checkbox unchecked.
- UI: the isolated browser workflow passed creation, generation settings,
  profile changes, saved configuration visibility, publication, immutable version
  inspection, reassignment confirmation, archive/restore, selection, deletion,
  and recovery from a deleted selection. No user league was modified by this test.
- Documentation: `docs/training_app.md` and `docs/site/jax/leagues.md` describe
  the trainer workflow, copied settings, sampling, evaluation, and lifecycle.

## Checks

- Focused backend and historical-evaluation tests: 53 passed, 87 deselected.
- Compiled JAX league sampling tests: 5 passed.
- Frontend scoring metric tests and production build: passed.
- Isolated league browser workflow and live metric-picker check: passed.
- `git diff --check`: passed.

The sandboxed backend run stalled in its local HTTP test and was stopped;
the complete focused suite above passed outside that sandbox.

## Previously completed compiled smoke and overhead measurement

The training kernel, roster sampler, and rollout integration have not changed
since these checks on 2026-11-27. The compiled autoregressive smoke completed one
PPO update, deploy evaluation, checkpoint save, and a same-run resume to update 2.
Policy, value, and total losses were finite. Saved checkpoints contained the
immutable league snapshot and roster RNG/assignments.

A bounded CPU comparison used 3v3, 16 environments, an 8-step rollout, a small
attention policy, 64 league teams, and ten timed rollouts after compilation:

- Compilation: random 133.98 s; league 139.96 s (+4.5%).
- Median rollout: random 86.04 ms; league 97.69 ms (+13.5%).
- Throughput: random 1,487.65; league 1,310.30 environment steps/s (-11.9%).
- Compiled argument storage: 125,031 versus 134,710 bytes (+7.7%).
- Compiled output storage: 214,824 versus 215,848 bytes (+0.5%).
- Compiled temporary storage: 312,480 versus 324,800 bytes (+3.9%).

These are executable buffer measurements, not peak process/GPU memory, and the
small CPU benchmark is not a production GPU throughput forecast. It measures
the feature cost, not an improvement in learning or policy strength.
