# Deterministic historical-evaluation Elo

`analytics/historical_eval_elo.py` builds an offline Elo ladder from the
deterministic matchup results already produced by the JAX historical-evaluation
harness. It does not load checkpoints or run new episodes.

## From an MLflow run

```bash
.env/bin/python analytics/historical_eval_elo.py \
  --run-id <MLFLOW_RUN_ID>
```

The script downloads `results/historical_opponent_evaluation.json` and writes:

- `elo_analysis/<run-id>/deterministic_elo.json`
- `elo_analysis/<run-id>/deterministic_elo.csv`

Use `--artifact-path` if the source artifact is stored elsewhere and
`--output-dir` to choose a different output directory.

## From a local artifact

```bash
.env/bin/python analytics/historical_eval_elo.py \
  --input-json /path/to/historical_opponent_evaluation.json \
  --output-dir /path/to/elo-output
```

The default is an order-independent batch fit centered on an initial rating of
`1000`, with an `800`-point Gaussian prior standard deviation. Override these
with `--initial-rating` and `--prior-stddev`.

To place a particular checkpoint at a familiar displayed rating without
changing any fitted gaps or matchup probabilities, anchor it after the fit:

```bash
.env/bin/python analytics/historical_eval_elo.py \
  --run-id <MLFLOW_RUN_ID> \
  --anchor-update 100 \
  --anchor-rating 400
```

Anchoring adds the same constant to every rating and to both endpoints of every
confidence interval. The JSON records the anchor and the resulting
`rating_offset`.

## Rating semantics

The default `--method batch` fits a regularized Bradley-Terry model on every
completed episode outcome. Its sufficient statistic for each matchup is:

```text
(candidate wins + 0.5 * candidate ties) / completed games
```

This produces Elo-scaled ratings whose differences correspond to fitted win
odds, while remaining independent of JSON or matchup order. The Gaussian prior
keeps estimates finite for nearly undefeated or winless checkpoints. The JSON
and CSV include approximate standard errors and 95% intervals from the fitted
posterior curvature.

For comparison with the original first pass, use chronological aggregate Elo:

```bash
.env/bin/python analytics/historical_eval_elo.py \
  --run-id <MLFLOW_RUN_ID> \
  --method chronological \
  --k-factor 32
```

Self-matches, non-deterministic records, truncated-only evaluations, exact
duplicates, and malformed records are skipped and counted by reason in the JSON
output.

The output also retains aggregate W/L/T, completed-game count, win score, and
mean point differential for each checkpoint. Elo is only a scalar summary;
inspect the underlying matchup matrix for non-transitive policies. Ratings in
disconnected matchup components are not globally comparable, and the output
labels their component IDs explicitly.
