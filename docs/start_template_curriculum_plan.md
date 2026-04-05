# Start-Template Curriculum Plan

## Goal

Add an optional start-template curriculum that broadens possession geometry and
defender coverage in a structured way while preserving the current spawning
strategy as the default fallback.

Target behavior:

1. keep the current run/config path unchanged unless the feature is enabled
2. support a weighted library of offensive + defensive start templates
3. resolve each template into exact `initial_positions` plus optional
   `ball_holder` / `shot_clock`
4. allow light stochastic jitter and left/right mirroring
5. mix template starts with the current spawn generator instead of replacing it
   unconditionally

## Why Change It

The current regime is finally producing strong low-level play signal:

- discriminator AUC is stable in the `~0.7+` range
- selector usage is healthier than earlier runs
- Playbook shows distinct possession families

But the remaining bottleneck appears to be start-state diversity:

1. **Current starts are still fairly narrow**
   - existing spawn logic is distance-based, not formation-based
   - many starts are shot-friendly in similar ways

2. **Learned play families are still dominated by a small number of spatial motifs**
   - quick-shot family
   - slower developed-possession family

3. **Selector preferences are averaged over a broad but weakly structured state distribution**
   - if some plays are only useful from certain formations or coverages, the
     selector may not get a clean enough signal to preserve them

The next structural step is therefore not:

- more intents
- multiselect reselection
- another discriminator redesign

It is:

- **structured start-state / coverage diversity**

## Current Support

The codebase already has the key primitive we need:

1. the env reset path already accepts exact overrides for:
   - `initial_positions`
   - `ball_holder`
   - `shot_clock`

2. reset precedence is already defined:
   - `options["initial_positions"]`
   - constructor fixed override
   - current `_generate_initial_positions()`

Relevant code:

- [basketworld_env_v2.py](/home/evanzamir/basketworld/basketworld/envs/basketworld_env_v2.py)
- [env_factory.py](/home/evanzamir/basketworld/train/env_factory.py)

Current limitation:

- there is no template schema
- no template library loader
- no reset-time template sampler
- no template-specific logging/diagnostics

## Design Choice (Recommended)

Implement a **weighted start-template library** with **probabilistic fallback to
the current spawn generator**.

Core principles:

1. **Default-off**
   - if template mode is disabled, behavior is identical to today

2. **Structured rather than fully random**
   - templates should anchor:
     - ball handler
     - teammates
     - defenders
   - do not constrain only the ball handler

3. **Stochastic rather than frozen**
   - templates generate starts with small jitter
   - they are not exact board snapshots every time

4. **Use both sides**
   - templates should describe:
     - offensive formation
     - defensive coverage/alignment

5. **Do not mix too many research changes at once**
   - keep selector multiselect off while validating this
   - keep the current discriminator/selector baseline otherwise unchanged

## Scope

### In Scope (v1)

- template schema and validation
- minimal dev-only board authoring tool for template creation
- optional CLI-driven template library for training
- reset-time sampling into exact `initial_positions`
- optional template-specific `ball_holder`
- optional template-specific `shot_clock`
- template mixing with current spawn logic
- basic logging of which template was used

### Out of Scope (v1)

- learned template generation
- learned in-game formation choice
- template-conditioned selector losses
- polished end-user template picker in the playable UI
- dynamic defender coverage logic beyond positions

## API / CLI Proposal

Add to `train/config.py`:

- `--start-template-enabled`
  - bool, default `false`
  - master on/off switch

- `--start-template-library`
  - path to YAML/JSON file
  - default `None`

- `--start-template-prob`
  - float in `[0, 1]`
  - probability that a reset uses a template instead of the current spawn logic
  - default `0.0`

- `--start-template-jitter-scale`
  - float, default `1.0`
  - global multiplier on per-player jitter radii

- `--start-template-mirror-prob`
  - float in `[0, 1]`
  - probability of mirroring a mirrorable template
  - default `0.5`

Optional later flag:

- `--start-template-strict`
  - fail on invalid libraries instead of silently disabling

Recommended first experiment:

- `start-template-enabled = true`
- `start-template-prob = 0.3`
- `start-template-jitter-scale = 1.0`
- `start-template-mirror-prob = 0.5`

That gives:

- 30% template starts
- 70% current spawning

## Template Schema Proposal

Use a small YAML file with normalized, team-local slots.

Why team-local slots instead of absolute player ids:

- avoids hardcoding current absolute index layout
- cleaner for future changes in players-per-side
- easier to read and author

Suggested schema:

```yaml
version: 1
players_per_side: 3
templates:
  - id: wing_entry_help
    weight: 1.0
    mirrorable: true
    shot_clock: 24
    ball_holder:
      team: offense
      slot: 1
    offense:
      - slot: 0
        anchor: [8, 11]
        jitter_radius: 1
      - slot: 1
        anchor: [7, 9]
        jitter_radius: 0
      - slot: 2
        anchor: [11, 9]
        jitter_radius: 1
    defense:
      - slot: 0
        anchor: [6, 9]
        jitter_radius: 0
        role: on_ball
      - slot: 1
        anchor: [9, 9]
        jitter_radius: 1
        role: help
      - slot: 2
        anchor: [12, 10]
        jitter_radius: 1
        role: weak_side_help
```

### Schema Semantics

Top-level:

- `version`
- `players_per_side`
- `templates`

Per template:

- `id`
- `weight`
- `mirrorable`
- optional `shot_clock`
- `ball_holder`
- `offense[]`
- `defense[]`

Per player anchor:

- `slot`
- `anchor`
  - axial `(q, r)` coordinates
- `jitter_radius`
- optional `role`
  - metadata only in v1

Important:

- `role` does not change gameplay in v1
- it is only descriptive/debug metadata for the template author

## Runtime Behavior

At env reset:

1. if explicit reset `options` contain:
   - `initial_positions`
   - `ball_holder`
   - `shot_clock`
   then honor them exactly and bypass template sampling

2. else if a constructor-level fixed override exists:
   - honor that fixed override

3. else if template mode is enabled and sampled:
   - resolve a template into exact start state

4. else:
   - fall back to the current `_generate_initial_positions()`

This preserves current semantics and keeps template mode backward-compatible.

### Template Resolution

If a template is chosen:

1. sample a template by weight
2. mirror it with probability `start_template_mirror_prob` if `mirrorable`
3. resolve offense anchors with jitter
4. resolve defense anchors with jitter
5. repair collisions or invalid cells if needed
6. emit exact `initial_positions`
7. set `ball_holder`
8. optionally set `shot_clock`

### Legality / Repair Rules

The resolver must enforce:

- on-court positions only
- unique positions
- no illegal basket-cell occupancy when dunks are disabled

Recommended repair strategy:

1. try jitter samples near the anchor
2. if still invalid, search nearest legal cell
3. if still impossible, fall back to current spawn logic and log the failure

Training should not crash because one sampled template is temporarily invalid.

## Integration Plan

### 1. New Template Module

Add a new module, e.g.:

- `basketworld/utils/start_templates.py`

Responsibilities:

- load YAML/JSON library
- validate schema
- normalize to runtime-friendly Python dicts
- sample a template by weight
- apply mirroring
- resolve to exact positions

Suggested functions:

- `load_start_template_library(path) -> dict`
- `validate_start_template_library(library, players_per_side) -> None`
- `sample_start_template(library, rng) -> dict`
- `resolve_start_template(env, template, jitter_scale, mirror) -> dict`

Resolved output should look like:

```python
{
    "template_id": "wing_entry_help",
    "mirrored": True,
    "initial_positions": [...],
    "ball_holder": 1,
    "shot_clock": 24,
}
```

### 2. Env Wiring

Add env fields:

- `start_template_enabled`
- `start_template_prob`
- `start_template_jitter_scale`
- `start_template_mirror_prob`
- `start_template_library`

Hook point:

- [basketworld_env_v2.py](/home/evanzamir/basketworld/basketworld/envs/basketworld_env_v2.py)
  reset path just before current position generation

Recommended logic:

- if no template is selected, keep current path exactly
- if template is selected, call the resolver and feed the result into the same
  position/ball-holder setup path used by existing overrides

### 3. Env Factory / Training Config

Pass the new args through:

- [env_factory.py](/home/evanzamir/basketworld/train/env_factory.py)

The library should be normalized into a simple serializable structure before it
is passed into subprocess env constructors.

That keeps SubprocVecEnv behavior predictable and avoids repeated ad hoc file IO
inside worker resets.

### 4. MLflow / Training Params

Log the new config fields so the UI and analysis can recover them:

- `start_template_enabled`
- `start_template_library`
- `start_template_prob`
- `start_template_jitter_scale`
- `start_template_mirror_prob`

### 5. UI / Backend Visibility

Needed early for debugging and authoring:

- show template settings in Training tab
- expose template id in debug state/info for replay and Playbook diagnostics

### 6. Minimal Template Authoring UI (Phase 1)

Add a dev-only authoring tool, preferably as another tab:

- `Template Mode`

Reason:

- authoring axial `(q, r)` coordinates by hand is too error-prone without visual
  guidance
- a thin board-driven authoring surface will speed up template creation
  materially

This should be **minimal**, not a polished full editor.

Goal:

- place offense and defense slots directly on the board
- choose ball-holder slot
- export one schema-valid template entry

Recommended first version:

1. enter `Template Mode`
2. toggle placement target:
   - offense slot `0/1/2`
   - defense slot `0/1/2`
3. click a board hex to assign the slot anchor
4. choose:
   - template id
   - weight
   - mirrorable yes/no
   - ball-holder slot
   - optional shot clock
   - per-slot jitter radius
5. click `Export Template`
6. receive YAML/JSON for one template entry

Minimum UI requirements:

- visible `(q, r)` coordinate guidance on hover and/or click
- clear highlighting of assigned offense/defense slots
- validation that all required slots are filled
- export that matches the training schema exactly

Important constraint:

- the UI does not define a separate format
- it is only an authoring surface over the same schema used by the runtime

Recommended implementation split:

1. frontend
   - add `Template Mode` to [PlayerControls.vue](/home/evanzamir/basketworld/app/frontend/src/components/PlayerControls.vue)
   - reuse board click handling already present in the app
   - keep draft template state client-side

2. backend
   - optional validation/export endpoint
   - return normalized schema-valid template payload

3. shared schema
   - export must round-trip through the same loader/validator used by training

## Diagnostics / Logging

At minimum, add per-reset info fields:

- `start_template_used`
- `start_template_id`
- `start_template_mirrored`
- `start_template_fallback_reason`

Why:

- lets us measure actual template usage
- lets us correlate selector preferences with template ids
- makes evaluation and debugging much easier

Nice-to-have metrics:

- per-template reset frequency
- selector intent distribution conditioned on template id
- per-template shot rollout rate / turnover rate / first-shot step

## Recommended First Template Set

Start with a small hand-authored library:

1. top entry, tight on-ball defense
2. left wing entry, help one pass away
3. right wing entry, help one pass away
4. corner-side entry, weak-side help loaded
5. elbow / high-post entry, on-ball gap coverage

For each:

- mirror left/right where appropriate
- keep jitter small
- do not try to cover every basketball situation immediately

The first goal is not realism maximalism.
It is to expose the model to a few clearly different structured starts.

## Recommended Next Experiment

Keep the current strong baseline as intact as possible:

- multiselect off
- same discriminator setup
- same selector exploration setup
- same intent count

Change only:

- enable template starts
- mix them with current spawning

Suggested config:

- `start-template-enabled = true`
- `start-template-prob = 0.3`
- `start-template-jitter-scale = 1.0`
- `start-template-mirror-prob = 0.5`

Success criteria:

1. discriminator AUC remains strong
2. selector still avoids collapse
3. different templates produce different selector preference profiles
4. underused intents become useful in at least some template families
5. Playbook shows richer than the current two-family split

## Testing Plan

### Unit tests

1. schema parse / validation
2. mirrored template resolution
3. jittered placement legality
4. collision repair
5. team-slot to absolute-player-index mapping

### Integration tests

1. env reset uses current spawning when template mode is disabled
2. env reset uses current spawning when template mode is enabled but not sampled
3. env reset uses template output when sampled
4. invalid template resolution falls back cleanly
5. training smoke run with template probability > 0 completes

### Behavioral tests

1. template-specific selector probability profiles differ
2. Playbook families differ by template
3. some previously underused intents become template-specific rather than globally dead

## Rollout Plan

### Phase 1

- schema
- loader
- validation
- resolved-template sampler
- minimal `Template Mode` board authoring/export tool

### Phase 2

- env reset integration
- logging / info fields
- training config / MLflow plumbing

### Phase 3

- small smoke training run
- verify fallback path and template usage metrics

### Phase 4

- real training run with `start-template-prob = 0.3`
- compare to current baseline

### Phase 5

- if successful, later expose richer template-specific diagnostics and UI controls

## When To Add Multiselect Back

Not in the first template experiment.

Recommended order:

1. validate one-play-per-possession behavior under richer start distributions
2. verify template-specific selector preferences
3. only then add a late multiselect schedule

Reason:

- otherwise we will not know whether the gain came from richer starts or from
  mid-possession play handoff dynamics

## Future Extension

If the template strategy works, later product features become much more natural:

- user-selectable starting template
- user-selectable play from that template
- template-conditioned playbook evaluation

That is not the immediate training objective, but it is a strong reason to
build the template system and the minimal authoring tool cleanly now.
