# MuJoCo Rebound Physics Prototype

This is the Milestone 1 sandbox for generating rebound data from a small MuJoCo scene. It is intentionally standalone from the JAX training runtime.

## Install

MuJoCo is optional for the main Basketworld app. Install it only when running this sidequest:

```bash
.env/bin/pip install mujoco
```

## Run

```bash
.env/bin/python analytics/rebound_physics/run_rebound_physics.py   --samples 200   --seed 0   --shot-x 4.0   --shot-y -4.5   --out-dir analytics/rebound_physics/outputs
```

## Shot Generation Modes

The prototype supports two shot-generation modes:

- `target_noise`: sample a noisy target near the rim, then analytically solve the launch velocity needed to hit that target. This is the original mode.
- `release_noise`: solve an ideal shot to a configured aim point, then perturb the release speed and release angles. Jumpers use a downward rim-entry angle. Bank layups instead use `--layup-board-impact-angle-degrees`, which controls whether the ball is rising, flat, or descending when it reaches the glass. Bank layups default to the upper corner of the regulation backboard target box, with a configurable reflection target available for experiments. This preserves the target-based mode while adding a more shooter-error-like mode.

Example release-noise run:

```bash
.env/bin/python analytics/rebound_physics/run_rebound_physics.py   --shot-model release_noise   --samples 200   --release-speed-noise-std 0.025   --release-lateral-angle-std-degrees 1.0   --release-vertical-angle-std-degrees 1.0
```

### Single-Location Shot-Type Profiles

`run_rebound_physics.py` also supports the same explicit shot-type profiles used by the dataset sweep, which makes it useful for calibrating one shot location before regenerating the full rebound table.

- `--shot-type jumper`: uses the base `--shot-z`, `--shot-model`, and low-level sampling flags.
- `--shot-type finger_roll`: uses a direct-to-rim close-finish profile (`release_noise`, rim target, descending target angle). This is intended for the centerline hex directly in front of the rim, where a bank-layup target is not appropriate. Tune it with the `--finger-roll-*` flags.
- `--shot-type layup`: uses the calibrated bank-layup profile (`release_noise`, higher release point, board-impact angle, larger release noise). By default it aims at the side-aware upper corner of the regulation target box: top-right for a right-side layup and top-left for a left-side layup. `--layup-target-kind backboard_reflection` is still available for mirror-geometry experiments; it reflects the horizontal path through the glass plane while keeping the vertical aim point on the regulation box above the rim. Tune it with the `--layup-*` flags.
- `--shot-type dunk`: uses the dunk profile (`target_noise`, near-rim release height, short flight time, small target noise). Tune it with the `--dunk-*` flags.
- `--shot-type custom`: equivalent to using only the base low-level sampler flags, useful when testing a one-off shot model.

Example one-hex layup calibration with heatmaps and a GIF:

```bash
.env/bin/python analytics/rebound_physics/run_rebound_physics.py \
  --shot-type layup \
  --shot-x 0.5 \
  --shot-y -0.5 \
  --samples 500 \
  --make-gif \
  --gif-trajectory-count 8
```

Example dunk profile smoke run:

```bash
.env/bin/python analytics/rebound_physics/run_rebound_physics.py \
  --shot-type dunk \
  --shot-x 0.0 \
  --shot-y 0.0 \
  --samples 200 \
  --make-gif
```

Useful plotting controls:

```bash
.env/bin/python analytics/rebound_physics/run_rebound_physics.py   --samples 500   --trajectory-stride 8   --max-plot-trajectories 60
```

For high-make shot profiles, the single-location script can also target a fixed number of misses instead of a fixed number of attempts. This is useful when calibrating layup/dunk rebound distributions or GIFs from missed shots:

```bash
.env/bin/python analytics/rebound_physics/run_rebound_physics.py \
  --shot-type layup \
  --shot-x 0.5 \
  --shot-y -0.5 \
  --misses 250 \
  --miss-target landing \
  --max-attempts 50000
```

`--miss-target landing` counts only misses with valid floor landings, `--miss-target catch` counts valid catch-height intercepts that are not behind the backboard, and `--miss-target any` counts every miss. Leave `--misses 0` to use the fixed `--samples` behavior.

## Physics Parameters

Distances are meters, time is seconds, velocities are meters/second, and spin values are converted to MuJoCo angular velocity components. The current CLI exposes the parameters most useful for calibration; additional fixed geometry defaults live in `PhysicsConfig`.

### Sampling And Shot Setup


| Parameter                                                                                                       | Role                                                                                                                                                                                                                                                                                                                     |
| --------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `--samples`                                                                                                     | Number of independent shots to simulate in fixed-attempt mode. More samples reduce heatmap noise and improve aggregate rebound statistics.                                                                                                                                                                               |
| `--misses` / `--target-misses`                                                                                  | If positive, switches `run_rebound_physics.py` to target-miss mode and runs until this many usable misses are collected.                                                                                                                                                                                                 |
| `--miss-target`                                                                                                 | Defines what counts toward `--misses`: `landing`, `catch`, or `any`.                                                                                                                                                                                                                                                     |
| `--max-attempts`                                                                                                | Safety cap for target-miss mode. `0` disables the cap.                                                                                                                                                                                                                                                                   |
| `--seed`                                                                                                        | Random seed for reproducible shot/noise sampling.                                                                                                                                                                                                                                                                        |
| `--shot-x`, `--shot-y`, `--shot-z`                                                                              | Initial ball release position. Changing this moves the shooter; it does not directly change the target unless the shot model computes a new ideal trajectory from that position.                                                                                                                                         |
| `--three-point-distance`, `--three-point-short-distance`, `--three-point-radius-meters`, `--meters-per-bw-unit` | Basketworld 3pt geometry used for landing heatmap overlays and dataset cell-to-meter conversion. If `--meters-per-bw-unit` is omitted, the scale is calibrated so the arc apex equals `--three-point-radius-meters` meters. The default is `7.24m`, the NBA max 3pt distance.                                            |
| `--draw-three-point-line` / `--no-draw-three-point-line`                                                        | Enables or disables the 3pt overlay on `rebound_physics_landing_heatmap.png`, `rebound_physics_missed_landing_heatmap.png`, `rebound_physics_missed_catch_heatmap.png`, and `rebound_physics_rim_outcomes.png`. Landing/catch heatmaps auto-expand their plot bounds to include all plotted points plus the 3pt context. |
| `--catch-height`                                                                                                | Height in meters used to estimate the in-air missed-shot rebound catch/intercept location. The simulator linearly interpolates the first descending trajectory crossing of this height; lower `--trajectory-stride` improves this diagnostic. Default is `2.6m`.                                                         |
| `--shot-model`                                                                                                  | Chooses the shot generation model. `target_noise` perturbs the target point near the rim. `release_noise` solves an ideal shot first, then perturbs release speed and angles.                                                                                                                                            |
| `--layup-target-kind`                                                                                           | Backboard aim point used by layup profiles in `run_rebound_physics.py` and `run_rebound_dataset.py`. Default is `backboard_box_upper_corner`, matching the common bank-layup sweet spot. Use `backboard_reflection` to test horizontal mirror geometry while keeping a realistic above-rim box target height.            |
| `--layup-board-impact-angle-degrees`                                                                            | Active bank-layup trajectory control. Positive values mean the ball is still rising at glass contact, `0` means near-apex contact, and negative values mean descending. This replaced the old jumper-style layup entry-angle behavior because realistic bank layups often hit the board before descending.               |
| `--finger-roll-shot-z`                                                                                          | Release height for the direct-to-rim close-finish profile used by `--shot-type finger_roll` and by centerline distance-1 cells in the dataset sweep.                                                                                                                                                                     |
| `--finger-roll-target-vertical-angle-degrees`                                                                   | Vertical angle at the rim target for finger rolls. Negative values mean the ball is descending through the rim plane.                                                                                                                                                                                                    |
| `--finger-roll-release-speed-noise-std`                                                                         | Fractional speed noise for finger rolls. Increase this to lower close-finish make rate without changing bank-layup calibration.                                                                                                                                                                                          |
| `--finger-roll-release-lateral-angle-std-degrees`, `--finger-roll-release-vertical-angle-std-degrees`           | Directional release noise for finger rolls. These control side and short/long misses for the front-of-rim close-finish profile.                                                                                                                                                                                          |


### Target-Noise Shot Model


| Parameter                                | Role                                                                                                                                     |
| ---------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| `--target-error-x-std`                   | Standard deviation of left/right target error around rim center. Larger values create wider misses across the rim plane.                 |
| `--target-error-y-std`                   | Standard deviation of short/long target error around rim center. This strongly affects front-rim, back-rim, and backboard contact rates. |
| `--target-error-z-std`                   | Standard deviation of target height error around rim height. Larger values change arc/rim contact behavior and make rate.                |
| `--flight-time-mean`                     | Mean time used by the analytical velocity solve from release point to noisy target. Higher values create slower, higher-arc shots.       |
| `--flight-time-std`                      | Shot-to-shot variation in solved flight time.                                                                                            |
| `--flight-time-min`, `--flight-time-max` | Clamps sampled flight time to avoid unrealistic line drives or overly high floaters.                                                     |


### Release-Noise Shot Model


| Parameter                                                | Role                                                                                                                                                                          |
| -------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--entry-angle-degrees`                                  | Mean downward entry angle used to solve the ideal jumper to rim center. Higher values create steeper arcs. This remains active for jumpers but is deprecated for bank layups. |
| `--entry-angle-std-degrees`                              | Shot-to-shot variation in the ideal entry angle before release perturbations.                                                                                                 |
| `--entry-angle-min-degrees`, `--entry-angle-max-degrees` | Bounds the ideal entry angle.                                                                                                                                                 |
| `--release-speed-noise-std`                              | Fractional noise applied to ideal release speed. For example, `0.01` is roughly 1% speed noise. Lowering this generally increases make rate.                                  |
| `--release-lateral-angle-std-degrees`                    | Left/right angular error applied to the release direction. This controls side misses.                                                                                         |
| `--release-vertical-angle-std-degrees`                   | Up/down angular error applied to the release direction. This controls short/long and arc errors.                                                                              |


### Spin


| Parameter         | Role                                                                                                                                                                                                                           |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `--backspin-mean` | Mean initial back/top spin around the shot-relative lateral axis. Positive keeps the existing backspin convention; negative produces topspin. Because it is shot-relative, the same value works from either side of the floor. |
| `--backspin-std`  | Shot-to-shot variation in back/top spin.                                                                                                                                                                                       |
| `--sidespin-mean` | Mean initial side spin / English around the vertical axis. Use a nonzero value when you want side spin to be clearly active in calibration runs.                                                                               |
| `--sidespin-std`  | Shot-to-shot variation in vertical-axis side spin. Larger values create more asymmetric bounces after rim, board, or floor contact.                                                                                            |


The GIF spin inset is a side-view diagnostic, so it is easiest to read for back/top spin. Vertical-axis side spin is still active in the physics, but a top-view diagnostic is a better visualization if you want to inspect side English directly.

### Scene Geometry And Make Logic


| `PhysicsConfig` field                                                                                    | Role                                                                                                                                                                                                                                                                                                                                        |
| -------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `floor_size`                                                                                             | Width/depth of the square MuJoCo floor used for diagnostics and landing capture.                                                                                                                                                                                                                                                            |
| `net_catch_made`                                                                                         | Effective net model. When enabled, made shots lose horizontal velocity after the made rim-plane crossing and drop vertically instead of continuing into the backboard/floor.                                                                                                                                                                |
| `net_downward_speed`                                                                                     | Downward velocity assigned by the effective net after a made shot.                                                                                                                                                                                                                                                                          |
| `rim_height`                                                                                             | Rim center height. NBA regulation is about 3.05 m.                                                                                                                                                                                                                                                                                          |
| `rim_radius`                                                                                             | Inner rim radius approximation. Regulation rim diameter is about 0.457 m, so radius is about 0.2286 m.                                                                                                                                                                                                                                      |
| `rim_tube_radius`                                                                                        | Thickness of the capsule segments used to approximate the rim.                                                                                                                                                                                                                                                                              |
| `rim_segments`                                                                                           | Number of capsule segments used to approximate the circular rim. More segments are smoother but slightly heavier.                                                                                                                                                                                                                           |
| `make_radius`                                                                                            | Horizontal rim-plane threshold used to classify a downward crossing as a made shot. This is intentionally smaller than rim radius because the ball has radius.                                                                                                                                                                              |
| `ball_radius`                                                                                            | Basketball radius. Regulation size 7 is about 0.121 m.                                                                                                                                                                                                                                                                                      |
| `ball_mass`                                                                                              | Basketball mass used by MuJoCo dynamics.                                                                                                                                                                                                                                                                                                    |
| `backboard_y`, `backboard_center_z_offset`, `backboard_width`, `backboard_height`, `backboard_thickness` | Backboard placement and dimensions. `backboard_center_z_offset` defaults to 0.381m, matching a regulation 42-inch backboard centered 15 inches above the rim. These affect bank shots and long rebounds after board contact.                                                                                                                |
| `backboard_box_width`                                                                                    | Width of the regulation backboard target box. The default is 0.6096m, corresponding to 24 inches. This is available for diagnostic upper-corner target tests.                                                                                                                                                                               |
| `backboard_box_center_z_offset`                                                                          | Height of the backboard target-box center above rim height. The default is 0.2286m, corresponding to the center of an 18-inch-high regulation target box whose bottom edge is at rim height. Bank-layup targets use these box dimensions directly, including the reflection target height, so they remain important calibration references. |
| `gravity`                                                                                                | Gravitational acceleration used by both analytical shot solving and MuJoCo simulation.                                                                                                                                                                                                                                                      |


### Contact And Bounce


| Parameter                   | Role                                                                                                                                                               |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `--contact-timeconst`       | MuJoCo `solref` time constant for contacts. It affects how quickly contacts resolve. In practice it interacts with damping to determine bounce feel and stability. |
| `--contact-dampratio`       | MuJoCo `solref` damping ratio for contacts. Lower values are bouncier; higher values absorb more energy. This is the main bounce calibration knob.                 |
| `--contact-solimp-width`    | MuJoCo `solimp` impedance width. Controls the softness transition around contacts.                                                                                 |
| `--contact-solimp-midpoint` | MuJoCo `solimp` midpoint. Changes where the impedance transition occurs.                                                                                           |
| `--contact-solimp-power`    | MuJoCo `solimp` exponent. Changes the shape of the impedance curve.                                                                                                |


### Rim And Backboard Contact Overrides

The global contact settings above are still the default for every contact surface. The rim and backboard can now override those values independently, which is useful when the floor bounce calibration looks right but rim/backboard misses are too lively or too dead.


| Parameter                             | Role                                                                                                                                                             |
| ------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--rim-contact-timeconst`             | Optional MuJoCo `solref` time constant for rim contacts only. If omitted, rim contacts use `--contact-timeconst`.                                                |
| `--rim-contact-dampratio`             | Optional MuJoCo `solref` damping ratio for rim contacts only. Lower values make rim bounces livelier; higher values absorb more energy.                          |
| `--rim-contact-solimp-width`          | Optional MuJoCo `solimp` width for rim contacts only.                                                                                                            |
| `--rim-contact-solimp-midpoint`       | Optional MuJoCo `solimp` midpoint for rim contacts only.                                                                                                         |
| `--rim-contact-solimp-power`          | Optional MuJoCo `solimp` power for rim contacts only.                                                                                                            |
| `--backboard-contact-timeconst`       | Optional MuJoCo `solref` time constant for backboard contacts only. If omitted, backboard contacts use `--contact-timeconst`.                                    |
| `--backboard-contact-dampratio`       | Optional MuJoCo `solref` damping ratio for backboard contacts only. Lower values make backboard rebounds livelier; higher values reduce long backboard rebounds. |
| `--backboard-contact-solimp-width`    | Optional MuJoCo `solimp` width for backboard contacts only.                                                                                                      |
| `--backboard-contact-solimp-midpoint` | Optional MuJoCo `solimp` midpoint for backboard contacts only.                                                                                                   |
| `--backboard-contact-solimp-power`    | Optional MuJoCo `solimp` power for backboard contacts only.                                                                                                      |


Example: preserve the floor calibration while adding softer rim/backboard contact behavior:

```bash
.env/bin/python analytics/rebound_physics/run_rebound_physics.py \
  --shot-model release_noise \
  --shot-x 0.0 \
  --shot-y -7.5 \
  --samples 500 \
  --contact-timeconst 0.030 \
  --contact-dampratio 0.060 \
  --rim-contact-dampratio 0.18 \
  --backboard-contact-dampratio 0.22
```

### Behind-Backboard Diagnostics

`rebound_physics_summary.json` and `rebound_dataset_summary.json` include diagnostics for missed shots whose first floor landing is behind the backboard. The threshold is `PhysicsConfig.backboard_y` by default, so a miss is counted as behind the backboard when `landing_y > backboard_y`.

Key fields:


| Field                                  | Meaning                                                                                                                           |
| -------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| `behind_backboard_y_threshold`         | The y-coordinate threshold used for the diagnostic.                                                                               |
| `behind_backboard_miss_count`          | Number of missed shots with a valid landing behind the backboard.                                                                 |
| `behind_backboard_miss_rate`           | `behind_backboard_miss_count / missed_landing_count`.                                                                             |
| `behind_backboard_by_first_contact`    | Behind-backboard misses grouped by first contact surface, useful for separating clean over/long misses from rim or board bounces. |
| `behind_backboard_by_contact_sequence` | Behind-backboard misses grouped by full contact sequence.                                                                         |
| `behind_backboard_by_rim_outcome`      | Behind-backboard misses grouped by rim outcome label.                                                                             |


If top-of-key or wing misses are landing behind the board too often, check these fields first. If the cases are mostly `rim` or `backboard` contact sequences, tune rim/backboard damping before changing shot noise. If they are mostly clean misses, the issue is usually release-speed or vertical-angle noise rather than contact physics.

### Simulation And Output


| Parameter                                                | Role                                                                                                                                                                                                                                                                                          |
| -------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--duration`                                             | Maximum simulated time per shot. Increase this if balls are still bouncing when samples end.                                                                                                                                                                                                  |
| `--net-catch-made` / `--no-net-catch-made`               | Enables or disables the effective net behavior for made shots. Missed shots are unaffected.                                                                                                                                                                                                   |
| `--net-downward-speed`                                   | Downward velocity used by the effective net after a make.                                                                                                                                                                                                                                     |
| `--backboard-center-z-offset`                            | Backboard center height above rim height in meters. The default is 0.381m; lowering this can let too many high misses clear the board.                                                                                                                                                        |
| `--timestep`                                             | MuJoCo integration step. Smaller values improve contact fidelity but cost more CPU.                                                                                                                                                                                                           |
| `--trajectory-stride`                                    | Records every Nth simulation step into JSON/plots. Lower values preserve smoother trajectories, improve catch-height interpolation fidelity, and produce larger files.                                                                                                                        |
| `--max-plot-trajectories`                                | Maximum number of trajectories drawn in static trajectory plots.                                                                                                                                                                                                                              |
| `--make-gif`                                             | Enables animated GIF rendering.                                                                                                                                                                                                                                                               |
| `--gif-shot-index`                                       | Forces a specific shot index into the GIF. If omitted, a representative sample is selected.                                                                                                                                                                                                   |
| `--gif-fps`                                              | GIF playback frame rate.                                                                                                                                                                                                                                                                      |
| `--gif-max-frames`                                       | Maximum frames retained per trajectory in the GIF.                                                                                                                                                                                                                                            |
| `--gif-trajectory-count`                                 | Number of trajectories overlaid in the GIF.                                                                                                                                                                                                                                                   |
| `--gif-spin-mode`                                        | Controls the tiny in-scene ball orientation overlay in GIFs. Default `none` keeps the court view clean. Use `seam` or `markers` only when debugging the regulation-size ball directly in the 3D scene.                                                                                        |
| `--gif-spin-inset` / `--no-gif-spin-inset`               | Draws the zoomed side-view spin inset in GIF frames. This is enabled by default and is the readable spin diagnostic.                                                                                                                                                                          |
| `--gif-spin-primary-only` / `--no-gif-spin-primary-only` | Draws in-scene spin only on the first selected GIF trajectory by default, avoiding clutter in multi-shot GIFs. Disable it when explicitly studying multiple simultaneous spins.                                                                                                               |
| `--gif-rim-inset` / `--no-gif-rim-inset`                 | Draws rim/backboard diagnostic closeups in GIF frames: a lower-right top-down rim view plus a left-side frontal backboard/rim view. They show rim/make thresholds, backboard context, recent ball trail, current ball footprint, and projected spin seams when orientation data is available. |
| `--gif-spin-alpha`                                       | Opacity of the spin overlay/inset seam. Lower values make the seam less intrusive.                                                                                                                                                                                                            |
| `--out-dir`                                              | Output directory for JSON, plots, and GIFs.                                                                                                                                                                                                                                                   |
| `--no-plot`                                              | Skips plot/GIF generation and writes only JSON summaries.                                                                                                                                                                                                                                     |


## Bounce Calibration

Contact bounciness is controlled through MuJoCo contact solver settings on `PhysicsConfig`, especially `contact_solref_dampratio`. Lower damping ratios produce more rebound. The standalone calibration runner drops a stationary ball and reports the first rebound peak.

Example: tune toward a ball that rebounds about 4 ft when dropped from 6 ft:

```bash
.env/bin/python analytics/rebound_physics/run_bounce_calibration.py \
  --drop-height-ft 6.0 \
  --target-bounce-height-ft 4.0 \
  --timeconst-grid 0.012,0.015,0.018,0.020,0.024,0.030 \
  --dampratio-grid 0.035,0.040,0.045,0.050,0.055,0.060,0.070,0.080,0.090,0.100,0.110
```

A local smoke sweep found `contact_solref_timeconst=0.030` and `contact_solref_dampratio=0.060` to rebound about 4.1 ft from a 6 ft drop with the current ball/floor model. Use those values as a starting point, not as final calibration.

The script writes `bounce_calibration_summary.json` and prints the closest solver time constant / damping ratio pairs. Heights are reported as floor-to-bottom-of-ball distances, which keeps the target intuitive for drop tests.

## Milestone 3 Dataset Sweep

`run_rebound_dataset.py` turns the MuJoCo shot simulator into a Basketworld-ready rebound dataset. It maps the current 9x8 half-court hex geometry into MuJoCo meters, uses left/right symmetry to simulate only canonical shot cells, filters made baskets out of the fitted rebound dataset, and writes transition matrices that can be used for fast BW inference.

The dataset now keeps separate rebound models for explicit shot types:

- `dunk`: the basket hex.
- `layup`: non-basket cells within `--layup-max-distance-hex`, default one hex from the basket.
- `jumper`: all other shot cells.

This avoids forcing basket-adjacent layups and long jumpers to share one landing table. The pooled transition arrays are still written for diagnostics/backward compatibility, but BW inference should prefer the per-shot-type tables under `shot_type_models/`.

The current layup defaults are intentionally much noisier than jumper defaults, calibrated in a 250-shot smoke sweep to land near a 60% make rate for the canonical one-hex layup cell. These are not final basketball constants; they are explicit knobs for Milestone 3 data generation.

Example smoke run:

```bash
.env/bin/python analytics/rebound_physics/run_rebound_dataset.py \
  --samples-per-cell 25 \
  --max-shot-cells 4 \
  --shot-model release_noise \
  --release-speed-noise-std 0.01 \
  --release-lateral-angle-std-degrees 0.5 \
  --release-vertical-angle-std-degrees 0.5 \
  --out-dir analytics/rebound_physics/outputs/dataset_smoke
```

Target-miss sampling is available when high make-rate cells need a stable miss distribution. In this mode, `--samples-per-cell` no longer controls attempts per cell; each canonical shot cell runs until `--misses-per-cell` target misses are accumulated. Use `--max-attempts-per-cell` as a safety cap when calibrating dunks or layups. The default `--miss-target landing` counts only missed shots with valid floor landing cells, which matches the landing transition model. Use `--miss-target catch` for the catch-height model or `--miss-target any` for raw missed-shot counts.

Example layup/dunk-friendly sweep with at least 250 landing misses per cell:

```bash
.env/bin/python analytics/rebound_physics/run_rebound_dataset.py \
  --misses-per-cell 250 \
  --miss-target landing \
  --max-attempts-per-cell 50000 \
  --include-dunk-cell \
  --out-dir analytics/rebound_physics/outputs/dataset_9x8_target_misses
```

Example full 9x8 sweep:

```bash
.env/bin/python analytics/rebound_physics/run_rebound_dataset.py \
  --samples-per-cell 500 \
  --court-rows 9 \
  --court-cols 8 \
  --three-point-radius-meters 7.24 \
  --catch-height 2.6 \
  --include-dunk-cell \
  --layup-max-distance-hex 1.0 \
  --layup-shot-z 2.50 \
  --layup-entry-angle-degrees 58.0 \
  --layup-entry-angle-std-degrees 7.0 \
  --layup-release-speed-noise-std 0.075 \
  --layup-release-lateral-angle-std-degrees 3.5 \
  --layup-release-vertical-angle-std-degrees 3.5 \
  --dunk-shot-z 3.35 \
  --shot-model release_noise \
  --contact-timeconst 0.030 \
  --contact-dampratio 0.060 \
  --out-dir analytics/rebound_physics/outputs/dataset_9x8
```

Dataset outputs:

- `rebound_dataset_raw_samples.jsonl`: every simulated shot, including makes. This is diagnostic data.
- `rebound_dataset_missed_samples.jsonl`: missed shots with valid first floor contact only. This is the data intended for fitting rebound landings. Each record also includes `catch_*` fields when present.
- `rebound_dataset_catch_samples.jsonl`: missed shots with valid catch-height targets, excluding catch points behind the backboard. This is the data intended for fitting the in-air catch target model.
- `rebound_dataset_summary.json`: court config, physics config, sampler config, sampling mode, shot-type sampler configs, make rates, landing rates, raw miss counts, catch-height rates, behind-backboard diagnostics, per-canonical-shot counts, `shot_type_counts`, and `shot_make_grid` / `shot_make_arrays` fields for attempts, makes, and FG% by shot cell.
- `rebound_canonical_shot_mapping.json`: mapping from requested/original shot cells to canonical right-side shot cells. At inference time, reflect the sampled rebound target back when `reflection_sign=-1`.
- `rebound_transition_counts.npy`: missed-shot landing counts with shape `[canonical_shot_cells, court_cells]`.
- `rebound_transition_probs.npy`: row-normalized landing probabilities.
- `rebound_transition_logits.npy`: log probabilities suitable for a lightweight categorical model or JAX lookup table.
- `rebound_transition_row_shot_cell_indices.npy`: canonical shot-cell index for each row of the transition arrays.
- `catch_model/rebound_transition_counts.npy`: missed-shot catch-height target counts with shape `[canonical_shot_cells, court_cells]`, excluding catch points behind the backboard.
- `catch_model/rebound_transition_probs.npy`: row-normalized catch-height target probabilities.
- `catch_model/rebound_transition_logits.npy`: log probabilities for the catch-height target model.
- `catch_model/rebound_transition_row_shot_cell_indices.npy`: canonical shot-cell rows for the catch-height target arrays.
- `shot_type_models/<shot_type>/rebound_transition_counts.npy`: missed-shot landing counts for one explicit shot type.
- `shot_type_models/<shot_type>/rebound_transition_probs.npy`: row-normalized landing probabilities for one explicit shot type.
- `shot_type_models/<shot_type>/rebound_transition_logits.npy`: log probabilities for one explicit shot type.
- `shot_type_models/<shot_type>/rebound_transition_row_shot_cell_indices.npy`: canonical shot-cell rows covered by that shot type.
- `shot_type_models/<shot_type>/catch_model/*.npy`: per-shot-type catch-height target transition arrays.
- `rebound_shot_fg_pct_heatmap.png`: single court heatmap of FG% by canonical shot location, labeled with make/attempt counts and the configured 3pt line.
- `heatmaps/shot_cell_q*_r*.png`: per-shot missed-rebound landing heatmaps on the BW hex court with the configured 3pt line overlaid.
- `catch_heatmaps/shot_cell_q*_r*.png`: per-shot missed-rebound catch-height target heatmaps on the BW hex court.

Coordinate convention:

- BW axial cells are converted to BW cartesian centers using the existing Basketworld hex geometry.
- MuJoCo `x` is lateral left/right relative to the rim.
- MuJoCo `y` is negative away from the rim. By default, `--meters-per-bw-unit` is derived from `--three-point-radius-meters 7.24`, so `4.25 * sqrt(3)` BW cartesian units maps to the NBA max 3pt distance. Pass `--meters-per-bw-unit 0.5` explicitly only for the older compact prototype scale.
- Canonicalization reflects negative lateral `x` to positive lateral `x`; it does not mirror samples into the dataset.

Generate a representative animated GIF:

```bash
.env/bin/python analytics/rebound_physics/run_rebound_physics.py \
  --samples 80 \
  --make-gif \
  --gif-max-frames 90 \
  --gif-trajectory-count 5
```

Outputs:

- `rebound_physics_samples.jsonl`: one trajectory summary per shot, including make/miss, rim-plane crossing point, contact sequence, contact points, landing point, and decimated trajectory points.
- `rebound_physics_summary.json`: aggregate make/contact/landing stats.
- `rebound_physics_landing_heatmap.png`: court-plane heatmap of first floor contacts for all shots, including made-shot post-basket floor contacts.
- `rebound_physics_missed_landing_heatmap.png`: court-plane heatmap of first floor contacts for missed shots only. This is the landing-model diagnostic to use when fitting miss-only rebound data.
- `rebound_physics_missed_catch_heatmap.png`: court-plane heatmap of the first descending missed-shot trajectory crossing at `--catch-height`. This approximates where an in-air rebound could be caught before the ball reaches the floor.
- `rebound_physics_contact_heatmaps.png`: rim top-view contact heatmap and backboard face-view contact heatmap.
- `rebound_physics_rim_outcomes.png`: top-view rim-plane crossing plot showing makes, rim/backboard misses, and clean misses.
- `rebound_physics_side_trajectories.png`: side-view shot arcs using distance along the shot line vs height.
- `rebound_physics_shooter_view_trajectories.png`: shooter-perspective shot paths using lateral offset vs height.
- `rebound_physics_3d_scene.png`: 3D diagnostic view of floor bounds, backboard, rim, shot origin, and sampled trajectories.
- `rebound_physics_typical_shot.gif`: optional animated 3D view of one or more representative shots when `--make-gif` is enabled. Use `--gif-trajectory-count N` to overlay multiple sampled trajectories in the same GIF. By default it includes a spin inset plus top-down and frontal rim/backboard closeups; disable both rim closeups with `--no-gif-rim-inset`.

## Milestone 4 Table Fit

`run_rebound_fit.py` converts a completed Milestone 3 sweep into a lightweight table model for BW/JAX inference. It does not rerun MuJoCo. It reads `rebound_dataset_raw_samples.jsonl`, filters the records, applies optional pseudocount smoothing, and writes row-normalized transition arrays.

Default behavior fits the catch-height target model:

```bash
.env/bin/python analytics/rebound_physics/run_rebound_fit.py   --dataset-dir analytics/rebound_physics/outputs/dataset_9x8   --target-cell-field catch_cell_index   --filter-behind-backboard   --pseudocount 0.05   --out-dir analytics/rebound_physics/outputs/dataset_9x8/fitted_catch_model
```

Filtering rules:

- Made shots are always excluded from the rebound fit.
- Records missing the selected target cell are excluded.
- With `--filter-behind-backboard`, selected targets behind the backboard are excluded. For `catch_cell_index`, this uses `catch_behind_backboard` plus the target y-coordinate. For `landing_cell_index`, this uses `landing_y > behind_backboard_y_threshold`.
- The raw sweep remains unchanged, so FG%, rim/contact diagnostics, and behind-backboard rates can still be inspected later.

Fit outputs:

- `rebound_fit_samples.jsonl`: filtered records actually used by the fitted table model.
- `rebound_fit_summary.json`: source paths, filter counts, pseudocount, target field, shot-type counts, row coverage, and artifact paths.
- `rebound_transition_counts.npy`: smoothed transition counts with shape `[canonical_shot_cells, court_cells]`.
- `rebound_transition_probs.npy`: row-normalized target probabilities.
- `rebound_transition_logits.npy`: log probabilities for fast categorical sampling.
- `rebound_transition_row_shot_cell_indices.npy`: canonical shot-cell index for each row.
- `shot_type_models/<shot_type>/*.npy`: the same arrays split by `dunk`, `finger_roll`, `layup`, and `jumper`.
- `rebound_canonical_shot_mapping.json`: copied from the source dataset so inference can reflect canonical predictions back to the original side.

Use `--target-cell-field landing_cell_index` only if you specifically want first-floor-contact rebounds instead of in-air catch targets.

## Current Scope

The prototype models a ball, floor, backboard, configurable contact solver settings, and an approximate rim built from capsule segments. It samples shots with either noisy launch targets around the rim or noisy release perturbations from an ideal trajectory, then records makes, rim-plane crossings, first contact type, contact sequence, rim/backboard/floor contact points, max height, landing position, and decimated ball trajectories.

This is not yet a calibrated basketball simulator. The purpose is to produce a controllable dataset that can later be fit into a lightweight rebound model for Basketworld inference.