# Court and geometry

BasketWorld stores player locations in axial hex coordinates \((q,r)\), while
the finite court is enumerated from a rectangular odd-row-offset grid. The
conversion makes neighbor movement and distance calculations simple without
giving up a rectangular array for court construction and rendering.

## Offset and axial coordinates

For offset column `col` and row `row`, BasketWorld uses:

\[
q = \mathrm{col} - \frac{\mathrm{row} - (\mathrm{row}\bmod 2)}{2},
\qquad
r = \mathrm{row}.
\]

The inverse conversion is:

\[
\mathrm{col} = q + \frac{r - (r\bmod 2)}{2},
\qquad
\mathrm{row} = r.
\]

The six axial movement deltas correspond to east, north-east, north-west,
west, south-west, and south-east.

## Hex distance

Axial coordinates map to cube coordinates as:

\[
(x,y,z) = (q,\,-q-r,\,r).
\]

The distance between two cells is:

\[
d(a,b) =
\frac{
  |q_a-q_b| +
  |(q_a+r_a)-(q_b+r_b)| +
  |r_a-r_b|
}{2}.
\]

The JAX static state precomputes a dense cell-to-cell distance matrix. Runtime
physics can therefore gather distances without Python loops or repeated
coordinate conversion.

## Cartesian projection

Angles and point-to-line distances need an ordinary Euclidean embedding. For
axial deltas, the JAX kernel uses:

\[
x = \sqrt{3}q + \frac{\sqrt{3}}{2}r,
\qquad
y = \frac{3}{2}r.
\]

Dividing Euclidean length by \(\sqrt{3}\) expresses it in approximately
hex-step units. Hex distance remains the authority for discrete reach and rule
tests; Cartesian geometry is used for arcs, alignment, and perpendicular
distance.

## Hoop and valid cells

The hoop is placed at the left edge of the offset court and vertically centered:

```text
basket_col = 0
basket_row = court_rows // 2
```

The resulting offset cell is converted to axial coordinates. Every legal court
cell, movement mask, lane mask, and shot classification is compiled into
lookup arrays in the same cell order.

When dunks are disabled, the basket cell cannot be entered. When dunks are
enabled, occupying the basket cell creates the distance-zero dunk opportunity.

## Three-point geometry

Three-point status is a cell mask, not just a raw hex-distance threshold. Court
construction projects each cell into Cartesian space and compares it with the
configured arc radius. An optional short-distance band produces corner-like
straight sections.

At runtime the mask determines shot value—two or three points—while the
continuous projected distance helps interpolate the base make probability.
Consequently, shot value and probability use related but distinct geometric
signals.

## Lane geometry

Offensive and defensive lane masks are precomputed from the configured lane
width and height. The step function only needs to look up whether each
position is in the corresponding mask and update its per-player counter.

Defensive lane occupancy becomes illegal only when the defender is not
guarding an offensive player within `defender_guard_distance`. Offensive lane
semantics distinguish the ball holder from other offensive players at the
threshold; see [Movement and rules](../physics/movement-rules.md).

## Spawn geometry

Random resets first sample distinct offense cells that satisfy configured
minimum and optional maximum hoop distance. Defenders are then matched to
offensive players and placed on available cells, preferring legal cells nearer
the corresponding offensive player while respecting court and spawn
constraints.

If the strict candidate set is too small, reset logic broadens the candidates
in stages rather than failing immediately. Start templates provide a separate,
explicit reset distribution when a research run needs reproducible formations.
