# Visual Guide: Court Coordinate Systems

## Three Layers of Coordinates

Your court uses three different coordinate systems that all represent the **same physical location**, just in different formats:

### Quick visual: 9×9 odd-r map

![9x9 odd-r grid showing offset (col,row) and axial (q,r) per hex](../docs/assets/odd_r_9x9_coordinates.png)

Each hex is labeled with its offset `(col,row)` on the first line and axial `(q,r)` on the second line. Rows increase downward; odd rows are shifted to the right (odd-r).

### Layer 1: Offset Coordinates (Human-Friendly)
This is what you see when rendered:
```
    col:  0   1   2   3  ...  14  15
row:
  0       .   .   .   .      .   .
  1         .   .   .   .      .   .
  2       .   .   .   .      .   .
  3         .   .   .   .      .   .
  ...
  8    [B] . - . - . - .  ...  -[x] .   ← Basket at (col=0, row=8)
  ...
  15        .   .   .   .      .   .
```

The **offset coordinate** basket is at `(0, 8)` - leftmost edge, vertical middle.

---

### Layer 2: Axial Coordinates (Math-Friendly)
The game logic works internally with axial coordinates:

```
For the same basket position:
  offset (col=0, row=8)
  
  Formula: q = col - (row - (row & 1)) // 2
           r = row
  
  q = 0 - (8 - 0) // 2 = 0 - 4 = -4
  r = 8
  
  Result: axial (-4, 8)
```

**Why the negative q?** Because in axial coordinates, the hex grid is rotated. The left side of the offset grid maps to negative q values.

---

### Layer 3: Normalized Observation (Network Input)
The observation vector normalizes to roughly [-1, 1]:

```
norm_den = max(court_width, court_height) = 16

q_normalized = -4 / 16 = -0.25
r_normalized =  8 / 16 = 0.50

Result in observation: [-0.25, 0.50]
```

This is what you see in the test output! ✓

---

## Coordinate Transformation Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                    COORDINATE TRANSFORMATION                        │
└─────────────────────────────────────────────────────────────────────┘

  PHYSICAL COURT (Offset)      GAME LOGIC (Axial)       NETWORK (Normalized)
  ═══════════════════════      ══════════════════       ═════════════════════
  
  col ∈ [0, 15]          →    q = f(col, row)    →    q_norm ∈ [-0.5, 0.9375]
  row ∈ [0, 15]               r = row                 r_norm ∈ [0.0, 0.9375]
  
  (0, 8) [Basket]        →    (-4, 8)            →    (-0.25, 0.50) ✓
  (7, 8) [Middle]        →    (3, 8)             →    (0.1875, 0.50)
  (15, 8) [Right edge]   →    (11, 8)            →    (0.6875, 0.50)
```

---

## Why Isn't the Basket at q = 0?

This is the key insight! Let's trace through several positions on row 8:

```
Offset (col, 8) → Axial (q, 8) → Normalized (q_norm, 0.5)
────────────────────────────────────────────────────────────
(0, 8)          → (-4, 8)       → (-0.25, 0.5)    ← Your basket!
(1, 8)          → (-3, 8)       → (-0.1875, 0.5)
(2, 8)          → (-2, 8)       → (-0.125, 0.5)
(3, 8)          → (-1, 8)       → (-0.0625, 0.5)
(4, 8)          → (0, 8)        → (0.0, 0.5)      ← Here's q=0!
(5, 8)          → (1, 8)        → (0.0625, 0.5)
...
(15, 8)         → (11, 8)       → (0.6875, 0.5)   ← Right edge
```

**So q = 0 in axial coordinates is NOT at the left edge!** It's 4 hexes to the right because of how the offset-to-axial conversion works.

---

## Understanding the Odd-R Offset System

The "odd-r offset" coordinate system alternates rows:

```
EVEN rows (0, 2, 4, 6, 8...):  Aligned normally
  col: 0   1   2   3   4   5
       ●───●───●───●───●───●

ODD rows (1, 3, 5, 7, 9...):   Shifted right by 0.5
       ●───●───●───●───●───●
  col: 0   1   2   3   4   5
```

The row 8 is EVEN, so it's aligned normally. But when we convert to axial:

```
# Row is even (8 & 1 = 0), so:
q = col - (8 - 0) // 2 = col - 4

# This shifts ALL q values left by 4
# Making the offset (0, 8) become axial (-4, 8)
```

---

## Verification: All Court Edges

Let's verify the corners to make sure our understanding is correct:

```
CORNER ANALYSIS (grid_size=16)
═════════════════════════════════════════════════════════════════

Position          Offset(col,row)  Axial(q,r)   Normalized
─────────────────────────────────────────────────────────────
Top-Left          (0, 0)           (0, 0)       (0.0, 0.0)
Top-Right         (15, 0)          (15, 0)      (0.9375, 0.0)
Bottom-Left       (0, 15)          (-8, 15)     (-0.5, 0.9375)
Bottom-Right      (15, 15)         (7, 15)      (0.4375, 0.9375)
Left-Middle       (0, 8)           (-4, 8)      (-0.25, 0.5)  ← BASKET
Right-Middle      (15, 8)          (11, 8)      (0.6875, 0.5)
```

Notice:
- **Top-left is (0,0) in axial** — No conversion needed there
- **Left-middle is (-4, 8)** — Shifted left by 4 because row 8 is even
- **Bottom-left is (-8, 15)** — Shifted left by 8 because row 15 is odd
- The asymmetry between left and right edges is due to the odd-r system

---

## The Big Picture

Your court layout with absolute coordinates looks like this:

```
                    Axial Space Visualization
                    (not to scale; showing q/r ranges)

            q increases →

        (-8,15) ∙ ∙ ∙ ∙ ∙ ∙ ∙ ∙ (7,15)    [Bottom edge]
           ∙                        ∙
           ∙                        ∙
           ∙                        ∙
      (-4,8)∙[B]                    ∙(11,8)    [Middle, Basket marked]
           ∙                        ∙
           ∙                        ∙
           ∙                        ∙
        (0,0) ∙ ∙ ∙ ∙ ∙ ∙ ∙ ∙ (15,0)    [Top edge]

r increases ↓
```

The basket **[B]** at offset (0, 8) becomes (-4, 8) in axial space because:
- Row 8 causes a **-4 offset** to q values
- Axial coordinates tilt the grid for mathematical convenience

---

## Why This Matters for Your Observations

In your new **absolute coordinate system**, the network sees:

```
All 6 players' positions in absolute axial coordinates
Example from test:
  [0.5625, 0.5, 0.3125, 0.25, 0.1875, 0.3125, ...]
   └─ Player 0 ─┘ └─ Player 1 ─┘ └─ Player 2 ─┘

Plus:
  Ball handler position: [0.1875, 0.3125]
  Basket position: [-0.25, 0.5]
  
Plus all the defensive information...
```

The network learns that:
- Positions near **negative q** are on the **left side of the court**
- Positions near **positive q** are on the **right side**
- The basket at **-0.25** is on the left edge
- This info helps it learn position-dependent strategies!
```

---

## Summary

| What | Value | In Normalized |
|---|---|---|
| Basket offset coords | (col=0, row=8) | — |
| Basket axial coords | (q=-4, r=8) | — |
| **Basket in observation** | **(-0.25, 0.5)** | ✓ This is what you see! |

The -0.25 is NOT a bug—it's mathematically correct! The odd-r offset system naturally produces negative q values for the left edge of the court. 🎯


