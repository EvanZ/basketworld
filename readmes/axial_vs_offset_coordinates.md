# Can you explain the difference between axial and offset coordinates for the hexagonal positions and how the math works?
---
Of course. This is a fantastic and fundamental question about working with hexagonal grids. It's a common point of confusion because the best way for a *human* to think about a grid isn't the best way for a *computer* to do math on it.

Think of it like this:
*   **Offset Coordinates** are like a street address (`123 Main Street`). It's easy for us to read and understand, especially on a rectangular map.
*   **Axial Coordinates** are like GPS coordinates (`40.7128° N, 74.0060° W`). They are perfect for precise calculations (distance, direction), but aren't very intuitive at a glance.

In our environment, we use **both** to get the best of both worlds. Let's break them down.

---

### 1. Offset Coordinates (The "Human-Friendly" System)

This is the most intuitive way to think about a hex grid. You imagine it as a standard square grid, but with every other row or column "pushed" or "offset" by half a tile. This is exactly what you see in our ASCII and visual renderers.

**How it Works:**
*   It uses a simple `(column, row)` system, just like a spreadsheet.
*   It's perfect for defining rectangular boundaries, which is why we use it for our court shape (`court_width`, `court_height`).
*   The "gotcha" is that the math for calculating distances or finding neighbors is surprisingly awkward because of the offset. Moving "up" might mean changing your row, but moving "northeast" might involve changing both your column and your row, and the change depends on whether you're in an even or odd row/column.

This is why we only use it for defining the shape of the court and for spawning players in a human-readable way (e.g., "top half of the court").

![Offset Grid](https://www.redblobgames.com/grids/hexagons/offset-doubled-q.png)
*(Image credit: Red Blob Games, a phenomenal resource on this topic)*

---

### 2. Axial Coordinates (The "Math-Friendly" System)

This is the system the environment uses for all its internal logic: player movement, distance calculations, and storing positions.

**How it Works:**
*   Instead of `(column, row)`, it uses a `(q, r)` coordinate pair.
*   These coordinates correspond to two of three axes on a "cubical" grid that is projected onto the 2D plane. The axes are 120 degrees apart.
*   The magic rule of this system is that for any hexagon, the sum of its three coordinates (`q`, `r`, and a theoretical `s`) is always zero: `q + r + s = 0`.
*   Because of this rule, we never need to store `s`. We can always calculate it (`s = -q - r`). By storing just `(q, r)`, we have all the information we need.

**Why is this so good for math?**
*   **Neighbors:** Finding a neighbor is always a matter of adding a constant vector. For example, moving East is always `(q+1, r)`. There are no "if you're in an even row" checks.
*   **Distance:** The distance between two hexagons in axial coordinates has a beautiful, simple formula:
    `distance = (abs(q1 - q2) + abs(r1 - r2) + abs(s1 - s2)) / 2`
    (Where `s1 = -q1-r1` and `s2 = -q2-r2`). This is what our `_hex_distance` method uses.

![Axial Grid](https://www.redblobgames.com/grids/hexagons/axial-q-r.png)
*(Image credit: Red Blob Games)*

---

### How the Conversion Math Works

To get the best of both worlds, we need to convert between the two systems. Here's how the functions in our code work. We are using a "pointy-topped" layout, and our offset system is "odd-q" (meaning odd-numbered columns are shifted down).

#### `_offset_to_axial(col, row)`

This function takes a human-friendly `(col, row)` and turns it into a math-friendly `(q, r)`.

```python
def _offset_to_axial(self, col: int, row: int) -> Tuple[int, int]:
    """Converts odd-q offset coordinates to axial coordinates."""
    q = col
    r = row - (col - (col & 1)) // 2
    return q, r
```
*   `q = col`: The `q` axis in an "odd-q" system conveniently aligns perfectly with the columns. So, `q` is just `col`.
*   `r = row - (col - (col & 1)) // 2`: This is the clever part.
    *   `(col & 1)` is a fast, bitwise way to check if `col` is odd. It returns `1` if odd, `0` if even.
    *   `(col - (col & 1))` effectively rounds the column number *down* to the nearest even number.
    *   `// 2`: We divide by two.
    *   **In English:** This line says, "The `r` coordinate is the row, but for every two columns we move to the right, we need to shift our `r` coordinate up by one to compensate for the grid's vertical stagger."

#### `_axial_to_offset(q, r)`

This function takes a math-friendly `(q, r)` and turns it back into a human-friendly `(col, row)`.

```python
def _axial_to_offset(self, q: int, r: int) -> Tuple[int, int]:
    """Converts axial coordinates to odd-q offset coordinates."""
    col = q
    row = r + (q - (q & 1)) // 2
    return col, row
```
*   `col = q`: Again, the column is simply the `q` value.
*   `row = r + (q - (q & 1)) // 2`: This is the exact reverse of the logic above. It adds back the offset to the `r` coordinate to find the correct row number.

### Summary

| Coordinate System | Best For                               | Pros                                       | Cons                                       |
| ----------------- | -------------------------------------- | ------------------------------------------ | ------------------------------------------ |
| **Offset** (`col`, `row`) | Defining rectangular maps, Spawning, Rendering | Intuitive for humans, Easy for rectangular shapes | Awkward math for distance and neighbors      |
| **Axial** (`q`, `r`)    | Game Logic, Movement, Distance         | Elegant and fast math, Simple neighbor-finding | Not intuitive for rectangular shapes |

We use **Offset** to define the court's boundaries and to render it, but we immediately convert to **Axial** for storing positions and performing all in-game calculations.