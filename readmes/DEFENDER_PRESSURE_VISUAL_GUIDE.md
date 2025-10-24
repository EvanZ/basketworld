# Defender Pressure Visual Guide

## How the Directional Check Works

### Concept
The ball handler is assumed to face the basket, and only defenders in their field of view (180° arc toward basket) can apply pressure to force turnovers.

### Visual Example

```
Basket at (0, 8)
═══

Ball Handler at (5, 8)
        🏀

Scenario 1: Defender IN FRONT (can apply pressure)
        
Basket ═══ ← → 🛡️ ← → 🏀
(0,8)        (3,8)    (5,8)

✅ Defender is between handler and basket
✅ cos(angle) > 0 (within 180° arc toward basket)
✅ Distance = 2 hexes → Turnover probability = 0.05 * exp(-1.0 * (2-1)) = 0.0184 (1.84%)


Scenario 2: Defender BEHIND (no pressure)

Basket ═══ ← → 🏀 ← → 🛡️
(0,8)        (5,8)    (7,8)

❌ Defender is behind the handler (away from basket)
❌ cos(angle) < 0 (outside 180° arc)
❌ No turnover check performed


Scenario 3: Defender to the SIDE (can apply pressure if within arc)

Basket ═══ ← → 🏀
(0,8)        (5,8)
             
             🛡️
           (5,10)

The angle from handler→defender vs handler→basket determines if pressure applies:
- If angle ≤ 90°: ✅ Pressure applies
- If angle > 90°: ❌ No pressure
```

## Exponential Decay Formula

```
turnover_probability = base_chance * exp(-lambda * max(0, distance - 1))
```

**Note:** Distance is measured in hexes. Minimum distance is 1 (adjacent). The formula uses `(distance - 1)` so that adjacent defenders (distance=1) get the full baseline probability.

### Example with defaults (base_chance=0.05, lambda=1.0):

| Distance | Formula | Probability | Percentage |
|----------|---------|-------------|------------|
| 1 (adjacent) | 0.05 * exp(0) | 0.0500 | 5.00% |
| 2 | 0.05 * exp(-1) | 0.0184 | 1.84% |
| 3 | 0.05 * exp(-2) | 0.0068 | 0.68% |
| 4 | 0.05 * exp(-3) | 0.0025 | 0.25% |

### Effect of Lambda Parameter:

**Lambda = 0.5 (slower decay)**
| Distance | Probability |
|----------|-------------|
| 1 (adjacent) | 5.00% |
| 2 | 3.03% |
| 3 | 1.84% |
| 4 | 1.11% |

**Lambda = 2.0 (faster decay)**
| Distance | Probability |
|----------|-------------|
| 1 (adjacent) | 5.00% |
| 2 | 0.68% |
| 3 | 0.09% |
| 4 | 0.01% |

## Implementation Details

### Step-by-Step Process:

1. **Get ball handler position** and calculate direction to basket
   ```python
   basket_vector = (basket.x - handler.x, basket.y - handler.y)
   ```

2. **For each defender within pressure distance:**
   ```python
   defender_vector = (defender.x - handler.x, defender.y - handler.y)
   ```

3. **Calculate angle using dot product:**
   ```python
   cos_angle = dot(defender_vector, basket_vector) / (||defender|| * ||basket||)
   ```

4. **Check if defender is in front:**
   ```python
   if cos_angle >= 0:  # Angle within [-90°, 90°]
       # Defender can apply pressure
   ```

5. **Calculate decay probability:**
   ```python
   prob = base_chance * exp(-lambda * max(0, distance - 1))
   ```
   Note: `(distance - 1)` ensures adjacent defenders (distance=1) get full baseline probability

6. **Roll for turnover:**
   ```python
   if random() < prob:
       # Turnover occurs, defender steals ball
   ```

## Why This Is More Realistic

### Before:
- ❌ Hard threshold: defender at distance 1 = 5% chance, distance 2 = 0% chance
- ❌ Defender behind player could force turnover
- ❌ Binary "in range" vs "out of range"

### After:
- ✅ Smooth probability decay with distance
- ✅ Only defenders that ball handler can see apply pressure
- ✅ More realistic basketball simulation where:
  - Ball handlers protect the ball from defenders in front
  - Defenders behind are not a threat
  - Closer defenders are more dangerous, but gradually

