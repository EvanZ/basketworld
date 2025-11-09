# Defensive Three-Second Rule Implementation Improvement Plan

## Executive Summary

The current implementation of the defensive three-second (illegal defense) rule in BasketWorld is **overly simplistic** compared to NBA rules. This plan outlines a comprehensive improvement strategy to bring the simulation much closer to actual NBA enforcement.

**Key Insight**: The NBA rule is highly contextual and dynamic—it's not simply "can't be in the lane for 3 seconds." It depends on:
- Whether the defender is **actively guarding** an opponent
- Whether that opponent is nearby or far away
- Whether an opponent is in the **act of shooting**
- Whether there's a **loss of team control** (turnover, etc.)
- Whether the defender is in the process of **leaving the lane**
- Many exception cases that suspend the count

---

## Current Implementation Analysis

### What We Do Well ✅
1. We track defenders in a designated lane area
2. We count consecutive steps in the lane
3. We penalize violations with scoring
4. We end the episode on violation
5. We mask the NOOP action to force defenders out

### What's Missing or Incomplete ❌

| NBA Rule Aspect | Current Status | Gap |
|---|---|---|
| **Actively Guarding Requirement** | ❌ Not implemented | Defenders can be anywhere in lane without guarding anyone |
| **Arm's Length Distance Check** | ❌ Not implemented | No concept of "actively guarding position" |
| **Defensive Team Control Loss** | ❌ Not implemented | No suspension when ball is turned over |
| **Shooting Exception** | ❌ Not implemented | Count doesn't suspend during shot attempts |
| **In-Motion Exit (Imminent)** | ❌ Not implemented | No grace period for defenders leaving lane |
| **Teammate Guarding Ball Handler** | ❌ Partial | We don't enable multiple defenders to satisfy the rule |
| **Double-Team Allowance** | ❌ Not implemented | Defenders can't coordinate to double-team |
| **Technical Foul Penalty** | ⚠️ Partial | Awards point but doesn't reset shot clock correctly |
| **Successful Shot Override** | ❌ Not implemented | Violations during successful shots should be ignored |
| **Per-Violation Precision** | ❌ Not implemented | Violations end episode; NBA would restart with ball at FT line |

---

## Proposed Improvement Roadmap

### Phase 1: Foundation - "Actively Guarding" Requirement (HIGH PRIORITY)
**Estimated Effort**: Medium | **Impact**: High | **Breaking Changes**: Minimal

#### 1.1 Define "Actively Guarding"
**What**: A defender is actively guarding an opponent when:
- Distance to opponent is **≤ arm's length** (e.g., 2 hexes in our coordinate system)
- The defender is in a **guarding stance** (facing opponent, positioned between opponent and basket)
- OR the defender is guarding the **ball holder** (with exceptions)

**Implementation**:
```python
def _is_actively_guarding(self, defender_id: int, opponent_id: int) -> bool:
    """Check if defender is actively guarding an opponent (within arm's length)."""
    if not self.positions or defender_id not in self.positions or opponent_id not in self.positions:
        return False
    
    dist = self._hex_distance(self.positions[defender_id], self.positions[opponent_id])
    # Arm's length in hex distance = 2 hexes (configurable)
    return dist <= self.active_guard_distance  # Default: 2
```

#### 1.2 Modify Lane Violation Check
**Current Logic**:
```
If defender in lane for > 3 seconds → violation
```

**New Logic**:
```
If defender in lane for > 3 seconds:
    AND not actively guarding any offensive player → violation
```

**Code Location**: `_process_simultaneous_actions()` around line 900

**Pseudocode**:
```python
def _check_defensive_lane_violation(self, defender_id: int) -> bool:
    if not self.illegal_defense_enabled:
        return False
    
    in_lane = tuple(self.positions[defender_id]) in self.defensive_lane_hexes
    steps = self._defender_in_key_steps.get(defender_id, 0)
    
    if not in_lane or steps <= self.three_second_max_steps:
        return False
    
    # Violation only if NOT actively guarding anyone
    for oid in self.offense_ids:
        if self._is_actively_guarding(defender_id, oid):
            return False
    
    return True
```

#### 1.3 Configuration Addition
Add new parameter to environment:
```python
active_guard_distance: int = 2  # Hexes, arm's length proxy
```

**Update Training Script**: Add CLI argument:
```bash
--active-guard-distance 2
```

---

### Phase 2: Exception Handling - Suspend Count (MEDIUM PRIORITY)
**Estimated Effort**: Medium | **Impact**: High | **Breaking Changes**: None

#### 2.1 Suspend Count During Shots
**Requirement**: "Count starts when offensive team is in control... suspended when player is in act of shooting"

**Implementation**:
- Track if any offensive player is currently in a **shoot action** (not just attempted shots, but the action frame)
- When `self._pending_shot_actions` is non-empty OR a shot was just executed → suspend counts

**Code Location**: Lane tracking in `step()` method

```python
def _update_defender_lane_tracking(self):
    """Update defensive lane violations, with exceptions."""
    for did in self.defense_ids:
        in_lane = tuple(self.positions[did]) in self.defensive_lane_hexes
        
        # Suspend count if:
        # 1. No offensive team control (turnover in progress)
        # 2. Offensive player is shooting
        # 3. Defender is actively guarding
        if self._should_suspend_lane_count():
            self._defender_in_key_steps[did] = 0
        elif in_lane:
            self._defender_in_key_steps[did] += 1
        else:
            self._defender_in_key_steps[did] = 0

def _should_suspend_lane_count(self) -> bool:
    """Check if lane counter should be suspended."""
    # Suspend if there's no offensive team control
    if self.last_loss_of_control_step and self.step_count - self.last_loss_of_control_step < 2:
        return True
    
    # Suspend if a shot is in progress
    # (This requires tracking shot state across frames)
    if hasattr(self, '_shot_in_progress') and self._shot_in_progress:
        return True
    
    return False
```

#### 2.2 Loss of Team Control Tracking
Add new state variable:
```python
self.last_loss_of_control_step = -999  # Updated when turnover occurs
```

Update in `_process_simultaneous_actions()` when turnover is detected:
```python
if action_results.get("turnovers"):
    self.last_loss_of_control_step = self.step_count
```

---

### Phase 3: Ball Handler Exception & Guarding Logic (MEDIUM PRIORITY)
**Estimated Effort**: Medium | **Impact**: Medium | **Breaking Changes**: Minimal

#### 3.1 Ball Handler Guarding Rule
**Requirement**: "If defender is guarding player with ball, he may be in lane. Once ball is passed, defender must actively guard or exit."

**Implementation**:
```python
def _can_defender_camp_on_ball_handler(self, defender_id: int) -> bool:
    """
    Defender guarding the ball holder gets exception to lane rule.
    Returns True if defender is guarding the ball holder (no arm's length requirement).
    """
    ball_holder = self.ball_holder
    if ball_holder is None:
        return False
    
    # Check if this defender is marking the ball holder
    # (can be done via a marking/assignment system, or simply by proximity)
    dist = self._hex_distance(self.positions[defender_id], self.positions[ball_holder])
    
    # Ball handler exception: larger distance allowed when guarding ball
    return dist <= self.ball_handler_guard_distance  # E.g., 4 hexes
```

#### 3.2 Update Violation Check
Modify `_check_defensive_lane_violation()`:
```python
def _check_defensive_lane_violation(self, defender_id: int) -> bool:
    in_lane = tuple(self.positions[defender_id]) in self.defensive_lane_hexes
    steps = self._defender_in_key_steps.get(defender_id, 0)
    
    if not in_lane or steps <= self.three_second_max_steps:
        return False
    
    # Exception 1: Actively guarding any offensive player within arm's length
    for oid in self.offense_ids:
        if self._is_actively_guarding(defender_id, oid):
            return False
    
    # Exception 2: Guarding the ball handler with extended distance
    if self._can_defender_camp_on_ball_handler(defender_id):
        return False
    
    return True
```

**New Parameters**:
```python
ball_handler_guard_distance: int = 4  # Larger distance allowed for ball handler
```

---

### Phase 4: Shot Clock & Possession Reset (LOW-MEDIUM PRIORITY)
**Estimated Effort**: Low | **Impact**: Medium | **Breaking Changes**: Yes

#### 4.1 Correct Technical Foul Penalty
**Current**: Awards 1 point, ends episode
**NBA Rule**: Technical foul → free throw, then ball at sideline at FT line extended, shot clock reset to max or 14 seconds (whichever is greater)

**Proposal** (Optional - maintains episode continuity):
Instead of ending episode on violation, we could:
1. Award offense 1 point (already do this ✓)
2. **Reset shot clock** to `max(original_shot_clock, 14)` 
3. **Reset to neutral possession** (ball at sideline)
4. **Continue episode** instead of ending

**Code Location**: Where we handle `defensive_lane_violations`

```python
if action_results.get("defensive_lane_violations"):
    # Award technical foul
    self.offense_score += 1
    
    # Option A: Continue play (more realistic)
    # Reset shot clock
    self.shot_clock = max(self.initial_shot_clock, 14)
    
    # Place ball at sideline FT line extended (configurable position)
    self.positions[self.ball_holder] = self.technical_foul_inbound_position
    
    # Don't end episode
    done = False
    
    # Option B: Current behavior (keep for backward compatibility)
    # done = True
```

**Trade-offs**:
- ✅ More realistic NBA simulation
- ✅ Episodes longer (more learning from one episode)
- ❌ Breaking change to existing training runs
- ❌ May require tuning of rewards/penalties

---

### Phase 5: Double-Team Coordination (LOW PRIORITY)
**Estimated Effort**: High | **Impact**: Low-Medium | **Breaking Changes**: Yes

#### 5.1 Defender Assignment System
**Concept**: Track which offensive player each defender is assigned to guard.

**Why**: Enables proper enforcement of "defenders may double-team" rule.

**Implementation Option A: Implicit via Distance**
```python
def _get_defender_assignment(self, defender_id: int) -> Optional[int]:
    """Get the offensive player this defender is closest to."""
    closest_oid = None
    closest_dist = float('inf')
    
    for oid in self.offense_ids:
        dist = self._hex_distance(self.positions[defender_id], self.positions[oid])
        if dist < closest_dist:
            closest_dist = dist
            closest_oid = oid
    
    return closest_oid if closest_dist <= self.max_guard_distance else None
```

**Implementation Option B: Explicit via Action**
```python
# Add an action type: "ASSIGN_DEFENSE"
# Allows defenders to explicitly choose which offensive player to guard
# More complex but more realistic
```

**For Now**: Skip Phase 5; Phase 1-3 cover 80% of the improvement.

---

### Phase 6: Visual & Feedback Enhancements (LOW PRIORITY)
**Estimated Effort**: Low | **Impact**: Low | **Breaking Changes**: None

#### 6.1 Violation Warning System
Add visual indicator in web UI when defender approaches violation:
- Show defender "heat" color when 1 step away from violation
- Display countdown timer for defender in lane
- Highlight which offensive player defender is guarding (if any)

#### 6.2 Action Results Enrichment
```python
{
    "defensive_lane_violation": {
        "player_id": 1,
        "reason": "not_actively_guarding",  # or "actively_guarding_exception"
        "steps_in_lane": 4,
        "guarding_players": [],  # Who was this defender near?
        "distance_to_nearest_opponent": 5,
    }
}
```

---

## Implementation Sequence & Dependencies

```
Phase 1 (Foundation)
├── Define actively_guarding()
├── Add active_guard_distance parameter
└── Modify violation check
    ↓
Phase 2 (Exceptions)
├── Add _should_suspend_lane_count()
├── Track loss_of_control_step
└── Suspend count on shots/turnovers
    ↓
Phase 3 (Ball Handler)
├── Implement _can_defender_camp_on_ball_handler()
├── Add ball_handler_guard_distance parameter
└── Update violation logic
    ↓
Phase 4 (Optional: Realistic Penalty)
├── Implement shot clock reset
├── Implement neutral ball position
└── Optionally extend episodes
    ↓
Phase 6 (Optional: Polish)
├── Add visual warnings
└── Enrich action results
```

**Phases 1-3** are **strongly recommended** (2-3 days of work)
**Phase 4** is **optional** but realistic (1 day, breaking change)
**Phase 5-6** are **nice-to-have** (low priority)

---

## Key Configuration Parameters to Add

```python
# Phase 1
active_guard_distance: int = 2            # Hexes (arm's length equivalent)

# Phase 2
loss_of_control_suspension_frames: int = 2  # How many frames to suspend after turnover

# Phase 3
ball_handler_guard_distance: int = 4      # Larger distance for ball handler guarding

# Phase 4
technical_foul_inbound_position: Tuple[int, int] = (0, 5)  # Sideline at FT line
technical_foul_min_shot_clock: int = 14   # Per NBA rules
continue_after_violation: bool = False    # False = current behavior (end episode)
```

---

## Expected Benefits

### Gameplay Improvements
1. **More Realistic Defense**: Defenders must make active guarding choices
2. **Better AI Learning**: Agents learn to communicate defensive assignments implicitly
3. **Reduced Artificial Violations**: No more random violations when defender is guarding someone
4. **Longer Episodes**: More learning per episode (if we continue after violation)

### Measurable Metrics
1. **Illegal Defense Violation Rate**: Should drop significantly
2. **Defense Positioning**: Defenders cluster around opponents (not random lane placement)
3. **Episode Length**: Longer episodes with more meaningful actions
4. **Win Rates**: More realistic gameplay may shift win rates

---

## Testing Strategy

### Unit Tests
```python
def test_actively_guarding_within_distance():
    """Verify _is_actively_guarding() works correctly."""

def test_no_violation_when_guarding():
    """Defender in lane but guarding → no violation."""

def test_violation_when_not_guarding():
    """Defender in lane but not guarding anyone → violation."""

def test_shooting_suspension():
    """Count suspended when offensive player shoots."""

def test_loss_of_control_suspension():
    """Count suspended after turnover."""

def test_ball_handler_exception():
    """Defender guarding ball holder doesn't violate."""
```

### Integration Tests
```python
# Create game where:
# 1. Defender stands in lane but guards no one → violation
# 2. Defender stands in lane but guards opponent → no violation
# 3. Defender guards ball holder in lane → no violation
# 4. Ball is passed, defender not guarding new handler → violation if steps exceeded
# 5. Offensive player shoots → suspension during shot
```

### Training Validation
- Train with improved rules for 1M steps
- Verify no crashes or invalid states
- Verify defense learning is more robust
- Compare win rates, violation rates, positioning distributions

---

## Backward Compatibility

**Breaking Changes**:
- Phase 1-3: Minimal (defenders will behave differently, but rules are clearer)
- Phase 4: Major (episode continuation changes reward structure)

**Migration Path**:
```python
# Old behavior flag for backward compatibility
use_legacy_illegal_defense: bool = False

if self.use_legacy_illegal_defense:
    # Simple: any time in lane > 3 seconds = violation
    return steps > self.three_second_max_steps
else:
    # New: with actively guarding exception
    return (steps > self.three_second_max_steps and 
            not self._is_actively_guarding_anyone(defender_id))
```

---

## Implementation Effort Estimate

| Phase | Effort | Duration | Priority |
|---|---|---|---|
| 1: Actively Guarding | Medium | 1-2 days | 🔴 HIGH |
| 2: Exception Handling | Medium | 1 day | 🟠 HIGH |
| 3: Ball Handler Exception | Medium | 1 day | 🟠 HIGH |
| 4: Realistic Penalty | Low | 0.5 day | 🟡 MEDIUM |
| 5: Defense Coordination | High | 2-3 days | 🟢 LOW |
| 6: Visual Feedback | Low | 0.5 day | 🟢 LOW |

**Total for Phases 1-3**: 3-4 days → **Highly Recommended**
**Total with Phase 4**: 3.5-4.5 days → **Reasonable if realistic sim is goal**

---

## Questions for Review

Before implementation, consider:

1. **Should we end episodes on violations?** (Current: Yes. NBA-realistic: No, continue play)
2. **What's "actively guarding" in our hex grid?** (Proposed: ≤2 hexes distance)
3. **Should ball handler exception be larger distance?** (Proposed: 4 hexes vs 2)
4. **Do we need defense coordination/assignments?** (Current: No. Valuable but complex)
5. **How important is realism vs. training stability?** (Affects Phase 4 decision)

---

## Next Steps

1. **Review & Feedback**: Get your thoughts on the plan
2. **Prioritize Phases**: Decide which phases to implement
3. **Define Parameters**: Decide on hex distance values, suspension timing, etc.
4. **Implement Phase 1**: Start with foundation (actively guarding requirement)
5. **Test & Validate**: Verify no crashes, rule logic correct
6. **Proceed to Phases 2-3**: Build remaining exceptions
7. **Train & Evaluate**: See how it affects agent learning and gameplay


