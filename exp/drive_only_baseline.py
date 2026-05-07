"""Drive-only baseline: reduction of Γ to A1-A3 (Driveplexity-style).

This is the canonical reduction described in the Pro-Action paper Section 3.4:
    "When x_A = ... = x_N ≡ 0, A=id, I=M=D=∅, π_fast ≡ 0, W=0, and x → x_C ≡ δ_i,
     Equation (master) reduces to δ_{t+1} = δ_t + λ - α·g(e_t, e_{t+1}),
     which is exactly A1-A3."

Implementation: a single scalar drive δ that:
- drifts upward by λ per round (autonomy/impulse drift)
- is reduced by α·g where g is action quality against last opponent action
- generates action via p(C) = σ(D(δ))

Free parameters held fixed (no learning, no coupling, no thermostats).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field


# ═══════════════════════════════════════════════════════════════════════════════
# Constants (single-thermostat reduction)
# ═══════════════════════════════════════════════════════════════════════════════

LAMBDA: float = 0.10  # drift per round
ALPHA: float = 0.20   # satiety gain
DELTA_INIT: float = 1.0   # start far from satiety
DELTA_MAX: float = 2.0
DELTA_MIN: float = 0.0
SIGMOID_GAIN: float = 3.0  # steepness of p(C) curve


# ═══════════════════════════════════════════════════════════════════════════════
# Per-episode state
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class DriveOnlyState:
    """Single-scalar drive state for one episode."""
    delta: float = DELTA_INIT


_EPISODE_STATES: dict[int, DriveOnlyState] = {}


def _get_or_create_state(opponent_actions: list[str]) -> DriveOnlyState:
    """Get/create per-episode state keyed by opponent_actions list identity."""
    key = id(opponent_actions)
    if key not in _EPISODE_STATES:
        _EPISODE_STATES[key] = DriveOnlyState()
    return _EPISODE_STATES[key]


# ═══════════════════════════════════════════════════════════════════════════════
# Action quality g (matches Driveplexity A1-A3)
# ═══════════════════════════════════════════════════════════════════════════════

def action_quality(action: str, last_opp: str) -> float:
    """g(e_t, e_{t+1}): scalar quality of (action, observation) pair.
    
    Returns value in [0, 1].
    """
    if action == "C" and last_opp == "C":
        return 0.9   # mutual cooperation: high satiety
    if action == "D" and last_opp == "D":
        return 0.4   # mutual defection: low satiety
    if action == "D" and last_opp == "C":
        return 0.7   # exploited cooperator: short-term gain
    if action == "C" and last_opp == "D":
        return 0.1   # got exploited: very low quality
    return 0.5


# ═══════════════════════════════════════════════════════════════════════════════
# Decision function
# ═══════════════════════════════════════════════════════════════════════════════

def drive_only_decide(t: int, opponent_actions: list[str], rng: random.Random) -> str:
    """Make one Drive-only decision.
    
    Single-scalar drive: δ_{t+1} = δ_t + λ - α·g
    Action: p(C) = σ(D(δ))
    
    Args:
        t: round index
        opponent_actions: opponent's past actions
        rng: seeded RNG for stochastic action sampling
    """
    s = _get_or_create_state(opponent_actions)
    
    # Update drive based on previous round's outcome
    if len(opponent_actions) > 0:
        last_opp = opponent_actions[-1]
        # Use the action implied by current drive level for quality computation
        # (we approximate by using the dominant action under current δ)
        implicit_action = "C" if s.delta > 1.0 else "D"
        g = action_quality(implicit_action, last_opp)
        
        # Drive update (A1-A3 master equation)
        s.delta = s.delta + LAMBDA - ALPHA * g
        s.delta = max(DELTA_MIN, min(DELTA_MAX, s.delta))
    
    # Action probability via sigmoid of drive
    # High drive → more cooperation (urge to act / take risk)
    p_C = 1.0 / (1.0 + math.exp(-SIGMOID_GAIN * (s.delta - 1.0)))
    
    return "C" if rng.random() < p_C else "D"
