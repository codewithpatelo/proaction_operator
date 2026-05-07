"""Faithful Keramati-Gutkin (2014) Homeostatic RL baseline for IPD.

Reference: Keramati M, Gutkin B. (2014). Homeostatic reinforcement learning for
integrating reward collection and physiological stability. eLife 3:e04811.

Key elements (per paper, eLife 04811):
- Internal homeostatic state H_t \\in R^d (here d=2 for IPD: cooperation_deficit, defect_avoidance).
- Setpoint H* (target equilibrium).
- Drive: D(H) = ( sum_i |h*_i - h_i|^n )^{1/m},  with n > m > 1 (we use n=4, m=2 as in Fig 1 of paper).
- Reward = D(H_t) - D(H_t + K_t)  where K_t is the homeostatic effect of action.
- Policy: epsilon-greedy Q-learning with this drive-derived reward.

Per-episode state is encapsulated in a closure indexed by (master_seed, opponent_policy)
so that learning persists across rounds within an episode but resets between episodes.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
# Constants (per Keramati-Gutkin 2014)
# ═══════════════════════════════════════════════════════════════════════════════

DRIVE_N = 4.0   # exponent inside (n > m > 1)
DRIVE_M = 2.0   # outer root
LEARNING_RATE = 0.2
DISCOUNT = 0.9
EPSILON = 0.10  # exploration rate

# Setpoint: high cooperation-readiness, low defect-tolerance
H_STAR = np.array([1.0, 0.0])

# State discretization for Q-table:
# state = (last_opp_C, last_opp_D)  → 4 states (00, 01, 10, 11)
# action = 0 (C) or 1 (D)
N_STATES = 4
N_ACTIONS = 2


# ═══════════════════════════════════════════════════════════════════════════════
# Drive function
# ═══════════════════════════════════════════════════════════════════════════════

def drive(H: np.ndarray, H_star: np.ndarray = H_STAR, n: float = DRIVE_N, m: float = DRIVE_M) -> float:
    """K-G 2014 drive: D(H) = (Σ |h*_i - h_i|^n)^(1/m)."""
    deficits = np.abs(H_star - H)
    return float(np.power(np.sum(np.power(deficits, n)), 1.0 / m))


def homeostatic_effect(action: str, last_opp: str) -> np.ndarray:
    """Effect K_t on internal state H given action and opponent's last action.
    
    - Cooperate when opponent cooperated: increases coop-readiness.
    - Defect when opponent cooperated: increases defect-tolerance (drift away from setpoint).
    - Cooperate when opponent defected: decreases coop-readiness (got exploited).
    - Defect when opponent defected: small increase in defect-tolerance.
    """
    if action == "C" and last_opp == "C":
        return np.array([+0.30, -0.05])  # towards setpoint
    if action == "D" and last_opp == "C":
        return np.array([-0.10, +0.30])  # away from setpoint
    if action == "C" and last_opp == "D":
        return np.array([-0.20, -0.05])  # exploited, drifts away
    if action == "D" and last_opp == "D":
        return np.array([-0.05, +0.10])  # mild drift
    return np.zeros(2)


def reward(H_t: np.ndarray, action: str, last_opp: str) -> tuple[float, np.ndarray]:
    """K-G reward: r = D(H_t) - D(H_t + K_t).
    
    Returns (reward, H_next).
    """
    K = homeostatic_effect(action, last_opp)
    H_next = H_t + K
    H_next = np.clip(H_next, -1.0, 2.0)  # keep bounded
    
    D_t = drive(H_t)
    D_next = drive(H_next)
    
    return D_t - D_next, H_next


# ═══════════════════════════════════════════════════════════════════════════════
# State encoding
# ═══════════════════════════════════════════════════════════════════════════════

def encode_state(opponent_actions: list[str]) -> int:
    """Encode last 2 opponent actions as 0..3 state.
    
    Bits: (last_round_was_D, second_last_round_was_D)
    """
    if len(opponent_actions) == 0:
        return 0  # initial state: assume both cooperated
    if len(opponent_actions) == 1:
        return 1 if opponent_actions[-1] == "D" else 0
    
    last = 1 if opponent_actions[-1] == "D" else 0
    prev = 1 if opponent_actions[-2] == "D" else 0
    return last + 2 * prev


# ═══════════════════════════════════════════════════════════════════════════════
# Per-episode HRRL agent
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class HRRLEpisodeState:
    """Mutable state for one HRRL episode."""
    Q: np.ndarray = field(default_factory=lambda: np.zeros((N_STATES, N_ACTIONS)))
    H: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.5]))  # start far from setpoint
    last_state: int = 0
    last_action: int | None = None


# Episode-scoped state: keyed by id() of opponent_actions list
# (we use the list identity to scope state to one episode)
_EPISODE_STATES: dict[int, HRRLEpisodeState] = {}


def _get_or_create_state(opponent_actions: list[str]) -> HRRLEpisodeState:
    """Get/create per-episode state keyed by opponent_actions list identity."""
    key = id(opponent_actions)
    if key not in _EPISODE_STATES:
        _EPISODE_STATES[key] = HRRLEpisodeState()
    return _EPISODE_STATES[key]


def _cleanup_episode_state(opponent_actions: list[str]) -> None:
    """Remove episode state (call after episode ends)."""
    key = id(opponent_actions)
    _EPISODE_STATES.pop(key, None)


# ═══════════════════════════════════════════════════════════════════════════════
# Decision function (called by runner)
# ═══════════════════════════════════════════════════════════════════════════════

def hrrl_decide(t: int, opponent_actions: list[str], rng: random.Random) -> str:
    """Make one HRRL decision.
    
    Args:
        t: round index (0-based)
        opponent_actions: list of opponent's past actions (this episode)
        rng: seeded random.Random for epsilon-greedy
    
    Returns:
        'C' or 'D'
    """
    s = _get_or_create_state(opponent_actions)
    
    # Encode current state from observed history
    state = encode_state(opponent_actions)
    
    # Update Q for previous transition (if any)
    if s.last_action is not None and len(opponent_actions) > 0:
        # We took s.last_action in s.last_state, then observed opponent_actions[-1]
        last_opp = opponent_actions[-1]
        action_str = "C" if s.last_action == 0 else "D"
        r, H_next = reward(s.H, action_str, last_opp)
        s.H = H_next
        
        # Q-update: Q(s,a) ← Q(s,a) + α [r + γ max_a' Q(s',a') − Q(s,a)]
        td_error = r + DISCOUNT * np.max(s.Q[state]) - s.Q[s.last_state, s.last_action]
        s.Q[s.last_state, s.last_action] += LEARNING_RATE * td_error
    
    # ε-greedy action selection
    if rng.random() < EPSILON:
        action_idx = rng.randint(0, 1)
    else:
        action_idx = int(np.argmax(s.Q[state]))
    
    # Persist for next call
    s.last_state = state
    s.last_action = action_idx
    
    # Cleanup if episode is ending (heuristic: if t is last round)
    # Note: runner does not signal end-of-episode here, so we rely on
    # episode_actions being a fresh list per episode (different id())
    
    return "C" if action_idx == 0 else "D"
