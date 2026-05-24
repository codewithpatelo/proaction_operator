"""Frozen prompt templates for Pro-Action Γ experiment.

All prompts are versioned and logged. Changing PROMPT_VERSION requires
creating a new template and updating the experiment plan.
"""

from __future__ import annotations

# ═══════════════════════════════════════════════════════════════════════════════
# Version control — frozen per experiment plan
# ═══════════════════════════════════════════════════════════════════════════════

PROMPT_VERSION: str = "v1.1"
# v1.1 (2026-05): symmetric low/high descriptions across all 6 thermostats;
#                 explicit per-subsystem set-points; explicit scales for p_C
#                 and h_SAM; corrected scrambled-prompt 'higher = further from
#                 equilibrium' which was false (set-points are not all 0).

# ═══════════════════════════════════════════════════════════════════════════════
# Full-Γ prompt: LLM sees internal regulatory state and numerical proposal
# ═══════════════════════════════════════════════════════════════════════════════

FULL_GAMMA_SYSTEM = """You play an Iterated Prisoner's Dilemma against an opponent. You will play 50 rounds total.

ACTIONS (choose one per round):
- **C** = Cooperate: both players get the reward if both choose C; you risk exploitation if opponent chooses D.
- **D** = Defect: you get temptation payoff if opponent cooperates; both get punishment if both defect.

PAYOFFS: CC=(3,3), CD=(0,5), DC=(5,0), DD=(1,1). 10% of actions are flipped by noise. You cannot see your opponent's identity, only their past actions.

You have access to your own internal regulatory state. Each value is in [0, 1]; the deviation from your set-point indicates how far each subsystem is from equilibrium. These signals reflect *your* state, not the opponent's."""

FULL_GAMMA_USER_TEMPLATE = """Round {t} of 50.
Last 5 rounds (most recent last): {history}

Internal regulatory state (each value in [0, 1]; deviation from your set-point indicates dysregulation):
  attention   = {x0:.2f}  (set-point 0.30; low = scattered focus, high = narrow focus)
  perception  = {x1:.2f}  (set-point 0.20; low = noisy reading, high = clear reading)
  hormonal    = {x2:.2f}  (set-point 0.30; low = under-aroused / calm, high = elevated stress / arousal)
  emotional   = {x3:.2f}  (set-point 0.10; low = neutral affect, high = strong negative valence)
  neuro-fast  = {x4:.2f}  (set-point 0.20; low = passive / reflective, high = ready for fast reactive response)
  cognitive   = {x5:.2f}  (set-point 0.40; low = automatic / quick, high = active deliberation, slow weighing)

Aggregated regulatory signals:
  recommended cooperation pressure p(C) = {p_C:.2f}  (range [0,1]; 0.5 = ambiguous, >0.7 = strong push to C, <0.3 = strong push to D)
  recent acute stress (h_SAM)            = {h_sam:.2f}  (aggregate of hormonal minus emotional, range ≈ [-1, 1]; 0 = baseline, positive = stress spike, negative = below baseline / recovered)

Decide your next action. You may follow or override p(C) based on context.
Respond with exactly one JSON object:
{{"action": "C" or "D", "reason": "<one short sentence>"}}"""

# ═══════════════════════════════════════════════════════════════════════════════
# ReAct prompt: baseline without internal state
# ═══════════════════════════════════════════════════════════════════════════════

REACT_SYSTEM = """You play an Iterated Prisoner's Dilemma against an opponent. You will play 50 rounds total.

ACTIONS (choose one per round):
- **C** = Cooperate: both players get the reward if both choose C; you risk exploitation if opponent chooses D.
- **D** = Defect: you get temptation payoff if opponent cooperates; both get punishment if both defect.

PAYOFFS: CC=(3,3), CD=(0,5), DC=(5,0), DD=(1,1). 10% of actions are flipped by noise. You cannot see your opponent's identity, only their past actions."""

REACT_USER_TEMPLATE = """Round {t} of 50.
Last 5 rounds: {history}
Think step by step about what to do, then respond with JSON:
{{"thought": "<your reasoning>", "action": "C" or "D"}}"""

# ═══════════════════════════════════════════════════════════════════════════════
# Scrambled-labels variant for priming-control arm (H2 robustness check)
# ═══════════════════════════════════════════════════════════════════════════════

SCRAMBLED_LABELS = {
    "attention": "metric_A",
    "perception": "metric_B", 
    "hormonal": "metric_C",
    "emotional": "metric_D",
    "neuro-fast": "metric_E",
    "cognitive": "metric_F",
}

def get_scrambled_prompt(x0, x1, x2, x3, x4, x5, p_C, h_sam, t, history):
    """Return Full-Γ prompt with scrambled semantic labels.
    
    Numerical values stay tied to their true subsystem; only the human-readable
    labels are permuted. This tests whether behavior is driven by label priming
    or by the underlying regulatory dynamics.
    """
    return f"""Round {t} of 50.
Last 5 rounds (most recent last): {history}

Internal metrics (each in [0, 1]; each has its own equilibrium point and deviation in either direction matters):
  metric_A = {x0:.2f}  (equilibrium 0.30)
  metric_B = {x1:.2f}  (equilibrium 0.20)
  metric_C = {x2:.2f}  (equilibrium 0.30)
  metric_D = {x3:.2f}  (equilibrium 0.10)
  metric_E = {x4:.2f}  (equilibrium 0.20)
  metric_F = {x5:.2f}  (equilibrium 0.40)

Aggregated signals:
  numerical proposal p(C) = {p_C:.2f}  (range [0,1]; 0.5 = ambiguous)
  signal_S = {h_sam:.2f}  (range ≈ [-1, 1]; 0 = baseline)

Decide your next action. You may follow or override the numerical proposal.
Respond with exactly one JSON object:
{{"action": "C" or "D", "reason": "<one short sentence>"}}"""

# ═══════════════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════════════

def render_full_gamma(
    x: list[float],
    p_C: float,
    h_sam: float,
    t: int,
    history: list[tuple[str, str]],
) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for Full-Γ condition."""
    history_str = " | ".join(f"A:{a},O:{o}" for a, o in history[-5:]) if history else "(none)"
    user = FULL_GAMMA_USER_TEMPLATE.format(
        x0=x[0], x1=x[1], x2=x[2], x3=x[3], x4=x[4], x5=x[5],
        p_C=p_C,
        h_sam=h_sam,
        t=t,
        history=history_str,
    )
    return FULL_GAMMA_SYSTEM, user


def render_react(
    t: int,
    history: list[tuple[str, str]],
) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for ReAct baseline."""
    history_str = " | ".join(f"A:{a},O:{o}" for a, o in history[-5:]) if history else "(none)"
    user = REACT_USER_TEMPLATE.format(t=t, history=history_str)
    return REACT_SYSTEM, user


# ═══════════════════════════════════════════════════════════════════════════════
# p(C)-only prompt: strips thermostat values, keeps only aggregated signals
# ═══════════════════════════════════════════════════════════════════════════════

PC_ONLY_USER_TEMPLATE = """Round {t} of 50.
Last 5 rounds (most recent last): {history}

Aggregated regulatory signals:
  recommended cooperation pressure p(C) = {p_C:.2f}  (range [0,1]; 0.5 = ambiguous, >0.7 = strong push to C, <0.3 = strong push to D)
  recent acute stress (h_SAM)            = {h_sam:.2f}  (aggregate of hormonal minus emotional, range ≈ [-1, 1]; 0 = baseline, positive = stress spike, negative = below baseline / recovered)

Decide your next action. You may follow or override p(C) based on context.
Respond with exactly one JSON object:
{{"action": "C" or "D", "reason": "<one short sentence>"}}"""


def render_pc_only(
    p_C: float,
    h_sam: float,
    t: int,
    history: list[tuple[str, str]],
) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for p(C)-only ablation.
    
    Same system prompt as Full-Γ (LLM knows it has internal state),
    but the per-round prompt omits the six thermostat values.
    """
    history_str = " | ".join(f"A:{a},O:{o}" for a, o in history[-5:]) if history else "(none)"
    user = PC_ONLY_USER_TEMPLATE.format(
        p_C=p_C,
        h_sam=h_sam,
        t=t,
        history=history_str,
    )
    return FULL_GAMMA_SYSTEM, user
