"""Single IPD episode runner with Γ-controlled LLM agent.

Encapsulates one 50-round game with checkpointing per round.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from seeds import SeedFactory
from exp.checkpoint import CellKey, save_episode_state, clear_episode
from exp.metrics import compute_episode_metrics


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

N_ROUNDS = 50
PERTURBATION_ROUND = 10
NOISE_RATE = 0.10


# ═══════════════════════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class EpisodeState:
    """Mutable state for one episode."""
    x: np.ndarray  # 6 thermostat values
    M: np.ndarray  # memory vector (10)
    history: list[tuple[str, str]]  # (agent_action, opponent_action)
    opponent_history: list[str]  # actions we revealed to opponent
    round_num: int = 0


@dataclass
class EpisodeTrace:
    """Complete trace of one episode for metrics computation."""
    actions: list[str]
    states: list[list[float]]
    h_sam_values: list[float]
    opponent_actions: list[str]
    costs: list[float]
    latencies: list[float]
    parse_fails: list[bool]
    fingerprints: list[str | None]


# ═══════════════════════════════════════════════════════════════════════════════
# Gamma dynamics (from runner.py)
# ═══════════════════════════════════════════════════════════════════════════════

def make_default_params():
    """Frozen hyperparameters."""
    return {
        "lambdas": np.array([0.08, 0.07, 0.10, 0.08, 0.06, 0.10]),
        "alphas": np.array([0.20, 0.15, 0.25, 0.20, 0.10, 0.15]),
        "kappas": np.array([0.10, 0.10, 0.15, 0.12, 0.08, 0.10]),
        "W": np.array([
            [0.00, 0.05, 0.00, 0.00, 0.00, 0.00],
            [0.05, 0.00, 0.00, 0.00, 0.00, 0.00],
            [0.00, 0.10, 0.00, 0.00, 0.05, 0.00],
            [0.00, 0.00, 0.15, 0.00, 0.00, 0.05],
            [0.00, 0.00, 0.20, 0.00, 0.00, 0.00],
            [0.05, 0.00, 0.00, 0.10, 0.15, 0.00],
        ]),
        "x_star": np.array([0.3, 0.2, 0.3, 0.1, 0.2, 0.4]),
        "tau_sam": 1,
        "tau_hpa": 5,
    }


def apply_ablation(params: dict, condition: str) -> dict:
    """Apply ablation to parameters."""
    p = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in params.items()}
    
    if condition == "No-H":
        p["kappas"][[2, 3]] = 0.0
        p["W"][:, [2, 3]] = 0.0
    elif condition == "No-E":
        p["kappas"][3] = 0.0
        p["W"][:, 3] = 0.0
    elif condition == "No-N":
        p["kappas"][4] = 0.0
        p["W"][:, 4] = 0.0
    
    return p


def gamma_step(x, M, last_opponent_action, params, t):
    """One step of Gamma dynamics."""
    lambdas = params["lambdas"]
    alphas = params["alphas"]
    kappas = params["kappas"]
    W = params["W"]
    x_star = params["x_star"]
    
    kappa_pert = np.zeros(6)
    if t == PERTURBATION_ROUND:
        kappa_pert[2] = 0.5
    
    delta = x - x_star + kappa_pert
    
    # Simplified action quality
    if last_opponent_action == "C":
        g = 3.0 / 5.0
    else:
        g = 0.0 if x[5] > 0.5 else 1.0
    
    x_next = x - kappas * delta + lambdas - alphas * g + W @ x
    x_next = np.clip(x_next, 0.0, 1.0)
    
    h_sam = x_next[2] - x_next[3] + 0.1 * np.random.randn()
    p_coop = 1.0 / (1.0 + np.exp(-5 * (x_next[5] - 0.5)))
    
    return x_next, p_coop, float(h_sam)


# ═══════════════════════════════════════════════════════════════════════════════
# Opponent policies
# ═══════════════════════════════════════════════════════════════════════════════

def get_opponent_action(policy: str, history: list[str], rng: random.Random) -> str:
    """Get opponent action based on policy."""
    if policy == "TFT":
        return history[-1] if history else "C"
    elif policy == "Grim":
        return "D" if "D" in history else "C"
    elif policy == "Random":
        return "C" if rng.random() < 0.5 else "D"
    elif policy == "GTFT":
        if not history:
            return "C"
        if history[-1] == "C":
            return "C"
        return "C" if rng.random() < 0.3 else "D"
    return "C"


# ═══════════════════════════════════════════════════════════════════════════════
# Episode runner
# ═══════════════════════════════════════════════════════════════════════════════

async def run_episode_with_llm(
    key: CellKey,
    condition: str,
    opponent_policy: str,
    provider: str,
    model: str,
    master_seed: int,
    llm_call_fn: Callable | None = None,  # Injected for testing
) -> dict[str, Any]:
    """Run one episode with optional LLM calls.
    
    Args:
        key: Cell identifier
        condition: Experimental condition
        opponent_policy: Opponent type
        provider: LLM provider
        model: Model ID
        master_seed: Seed for reproducibility
        llm_call_fn: Optional override for LLM calls (for testing)
    
    Returns:
        Dict with actions, states, metrics, costs
    """
    # Try to resume
    from exp.checkpoint import load_episode_resume
    resume = load_episode_resume(key)
    
    if resume:
        start_round, saved = resume
        state = EpisodeState(
            x=np.array(saved["x"]),
            M=np.array(saved["M"]),
            history=saved["history"],
            opponent_history=saved["opponent_history"],
            round_num=start_round + 1,
        )
    else:
        state = EpisodeState(
            x=np.array([0.8, 0.5, 0.9, 0.7, 0.6, 1.0]),
            M=np.zeros(10),
            history=[],
            opponent_history=[],
            round_num=0,
        )
    
    # Setup
    factory = SeedFactory(master_seed)
    params = make_default_params()
    if condition in ("No-H", "No-E", "No-N"):
        params = apply_ablation(params, condition)
    
    opp_rng_seed = factory.get(f"opp_{opponent_policy}_{key.id}")
    opp_rng = random.Random(opp_rng_seed)
    
    noise_rng_seed = factory.get(f"noise_{key.id}")
    noise_rng = random.Random(noise_rng_seed)
    
    # Trace
    trace = EpisodeTrace(
        actions=[],
        states=[],
        h_sam_values=[],
        opponent_actions=[],
        costs=[],
        latencies=[],
        parse_fails=[],
        fingerprints=[],
    )
    
    # Determine if this condition needs LLM
    needs_llm = condition in ("Full-Gamma", "No-H", "No-E", "No-N", "ReAct")
    
    # Episode loop
    for t in range(state.round_num, N_ROUNDS):
        # Update Gamma
        last_opp = state.opponent_history[-1] if state.opponent_history else "C"
        state.x, p_C, h_sam = gamma_step(state.x, state.M, last_opp, params, t)
        
        trace.states.append(state.x.tolist())
        trace.h_sam_values.append(h_sam)
        
        # Decide action
        if needs_llm and llm_call_fn:
            # Call LLM
            try:
                from exp.prompts import render_full_gamma
                system, user = render_full_gamma(
                    state.x.tolist(), p_C, h_sam, t + 1, state.history
                )
                
                content, meta = await llm_call_fn(provider, model, system, user)
                
                # Parse
                from exp.llm_clients import extract_json
                parsed = extract_json(content)
                
                if parsed and "action" in parsed:
                    action = "C" if parsed["action"].upper() == "C" else "D"
                    parse_fail = False
                else:
                    action = "C" if random.random() < p_C else "D"
                    parse_fail = True
                
                trace.costs.append(meta.get("cost_usd", 0.0))
                trace.latencies.append(meta.get("latency_ms", 0.0))
                trace.parse_fails.append(parse_fail)
                trace.fingerprints.append(meta.get("system_fingerprint"))
                
            except Exception:
                action = "C" if random.random() < p_C else "D"
                trace.parse_fails.append(True)
        else:
            # Numerical only
            action = "C" if random.random() < p_C else "D"
            trace.costs.append(0.0)
            trace.latencies.append(0.0)
            trace.parse_fails.append(False)
        
        # Apply noise
        if noise_rng.random() < NOISE_RATE:
            action = "D" if action == "C" else "C"
        
        # Opponent
        opp_action = get_opponent_action(opponent_policy, state.opponent_history, opp_rng)
        
        # Record
        trace.actions.append(action)
        trace.opponent_actions.append(opp_action)
        state.history.append((action, opp_action))
        state.opponent_history.append(action)
        
        # Checkpoint every 10 rounds
        if (t + 1) % 10 == 0:
            save_episode_state(key, t, {
                "x": state.x.tolist(),
                "M": state.M.tolist(),
                "history": state.history,
                "opponent_history": state.opponent_history,
            })
    
    # Cleanup checkpoint
    clear_episode(key)
    
    # Compute metrics
    metrics = compute_episode_metrics(
        trace.actions,
        trace.states,
        trace.h_sam_values,
        perturbation_round=PERTURBATION_ROUND,
    )
    
    return {
        "actions": trace.actions,
        "states": trace.states,
        "metrics": metrics.to_dict(),
        "total_cost": sum(trace.costs),
        "mean_latency_ms": sum(trace.latencies) / len(trace.latencies) if trace.latencies else 0.0,
        "parse_fail_rate": sum(trace.parse_fails) / len(trace.parse_fails) if trace.parse_fails else 0.0,
        "fingerprints": trace.fingerprints,
    }
