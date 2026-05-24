"""Condition wiring with LLM hooks.

Maps experimental conditions to their Gamma configurations and LLM modes.
"""

from __future__ import annotations

from typing import Callable, Any
import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
# Condition registry
# ═══════════════════════════════════════════════════════════════════════════════

CONDITIONS: dict[str, dict[str, Any]] = {
    "Full-Gamma": {
        "description": "Six thermostats, fast+slow, explicit τ_HPA ≫ τ_SAM",
        "ablation": None,
        "needs_llm": True,
        "gamma_params": "default",
    },
    "No-H": {
        "description": "Hormonal thermostat ablated; no fast route, no HPA delay",
        "ablation": "hormonal",
        "needs_llm": True,
        "gamma_params": "no_h",
    },
    "No-E": {
        "description": "Emotion thermostat ablated; valence ≡ 0",
        "ablation": "emotional",
        "needs_llm": True,
        "gamma_params": "no_e",
    },
    "No-N": {
        "description": "Neuro-fast ablated; π_slow ≡ 1",
        "ablation": "neuro_fast",
        "needs_llm": True,
        "gamma_params": "no_n",
    },
    "Drive-only": {
        "description": "Drive-based policy, no learning (A1–A3 reduction)",
        "ablation": "drive_only",
        "needs_llm": False,  # Strictly numerical
        "gamma_params": "drive_only",
    },
    "HRRL": {
        "description": "Faithful Keramati–Gutkin 2014 HRRL baseline",
        "ablation": None,
        "needs_llm": False,  # Strictly numerical (tabular Q)
        "gamma_params": "hrrl",
    },
    "ReAct": {
        "description": "Baseline without internal regulation (thought→action)",
        "ablation": None,
        "needs_llm": True,
        "gamma_params": None,  # No Gamma
    },
    "Scrambled-Gamma": {
        "description": "Full-Γ with scrambled semantic labels (priming control)",
        "ablation": None,
        "needs_llm": True,
        "gamma_params": "default",
    },
    "pC-only": {
        "description": "Full-Γ controller, prompt stripped to p(C) + h_SAM only",
        "ablation": None,
        "needs_llm": True,
        "gamma_params": "default",
    },
    "Random-Gamma": {
        "description": "W entries permuted; same τ_k, κ, λ, α (coupling-structure ablation)",
        "ablation": "random_W",
        "needs_llm": True,
        "gamma_params": "random_W",
    },
    "Collapse-NC": {
        "description": "N and C merged into one regulator (5-vs-6 subsystem test)",
        "ablation": "collapse_NC",
        "needs_llm": True,
        "gamma_params": "collapse_NC",
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# Parameter generators
# ═══════════════════════════════════════════════════════════════════════════════

def make_default_params():
    """Default Γ parameters (calibrated on seed 42)."""
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


def apply_ablation(params: dict, ablation_type: str) -> dict:
    """Apply ablation to parameters."""
    p = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in params.items()}
    
    if ablation_type == "hormonal" or ablation_type == "no_h":
        # No-H: ablate hormonal and emotional
        p["kappas"][[2, 3]] = 0.0
        p["W"][:, [2, 3]] = 0.0
    elif ablation_type == "emotional" or ablation_type == "no_e":
        # No-E: ablate emotional only
        p["kappas"][3] = 0.0
        p["W"][:, 3] = 0.0
    elif ablation_type == "neuro_fast" or ablation_type == "no_n":
        # No-N: ablate neuro-fast
        p["kappas"][4] = 0.0
        p["W"][:, 4] = 0.0
    elif ablation_type == "drive_only":
        # Drive-only: all couplings off
        p["W"] = np.zeros((6, 6))
        p["kappas"] = np.zeros(6)
        p["kappas"][5] = 0.10  # only cognitive has weak return
    elif ablation_type == "random_W":
        # Random-Γ: permute W entries, keep diagonal zero
        import numpy as np
        rng = np.random.default_rng(42)
        W_flat = p["W"].copy()
        mask = ~np.eye(6, dtype=bool)
        vals = W_flat[mask].copy()
        rng.shuffle(vals)
        W_flat[mask] = vals
        p["W"] = W_flat
    elif ablation_type == "collapse_NC":
        # Collapse-NC: average N (idx 4) and C (idx 5) rows/cols
        p["W"][4, :] = (p["W"][4, :] + p["W"][5, :]) / 2
        p["W"][5, :] = p["W"][4, :]
        p["W"][:, 4] = (p["W"][:, 4] + p["W"][:, 5]) / 2
        p["W"][:, 5] = p["W"][:, 4]
        p["kappas"][4] = (p["kappas"][4] + p["kappas"][5]) / 2
        p["kappas"][5] = p["kappas"][4]
    
    return p


def get_params_for_condition(condition: str) -> dict | None:
    """Get Gamma parameters for a condition."""
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition: {condition}")
    
    cfg = CONDITIONS[condition]
    
    if cfg["gamma_params"] is None:
        # ReAct has no Gamma
        return None
    
    if cfg["gamma_params"] == "default":
        return make_default_params()
    
    base = make_default_params()
    return apply_ablation(base, cfg["ablation"])


# ═══════════════════════════════════════════════════════════════════════════════
# LLM mode detection
# ═══════════════════════════════════════════════════════════════════════════════

def condition_needs_llm(condition: str) -> bool:
    """Return True if condition requires LLM calls."""
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition: {condition}")
    return CONDITIONS[condition]["needs_llm"]


def get_llm_prompt_renderer(condition: str) -> Callable | None:
    """Get prompt renderer for condition.
    
    Returns None for numerical-only conditions.
    """
    if not condition_needs_llm(condition):
        return None
    
    if condition == "ReAct":
        from exp.prompts import render_react
        return render_react
    elif condition == "Scrambled-Gamma":
        from exp.prompts import get_scrambled_prompt
        return get_scrambled_prompt
    elif condition == "pC-only":
        from exp.prompts import render_pc_only
        return render_pc_only
    else:
        # Full-Gamma, No-H, No-E, No-N, Random-Gamma, Collapse-NC use Full-Gamma prompt
        from exp.prompts import render_full_gamma
        return render_full_gamma


# ═══════════════════════════════════════════════════════════════════════════════
# Condition list utilities
# ═══════════════════════════════════════════════════════════════════════════════

def list_conditions() -> list[str]:
    """Return all condition names."""
    return list(CONDITIONS.keys())


def list_llm_conditions() -> list[str]:
    """Return conditions that need LLM."""
    return [c for c, cfg in CONDITIONS.items() if cfg["needs_llm"]]


def list_numerical_conditions() -> list[str]:
    """Return conditions that are numerical-only."""
    return [c for c, cfg in CONDITIONS.items() if not cfg["needs_llm"]]


def describe_condition(condition: str) -> str:
    """Get human-readable description."""
    if condition not in CONDITIONS:
        return f"Unknown condition: {condition}"
    return CONDITIONS[condition]["description"]
