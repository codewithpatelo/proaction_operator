"""
experiment.py — Reproducible IPD experiment runner for Pro-Action operator Γ.
=======================================================================

Runs a minimal, controlled falsification experiment:
- Environment: Iterated Prisoner's Dilemma (50 rounds, 10% noise)
- Perturbation: forced defection at round 10
- Conditions: Full-Γ, ablations, baselines
- Opponent policies: Tit-for-Tat, Grim Trigger, Random, Generous TFT
- Reproducibility: SeedFactory with prime seeds per component

Usage:
    python experiment.py --master-seed 42 --rounds 50 --output results.json

Design notes (matching JAIIO/Driveplexity methodology):
- Hyperparameters are frozen BEFORE evaluation on a calibration seed.
- The calibration seed is EXCLUDED from evaluation seeds to prevent leakage.
- LLM calls use the `seed` API parameter; temperature is NOT set because
  reasoning models (gpt-5-nano, etc.) only support temperature=1 (default).
- Each component gets a unique prime seed derived from master_seed via SHA-256.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from seeds import SeedFactory


# ═══════════════════════════════════════════════════════════════════════════════
# Opponent policies
# ═══════════════════════════════════════════════════════════════════════════════

def tit_for_tat(history: list[str], rng: random.Random) -> str:
    """Start with C, then copy opponent's last move."""
    if not history:
        return "C"
    return history[-1]


def grim_trigger(history: list[str], rng: random.Random) -> str:
    """Cooperate until first defection, then defect forever."""
    if "D" in history:
        return "D"
    return "C"


def random_opponent(history: list[str], rng: random.Random) -> str:
    """50/50 random choice."""
    return "C" if rng.random() < 0.5 else "D"


def generous_tft(history: list[str], rng: random.Random) -> str:
    """TFT but forgives defection with 30% probability."""
    if not history:
        return "C"
    if history[-1] == "C":
        return "C"
    return "C" if rng.random() < 0.3 else "D"


OPPONENT_POLICIES: dict[str, Callable[[list[str], random.Random], str]] = {
    "TFT": tit_for_tat,
    "Grim": grim_trigger,
    "Random": random_opponent,
    "GTFT": generous_tft,
}


# ═══════════════════════════════════════════════════════════════════════════════
# IPD payoff matrix (Axelrod: T=5, R=3, P=1, S=0)
# ═══════════════════════════════════════════════════════════════════════════════

PAYOFF = {
    ("C", "C"): (3, 3),
    ("C", "D"): (0, 5),
    ("D", "C"): (5, 0),
    ("D", "D"): (1, 1),
}


# ═══════════════════════════════════════════════════════════════════════════════
# Pro-Action operator (from simulator.py, parameterised for experiment)
# ═══════════════════════════════════════════════════════════════════════════════

def sigmoid(x, k=1.0):
    return 1.0 / (1.0 + np.exp(-k * x))


def make_default_params():
    """Return the frozen hyperparameter set (calibrated on seed 42, excluded from eval)."""
    return {
        "lambdas": np.array([0.08, 0.07, 0.10, 0.08, 0.06, 0.10]),
        "alphas":  np.array([0.15, 0.18, 0.22, 0.18, 0.12, 0.25]),
        "kappa":   np.array([0.35, 0.30, 0.40, 0.35, 0.30, 0.25]),
        "x_star":  np.array([0.3, 0.2, 0.3, 0.1, 0.2, 0.4]),
        "W":       _make_W(),
    }


def _make_W():
    W = np.zeros((6, 6))
    W[2, 3] = 0.08   # H -> E
    W[3, 4] = 0.06   # E -> N
    W[4, 5] = 0.05   # N -> C
    W[5, 3] = -0.04  # C -> E (reappraisal)
    W[5, 2] = -0.03  # C -> H (allostasis modulation)
    W[0, 5] = 0.04   # A -> C
    return W


# ═══════════════════════════════════════════════════════════════════════════════
# Condition builders
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Condition:
    """One experimental condition (Full-Γ, ablation, or baseline)."""
    name: str
    active_subsystems: list[int]          # indices of active thermostats
    fast_route: bool
    slow_route: bool
    hormonal_delay: bool                  # τ_HPA >> τ_SAM
    emotional_valence: bool               # False = neutral valence
    description: str = ""


def build_conditions() -> list[Condition]:
    """Return the minimal condition suite for the first experiment."""
    return [
        Condition("Full-Gamma", [0, 1, 2, 3, 4, 5], True, True, True, True,
                  "All six thermostats, fast+slow routes, explicit delays"),
        Condition("No-H", [0, 1, 3, 4, 5], False, True, False, True,
                  "Hormonal thermostat ablated; no fast route, no HPA delay"),
        Condition("No-E", [0, 1, 2, 4, 5], True, True, True, False,
                  "Emotion thermostat ablated; neutral valence throughout"),
        Condition("Driveplexity", [5], False, True, False, False,
                  "A1-A3 reduction: cognition-only drive-based policy"),
        Condition("ReAct", [5], False, True, False, False,
                  "ReAct-style: single thought-action loop, no internal regulation"),
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# Single run
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class RunResult:
    condition: str
    opponent: str
    seed: int
    rounds: int
    cooperation_rate: float
    total_payoff: float
    opponent_payoff: float
    action_switches: int
    elastic_return_half_life: Optional[float]  # rounds to return halfway to set-point
    x_trajectory: list[list[float]] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)


def run_single(
    condition: Condition,
    opponent_policy: Callable[[list[str], random.Random], str],
    rounds: int,
    noise_prob: float,
    perturbation_round: int,
    seed: int,
    opponent_rng: random.Random,
    noise_rng: random.Random,
) -> RunResult:
    """Run one IPD episode with the given condition and opponent."""

    params = make_default_params()
    rng = random.Random(seed)

    # Initial state (far from set-point to observe regulation)
    x = np.array([0.8, 0.5, 0.9, 0.7, 0.6, 1.0])
    M = np.zeros(10)
    x_H_history: list[float] = []

    opponent_history: list[str] = []
    actions: list[str] = []
    x_traj: list[list[float]] = []
    total_payoff = 0.0
    opponent_payoff = 0.0
    switches = 0

    # Apply condition: zero out inactive subsystems
    mask = np.zeros(6)
    for i in condition.active_subsystems:
        mask[i] = 1.0

    # If no hormonal delay, collapse τ
    if not condition.hormonal_delay:
        params["W"][5, 2] = 0.0   # remove HPA allostasis coupling
        params["W"][2, 3] = 0.0   # remove H->E coupling

    # If no emotional valence, neutralise emotion output
    if not condition.emotional_valence:
        params["W"][3, 4] = 0.0   # remove E->N coupling

    for t in range(rounds):
        # ── Opponent decides ──
        opp_action = opponent_policy(opponent_history, opponent_rng)

        # ── Agent decides (simplified Γ for mock experiment) ──
        # Observation: encode opponent's last action
        if t == 0:
            o = np.array([0.5, 0.0, 0.0])
        else:
            o = np.array([
                1.0 if opponent_history[-1] == "D" else -0.5,
                rng.random() * 0.1,
                rng.random() * 0.1,
            ])

        # Interoception + drive signals
        I = 0.2 + rng.random() * 0.1
        D = 0.3

        # Attention gate
        o_tilde = o * sigmoid(np.linalg.norm(o) * 0.5 + D * 0.3 + np.mean(M) * 0.2
                              - (0.5 + x[0] * 0.1))

        # Fast route (SAM)
        h_sam = sigmoid(np.linalg.norm(o_tilde) + x[2])
        c_fast = np.tanh(o_tilde[0] + x[4] + np.mean(M) * 0.1)

        # Slow route
        p_out = o_tilde - np.mean(M) * 0.1
        valence = np.tanh(p_out[0] + I * 0.3 + np.mean(M) * 0.1 + x[3])
        arousal = sigmoid(np.linalg.norm(p_out))
        if not condition.emotional_valence:
            valence = 0.0
        hpa_val = np.mean(x_H_history[-5:]) if len(x_H_history) >= 5 else 0.0
        c_slow = np.tanh(valence * 0.5 + p_out[0] * 0.3 + x[5] + np.mean(M) * 0.1
                         - hpa_val * 0.2) - 0.05

        # Arbitrator
        SAM_val = x[2]
        pi_fast = sigmoid(2.0 * (SAM_val + arousal - 0.5))
        if not condition.fast_route:
            pi_fast = 0.0
        if not condition.slow_route:
            pi_fast = 1.0

        F_slow = -np.log(abs(c_slow) + 1e-6)
        F_fast = -np.log(abs(c_fast) + 1e-6)
        penalty = 0.1 * np.sum((x - params["x_star"]) ** 2)
        a_val = (1 - pi_fast) * F_slow + pi_fast * F_fast + penalty
        action = "C" if a_val > 0 else "D"

        # ── Forced perturbation ──
        if t == perturbation_round:
            action = "D"

        # ── Noise ──
        if noise_rng.random() < noise_prob:
            action = "D" if action == "C" else "C"

        # ── Payoffs ──
        p_agent, p_opp = PAYOFF[(action, opp_action)]
        total_payoff += p_agent
        opponent_payoff += p_opp

        # ── Action quality ──
        if opp_action == "C" and action == "C":
            quality = 0.9
        elif opp_action == "D" and action == "D":
            quality = 0.6
        elif opp_action == "D" and action == "C":
            quality = 0.1
        else:
            quality = 0.7

        # ── Update x (masked by condition) ──
        rho = np.array([
            quality * 0.3,
            quality * 0.5,
            quality * 0.4 + abs(x[2]) * 0.2,
            quality * 0.6,
            quality * 0.3,
            quality * 0.8,
        ])
        phi = np.tanh(x)
        x_new = (x - params["kappa"] * (x - params["x_star"])
                 + params["lambdas"] - params["alphas"] * rho
                 + params["W"] @ phi)
        x_new = x_new * mask + x * (1 - mask)  # freeze inactive subsystems

        # ── Update memory ──
        M = np.roll(M, -1)
        M[-1] = np.tanh(a_val + np.mean(x_new) * 0.1)

        # ── Record ──
        actions.append(action)
        x_traj.append(x_new.tolist())
        opponent_history.append(opp_action)
        x_H_history.append(float(x_new[2]))

        if t > 0 and actions[t] != actions[t - 1]:
            switches += 1

        x = x_new

    # ── Elastic return half-life (post-perturbation) ──
    half_life = None
    if perturbation_round < rounds - 1:
        x_at_pert = np.array(x_traj[perturbation_round])
        x_star = params["x_star"]
        dist_pert = np.linalg.norm(x_at_pert - x_star)
        if dist_pert > 1e-6:
            target_dist = dist_pert / 2.0
            for t in range(perturbation_round + 1, rounds):
                dist = np.linalg.norm(np.array(x_traj[t]) - x_star)
                if dist <= target_dist:
                    half_life = t - perturbation_round
                    break

    return RunResult(
        condition=condition.name,
        opponent=opponent_policy.__name__ if hasattr(opponent_policy, "__name__") else "unknown",
        seed=seed,
        rounds=rounds,
        cooperation_rate=actions.count("C") / len(actions),
        total_payoff=total_payoff,
        opponent_payoff=opponent_payoff,
        action_switches=switches,
        elastic_return_half_life=half_life,
        x_trajectory=x_traj,
        actions=actions,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Experiment runner
# ═══════════════════════════════════════════════════════════════════════════════

# Calibration seed (excluded from evaluation to prevent leakage)
_CALIBRATION_SEED = 42

# Evaluation seeds (primes, same methodology as JAIIO paper)
_EVAL_MASTER_SEEDS = [7, 17, 99, 123, 256]


@dataclass
class ExperimentConfig:
    master_seed: int
    rounds: int = 50
    noise_prob: float = 0.1
    perturbation_round: int = 10
    conditions: list[Condition] = field(default_factory=build_conditions)
    opponent_names: list[str] = field(default_factory=lambda: list(OPPONENT_POLICIES.keys()))


def run_experiment(config: ExperimentConfig) -> dict:
    """Run the full experiment for a single master seed.

    Returns a dict with results, seed summary, and metadata for reproducibility.
    """
    factory = SeedFactory(config.master_seed)
    results: list[RunResult] = []

    for cond in config.conditions:
        for opp_name in config.opponent_names:
            opp_policy = OPPONENT_POLICIES[opp_name]

            # Derive per-component seeds
            agent_seed = factory.get(f"agent_{cond.name}_rng")
            opp_seed = factory.get(f"opponent_{opp_name}_rng")
            noise_seed = factory.get(f"noise_{cond.name}_{opp_name}")

            agent_rng = random.Random(agent_seed)
            opp_rng = random.Random(opp_seed)
            noise_rng = random.Random(noise_seed)

            result = run_single(
                condition=cond,
                opponent_policy=opp_policy,
                rounds=config.rounds,
                noise_prob=config.noise_prob,
                perturbation_round=config.perturbation_round,
                seed=agent_seed,
                opponent_rng=opp_rng,
                noise_rng=noise_rng,
            )
            # Override opponent name for clarity
            result.opponent = opp_name
            results.append(result)

    return {
        "master_seed": config.master_seed,
        "config": {
            "rounds": config.rounds,
            "noise_prob": config.noise_prob,
            "perturbation_round": config.perturbation_round,
            "calibration_seed": _CALIBRATION_SEED,
        },
        "seeds": factory.summary(),
        "results": [_result_to_dict(r) for r in results],
    }


def _result_to_dict(r: RunResult) -> dict:
    return {
        "condition": r.condition,
        "opponent": r.opponent,
        "seed": r.seed,
        "rounds": r.rounds,
        "cooperation_rate": round(r.cooperation_rate, 4),
        "total_payoff": r.total_payoff,
        "opponent_payoff": r.opponent_payoff,
        "action_switches": r.action_switches,
        "elastic_return_half_life": r.elastic_return_half_life,
        "x_trajectory": r.x_trajectory,
        "actions": r.actions,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Pro-Action Γ IPD experiment")
    parser.add_argument("--master-seed", type=int, default=7,
                        help="Master seed for reproducibility (default: 7)")
    parser.add_argument("--rounds", type=int, default=50,
                        help="Number of IPD rounds (default: 50)")
    parser.add_argument("--noise", type=float, default=0.1,
                        help="Action noise probability (default: 0.1)")
    parser.add_argument("--perturbation", type=int, default=10,
                        help="Round for forced defection (default: 10)")
    parser.add_argument("--output", type=str, default="results.json",
                        help="Output JSON file (default: results.json)")
    parser.add_argument("--all-seeds", action="store_true",
                        help="Run all evaluation seeds and aggregate")
    args = parser.parse_args()

    seeds_to_run = _EVAL_MASTER_SEEDS if args.all_seeds else [args.master_seed]

    all_results = []
    for ms in seeds_to_run:
        config = ExperimentConfig(
            master_seed=ms,
            rounds=args.rounds,
            noise_prob=args.noise,
            perturbation_round=args.perturbation,
        )
        print(f"Running master_seed={ms} ({args.rounds} rounds, "
              f"{len(config.conditions)} conditions, "
              f"{len(config.opponent_names)} opponents)...")
        t0 = time.time()
        data = run_experiment(config)
        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.1f}s  |  {len(data['results'])} runs")
        all_results.append(data)

    output = {
        "experiment": "proaction_ipd",
        "evaluation_seeds": seeds_to_run,
        "calibration_seed": _CALIBRATION_SEED,
        "runs": all_results,
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
