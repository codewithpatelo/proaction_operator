"""Main experiment runner with phases (calibration→preflight→benchmark).

Implements idempotent execution, checkpointing, and progressive reporting.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Load .env BEFORE importing modules that read API keys at class init time
try:
    from dotenv import load_dotenv
    # Search parent dirs for .env (langclaw_experiment/.env)
    _here = Path(__file__).resolve()
    for _parent in [_here.parent, _here.parent.parent, _here.parent.parent.parent]:
        _env = _parent / ".env"
        if _env.exists():
            load_dotenv(_env)
            break
except ImportError:
    pass  # dotenv optional; relies on OS env vars

# Local imports
from exp.prompts import render_full_gamma, render_react, render_pc_only, get_scrambled_prompt, PROMPT_VERSION
from exp.checkpoint import (
    CellKey, CellResult, load_cells, save_cell, is_cell_complete,
    save_episode_state, load_episode_resume, clear_episode,
    check_writable,
)
from exp.budget import (
    record_call, get_summary as get_budget_summary, reset_budget,
    PROVIDER_CAPS,
)
from exp.metrics import compute_episode_metrics, EpisodeMetrics
from exp.red_flags import check_cell, check_aggregate, should_halt, summarize_flags
from exp.llm_clients import call_llm, ParseError, RateLimitError, TransientError


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

N_ROUNDS = 50
PERTURBATION_ROUND = 10
NOISE_RATE = 0.10

# Eval seeds (10 total, calibration seed 42 is excluded)
EVAL_SEEDS = [7, 17, 99, 123, 256, 511, 1024, 2048, 4096, 8192]
CALIBRATION_SEED = 42
VALIDATION_SEED = 999  # held-out for calibration validation

# Conditions (baselines prioritized for critical comparison)
CONDITIONS = [
    "ReAct",         # CRITICAL: baseline without internal regulation
    "Drive-only",    # CRITICAL: Driveplexity reduction (numerical, $0)
    "HRRL",          # CRITICAL: Keramati-Gutkin 2014 baseline (numerical, $0)
    "Collapse-NC",    # E4: 6-vs-5 subsystem ablation (REV 2 #5) - prioritize over primary conditions
    "Full-Gamma",
    "No-H",
    "No-E", 
    "No-N",
    "Random-Gamma",   # E2: parameter-count control (REV 5 #3)
    "pC-only",        # E5: prompt stripped to p(C) + h_SAM only
    "Scrambled-Gamma", # E6: scrambled semantic labels (priming control)
]

# LLM-textual conditions (require LLM calls)
LLM_CONDITIONS = {"Full-Gamma", "No-H", "No-E", "No-N",
                  "Random-Gamma", "Collapse-NC", "ReAct",
                  "pC-only", "Scrambled-Gamma"}

# Numerical-only conditions (run once per seed, no provider duplication)
NUMERICAL_CONDITIONS = {"Drive-only", "HRRL"}

# Conditions with ablations to default Gamma
ABLATION_CONDITIONS = {"No-H", "No-E", "No-N", "Random-Gamma", "Collapse-NC"}

# Opponents
OPPONENTS = ["TFT", "Grim", "Random", "GTFT"]

# Provider-model mapping
PROVIDER_MODELS = {
    "openai": "gpt-5-nano",
    "anthropic": "claude-haiku-4-5",
    "deepseek": "deepseek-v4-flash",
}

# Synthetic provider tag for numerical-only cells
NUMERICAL_PROVIDER = "numerical"

# Reports
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Gamma operator (numerical)
# ═══════════════════════════════════════════════════════════════════════════════

import numpy as np
from seeds import SeedFactory


def make_default_params():
    """Frozen hyperparameters (calibrated on seed 42)."""
    return {
        "lambdas": np.array([0.08, 0.07, 0.10, 0.08, 0.06, 0.10]),
        "alphas": np.array([0.20, 0.15, 0.25, 0.20, 0.10, 0.15]),
        "kappas": np.array([0.10, 0.10, 0.15, 0.12, 0.08, 0.10]),
        "W": np.array([
            [0.00, 0.05, 0.00, 0.00, 0.00, 0.00],
            [0.05, 0.00, 0.00, 0.00, 0.00, 0.00],
            [0.00, 0.10, 0.00, 0.00, 0.05, 0.00],  # SAM
            [0.00, 0.00, 0.15, 0.00, 0.00, 0.05],  # HPA
            [0.00, 0.00, 0.20, 0.00, 0.00, 0.00],  # fast
            [0.05, 0.00, 0.00, 0.10, 0.15, 0.00],  # slow
        ]),
        "x_star": np.array([0.3, 0.2, 0.3, 0.1, 0.2, 0.4]),
        "tau_sam": 1,   # τ_SAM
        "tau_hpa": 5,   # τ_HPA (informed by TSST literature)
    }


def apply_ablation(params: dict, condition: str, master_seed: int = 0) -> dict:
    """Return parameter set with ablation applied.
    
    Args:
        params: base parameter dict
        condition: ablation name
        master_seed: used for Random-Gamma to get a deterministic permutation per seed
    """
    p = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in params.items()}
    
    if condition == "No-H":
        p["kappas"][2] = 0.0  # hormonal
        p["kappas"][3] = 0.0  # emotional
        p["W"][:, 2] = 0.0
        p["W"][:, 3] = 0.0
    elif condition == "No-E":
        p["kappas"][3] = 0.0  # emotional only
        p["W"][:, 3] = 0.0
    elif condition == "No-N":
        p["kappas"][4] = 0.0  # neuro-fast
        p["W"][:, 4] = 0.0
    elif condition == "Random-Gamma":
        # PARAMETER-COUNT CONTROL BASELINE.
        #
        # Random-Gamma is NOT a degraded Full-Gamma; it is an independent
        # experimental condition. It preserves everything about Full-Gamma
        # EXCEPT the biological wiring of the coupling matrix W:
        #   - same 6 subsystems
        #   - same λ_k, α_k, κ_k, x*_k (per-subsystem hyperparameters)
        #   - same τ_SAM=1, τ_HPA=5 (multi-timescale dynamics intact)
        #   - same 30 non-zero entries in W
        #   - W entries permuted (e.g. SAM→E swapped with C→A, etc.)
        #
        # This isolates the contribution of the *biologically-motivated wiring*
        # from the contribution of *raw parameter capacity*. If Full-Gamma
        # outperforms Random-Gamma on P1-P3, the structure matters; if not,
        # the paper's claim that the wiring is necessary is falsified.
        # Addresses REV 5 #3 (parameter-count confound).
        rng = np.random.default_rng(seed=master_seed * 31 + 7)  # deterministic per seed
        flat = p["W"].flatten()
        rng.shuffle(flat)
        p["W"] = flat.reshape(p["W"].shape)
    elif condition == "Collapse-NC":
        # Collapse neuropsych (idx 4) and cognitive (idx 5) into one regulator.
        # Tests whether the 6-subsystem partition is necessary, or whether 5
        # would do (REV 2 #5, REV 3 #4).
        # We force x_N == x_C by averaging their dynamics: equal κ, λ, α, and
        # symmetrize their rows/cols in W.
        avg_k = 0.5 * (p["kappas"][4] + p["kappas"][5])
        avg_l = 0.5 * (p["lambdas"][4] + p["lambdas"][5])
        avg_a = 0.5 * (p["alphas"][4] + p["alphas"][5])
        avg_xs = 0.5 * (p["x_star"][4] + p["x_star"][5])
        p["kappas"][4] = p["kappas"][5] = avg_k
        p["lambdas"][4] = p["lambdas"][5] = avg_l
        p["alphas"][4] = p["alphas"][5] = avg_a
        p["x_star"][4] = p["x_star"][5] = avg_xs
        # Symmetrize W so rows 4,5 and cols 4,5 are interchangeable.
        # The 4 corner cells {(4,4),(4,5),(5,4),(5,5)} must all share one value
        # (the mean of the originals); off-diagonal entries get row/col means.
        W = p["W"]
        corner_mean = 0.25 * (W[4, 4] + W[4, 5] + W[5, 4] + W[5, 5])
        for j in range(6):
            if j in (4, 5):
                continue
            row_avg = 0.5 * (W[4, j] + W[5, j])
            W[4, j] = W[5, j] = row_avg
            col_avg = 0.5 * (W[j, 4] + W[j, 5])
            W[j, 4] = W[j, 5] = col_avg
        W[4, 4] = W[4, 5] = W[5, 4] = W[5, 5] = corner_mean
        p["W"] = W
    
    return p


def gamma_step(x, M, last_opponent_action, params, t, np_rng):
    """One step of Gamma dynamics.
    
    Args:
        x: current 6-dim thermostat vector
        M: memory vector
        last_opponent_action: 'C' or 'D' from opponent's previous round (or 'C' if first round)
        params: hyperparameter dict
        t: round index
        np_rng: seeded numpy.random.Generator for h_sam noise
    """
    lambdas = params["lambdas"]
    alphas = params["alphas"]
    kappas = params["kappas"]
    W = params["W"]
    x_star = params["x_star"]
    
    # Perturbation kappa (applied at t=10)
    kappa_pert = np.zeros(6)
    if t == PERTURBATION_ROUND:
        kappa_pert[2] = 0.5  # stress injection to hormonal
    
    # Drive deficit
    delta = x - x_star + kappa_pert
    
    # Action quality g (computed against OPPONENT's last action, per IPD rules)
    if last_opponent_action == "C":
        g = 3.0 / 5.0  # normalized payoff
    else:
        g = 0.0 if x[5] > 0.5 else 1.0  # simplified
    
    # Update equation
    x_next = x - kappas * delta + lambdas - alphas * g + W @ x
    
    # Clip to [0, 1]
    x_next = np.clip(x_next, 0.0, 1.0)
    
    # SAM signal (fast) — uses seeded Generator (not global state)
    h_sam = x_next[2] - x_next[3] + 0.1 * np_rng.standard_normal()
    
    # Cooperation probability (cognitive thermostat)
    p_coop = 1.0 / (1.0 + np.exp(-5 * (x_next[5] - 0.5)))
    
    return x_next, p_coop, float(h_sam)


# ═══════════════════════════════════════════════════════════════════════════════
# Episode runner
# ═══════════════════════════════════════════════════════════════════════════════

async def run_episode(
    key: CellKey,
    condition: str,
    opponent_policy: str,
    provider: str,
    model: str,
    master_seed: int,
) -> CellResult:
    """Run one 50-round IPD episode.
    
    IPD information rules (Axelrod-faithful):
    - Each round: agents choose simultaneously without seeing the other.
    - After the round: BOTH players observe the post-noise action of the opponent.
    - The agent's gamma_step regulates against `opponent_actions[-1]` (what opponent did),
      while the opponent policy receives `agent_actions_revealed[-1]` (what opponent saw).
    
    Returns CellResult with all metrics.
    """
    # Check for resume
    resume = load_episode_resume(key)
    start_round = 0
    state = None
    
    if resume:
        start_round, state = resume
        start_round += 1  # resume from next round
    
    # Initialize seed factory
    factory = SeedFactory(master_seed)
    
    # Initialize state
    if state is None:
        x = np.array([0.8, 0.5, 0.9, 0.7, 0.6, 1.0])  # far from set-point
        M = np.zeros(10)
        history: list[tuple[str, str]] = []
        agent_actions_revealed: list[str] = []   # what the OPPONENT sees (post-noise our actions)
        opponent_actions: list[str] = []          # what the OPPONENT actually did
    else:
        x = np.array(state["x"])
        M = np.array(state["M"])
        history = state["history"]
        agent_actions_revealed = state["agent_actions_revealed"]
        opponent_actions = state["opponent_actions"]
    
    # Get parameters (with ablation if needed)
    params = make_default_params()
    if condition in ABLATION_CONDITIONS:
        params = apply_ablation(params, condition, master_seed=master_seed)
    
    # Opponent RNG
    opponent_rng_seed = factory.get(f"opponent_{opponent_policy}_rng")
    opponent_rng = __import__("random").Random(opponent_rng_seed)
    
    # Noise RNG (Python random)
    noise_rng_seed = factory.get(f"noise_{key.id}")
    noise_rng = __import__("random").Random(noise_rng_seed)
    
    # Action fallback RNG (Python random)
    fallback_rng_seed = factory.get(f"fallback_{key.id}")
    fallback_rng = __import__("random").Random(fallback_rng_seed)
    
    # Numpy Generator for h_sam noise (seeded, isolated from global state)
    np_rng_seed = factory.get(f"np_{key.id}")
    np_rng = np.random.default_rng(np_rng_seed)
    
    # Tracking
    actions: list[str] = []
    states: list[list[float]] = []
    h_sam_values: list[float] = []
    latencies: list[float] = []
    parse_fails = 0
    fingerprints: list[str | None] = []
    total_cost = 0.0
    
    # Episode loop
    for t in range(start_round, N_ROUNDS):
        # Update Gamma state — uses OPPONENT's last action (per IPD rules)
        last_opp_action = opponent_actions[-1] if opponent_actions else "C"
        x, p_C, h_sam = gamma_step(x, M, last_opp_action, params, t, np_rng)
        
        states.append(x.tolist())
        h_sam_values.append(h_sam)
        
        # Decide action
        if condition in LLM_CONDITIONS:
            # LLM-textual mode
            if condition == "ReAct":
                system_prompt, user_prompt = render_react(t + 1, history)
            elif condition == "pC-only":
                system_prompt, user_prompt = render_pc_only(p_C, h_sam, t + 1, history)
            elif condition == "Scrambled-Gamma":
                system_prompt, user_prompt = get_scrambled_prompt(
                    x.tolist(), p_C, h_sam, t + 1, history
                )
            else:
                system_prompt, user_prompt = render_full_gamma(
                    x.tolist(), p_C, h_sam, t + 1, history
                )
            
            try:
                content, meta = await call_llm(
                    provider, model, system_prompt, user_prompt,
                    seed=factory.get(f"llm_seed_{t}"),
                )
                
                # Parse JSON
                from exp.llm_clients import extract_json
                parsed = extract_json(content)
                
                if parsed and "action" in parsed:
                    action = "C" if parsed["action"].upper() == "C" else "D"
                else:
                    # Default to numerical proposal (seeded fallback)
                    action = "C" if fallback_rng.random() < p_C else "D"
                    parse_fails += 1
                
                latencies.append(meta.get("latency_ms", 0.0))
                fingerprints.append(meta.get("system_fingerprint"))
                
                # Record cost
                cost = record_call(
                    provider, model,
                    meta.get("prompt_tokens", 0),
                    meta.get("completion_tokens", 0),
                )
                total_cost += cost
                
            except Exception:
                # Fallback to numerical (seeded)
                action = "C" if fallback_rng.random() < p_C else "D"
                parse_fails += 1
        elif condition == "HRRL":
            # Faithful Keramati-Gutkin Q-learning baseline
            from exp.hrrl_baseline import hrrl_decide
            action = hrrl_decide(t, opponent_actions, fallback_rng)
        elif condition == "Drive-only":
            # A1-A3 reduction: single drive scalar, no coupling
            from exp.drive_only_baseline import drive_only_decide
            action = drive_only_decide(t, opponent_actions, fallback_rng)
        else:
            # Generic numerical fallback (shouldn't reach here)
            action = "C" if fallback_rng.random() < p_C else "D"
        
        # Apply noise to OUR action (this is what opponent will observe)
        revealed_action = action
        if noise_rng.random() < NOISE_RATE:
            revealed_action = "D" if action == "C" else "C"
        
        # Get opponent action — opponent sees our REVEALED (post-noise) actions
        opp_action = get_opponent_action(opponent_policy, agent_actions_revealed, opponent_rng)
        # Apply noise to opponent's action too (symmetric IPD with noise)
        if noise_rng.random() < NOISE_RATE:
            opp_action = "D" if opp_action == "C" else "C"
        
        # Record both actions
        actions.append(revealed_action)  # log post-noise (what was actually played)
        agent_actions_revealed.append(revealed_action)
        opponent_actions.append(opp_action)
        history.append((revealed_action, opp_action))
        
        # Checkpoint every 10 rounds
        if (t + 1) % 10 == 0:
            save_episode_state(key, t, {
                "x": x.tolist(),
                "M": M.tolist(),
                "history": history,
                "agent_actions_revealed": agent_actions_revealed,
                "opponent_actions": opponent_actions,
            })
    
    # Compute metrics
    metrics = compute_episode_metrics(
        actions, states, h_sam_values,
        perturbation_round=PERTURBATION_ROUND,
    )
    
    # Cleanup episode checkpoint
    clear_episode(key)
    
    # Check for parse fail rate
    parse_fail_rate = parse_fails / N_ROUNDS if N_ROUNDS > 0 else 0.0
    
    # Determine status
    status = "ok"
    if parse_fail_rate > 0.05:
        status = "degraded"
    
    return CellResult(
        key=key,
        rounds=len(actions),
        cooperation_rate=metrics.cooperation_rate,
        action_volatility=metrics.action_volatility,
        half_life=metrics.half_life,
        curvature_beta2=metrics.curvature_beta2,
        cross_lag_peak=metrics.cross_lag_peak,
        coop_recovery_delay=metrics.coop_recovery_delay,
        cost_usd=total_cost,
        latency_mean_ms=sum(latencies) / len(latencies) if latencies else 0.0,
        parse_fail_rate=parse_fail_rate,
        status=status,
        meta={
            "fingerprints": fingerprints,
            "latency_list": latencies,
        },
    )


def get_opponent_action(policy: str, history: list[str], rng) -> str:
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
    else:
        return "C"


# ═══════════════════════════════════════════════════════════════════════════════
# Phase runners
# ═══════════════════════════════════════════════════════════════════════════════

async def run_calibration() -> bool:
    """Calibration phase: validate hyperparameters on seed 42 (numerical only, no LLM).
    
    Uses Drive-only as the calibration condition because it exercises the same
    Gamma dynamics + noise + opponent loop without spending API credits.
    """
    print("[PHASE] Calibration (seed 42, numerical only)...")
    
    # Use Drive-only for calibration: numerical, no LLM, exercises full pipeline
    key = CellKey("Drive-only", "TFT", CALIBRATION_SEED, NUMERICAL_PROVIDER)
    
    try:
        result = await run_episode(
            key, "Drive-only", "TFT", NUMERICAL_PROVIDER, "none", CALIBRATION_SEED
        )
        
        # Smoke test criteria (relaxed for numerical baseline)
        if result.cooperation_rate < 0.05 or result.cooperation_rate > 0.95:
            print(f"  FAIL: degenerate cooperation {result.cooperation_rate:.2f}")
            return False
        
        # Half-life is computed from Gamma states; Drive-only has no Gamma
        # so we just check the cell completed without crashing
        print(f"  OK: coop={result.cooperation_rate:.2f}, switches={result.action_volatility}")
        return True
        
    except Exception as e:
        import traceback
        print(f"  FAIL: {e}")
        traceback.print_exc()
        return False


async def run_preflight() -> bool:
    """Preflight phase: one cell with real LLM."""
    print("[PHASE] Preflight (one real LLM cell)...")
    
    # Try cheapest provider first
    for provider in ["deepseek", "openai", "anthropic"]:
        model = PROVIDER_MODELS[provider]
        key = CellKey("Full-Gamma", "TFT", EVAL_SEEDS[0], provider)
        
        try:
            result = await run_episode(
                key, "Full-Gamma", "TFT", provider, model, EVAL_SEEDS[0]
            )
            
            # Check basic sanity
            if result.parse_fail_rate > 0.5:
                print(f"  WARN {provider}: high parse fail {result.parse_fail_rate:.1%}")
                continue
            
            print(f"  OK {provider}: cost=${result.cost_usd:.3f}, coop={result.cooperation_rate:.2f}")
            return True
            
        except Exception as e:
            print(f"  FAIL {provider}: {e}")
            continue
    
    print("  CRITICAL: No provider passed preflight")
    return False


def _enumerate_cells(
    conditions: list[str],
    providers: list[str],
) -> list[tuple[str, str, str, int, str]]:
    """Enumerate (condition, opponent, provider, seed, model) tuples.
    
    Numerical-only conditions (HRRL, Drive-only) are run ONCE with provider='numerical',
    not duplicated per LLM provider.
    
    Cells are INTERLEAVED by provider so consecutive cells (which run together in
    a parallel batch) hit different providers, distributing API load across all
    providers simultaneously instead of bursting one provider at a time.
    """
    cells = []
    
    for condition in conditions:
        if condition in NUMERICAL_CONDITIONS:
            for opponent in OPPONENTS:
                for seed in EVAL_SEEDS:
                    cells.append((condition, opponent, NUMERICAL_PROVIDER, seed, "none"))
        elif condition in LLM_CONDITIONS:
            # Build per-provider lists then interleave to spread API load
            per_provider: dict[str, list] = {}
            for provider in providers:
                model = PROVIDER_MODELS[provider]
                per_provider[provider] = [
                    (condition, opponent, provider, seed, model)
                    for opponent in OPPONENTS for seed in EVAL_SEEDS
                ]
            # Interleave: round-robin across providers
            max_len = max(len(v) for v in per_provider.values())
            for i in range(max_len):
                for provider in providers:
                    if i < len(per_provider[provider]):
                        cells.append(per_provider[provider][i])
        else:
            print(f"  WARN: unknown condition {condition}, skipping")
    
    return cells


async def _run_one_cell(
    cell_tuple: tuple,
    skipped_providers: set[str],
) -> tuple:
    """Run a single cell; returns (key, result, error_str) for batch processing."""
    condition, opponent, provider, seed, model = cell_tuple
    key = CellKey(condition, opponent, seed, provider)

    if is_cell_complete(key) or provider in skipped_providers:
        return (key, "skip", None)

    try:
        result = await run_episode(key, condition, opponent, provider, model, seed)
        return (key, result, None)
    except Exception as e:
        return (key, None, str(e))


async def run_benchmark(
    conditions: list[str] | None = None,
    providers: list[str] | None = None,
) -> int:
    """Benchmark phase: full grid with numerical-condition deduplication.
    
    Cells are executed in concurrent batches (BATCH_SIZE=6) for ~4-6x speedup.
    Each cell internally runs 50 serial LLM rounds, but multiple cells can
    run in parallel across different providers / within the same provider's
    concurrency limit.
    
    Returns exit code (0=ok, 75=paused, 86=halt).
    """
    conditions = conditions or CONDITIONS
    providers = providers or list(PROVIDER_MODELS.keys())
    
    cells = _enumerate_cells(conditions, providers)
    total_cells = len(cells)
    
    n_llm = sum(1 for c in cells if c[2] != NUMERICAL_PROVIDER)
    n_num = total_cells - n_llm
    print(f"[PHASE] Benchmark: {total_cells} total cells ({n_llm} LLM, {n_num} numerical)")
    
    completed = 0
    skipped_providers: set[str] = set()
    consecutive_fails: dict[str, int] = {}
    global_consec_fails = 0
    MAX_CONSEC_FAILS = 5
    MAX_GLOBAL_CONSEC = 8
    MAX_PROVIDERS_SKIPPED = 2
    
    print(f"         Worker pool: 8 cells in flight at all times")
    
    status_dir = Path("checkpoints/status")
    status_dir.mkdir(parents=True, exist_ok=True)
    
    def write_status(state: str, info: dict | None = None) -> None:
        import time
        payload = {"state": state, "ts": time.time(), "completed": completed,
                   "total": total_cells, "skipped_providers": list(skipped_providers)}
        if info:
            payload.update(info)
        try:
            with open(status_dir / "benchmark.json", "w") as f:
                json.dump(payload, f)
        except Exception:
            pass
    
    write_status("running")
    
    # Pre-filter: skip already-completed and exhausted-provider cells
    pending = []
    for cell in cells:
        condition, opponent, provider, seed, model = cell
        key = CellKey(condition, opponent, seed, provider)
        if is_cell_complete(key):
            completed += 1
            continue
        if provider in skipped_providers:
            continue
        pending.append(cell)
    
    print(f"         {completed} already done, {len(pending)} pending")
    
    # Worker-pool: maintain MAX_INFLIGHT cells in flight at all times.
    # When one finishes, immediately launch the next pending cell. This avoids
    # idle wait time on heterogeneous-latency cells (e.g. fast DeepSeek waiting
    # for slow OpenAI reasoning model in the same batch).
    MAX_INFLIGHT = 8  # ~3 cells per provider on average
    cell_queue = list(pending)  # mutable queue
    inflight: dict[asyncio.Task, tuple] = {}
    halt_code: int | None = None
    
    def launch_next() -> bool:
        """Pop the next valid cell off the queue and launch it. Returns False if queue empty."""
        while cell_queue:
            cell = cell_queue.pop(0)
            condition, opp, prov, seed, model = cell
            if prov in skipped_providers:
                continue  # provider died; skip
            task = asyncio.create_task(_run_one_cell(cell, skipped_providers))
            inflight[task] = cell
            return True
        return False
    
    # Initial fill: launch up to MAX_INFLIGHT
    for _ in range(MAX_INFLIGHT):
        if not launch_next():
            break
    
    last_progress_print = time.time()
    
    while inflight and halt_code is None:
        done, _pending_set = await asyncio.wait(
            inflight.keys(), return_when=asyncio.FIRST_COMPLETED, timeout=300
        )
        
        if not done:
            # 5-min idle timeout — log heartbeat, continue waiting
            print(f"  [heartbeat] {len(inflight)} cells in flight, "
                  f"completed={completed}/{total_cells}", flush=True)
            write_status("running", {"heartbeat": time.time()})
            continue
        
        for task in done:
            cell = inflight.pop(task)
            try:
                raw = task.result()
            except Exception as e:
                print(f"  TASK-EXC for {cell}: {e}", flush=True)
                global_consec_fails += 1
                # Try to launch next
                launch_next()
                continue
            
            key, result_or_skip, err = raw
            provider = key.model  # CellKey stores provider name in 'model' field
            
            if result_or_skip == "skip":
                launch_next()
                continue
            
            if err:
                print(f"  ERROR in {key.id}: {err[:300]}", flush=True)
                consecutive_fails[provider] = consecutive_fails.get(provider, 0) + 1
                global_consec_fails += 1
                
                if "INSUFFICIENT_FUNDS" in err:
                    skipped_providers.add(provider)
                    print(f"  >>> Provider '{provider}' exhausted. Skipping.", flush=True)
                    with open("provider_budget_exhausted.flag", "a") as f:
                        f.write(f"{provider} exhausted at cell {key.id}\n")
                    write_status("running", {"warning": f"{provider}_exhausted"})
                
                elif consecutive_fails[provider] >= MAX_CONSEC_FAILS:
                    print(f"  >>> Provider '{provider}' max fails. Skipping.", flush=True)
                    skipped_providers.add(provider)
                    write_status("running", {"warning": f"{provider}_max_fails"})
                
                if len(skipped_providers) >= MAX_PROVIDERS_SKIPPED:
                    print(f"  HALT: {len(skipped_providers)} providers skipped. Pausing.",
                          flush=True)
                    write_status("paused", {
                        "reason": "too_many_providers_skipped",
                        "skipped": sorted(skipped_providers),
                    })
                    halt_code = 75
                    break
                
                if global_consec_fails >= MAX_GLOBAL_CONSEC:
                    print(f"  HALT: {global_consec_fails} global consecutive failures.",
                          flush=True)
                    write_status("paused", {
                        "reason": "global_consecutive_failures",
                        "n_consecutive": global_consec_fails,
                        "last_error": err[:300],
                    })
                    halt_code = 75
                    break
                
                launch_next()
                continue
            
            # Success
            result = result_or_skip
            save_cell(result)
            completed += 1
            consecutive_fails[provider] = 0
            global_consec_fails = 0
            
            flags = check_cell(key.id, {
                "metrics": result.to_dict(),
                "meta": result.meta,
            })
            
            if should_halt(flags):
                print(f"  HALT: critical red flag in {key.id}", flush=True)
                write_status("halted", {"reason": f"red_flag in {key.id}"})
                halt_code = 86
                break
            
            now = time.time()
            if completed % 5 == 0 or (now - last_progress_print) > 60:
                print(f"  Progress: {completed}/{total_cells} cells "
                      f"({len(inflight)} inflight, "
                      f"skipped: {sorted(skipped_providers) or 'none'})", flush=True)
                write_status("running")
                last_progress_print = now
            
            launch_next()
    
    # If halting, cancel any remaining inflight tasks gracefully
    if halt_code is not None:
        for task in inflight:
            task.cancel()
        await asyncio.gather(*inflight.keys(), return_exceptions=True)
        return halt_code
    
    print(f"  Complete: {completed}/{total_cells} cells "
          f"(skipped providers: {sorted(skipped_providers) or 'none'})", flush=True)
    write_status("done")
    return 0


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Pro-Action Γ experiment runner")
    parser.add_argument("--calibration", action="store_true", help="Run calibration phase")
    parser.add_argument("--preflight", action="store_true", help="Run preflight phase")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark phase")
    parser.add_argument("--all", action="store_true", help="Run all phases")
    parser.add_argument("--conditions", nargs="+", help="Subset of conditions")
    parser.add_argument("--providers", nargs="+", help="Subset of providers")
    parser.add_argument("--reset-budget", action="store_true", help="Reset budget state")
    
    args = parser.parse_args()
    
    if args.reset_budget:
        reset_budget()
        print("Budget state reset")
        return 0
    
    # Check writable
    if not check_writable():
        print("CRITICAL: Checkpoint directory not writable")
        return 86
    
    # Run phases
    if args.calibration or args.all:
        ok = asyncio.run(run_calibration())
        if not ok:
            return 86
    
    if args.preflight or args.all:
        ok = asyncio.run(run_preflight())
        if not ok:
            return 86
    
    if args.benchmark or args.all:
        code = asyncio.run(run_benchmark(args.conditions, args.providers))
        return code
    
    if not any([args.calibration, args.preflight, args.benchmark, args.all]):
        parser.print_help()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
