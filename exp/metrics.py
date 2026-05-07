"""Per-cell metrics computation and BFI calculation.

Implements hypothesis-specific metrics (H1, H2, H3) and BFI with bootstrap CIs.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import stats


# ═══════════════════════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class EpisodeMetrics:
    """Metrics computed from one 50-round episode."""
    cooperation_rate: float
    action_volatility: int  # number of switches
    half_life: float | None  # H1: elastic return
    curvature_beta2: float | None  # H2: inverted-U curvature
    cross_lag_peak: int | None  # diagnostic only (internal-state P3, circular)
    coop_recovery_delay: int | None  # H3: delayed reappraisal (EXTERNAL observable)
    mean_h_sam: float
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "cooperation_rate": self.cooperation_rate,
            "action_volatility": self.action_volatility,
            "half_life": self.half_life,
            "curvature_beta2": self.curvature_beta2,
            "cross_lag_peak": self.cross_lag_peak,
            "coop_recovery_delay": self.coop_recovery_delay,
            "mean_h_sam": self.mean_h_sam,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Episode-level metrics
# ═══════════════════════════════════════════════════════════════════════════════

def compute_episode_metrics(
    actions: list[str],
    states: list[list[float]],  # x_t for each round
    h_sam_values: list[float],
    perturbation_round: int = 10,
    set_point: list[float] | None = None,
) -> EpisodeMetrics:
    """Compute all metrics from one episode trace.
    
    Args:
        actions: Agent's chosen actions ('C' or 'D')
        states: Internal state vector x_t for each round
        h_sam_values: SAM stress signal for each round
        perturbation_round: Round where forced defection occurred
        set_point: Target set-point x* (default from paper)
    """
    if set_point is None:
        set_point = [0.3, 0.2, 0.3, 0.1, 0.2, 0.4]
    
    n_rounds = len(actions)
    
    # Primary metric: cooperation rate
    coop_count = sum(1 for a in actions if a == "C")
    cooperation_rate = coop_count / n_rounds if n_rounds > 0 else 0.0
    
    # Primary metric: action volatility (switches)
    switches = sum(
        1 for i in range(1, len(actions)) if actions[i] != actions[i-1]
    )
    
    # H1: Elastic return half-life (post-perturbation)
    half_life = compute_half_life(states, set_point, perturbation_round)
    
    # H2: Inverted-U curvature (quadratic fit of coop rate vs h_sam)
    curvature = compute_curvature(actions, h_sam_values)
    
    # H3 (diagnostic): Cross-lag correlation peak (internal — kept for plots only)
    cross_lag = compute_cross_lag_peak(states)
    
    # H3 (PRE-REGISTERED): External cooperation recovery delay post-perturbation
    coop_recovery = compute_coop_recovery_delay(
        actions, perturbation_round=perturbation_round
    )
    
    # Aggregate h_sam for reporting
    mean_h_sam = np.mean(h_sam_values) if h_sam_values else 0.0
    
    return EpisodeMetrics(
        cooperation_rate=cooperation_rate,
        action_volatility=switches,
        half_life=half_life,
        curvature_beta2=curvature,
        cross_lag_peak=cross_lag,
        coop_recovery_delay=coop_recovery,
        mean_h_sam=mean_h_sam,
    )


def compute_half_life(
    states: list[list[float]],
    set_point: list[float],
    perturbation_round: int,
) -> float | None:
    """Compute rounds to halve distance to set-point after perturbation.
    
    Returns None if convergence not achieved within episode.
    """
    if perturbation_round >= len(states):
        return None
    
    # Distance at perturbation
    x_p = np.array(states[perturbation_round])
    x_star = np.array(set_point)
    dist_p = np.linalg.norm(x_p - x_star)
    
    if dist_p == 0:
        return 0.0  # Already at set-point
    
    target_dist = dist_p / 2.0
    
    # Find first round where distance <= target
    for t in range(perturbation_round + 1, len(states)):
        x_t = np.array(states[t])
        dist_t = np.linalg.norm(x_t - x_star)
        if dist_t <= target_dist:
            return float(t - perturbation_round)
    
    return None  # Did not halve within episode


def compute_curvature(
    actions: list[str],
    h_sam_values: list[float],
) -> float | None:
    """Fit quadratic: cooperation_rate ~ beta0 + beta1*h_sam + beta2*h_sam^2.
    
    Returns beta2 (curvature coefficient). Positive = U-shape, negative = inverted-U.
    """
    if len(actions) < 10 or len(h_sam_values) < 10:
        return None
    
    # Binary cooperation (1=C, 0=D)
    coop_binary = np.array([1.0 if a == "C" else 0.0 for a in actions])
    h_sam = np.array(h_sam_values[:len(actions)])
    
    # Design matrix for quadratic fit
    X = np.column_stack([
        np.ones(len(h_sam)),
        h_sam,
        h_sam ** 2,
    ])
    
    try:
        # OLS fit
        beta = np.linalg.lstsq(X, coop_binary, rcond=None)[0]
        return float(beta[2])  # curvature coefficient
    except np.linalg.LinAlgError:
        return None


def compute_coop_recovery_delay(
    actions: list[str],
    perturbation_round: int,
    window: int = 5,
    max_lag: int = 15,
) -> int | None:
    """External observable for P3 (delayed reappraisal).
    
    Measures rounds after `perturbation_round` until cooperation rate recovers
    to ≥50% of the (pre_shock − post_shock_minimum) gap.
    
    This is fully external (only uses actions, never internal states), so it
    cannot be tautologically derived from any τ_k hyperparameter and addresses
    REV 5 #1 / REV 2 #3 (P3 circularity).
    
    Args:
        actions: list of "C"/"D" actions across the episode
        perturbation_round: round at which the stress event occurs
        window: window size (rounds) for moving cooperation rate
        max_lag: maximum rounds to search for recovery
    
    Returns:
        delay (in rounds) for cooperation to recover halfway, or None if no recovery.
    """
    if len(actions) < perturbation_round + max_lag:
        return None
    
    # Pre-shock baseline (window before perturbation)
    pre_start = max(0, perturbation_round - window)
    pre_actions = actions[pre_start:perturbation_round]
    if not pre_actions:
        return None
    pre_coop = sum(1 for a in pre_actions if a == "C") / len(pre_actions)
    
    # Post-shock minimum (find the lowest-coop window after perturbation)
    post_coop_curve = []
    for t in range(perturbation_round, min(perturbation_round + max_lag, len(actions) - window + 1)):
        win = actions[t:t + window]
        post_coop_curve.append(sum(1 for a in win if a == "C") / len(win))
    
    if not post_coop_curve:
        return None
    
    min_coop = min(post_coop_curve)
    if pre_coop - min_coop < 0.05:  # no meaningful drop → no perturbation impact
        return None
    
    # Half-recovery threshold
    threshold = min_coop + 0.5 * (pre_coop - min_coop)
    
    # Find first lag (after the minimum) where coop ≥ threshold
    min_idx = post_coop_curve.index(min_coop)
    for lag in range(min_idx, len(post_coop_curve)):
        if post_coop_curve[lag] >= threshold:
            return lag
    
    return None  # no recovery within max_lag


def compute_cross_lag_peak(
    states: list[list[float]],
    max_lag: int = 10,
) -> int | None:
    """[INTERNAL — kept for diagnostic only]
    
    Find lag where cognition-emotion correlation peaks. NOTE: this is an
    internal-state observable and is therefore vulnerable to the circularity
    critique (peak ≈ τ_HPA by construction). Kept for diagnostic plots, but
    the pre-registered P3 test now uses `compute_coop_recovery_delay` (external).
    
    Cognition = states[5], Emotion = states[3] (from paper ordering).
    Returns lag with maximum correlation, or None if insufficient data.
    """
    if len(states) < max_lag + 5:
        return None
    
    cognition = np.array([s[5] for s in states])
    emotion = np.array([s[3] for s in states])
    
    # Compute cross-correlation at different lags
    # corr(cognition[t], emotion[t-lag])
    correlations = []
    with np.errstate(invalid="ignore", divide="ignore"):
        for lag in range(min(max_lag, len(states) - 5)):
            if lag == 0:
                c = np.corrcoef(cognition, emotion)[0, 1]
            else:
                c = np.corrcoef(cognition[lag:], emotion[:-lag])[0, 1]
            correlations.append(c if not np.isnan(c) else 0.0)
    
    if not correlations:
        return None
    
    return int(np.argmax(correlations))


# ═══════════════════════════════════════════════════════════════════════════════
# BFI (Behavioral Fidelity Index)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_bfi(
    agent_coop_rates: list[float],
    agent_volatilities: list[float],
    ref_coop_rates: list[float],
    ref_volatilities: list[float],
    lambda_param: float = 0.5,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> tuple[float, tuple[float, float]]:
    """Compute BFI with bootstrap confidence interval.
    
    BFI = exp(-W1(agent, ref) / lambda) averaged over marginals.
    
    Returns:
        (bfi_mean, (ci_lower, ci_upper))
    """
    rng = np.random.default_rng(seed)
    n = len(agent_coop_rates)
    
    if n == 0:
        return 0.0, (0.0, 0.0)
    
    # Compute observed BFI
    w1_coop = stats.wasserstein_distance(agent_coop_rates, ref_coop_rates)
    w1_vol = stats.wasserstein_distance(agent_volatilities, ref_volatilities)
    
    bfi_coop = math.exp(-w1_coop / lambda_param)
    bfi_vol = math.exp(-w1_vol / lambda_param)
    
    bfi_mean = (bfi_coop + bfi_vol) / 2.0
    
    # Bootstrap CI
    bfi_samples = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        a_coop = [agent_coop_rates[i] for i in idx]
        a_vol = [agent_volatilities[i] for i in idx]
        
        w1_c = stats.wasserstein_distance(a_coop, ref_coop_rates)
        w1_v = stats.wasserstein_distance(a_vol, ref_volatilities)
        
        bfi_c = math.exp(-w1_c / lambda_param)
        bfi_v = math.exp(-w1_v / lambda_param)
        bfi_samples.append((bfi_c + bfi_v) / 2.0)
    
    ci_lower = float(np.percentile(bfi_samples, 2.5))
    ci_upper = float(np.percentile(bfi_samples, 97.5))
    
    return bfi_mean, (ci_lower, ci_upper)


# ═══════════════════════════════════════════════════════════════════════════════
# Aggregation utilities
# ═══════════════════════════════════════════════════════════════════════════════

def aggregate_by_condition(
    results: list[dict[str, Any]],
) -> dict[str, dict[str, list[float]]]:
    """Aggregate metrics by condition.
    
    Returns dict[condition] -> {metric_name: [values]}.
    """
    by_cond: dict[str, dict[str, list]] = {}
    
    for r in results:
        cond = r.get("condition", "unknown")
        if cond not in by_cond:
            by_cond[cond] = {
                "cooperation_rate": [],
                "action_volatility": [],
                "half_life": [],
                "curvature_beta2": [],
                "cross_lag_peak": [],
            }
        
        m = r.get("metrics", {})
        by_cond[cond]["cooperation_rate"].append(m.get("cooperation_rate", 0.0))
        by_cond[cond]["action_volatility"].append(m.get("action_volatility", 0))
        
        hl = m.get("half_life")
        if hl is not None:
            by_cond[cond]["half_life"].append(hl)
        
        curv = m.get("curvature_beta2")
        if curv is not None:
            by_cond[cond]["curvature_beta2"].append(curv)
        
        cl = m.get("cross_lag_peak")
        if cl is not None:
            by_cond[cond]["cross_lag_peak"].append(float(cl))
    
    return by_cond
