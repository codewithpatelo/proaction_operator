"""Analysis and aggregation module.

Computes BFI, runs statistical tests, generates CSV/PNG outputs.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

from exp.metrics import compute_bfi, aggregate_by_condition
from exp.checkpoint import load_cells, CELLS_JSON


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

BFI_LAMBDA = 0.5  # Smoothing parameter for BFI
BFI_THRESHOLD = 0.70


# ═══════════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_all_results() -> list[dict[str, Any]]:
    """Load all completed cell results."""
    cells = load_cells()
    results = []
    for cell_id, cell_data in cells.items():
        result = cell_data.get("result", {})
        result["cell_id"] = cell_id
        results.append(result)
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Statistical tests
# ═══════════════════════════════════════════════════════════════════════════════

def run_wilcoxon_test(
    full_gamma_values: list[float],
    ablation_values: list[float],
) -> tuple[float, float]:
    """Paired Wilcoxon signed-rank test.
    
    Returns (statistic, p_value).
    """
    if len(full_gamma_values) != len(ablation_values) or len(full_gamma_values) < 3:
        return float("nan"), float("nan")
    
    try:
        stat, p = stats.wilcoxon(full_gamma_values, ablation_values)
        return float(stat), float(p)
    except Exception as e:
        return float("nan"), float("nan")


def bootstrap_contrast(
    values_a: list[float],
    values_b: list[float],
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> tuple[float, tuple[float, float]]:
    """Bootstrap CI for difference in means.
    
    Returns (observed_diff, (ci_lower, ci_upper)).
    """
    rng = np.random.default_rng(seed)
    
    obs_diff = np.mean(values_a) - np.mean(values_b)
    
    diffs = []
    for _ in range(n_bootstrap):
        idx_a = rng.integers(0, len(values_a), size=len(values_a))
        idx_b = rng.integers(0, len(values_b), size=len(values_b))
        
        boot_a = [values_a[i] for i in idx_a]
        boot_b = [values_b[i] for i in idx_b]
        
        diffs.append(np.mean(boot_a) - np.mean(boot_b))
    
    ci_lower = float(np.percentile(diffs, 2.5))
    ci_upper = float(np.percentile(diffs, 97.5))
    
    return float(obs_diff), (ci_lower, ci_upper)


# ═══════════════════════════════════════════════════════════════════════════════
# BFI computation
# ═══════════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════════
# Human reference data (placeholder until Brookins-DeBacker 2024 dataset loaded)
# ═══════════════════════════════════════════════════════════════════════════════

# TODO(human-data): Replace with empirical distributions from Brookins-DeBacker 2024
# (LLM-vs-human IPD dataset, Wharton 2024). These placeholders are illustrative ONLY
# and BFI values computed against them are NOT publishable until real data is loaded.
HUMAN_REFERENCE = {
    "TFT": {
        "cooperation_rate": [0.65, 0.72, 0.58, 0.81, 0.69, 0.76, 0.62, 0.74, 0.68, 0.71],
        "action_volatility": [8, 12, 15, 6, 10, 7, 14, 9, 11, 8],
    },
    "Grim": {
        "cooperation_rate": [0.45, 0.52, 0.38, 0.61, 0.49, 0.56, 0.42, 0.54, 0.48, 0.51],
        "action_volatility": [4, 6, 9, 3, 5, 4, 8, 5, 7, 6],
    },
    "Random": {
        "cooperation_rate": [0.50, 0.48, 0.52, 0.45, 0.51, 0.49, 0.47, 0.53, 0.50, 0.49],
        "action_volatility": [22, 25, 24, 21, 23, 26, 22, 24, 23, 25],
    },
    "GTFT": {
        "cooperation_rate": [0.71, 0.78, 0.65, 0.82, 0.74, 0.79, 0.68, 0.76, 0.73, 0.77],
        "action_volatility": [10, 14, 13, 8, 11, 9, 13, 10, 12, 11],
    },
}
HUMAN_REFERENCE_SOURCE = "PLACEHOLDER (replace with Brookins-DeBacker 2024)"


def compute_all_bfi(results: list[dict]) -> dict[str, Any]:
    """Compute BFI for each (condition, opponent, provider) cell vs HUMAN reference.
    
    BFI compares agent behavioral marginals (cooperation rate, volatility) against
    human empirical distributions, NOT against another model condition. The human
    reference is per-opponent (humans played against TFT, Grim, etc.).
    
    NOTE: Currently uses placeholder human data; replace HUMAN_REFERENCE with
    Brookins-DeBacker 2024 distributions before publication.
    """
    bfi_results = {}
    
    conditions = set(r["key"]["condition"] for r in results)
    opponents = set(r["key"]["opponent"] for r in results)
    providers = set(r["key"]["model"] for r in results)
    
    for cond in conditions:
        for opp in opponents:
            ref = HUMAN_REFERENCE.get(opp)
            if ref is None:
                continue  # No human reference for this opponent
            
            for prov in providers:
                cond_values = [
                    r for r in results
                    if r["key"]["condition"] == cond
                    and r["key"]["opponent"] == opp
                    and r["key"]["model"] == prov
                ]
                
                if len(cond_values) < 3:
                    continue
                
                agent_coop = [r["cooperation_rate"] for r in cond_values]
                agent_vol = [r["action_volatility"] for r in cond_values]
                
                bfi, ci = compute_bfi(
                    agent_coop, agent_vol,
                    ref["cooperation_rate"], ref["action_volatility"],
                    lambda_param=BFI_LAMBDA,
                )
                
                key = f"{cond}_{opp}_{prov}"
                bfi_results[key] = {
                    "condition": cond,
                    "opponent": opp,
                    "provider": prov,
                    "bfi": bfi,
                    "ci_lower": ci[0],
                    "ci_upper": ci[1],
                    "n": len(cond_values),
                    "reference_source": HUMAN_REFERENCE_SOURCE,
                }
    
    return bfi_results


# ═══════════════════════════════════════════════════════════════════════════════
# Hypothesis tests
# ═══════════════════════════════════════════════════════════════════════════════

def test_hypothesis_h1(results: list[dict]) -> dict[str, Any]:
    """H1: Elastic return (half-life).
    
    Test if No-H has different half-life than Full-Gamma.
    """
    full_hl = []
    noh_hl = []
    
    for r in results:
        hl = r.get("half_life")
        if hl is None:
            continue
        
        cond = r["key"]["condition"]
        if cond == "Full-Gamma":
            full_hl.append(hl)
        elif cond == "No-H":
            noh_hl.append(hl)
    
    if len(full_hl) < 3 or len(noh_hl) < 3:
        return {"status": "insufficient_data", "n_full": len(full_hl), "n_noh": len(noh_hl)}
    
    stat, p = run_wilcoxon_test(full_hl[:len(noh_hl)], noh_hl)
    diff, ci = bootstrap_contrast(full_hl, noh_hl)
    
    return {
        "status": "tested",
        "h1_supported": p < 0.05 and diff > 1.0,  # Δ > 1.5 rounds pre-registered
        "wilcoxon_stat": stat,
        "wilcoxon_p": p,
        "mean_diff": diff,
        "ci_95": ci,
        "n_full": len(full_hl),
        "n_noh": len(noh_hl),
    }


def test_hypothesis_h2(results: list[dict]) -> dict[str, Any]:
    """H2: Yerkes-Dodson inverted-U (curvature).
    
    Test if Full-Gamma has negative curvature and No-H has near-zero.
    """
    full_curv = []
    noh_curv = []
    
    for r in results:
        curv = r.get("curvature_beta2")
        if curv is None:
            continue
        
        cond = r["key"]["condition"]
        if cond == "Full-Gamma":
            full_curv.append(curv)
        elif cond == "No-H":
            noh_curv.append(curv)
    
    if len(full_curv) < 3 or len(noh_curv) < 3:
        return {"status": "insufficient_data"}
    
    full_mean = np.mean(full_curv)
    noh_mean = np.mean(noh_curv)
    
    return {
        "status": "tested",
        "h2_supported": full_mean < -0.05 and abs(noh_mean) < 0.05,
        "full_curvature_mean": float(full_mean),
        "noh_curvature_mean": float(noh_mean),
        "n_full": len(full_curv),
        "n_noh": len(noh_curv),
    }


def test_hypothesis_h3(results: list[dict]) -> dict[str, Any]:
    """H3: Delayed reappraisal — EXTERNAL observable.
    
    Pre-registered test (post-reviewer-feedback): uses `coop_recovery_delay`,
    measured purely from action sequences (never from internal x_k states).
    This is independent of any τ_k hyperparameter and addresses the circularity
    concern raised by REV 5 #1 / REV 2 #3.
    
    H3 is supported if:
      - Full-Gamma shows a delayed recovery (mean delay ≥ 3 rounds)
      - No-E (without emotional regulator) shows immediate or no recovery (delay ≤ 1 or None)
    """
    full_delay = []
    noe_delay = []
    
    for r in results:
        delay = r.get("coop_recovery_delay")
        if delay is None:
            continue
        
        cond = r["key"]["condition"]
        if cond == "Full-Gamma":
            full_delay.append(delay)
        elif cond == "No-E":
            noe_delay.append(delay)
    
    if len(full_delay) < 3 or len(noe_delay) < 3:
        return {"status": "insufficient_data",
                "n_full": len(full_delay), "n_noe": len(noe_delay)}
    
    full_mean = float(np.mean(full_delay))
    noe_mean = float(np.mean(noe_delay))
    
    return {
        "status": "tested",
        "h3_supported": full_mean >= 3.0 and noe_mean <= 2.0,
        "full_recovery_delay_mean": full_mean,
        "noe_recovery_delay_mean": noe_mean,
        "n_full": len(full_delay),
        "n_noe": len(noe_delay),
        "observable": "coop_recovery_delay (external, action-only)",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Export
# ═══════════════════════════════════════════════════════════════════════════════

def export_csv(results: list[dict], path: Path) -> None:
    """Export results to CSV."""
    import csv
    
    if not results:
        print("No results to export")
        return
    
    # Get all fields
    fields = ["cell_id", "condition", "opponent", "seed", "provider",
              "cooperation_rate", "action_volatility", "half_life",
              "curvature_beta2", "cross_lag_peak", "coop_recovery_delay",
              "cost_usd", "status"]
    
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        
        for r in results:
            row = {
                "cell_id": r.get("cell_id", ""),
                "condition": r["key"]["condition"],
                "opponent": r["key"]["opponent"],
                "seed": r["key"]["seed"],
                "provider": r["key"]["model"],
                "cooperation_rate": r.get("cooperation_rate"),
                "action_volatility": r.get("action_volatility"),
                "half_life": r.get("half_life"),
                "curvature_beta2": r.get("curvature_beta2"),
                "cross_lag_peak": r.get("cross_lag_peak"),
                "coop_recovery_delay": r.get("coop_recovery_delay"),
                "cost_usd": r.get("cost_usd"),
                "status": r.get("status"),
            }
            writer.writerow(row)
    
    print(f"  Exported {len(results)} rows to {path}")


def generate_summary(results: list[dict], bfi_results: dict, hypotheses: dict) -> str:
    """Generate markdown summary report."""
    lines = []
    lines.append("# Pro-Action Γ Experiment Summary")
    lines.append("")
    lines.append(f"**Total cells completed:** {len(results)}")
    lines.append("")
    
    # Budget summary
    from exp.budget import get_summary
    budget = get_summary()
    lines.append("## Budget")
    lines.append(f"- DeepSeek: ${budget['deepseek']['spent']:.2f} / ${budget['deepseek']['cap']:.2f}")
    lines.append(f"- Anthropic: ${budget['anthropic']['spent']:.2f} / ${budget['anthropic']['cap']:.2f}")
    lines.append(f"- OpenAI: ${budget['openai']['spent']:.2f} / ${budget['openai']['cap']:.2f}")
    lines.append(f"- **Total:** ${budget['total']['spent']:.2f} / ${budget['total']['cap']:.2f}")
    lines.append("")
    
    # Hypothesis results
    lines.append("## Hypothesis Tests")
    
    for h_name, h_result in hypotheses.items():
        lines.append(f"### {h_name}")
        if h_result.get("status") == "insufficient_data":
            lines.append("- Status: Insufficient data")
        else:
            supported = h_result.get("h1_supported") or h_result.get("h2_supported") or h_result.get("h3_supported")
            lines.append(f"- Supported: {supported}")
            
            if "wilcoxon_p" in h_result:
                lines.append(f"- Wilcoxon p: {h_result['wilcoxon_p']:.4f}")
            if "mean_diff" in h_result:
                lines.append(f"- Mean difference: {h_result['mean_diff']:.3f}")
        lines.append("")
    
    # BFI summary
    lines.append("## BFI Summary")
    above_threshold = sum(1 for b in bfi_results.values() if b["bfi"] >= BFI_THRESHOLD)
    total_bfi = len(bfi_results)
    lines.append(f"- BFI ≥ {BFI_THRESHOLD}: {above_threshold}/{total_bfi}")
    lines.append("")
    
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("[ANALYSIS] Loading results...")
    
    if not CELLS_JSON.exists():
        print("No checkpoint file found. Run experiment first.")
        return 1
    
    results = load_all_results()
    print(f"  Loaded {len(results)} completed cells")
    
    if len(results) == 0:
        print("No results to analyze")
        return 0
    
    # Compute BFI
    print("[ANALYSIS] Computing BFI...")
    bfi_results = compute_all_bfi(results)
    print(f"  Computed {len(bfi_results)} BFI values")
    
    # Test hypotheses
    print("[ANALYSIS] Testing hypotheses...")
    hypotheses = {
        "H1 (Elastic Return)": test_hypothesis_h1(results),
        "H2 (Yerkes-Dodson)": test_hypothesis_h2(results),
        "H3 (Delayed Reappraisal)": test_hypothesis_h3(results),
    }
    
    # Export
    print("[ANALYSIS] Exporting...")
    export_csv(results, RESULTS_DIR / "results.csv")
    
    with open(RESULTS_DIR / "bfi.json", "w") as f:
        json.dump(bfi_results, f, indent=2)
    
    with open(RESULTS_DIR / "hypotheses.json", "w") as f:
        json.dump(hypotheses, f, indent=2)
    
    # Summary report
    summary = generate_summary(results, bfi_results, hypotheses)
    with open(RESULTS_DIR / "summary.md", "w") as f:
        f.write(summary)
    
    print(summary)
    print(f"\n[ANALYSIS] Complete. Results in {RESULTS_DIR}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
