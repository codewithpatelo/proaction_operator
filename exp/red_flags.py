"""Red-flag detector for anomalous experiment results.

Implements post-run health checks and LLM-as-judge explanation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# ═══════════════════════════════════════════════════════════════════════════════
# Thresholds
# ═══════════════════════════════════════════════════════════════════════════════

DEGENERATE_COOP_MIN = 0.05  # cooperation < 5% or > 95%
DEGENERATE_COOP_MAX = 0.99
MAX_VOLATILITY = 40  # >40 switches in 50 rounds = near-random
MAX_PARSE_FAIL_RATE = 0.25  # >25% JSON failures in one cell
MAX_FINGERPRINTS_PER_CELL = 2  # OpenAI system fingerprint drift


# ═══════════════════════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class RedFlag:
    level: str  # "warning" | "critical"
    category: str
    message: str
    cell_id: str | None = None
    details: dict[str, Any] | None = None


# ═══════════════════════════════════════════════════════════════════════════════
# Detection functions
# ═══════════════════════════════════════════════════════════════════════════════

def check_degenerate_behavior(
    cell_id: str,
    cooperation_rate: float,
    actions: list[str] | None = None,
) -> RedFlag | None:
    """Check for all-C or all-D behavior."""
    if cooperation_rate < DEGENERATE_COOP_MIN:
        return RedFlag(
            level="critical",
            category="degenerate_behavior",
            message=f"Cell {cell_id}: cooperation rate {cooperation_rate:.2f} near zero",
            cell_id=cell_id,
            details={"cooperation_rate": cooperation_rate, "pattern": "all_defect"},
        )
    if cooperation_rate > DEGENERATE_COOP_MAX:
        return RedFlag(
            level="critical",
            category="degenerate_behavior",
            message=f"Cell {cell_id}: cooperation rate {cooperation_rate:.2f} near one",
            cell_id=cell_id,
            details={"cooperation_rate": cooperation_rate, "pattern": "all_cooperate"},
        )
    return None


def check_high_volatility(
    cell_id: str,
    volatility: int,
) -> RedFlag | None:
    """Check for excessive action switching (near-random)."""
    if volatility > MAX_VOLATILITY:
        return RedFlag(
            level="warning",
            category="high_volatility",
            message=f"Cell {cell_id}: {volatility} switches in 50 rounds",
            cell_id=cell_id,
            details={"volatility": volatility, "threshold": MAX_VOLATILITY},
        )
    return None


def check_parse_failures(
    cell_id: str,
    parse_fail_rate: float,
) -> RedFlag | None:
    """Check for excessive JSON parse failures."""
    if parse_fail_rate > MAX_PARSE_FAIL_RATE:
        return RedFlag(
            level="critical",
            category="parse_failures",
            message=f"Cell {cell_id}: {parse_fail_rate:.1%} JSON parse failures",
            cell_id=cell_id,
            details={"parse_fail_rate": parse_fail_rate, "threshold": MAX_PARSE_FAIL_RATE},
        )
    return None


def check_fingerprint_drift(
    cell_id: str,
    fingerprints: list[str | None],
) -> RedFlag | None:
    """Check for OpenAI system fingerprint drift within one cell."""
    unique_fps = set(f for f in fingerprints if f is not None)
    if len(unique_fps) > MAX_FINGERPRINTS_PER_CELL:
        return RedFlag(
            level="warning",
            category="fingerprint_drift",
            message=f"Cell {cell_id}: {len(unique_fps)} distinct fingerprints",
            cell_id=cell_id,
            details={"fingerprints": list(unique_fps)},
        )
    return None


def check_budget_exhaustion(
    provider: str,
    spent: float,
    cap: float,
) -> RedFlag | None:
    """Check if provider budget nearly exhausted."""
    ratio = spent / cap if cap > 0 else 0.0
    if ratio > 0.90:
        return RedFlag(
            level="critical",
            category="budget_exhaustion",
            message=f"Provider {provider}: {ratio:.1%} of budget spent",
            details={"provider": provider, "spent": spent, "cap": cap, "ratio": ratio},
        )
    if ratio > 0.75:
        return RedFlag(
            level="warning",
            category="budget_exhaustion",
            message=f"Provider {provider}: {ratio:.1%} of budget spent",
            details={"provider": provider, "spent": spent, "cap": cap, "ratio": ratio},
        )
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# Cell-level check
# ═══════════════════════════════════════════════════════════════════════════════

def check_cell(
    cell_id: str,
    result: dict[str, Any],
) -> list[RedFlag]:
    """Run all red-flag checks on one cell result.
    
    Returns list of flags (empty if clean).
    """
    flags = []
    
    metrics = result.get("metrics", {})
    meta = result.get("meta", {})
    rounds = metrics.get("rounds", 0)
    
    if rounds < 50:
        flags.append(RedFlag(
            level="warning",
            category="incomplete_cell",
            message=f"Cell {cell_id}: only {rounds} rounds completed",
            cell_id=cell_id,
            details={"rounds": rounds, "expected_rounds": 50},
        ))
        return flags
    
    # Degenerate behavior
    flag = check_degenerate_behavior(
        cell_id,
        metrics.get("cooperation_rate", 0.5),
    )
    if flag:
        flags.append(flag)
    
    # High volatility
    flag = check_high_volatility(
        cell_id,
        metrics.get("action_volatility", 0),
    )
    if flag:
        flags.append(flag)
    
    # Parse failures
    flag = check_parse_failures(
        cell_id,
        meta.get("parse_fail_rate", 0.0),
    )
    if flag:
        flags.append(flag)
    
    # Fingerprint drift (OpenAI only)
    fps = meta.get("fingerprints", [])
    if fps:
        flag = check_fingerprint_drift(cell_id, fps)
        if flag:
            flags.append(flag)
    
    return flags


# ═══════════════════════════════════════════════════════════════════════════════
# Aggregate checks
# ═══════════════════════════════════════════════════════════════════════════════

def check_aggregate(
    results: list[dict[str, Any]],
    budget_summary: dict[str, Any],
) -> list[RedFlag]:
    """Check for aggregate anomalies across all completed cells."""
    flags = []
    
    # Count degenerate cells per condition
    from collections import defaultdict
    degenerate_by_cond = defaultdict(int)
    
    for r in results:
        cond = r.get("condition", "unknown")
        coop = r.get("metrics", {}).get("cooperation_rate", 0.5)
        if coop < DEGENERATE_COOP_MIN or coop > DEGENERATE_COOP_MAX:
            degenerate_by_cond[cond] += 1
    
    # Flag if >2 cells in same condition are degenerate
    for cond, count in degenerate_by_cond.items():
        if count > 2:
            flags.append(RedFlag(
                level="critical",
                category="aggregate_degenerate",
                message=f"Condition {cond}: {count} cells with degenerate behavior",
                details={"condition": cond, "degenerate_count": count},
            ))
    
    # Budget checks per provider
    for provider in ["deepseek", "anthropic", "openai"]:
        p_budget = budget_summary.get(provider, {})
        spent = p_budget.get("spent", 0.0)
        cap = p_budget.get("cap", 1.0)
        flag = check_budget_exhaustion(provider, spent, cap)
        if flag:
            flags.append(flag)
    
    return flags


# ═══════════════════════════════════════════════════════════════════════════════
# LLM-as-judge explanation (cheap, cached)
# ═══════════════════════════════════════════════════════════════════════════════

EXPLANATION_CACHE = Path("prompts_cache/red_flag_explanations.json")
EXPLANATION_CACHE.parent.mkdir(parents=True, exist_ok=True)


def load_explanation_cache() -> dict[str, str]:
    """Load cached explanations."""
    if not EXPLANATION_CACHE.exists():
        return {}
    with open(EXPLANATION_CACHE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_explanation_cache(cache: dict[str, str]) -> None:
    """Save explanation cache."""
    with open(EXPLANATION_CACHE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)


async def explain_flags_with_llm(
    flags: list[RedFlag],
    provider: str = "openai",
    model: str = "gpt-5-nano",
) -> dict[str, str]:
    """Get LLM explanation of red flags (cheap, cached).
    
    Returns dict mapping flag signature to explanation text.
    """
    import asyncio
    
    cache = load_explanation_cache()
    explanations = {}
    
    for flag in flags:
        # Create cache key from flag content
        sig = f"{flag.category}:{flag.level}:{flag.message[:50]}"
        
        if sig in cache:
            explanations[sig] = cache[sig]
            continue
        
        # Generate explanation via LLM (one call per unique flag type)
        prompt = f"""Analyze this experiment anomaly and provide a one-sentence probable cause:

Category: {flag.category}
Level: {flag.level}
Message: {flag.message}
Details: {flag.details}

Probable cause:"""
        
        try:
            from exp.llm_clients import call_llm
            content, _ = await call_llm(provider, model, None, prompt)
            cache[sig] = content.strip()
            explanations[sig] = cache[sig]
        except Exception as e:
            cache[sig] = f"[LLM explanation failed: {e}]"
            explanations[sig] = cache[sig]
        
        # Rate limit self
        await asyncio.sleep(0.5)
    
    save_explanation_cache(cache)
    return explanations


# ═══════════════════════════════════════════════════════════════════════════════
# Decision
# ═══════════════════════════════════════════════════════════════════════════════

def should_halt(flags: list[RedFlag]) -> bool:
    """Return True if any critical flag requires human review (rc=86)."""
    return any(f.level == "critical" for f in flags)


def summarize_flags(flags: list[RedFlag]) -> dict[str, Any]:
    """Return summary dict for reporting."""
    by_level = {"critical": 0, "warning": 0}
    by_category: dict[str, int] = {}
    
    for f in flags:
        by_level[f.level] += 1
        by_category[f.category] = by_category.get(f.category, 0) + 1
    
    return {
        "total": len(flags),
        "by_level": by_level,
        "by_category": by_category,
        "halt_required": should_halt(flags),
    }
