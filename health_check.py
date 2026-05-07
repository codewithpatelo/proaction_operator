"""Periodic health check for running benchmark data.

Usage:
    python health_check.py           # one-shot analysis
    python health_check.py --watch  # loop every 30 min (for manual long runs)

Checks cells.json for anomalies and uses an LLM (gpt-5.5-nano via OpenAI)
for intelligent triage when issues are detected.
"""

import asyncio
import json
import os
import sys
import time

# Fix Windows console encoding for emoji/unicode output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, ".")

CHECKPOINTS = Path("checkpoints")
CELLS_JSON = CHECKPOINTS / "cells.json"
HEALTH_LOG = Path("logs/health_check.log")
REPORT_PATH = Path("logs/health_report.md")


def load_cells() -> dict:
    if not CELLS_JSON.exists():
        return {}
    try:
        with open(CELLS_JSON, "r") as f:
            return json.load(f)
    except Exception as e:
        return {"_error": str(e)}


def compute_stats(cells: dict) -> dict:
    if not cells or "_error" in cells:
        return {"status": "no_data", "error": cells.get("_error", "no cells.json")}

    n = len(cells)
    by_provider = defaultdict(list)
    by_condition = defaultdict(list)
    statuses = defaultdict(int)
    parse_fails = []
    coops = []
    costs = []
    anomalous = []

    for cid, cell in cells.items():
        k = cell.get("key", {})
        prov = k.get("model", "?")
        cond = k.get("condition", "?")
        by_provider[prov].append(cell)
        by_condition[cond].append(cell)
        statuses[cell.get("status", "?")] += 1

        pf = cell.get("parse_fail_rate")
        if pf is not None:
            parse_fails.append(pf)
            if pf > 0.30:
                anomalous.append((cid, f"parse_fail_rate={pf:.1%}"))

        cr = cell.get("cooperation_rate")
        if cr is not None:
            coops.append(cr)
            if cr in (0.0, 1.0):
                anomalous.append((cid, f"coop_rate={cr} (stuck)"))

        cost = cell.get("cost_usd", 0)
        costs.append(cost)
        if cost > 0.05:  # unusually high for a single cell
            anomalous.append((cid, f"cost=${cost:.4f} (high)"))

    # Check for missing/incomplete baselines (critical anomaly)
    baseline_targets = {
        "ReAct": 120,      # 4 opponents × 3 providers × 10 seeds
        "Drive-only": 40,   # 4 opponents × 10 seeds (numerical)
        "HRRL": 40,         # 4 opponents × 10 seeds (numerical)
    }
    for bc, target in baseline_targets.items():
        current = len(by_condition.get(bc, []))
        if current == 0:
            anomalous.append((f"condition:{bc}", f"baseline {bc} has 0 cells - NOT RUN"))
        elif current < target:
            anomalous.append((f"condition:{bc}", f"baseline {bc} incomplete: {current}/{target} cells"))

    stats = {
        "status": "ok",
        "n_cells": n,
        "by_status": dict(statuses),
        "providers": {p: len(v) for p, v in by_provider.items()},
        "conditions": {c: len(v) for c, v in by_condition.items()},
        "parse_fail": {"mean": sum(parse_fails)/len(parse_fails) if parse_fails else 0,
                       "max": max(parse_fails) if parse_fails else 0,
                       "n_above_30pct": sum(1 for p in parse_fails if p > 0.30)},
        "cooperation": {"mean": sum(coops)/len(coops) if coops else 0,
                        "min": min(coops) if coops else 0,
                        "max": max(coops) if coops else 0,
                        "n_extreme": sum(1 for c in coops if c in (0.0, 1.0))},
        "cost": {"total": sum(costs), "mean": sum(costs)/len(costs) if costs else 0,
                 "max": max(costs) if costs else 0},
        "anomalous_cells": anomalous,
    }

    # Overall health score
    score = 100
    if stats["parse_fail"]["n_above_30pct"] > 3:
        score -= 30
    if stats["cooperation"]["n_extreme"] > 5:
        score -= 30
    if stats["cost"]["max"] > 0.05:
        score -= 20
    if statuses.get("failed", 0) > 2:
        score -= 20

    stats["health_score"] = max(0, score)
    stats["needs_attention"] = score < 70 or len(anomalous) > 0
    return stats


async def llm_triage(stats: dict) -> str:
    """Call OpenAI gpt-5.5 for intelligent triage when anomalies detected.
    
    Per OpenAI docs (developers.openai.com/api/docs/models/gpt-5.5, May 2026):
      - gpt-5.5 is a reasoning model
      - temperature and max_tokens are NOT supported (fixed internally)
      - supports reasoning.effort: none/low/medium/high/xhigh (we use 'low'
        for fast, cheap triage — anomaly detection doesn't need deep reasoning)
      - JSON output format is supported via response_format
    """
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPEN_AI_API_KEY")
    if not api_key:
        return "No OPENAI_API_KEY available for LLM triage."

    try:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=api_key)
    except ImportError:
        return "openai package not installed."

    system = (
        "You are an experiment-health triage assistant. "
        "Given statistics from a multi-condition LLM-agent benchmark, identify "
        "the most likely root cause and recommend ONE concrete action. "
        "Be terse. Focus on: API issues, code bugs, or data quality. "
        "Respond ONLY with valid JSON of the form: "
        '{"verdict": "ok|warning|critical", '
        '"root_cause": "<short string>", '
        '"recommendation": "<one concrete action>", '
        '"notes": ["<bullet>", "<bullet>", "<bullet>"]}'
    )

    user = json.dumps({
        "health_score": stats["health_score"],
        "n_cells": stats["n_cells"],
        "parse_fail_mean": stats["parse_fail"]["mean"],
        "parse_fail_max": stats["parse_fail"]["max"],
        "n_parse_anomalies": stats["parse_fail"]["n_above_30pct"],
        "coop_mean": stats["cooperation"]["mean"],
        "n_stuck_cells": stats["cooperation"]["n_extreme"],
        "total_cost": stats["cost"]["total"],
        "max_cell_cost": stats["cost"]["max"],
        "by_status": stats["by_status"],
        "top_anomalies": stats["anomalous_cells"][:10],
    }, indent=2)

    try:
        response = await client.chat.completions.create(
            model="gpt-5.5",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={"type": "json_object"},
            reasoning_effort="low",  # fast triage, not deep analysis
        )
        content = response.choices[0].message.content or ""
        if not content:
            return "(empty LLM response)"
        # Parse and re-format as readable text
        try:
            parsed = json.loads(content)
            lines = [
                f"**Verdict:** {parsed.get('verdict', '?')}",
                f"**Root cause:** {parsed.get('root_cause', '?')}",
                f"**Recommendation:** {parsed.get('recommendation', '?')}",
                "",
                "**Notes:**",
            ]
            for note in parsed.get("notes", [])[:5]:
                lines.append(f"- {note}")
            return "\n".join(lines)
        except json.JSONDecodeError:
            return content  # fallback to raw
    except Exception as e:
        return f"LLM triage failed: {e}"


def format_report(stats: dict, llm_advice: str, timestamp: str) -> str:
    lines = [
        f"# Health Report — {timestamp}",
        "",
        f"**Health Score:** {stats['health_score']}/100  ",
        f"**Cells Completed:** {stats['n_cells']}  ",
        f"**Total Cost:** ${stats['cost']['total']:.3f}  ",
        f"**Needs Attention:** {'YES ⚠️' if stats['needs_attention'] else 'No ✅'}",
        "",
        "## Summary Stats",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Parse fail (mean) | {stats['parse_fail']['mean']:.1%} |",
        f"| Parse fail (max)  | {stats['parse_fail']['max']:.1%} |",
        f"| Coop rate (mean)  | {stats['cooperation']['mean']:.3f} |",
        f"| Coop rate (extreme 0/1) | {stats['cooperation']['n_extreme']} |",
        f"| Cell cost (mean)  | ${stats['cost']['mean']:.4f} |",
        f"| Cell cost (max)   | ${stats['cost']['max']:.4f} |",
        "",
        "## Distribution by Status",
        "",
    ]
    for st, cnt in sorted(stats["by_status"].items()):
        lines.append(f"- **{st}**: {cnt}")

    lines += [
        "",
        "## Distribution by Provider",
        "",
    ]
    for prov, cnt in sorted(stats["providers"].items()):
        lines.append(f"- **{prov}**: {cnt}")

    lines += [
        "",
        "## Anomalous Cells",
        "",
    ]
    if stats["anomalous_cells"]:
        for cid, reason in stats["anomalous_cells"][:15]:
            lines.append(f"- `{cid}` — {reason}")
    else:
        lines.append("None detected.")

    lines += [
        "",
        "## LLM Triage Advice",
        "",
        llm_advice,
        "",
        "---",
        f"_Auto-generated by health_check.py_",
    ]

    return "\n".join(lines)


async def run_once() -> dict:
    cells = load_cells()
    stats = compute_stats(cells)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    llm_advice = ""
    if stats["needs_attention"]:
        print(f"[{timestamp}] Anomalies detected ({len(stats['anomalous_cells'])}). Running LLM triage...")
        llm_advice = await llm_triage(stats)
    else:
        llm_advice = "No anomalies — benchmark running within expected parameters."

    report = format_report(stats, llm_advice, timestamp)

    # Write report
    HEALTH_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)

    # Append one-liner to log
    with open(HEALTH_LOG, "a", encoding="utf-8") as f:
        attn = "⚠️" if stats["needs_attention"] else "✅"
        f.write(f"{timestamp} | score={stats['health_score']}/100 | cells={stats['n_cells']} | "
                f"cost=${stats['cost']['total']:.3f} | anomalies={len(stats['anomalous_cells'])} {attn}\n")

    print(report)
    return stats


async def watch_loop(interval_minutes: int = 30) -> None:
    print(f"Health-check watch mode: every {interval_minutes} minutes. Ctrl+C to stop.")
    while True:
        try:
            await run_once()
        except Exception as e:
            print(f"Health check error: {e}")
        try:
            await asyncio.sleep(interval_minutes * 60)
        except asyncio.CancelledError:
            break


def main():
    parser = argparse.ArgumentParser(description="Benchmark health monitor")
    parser.add_argument("--watch", action="store_true", help="Run in loop every 30 min")
    parser.add_argument("--interval", type=int, default=30, help="Minutes between checks")
    args = parser.parse_args()

    if args.watch:
        asyncio.run(watch_loop(args.interval))
    else:
        asyncio.run(run_once())


if __name__ == "__main__":
    import argparse
    main()
