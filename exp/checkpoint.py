"""Checkpoint management for resilient experiment execution.

Implements atomic writes, resume capability, and idempotent cell completion.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from dataclasses import dataclass, asdict


# ═══════════════════════════════════════════════════════════════════════════════
# Paths
# ═══════════════════════════════════════════════════════════════════════════════

CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)

CELLS_JSON = CHECKPOINT_DIR / "cells.json"
RUN_PREFIX = CHECKPOINT_DIR / "run_"


# ═══════════════════════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CellKey:
    """Unique identifier for one experimental cell."""
    condition: str
    opponent: str
    seed: int
    model: str
    
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
    
    @property
    def id(self) -> str:
        return f"{self.condition}_{self.opponent}_s{self.seed}_{self.model}"


@dataclass  
class CellResult:
    """Result of one completed cell."""
    key: CellKey
    rounds: int
    cooperation_rate: float
    action_volatility: int
    half_life: float | None
    curvature_beta2: float | None
    cross_lag_peak: int | None
    coop_recovery_delay: int | None
    cost_usd: float
    latency_mean_ms: float
    parse_fail_rate: float
    status: str  # "ok", "degraded", "failed"
    meta: dict[str, Any]
    
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ═══════════════════════════════════════════════════════════════════════════════
# Atomic file operations
# ═══════════════════════════════════════════════════════════════════════════════

def atomic_write_json(path: Path, data: Any) -> None:
    """Write JSON atomically using tmp+rename pattern."""
    tmp_path = path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    os.replace(tmp_path, path)


def atomic_append_jsonl(path: Path, record: dict) -> None:
    """Append one JSON line atomically."""
    tmp_path = path.with_suffix(".tmp")
    line = json.dumps(record, default=str) + "\n"
    
    # Read existing if present
    existing = ""
    if path.exists():
        existing = path.read_text(encoding="utf-8")
    
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(existing + line)
    
    os.replace(tmp_path, path)


# ═══════════════════════════════════════════════════════════════════════════════
# Cell-level checkpointing
# ═══════════════════════════════════════════════════════════════════════════════

def load_cells() -> dict[str, dict]:
    """Load completed cells index."""
    if not CELLS_JSON.exists():
        return {}
    with open(CELLS_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def save_cell(result: CellResult) -> None:
    """Record a completed cell atomically."""
    cells = load_cells()
    cells[result.key.id] = {
        "key": result.key.to_dict(),
        "result": asdict(result),
        "timestamp": time_stamp(),
    }
    atomic_write_json(CELLS_JSON, cells)


def is_cell_complete(key: CellKey) -> bool:
    """Check if cell already exists in index with full rounds."""
    cells = load_cells()
    if key.id not in cells:
        return False
    # Verify the cell actually completed 50 rounds (not a partial/crashed run)
    return cells[key.id].get("result", {}).get("rounds", 0) >= 50


# ═══════════════════════════════════════════════════════════════════════════════
# Episode-level checkpointing (per-round resume within one cell)
# ═══════════════════════════════════════════════════════════════════════════════

def episode_path(key: CellKey) -> Path:
    return RUN_PREFIX / f"{key.id}.json"


def save_episode_state(key: CellKey, round_num: int, state: dict) -> None:
    """Save mid-episode state for crash recovery."""
    path = episode_path(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    data = {
        "key": key.to_dict(),
        "last_completed_round": round_num,
        "state": state,
        "timestamp": time_stamp(),
    }
    atomic_write_json(path, data)


def load_episode_resume(key: CellKey) -> tuple[int, dict] | None:
    """Return (last_completed_round, state) if episode exists, else None."""
    path = episode_path(key)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["last_completed_round"], data["state"]


def clear_episode(key: CellKey) -> None:
    """Remove episode checkpoint after successful cell completion."""
    path = episode_path(key)
    if path.exists():
        path.unlink()


# ═══════════════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════════════

import datetime

def time_stamp() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


# ═══════════════════════════════════════════════════════════════════════════════
# Verification
# ═══════════════════════════════════════════════════════════════════════════════

def check_writable() -> bool:
    """Verify checkpoint directory is writable."""
    try:
        test_file = CHECKPOINT_DIR / ".write_test"
        test_file.write_text("test")
        test_file.unlink()
        return True
    except Exception:
        return False
