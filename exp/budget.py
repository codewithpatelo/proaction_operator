"""Budget tracking and hard caps per provider.

Implements provider-level caps to prevent auto-recharge overspend.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import datetime


# ═══════════════════════════════════════════════════════════════════════════════
# Provider caps (matched to user's actual wallet)
# ═══════════════════════════════════════════════════════════════════════════════

PROVIDER_CAPS: dict[str, float] = {
    "deepseek": 4.75,    # $5 hard, 5% headroom
    "anthropic": 18.00,  # $20 + auto-recharge, stay below threshold
    "openai": 13.15,     # $13.84 demand balance, 5% headroom
}

GLOBAL_CAP: float = 50.0  # Total across all providers

# Price tables (per million tokens) — updated as of plan
PRICES: dict[str, dict[str, tuple[float, float]]] = {
    "deepseek": {
        "deepseek-v4-flash": (0.14, 0.28),
        "deepseek-v4-pro": (0.50, 2.00),
    },
    "anthropic": {
        "claude-haiku-4-5": (1.00, 5.00),
        "claude-3-5-haiku-latest": (0.50, 2.50),  # fallback
    },
    "openai": {
        "gpt-5-nano": (0.05, 0.40),
    },
}

# ═══════════════════════════════════════════════════════════════════════════════
# State management
# ═══════════════════════════════════════════════════════════════════════════════

BUDGET_FILE = Path("budget_state.json")


@dataclass
class BudgetState:
    deepseek_spent: float = 0.0
    anthropic_spent: float = 0.0
    openai_spent: float = 0.0
    last_updated: str = ""
    
    @property
    def total(self) -> float:
        return self.deepseek_spent + self.anthropic_spent + self.openai_spent
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: dict) -> "BudgetState":
        return cls(**d)


def load_budget() -> BudgetState:
    """Load budget state from disk."""
    if not BUDGET_FILE.exists():
        return BudgetState(last_updated=now())
    with open(BUDGET_FILE, "r", encoding="utf-8") as f:
        return BudgetState.from_dict(json.load(f))


def save_budget(state: BudgetState) -> None:
    """Atomically write budget state."""
    state.last_updated = now()
    tmp = BUDGET_FILE.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state.to_dict(), f, indent=2)
    os.replace(tmp, BUDGET_FILE)


def now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


# ═══════════════════════════════════════════════════════════════════════════════
# Cost calculation
# ═══════════════════════════════════════════════════════════════════════════════

def estimate_cost(provider: str, model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Estimate cost in USD for one call."""
    provider_prices = PRICES.get(provider, {})
    prices = provider_prices.get(model, (0.0, 0.0))
    in_price, out_price = prices
    
    return (prompt_tokens * in_price + completion_tokens * out_price) / 1_000_000


def record_call(provider: str, model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Record a call and return its cost. Triggers exit if cap exceeded."""
    cost = estimate_cost(provider, model, prompt_tokens, completion_tokens)
    
    state = load_budget()
    
    # Update provider spent
    if provider == "deepseek":
        state.deepseek_spent += cost
    elif provider == "anthropic":
        state.anthropic_spent += cost
    elif provider == "openai":
        state.openai_spent += cost
    else:
        # Unknown provider — log but don't count against caps
        pass
    
    # Check caps
    provider_cap = PROVIDER_CAPS.get(provider, float("inf"))
    provider_spent = getattr(state, f"{provider}_spent", 0.0)
    
    if provider_spent > provider_cap:
        save_budget(state)
        flag_path = Path("provider_budget_exhausted.flag")
        flag_path.write_text(f"{provider} cap {provider_cap} exceeded at {now()}")
        sys.exit(86)
    
    if state.total > GLOBAL_CAP * 0.95:
        save_budget(state)
        sys.exit(86)
    
    save_budget(state)
    return cost


# ═══════════════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════════════

def get_remaining_budget(provider: str) -> float:
    """Return remaining budget for a provider."""
    state = load_budget()
    cap = PROVIDER_CAPS.get(provider, 0.0)
    spent = getattr(state, f"{provider}_spent", 0.0)
    return max(0.0, cap - spent)


def get_summary() -> dict[str, Any]:
    """Return full budget summary for reporting."""
    state = load_budget()
    return {
        "deepseek": {
            "spent": state.deepseek_spent,
            "cap": PROVIDER_CAPS["deepseek"],
            "remaining": get_remaining_budget("deepseek"),
        },
        "anthropic": {
            "spent": state.anthropic_spent,
            "cap": PROVIDER_CAPS["anthropic"],
            "remaining": get_remaining_budget("anthropic"),
        },
        "openai": {
            "spent": state.openai_spent,
            "cap": PROVIDER_CAPS["openai"],
            "remaining": get_remaining_budget("openai"),
        },
        "total": {
            "spent": state.total,
            "cap": GLOBAL_CAP,
            "remaining": max(0.0, GLOBAL_CAP - state.total),
        },
        "last_updated": state.last_updated,
    }


def reset_budget() -> None:
    """Reset budget state (use with caution)."""
    if BUDGET_FILE.exists():
        BUDGET_FILE.unlink()
