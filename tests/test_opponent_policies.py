"""Smoke test #3: Opponent policies match canonical traces."""

import random
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiment import tit_for_tat, grim_trigger, random_opponent, generous_tft


def test_tft():
    """TFT: C, then mirror."""
    rng = random.Random(42)
    
    # Start with C
    assert tit_for_tat([], rng) == "C"
    
    # Mirror opponent's last action
    assert tit_for_tat(["C"], rng) == "C"
    assert tit_for_tat(["D"], rng) == "D"
    assert tit_for_tat(["C", "C", "D"], rng) == "D"
    
    print("  ✓ TFT OK")


def test_grim():
    """Grim: C until first D, then always D."""
    rng = random.Random(42)
    
    assert grim_trigger([], rng) == "C"
    assert grim_trigger(["C", "C"], rng) == "C"
    assert grim_trigger(["C", "C", "D"], rng) == "D"
    assert grim_trigger(["C", "C", "D", "C"], rng) == "D"  # Never forgives
    
    print("  ✓ Grim OK")


def test_random():
    """Random: 50/50 distribution."""
    rng = random.Random(123)
    
    actions = [random_opponent([], rng) for _ in range(1000)]
    c_rate = sum(1 for a in actions if a == "C") / len(actions)
    
    assert 0.45 <= c_rate <= 0.55, f"Random bias detected: {c_rate:.3f}"
    print(f"  ✓ Random OK (C rate: {c_rate:.3f})")


def test_gtft():
    """GTFT: TFT but forgives with 30% prob."""
    rng = random.Random(456)
    
    # Starts with C
    assert generous_tft([], rng) == "C"
    
    # Continues C after opponent C
    assert generous_tft(["C"], rng) == "C"
    
    # After D, might forgive (test statistically)
    n_trials = 1000
    responses = [generous_tft(["D"], random.Random(i)) for i in range(n_trials)]
    forgive_rate = sum(1 for a in responses if a == "C") / n_trials
    
    # Should be ~30% forgive rate
    assert 0.25 <= forgive_rate <= 0.35, f"GTFT forgive rate {forgive_rate:.3f} out of range"
    print(f"  ✓ GTFT OK (forgive rate: {forgive_rate:.3f})")


def main():
    print("[SMOKE TEST #3] Opponent Policies")
    test_tft()
    test_grim()
    test_random()
    test_gtft()
    print("[PASS]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
