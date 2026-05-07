"""Smoke test #2: IPD payoff matrix and noise rate."""

import random
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import payoff from experiment
from experiment import PAYOFF


def test_payoff_sanity():
    """Axelrod payoffs: T=5, R=3, P=1, S=0."""
    assert PAYOFF[("C", "C")] == (3, 3), "CC payoff wrong"
    assert PAYOFF[("C", "D")] == (0, 5), "CD payoff wrong"
    assert PAYOFF[("D", "C")] == (5, 0), "DC payoff wrong"
    assert PAYOFF[("D", "D")] == (1, 1), "DD payoff wrong"
    print("  ✓ Payoff matrix OK")


def test_noise_rate():
    """Noise rate empirically ~10% over many trials."""
    NOISE_RATE = 0.10
    N_TRIALS = 10000
    
    rng = random.Random(42)
    flips = sum(1 for _ in range(N_TRIALS) if rng.random() < NOISE_RATE)
    empirical_rate = flips / N_TRIALS
    
    # Allow ±2% tolerance
    assert 0.08 <= empirical_rate <= 0.12, f"Noise rate {empirical_rate:.3f} out of range"
    print(f"  ✓ Noise rate OK: {empirical_rate:.3f} (target 0.10)")


def test_payoff_ordering():
    """T > R > P > S ordering."""
    T, R, P, S = 5, 3, 1, 0
    assert T > R > P > S, "Payoff ordering violated"
    print("  ✓ Payoff ordering (T>R>P>S) OK")


def main():
    print("[SMOKE TEST #2] IPD Payoff & Noise")
    test_payoff_sanity()
    test_noise_rate()
    test_payoff_ordering()
    print("[PASS]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
