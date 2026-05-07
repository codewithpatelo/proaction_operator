"""Smoke test #1: SeedFactory determinism and collision resistance."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from seeds import SeedFactory, is_prime


def test_seeds_deterministic():
    """Same master seed produces same component seeds."""
    factory1 = SeedFactory(42)
    factory2 = SeedFactory(42)
    
    for component in ["a", "b", "c", "test_component_1"]:
        assert factory1.get(component) == factory2.get(component), f"Mismatch for {component}"
    
    print("  ✓ Determinism OK")


def test_seeds_prime():
    """All generated seeds are prime."""
    factory = SeedFactory(12345)
    
    for component in [f"comp_{i}" for i in range(50)]:
        seed = factory.get(component)
        assert is_prime(seed), f"Seed {seed} for {component} is not prime"
    
    print("  ✓ Primality OK")


def test_seeds_collision_resistance():
    """Different components get different seeds."""
    factory = SeedFactory(99999)
    seeds = [factory.get(f"comp_{i}") for i in range(100)]
    
    assert len(set(seeds)) == len(seeds), f"Collision! Only {len(set(seeds))} unique out of {len(seeds)}"
    print(f"  ✓ Collision resistance OK ({len(seeds)} unique seeds)")


def main():
    print("[SMOKE TEST #1] SeedFactory")
    test_seeds_deterministic()
    test_seeds_prime()
    test_seeds_collision_resistance()
    print("[PASS]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
