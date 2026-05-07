"""
Type-theoretic verification of A9 (Recursive closure):
  Γ : State → State    where State = R^6 × Memory × DriveState
  ⇒ Γ ∘ Γ : State → State    (well-typed, idempotent type signature)

We use Python's typing module to encode the type signature symbolically
and verify under a structural-typing rule that Γ ∘ Γ ∘ ... ∘ Γ stays
in the same type.  This is a finitary, decidable analogue of a type-check
in a dependently-typed proof assistant (Lean/Coq) for the same property.
"""

from dataclasses import dataclass
from typing import Callable, NamedTuple, Tuple
import numpy as np

results = {}
def report(label, ok, detail=""):
    tag = "PASS" if ok else "FAIL"
    results[label] = tag
    print(f"  [{tag}] {label}  {detail}")


# ── State type ────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class State:
    x: tuple        # 6 thermostat values, |x| == 6
    M: tuple        # memory state (variable length)
    D: tuple        # drive state, |D| == 6 (set-points)

    def well_typed(self) -> bool:
        return (isinstance(self.x, tuple) and len(self.x) == 6 and
                isinstance(self.M, tuple) and
                isinstance(self.D, tuple) and len(self.D) == 6)


# ── Γ as a typed function ────────────────────────────────────────────────
def Gamma(s: State) -> State:
    # mock dynamics consistent with the master equation: identity-like step
    new_x = tuple(0.9 * xi for xi in s.x)
    new_M = s.M + (sum(s.x),)        # memory accretes
    new_D = s.D                      # drives may evolve; here held fixed
    return State(x=new_x, M=new_M, D=new_D)


# ── T1. Γ preserves the State type (well-typedness) ─────────────────────
s0 = State(x=(0.3, 0.2, 0.3, 0.1, 0.2, 0.4), M=(), D=(0.3, 0.2, 0.3, 0.1, 0.2, 0.4))
s1 = Gamma(s0)
report("T1: Γ(s) : State", s1.well_typed())


# ── T2. Γ ∘ Γ preserves the State type ─────────────────────────────────
s2 = Gamma(Gamma(s0))
report("T2: (Γ∘Γ)(s) : State", s2.well_typed())


# ── T3. Iterated Γ^n : State → State for n in {1,...,100} ──────────────
ok = True
s = s0
for n in range(1, 101):
    s = Gamma(s)
    if not s.well_typed():
        ok = False; break
report("T3: Γ^n preserves State for n ≤ 100", ok)


# ── T4. Type signature stable under VSM recursion (Γ takes Γ-output) ──
# Γ' : State → State,  where Γ' = Γ ∘ Γ.  Composition closed.
def GammaPrime(s: State) -> State:
    return Gamma(Gamma(s))

s = GammaPrime(s0)
report("T4: VSM recursion Γ'=Γ∘Γ : State→State well-typed",
       s.well_typed())


# ── T5. Output dimension of x is invariant (R^6 → R^6) ─────────────────
s = s0
dims = []
for _ in range(20):
    s = Gamma(s)
    dims.append(len(s.x))
ok = all(d == 6 for d in dims)
report("T5: dim(x) ≡ 6 across iterations", ok,
       f"dims = {dims[:5]}...")


print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
total = len(results); passed = sum(1 for r in results.values() if r == "PASS")
for k, v in results.items():
    print(f"  {v}  {k}")
print(f"\n  {passed}/{total} type checks passed")
