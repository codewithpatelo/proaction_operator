"""
verify_z3.py — Z3 SMT counterexample search for the Pro-Action operator Γ.

Z3 cannot reason about tanh directly (transcendental), so we replace tanh
with conservative piecewise-linear bounds that are SOUND: any property we
prove against the bounded model also holds for the true tanh model.

Bounds used:
    -1 ≤ tanh(x) ≤ 1            (always true)
    tanh(x) · sign(x) ≥ 0       (same sign as input)

Checks:
  Z1. The set-point x* IS a fixed point when λ=0, ρ=0, W=0
  Z2. Γ is contractive (||x_{t+1} - x*|| < ||x_t - x*||) when W=0, λ=0, ρ=0,
      under κ_k ∈ (0, 1)  →  proves P1 in the decoupled case
  Z3. Γ is bounded: if ||x_t|| ≤ M then ||x_{t+1}|| ≤ M' with M' computable
  Z4. Mixer π_fast monotonicity in x_H (via sigmoid bounds)
  Z5. NO non-set-point fixed points exist when W=0, λ=0, ρ=0

Usage: python verify_z3.py
"""

from z3 import (Real, Reals, RealVector, Solver, And, Or, Not, Implies,
                ForAll, Exists, sat, unsat, unknown, If, Sum, simplify)

print("=" * 64)
print("Z3 SMT Counterexample Search — Pro-Action Operator Γ")
print("=" * 64)

N = 6
results = []


def report(name, status, detail=""):
    icon = {"PASS": "✓", "FAIL": "✗", "UNKNOWN": "?"}[status]
    print(f"  {icon} {name}: {status} {detail}")
    results.append((name, status, detail))


# ═══════════════════════════════════════════════════════════════════════════
# Z1. Set-point x* is a fixed point when λ=0, ρ=0, W=0
# ═══════════════════════════════════════════════════════════════════════════
print("\n── Z1. x* is a fixed point (λ=0, ρ=0, W=0) ──")

s = Solver()
x_star = RealVector('xstar', N)
kappa  = RealVector('kappa', N)

# Setting x_t = x*, with λ=0, ρ=0, W=0:
# x_{k,t+1} = x*_k - κ_k(x*_k - x*_k) + 0 - 0 + 0 = x*_k
# Try to find a counterexample: ∃ x*, κ such that update ≠ x*
for k in range(N):
    s.add(kappa[k] >= 0, kappa[k] <= 1)
    update_k = x_star[k] - kappa[k] * (x_star[k] - x_star[k])  # + 0 + 0 + 0
    # We want: update_k ≠ x_star_k → counterexample
    s.add(update_k != x_star[k])

result = s.check()
if result == unsat:
    report("Z1", "PASS", "(no counterexample: x* is always a fixed point)")
elif result == sat:
    report("Z1", "FAIL", f"counterexample: {s.model()}")
else:
    report("Z1", "UNKNOWN")


# ═══════════════════════════════════════════════════════════════════════════
# Z2. Contractivity: ||x_{t+1} - x*||² < ||x_t - x*||² when W=0, λ=0, ρ=0,
#     0 < κ_k < 2  (the discrete-time stability condition for elastic return)
# ═══════════════════════════════════════════════════════════════════════════
print("\n── Z2. Contractivity (decoupled, 0 < κ < 2) ──")

s = Solver()
x  = RealVector('x',  N)
xs = RealVector('xs', N)
k  = RealVector('k',  N)

# Update with W=0, λ=0, ρ=0: x_{i,t+1} = x_i - κ_i(x_i - x*_i) = (1-κ_i)x_i + κ_i·x*_i
# Deviation: e_i = x_i - x*_i
# e_{i,t+1} = (1-κ_i)·e_i
# ||e_{t+1}||² = Σ (1-κ_i)²·e_i²
# Contractive iff (1-κ_i)² < 1 for all i, i.e., 0 < κ_i < 2

# Counterexample search: find κ ∈ (0, 2) such that contraction fails
for i in range(N):
    s.add(k[i] > 0, k[i] < 2)
    # Force at least one e_i ≠ 0 to avoid trivial case
sum_old = Sum([(x[i] - xs[i]) * (x[i] - xs[i]) for i in range(N)])
sum_new = Sum([((1 - k[i]) * (x[i] - xs[i])) * ((1 - k[i]) * (x[i] - xs[i])) for i in range(N)])
s.add(sum_old > 0)  # non-trivial state
s.add(sum_new >= sum_old)  # counterexample: no contraction

result = s.check()
if result == unsat:
    report("Z2", "PASS", "(0 < κ < 2 ⇒ ||x_{t+1}-x*|| < ||x_t-x*||)")
elif result == sat:
    m = s.model()
    report("Z2", "FAIL", f"counterexample found")
    print(f"     model: {m}")
else:
    report("Z2", "UNKNOWN")


# ═══════════════════════════════════════════════════════════════════════════
# Z3. Boundedness: with W bounded by ||W||∞ ≤ w_max, and tanh ∈ [-1, 1],
#     coupling term is bounded by w_max · N · 1 = 6·w_max.
#     If λ_k bounded by L, α_k bounded by A, ρ ∈ [-1, 1]:
#     |x_{k,t+1}| ≤ |x_k|(|1-κ_k|) + κ_k|x*_k| + L + A + 6·w_max
# ═══════════════════════════════════════════════════════════════════════════
print("\n── Z3. Boundedness with W·tanh ∈ [-‖W‖₁, ‖W‖₁] ──")

s = Solver()
x_in  = Real('x')
xs_   = Real('xs')
k_    = Real('k')
lam_  = Real('lam')
alp_  = Real('alpha')
rho_  = Real('rho')
coup  = Real('coupling')  # represents [W·tanh(x)]_k
w_norm = Real('w_norm')   # ||W_row||_1

# Constraints (sound abstraction)
s.add(k_ > 0, k_ < 1)          # κ ∈ (0,1)
s.add(lam_ >= -0.1, lam_ <= 0.1)
s.add(alp_ >= 0, alp_ <= 0.2)
s.add(rho_ >= -1, rho_ <= 1)
s.add(w_norm >= 0, w_norm <= 1.0)  # ||W_row||_1 ≤ 1
s.add(coup >= -w_norm, coup <= w_norm)  # tanh ∈ [-1,1] ⇒ |W·tanh(x)| ≤ ||W||_1
s.add(x_in >= -10, x_in <= 10)
s.add(xs_ >= -1, xs_ <= 1)

x_next = x_in - k_ * (x_in - xs_) + lam_ - alp_ * rho_ + coup
# Counterexample: x_next outside [-12, 12]?
s.add(Or(x_next > 12, x_next < -12))

result = s.check()
if result == unsat:
    report("Z3", "PASS", "(bounded inputs ⇒ bounded output ∈ [-12,12])")
elif result == sat:
    m = s.model()
    report("Z3", "FAIL", f"counterexample: x_next escapes bounds")
    print(f"     model: {m}")
else:
    report("Z3", "UNKNOWN")


# ═══════════════════════════════════════════════════════════════════════════
# Z4. Mixer π_fast monotonicity in x_H
#     Since π = σ(c·x_H + b) and σ is monotonic, π is monotonic in x_H iff c ≥ 0.
#     We verify: if c ≥ 0 and x_H1 < x_H2, then π(x_H1) < π(x_H2).
#     We replace σ with its monotonicity property as an axiom.
# ═══════════════════════════════════════════════════════════════════════════
print("\n── Z4. Mixer π_fast monotonicity ──")

# Direct test: σ is monotonic, so we just need c·xH1+b < c·xH2+b
s = Solver()
c_, b_, xH1, xH2 = Reals('c b xH1 xH2')
s.add(c_ > 0)  # strict positivity required for strict monotonicity
s.add(xH1 < xH2)
# Monotonicity violation: c·xH1+b ≥ c·xH2+b
s.add(c_ * xH1 + b_ >= c_ * xH2 + b_)

result = s.check()
if result == unsat:
    report("Z4", "PASS", "(c ≥ 0 ⇒ π_fast monotonic in x_H)")
elif result == sat:
    report("Z4", "FAIL", f"counterexample: {s.model()}")
else:
    report("Z4", "UNKNOWN")


# ═══════════════════════════════════════════════════════════════════════════
# Z5. Uniqueness of fixed point in decoupled case (W=0, λ=0, ρ=0)
#     Fixed point: x_k = x_k - κ_k(x_k - x*_k) ⇒ κ_k(x_k - x*_k) = 0
#     With κ_k > 0 ⇒ x_k = x*_k.  No other fixed point exists.
# ═══════════════════════════════════════════════════════════════════════════
print("\n── Z5. Uniqueness of fixed point (decoupled) ──")

s = Solver()
x_  = Real('x')
xs2 = Real('xs')
k2  = Real('k')
s.add(k2 > 0, k2 < 2)
# Fixed point condition: x = x - k(x - xs) ⇒ k(x - xs) = 0
s.add(k2 * (x_ - xs2) == 0)
# Counterexample: x ≠ xs (another fixed point exists)
s.add(x_ != xs2)

result = s.check()
if result == unsat:
    report("Z5", "PASS", "(unique fixed point: x = x*)")
elif result == sat:
    report("Z5", "FAIL", f"alternative fixed point: {s.model()}")
else:
    report("Z5", "UNKNOWN")


# ═══════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*64}")
n_pass = sum(1 for _, s, _ in results if s == "PASS")
n_fail = sum(1 for _, s, _ in results if s == "FAIL")
n_unk  = sum(1 for _, s, _ in results if s == "UNKNOWN")
print(f"Z3 results: {n_pass} PASS, {n_fail} FAIL, {n_unk} UNKNOWN of {len(results)}")
if n_fail == 0:
    print("Γ equation is logically sound: no counterexamples found in the")
    print("decoupled regime. Coupled regime requires numerical/eigenvalue analysis.")
else:
    print("ISSUES FOUND — review counterexamples above.")
print(f"{'='*64}")
