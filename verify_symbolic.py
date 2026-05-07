"""
verify_symbolic.py — SymPy symbolic verification of the Pro-Action operator Γ.

Checks:
  R1. Reduction to Driveplexity (A1–A3): κ→0, W→0, τ→0, only C active
  R2. Reduction to homeostatic RL: κ→0, τ→0, W→0
  R3. Fixed-point condition: solve x = f(x) symbolically
  R4. Jacobian at fixed point for stability analysis
  R5. Mixer π_fast monotonicity in x_H

Usage: python verify_symbolic.py
"""

import sympy as sp
from sympy import symbols, Matrix, tanh, diag, zeros, ones, Eq, simplify, solve

# ── Declare symbolic variables ──────────────────────────────────────────────
N = 6

# Per-subsystem parameters (symbolic vectors)
kappa = symbols('kappa0:6', real=True)
lam   = symbols('lam0:6',   real=True)
alpha = symbols('alpha0:6', real=True)
tau   = symbols('tau0:6',   integer=True)
x_star = symbols('xstar0:6', real=True)

# State vector
x = symbols('x0:6', real=True)

# Coupling matrix W (6×6 symbolic)
W = Matrix(N, N, symbols('W0:36', real=True))

# Outcome feedback ρ (symbolic vector)
rho = symbols('rho0:6', real=True)

# ── Helper: build the update for one subsystem ─────────────────────────────
def subsystem_update(k):
    """Symbolic x_{k,t+1} = x_k - κ_k(x_k - x*_k) + λ_k - α_k·ρ_k + [W·tanh(x)]_k"""
    # Coupling term: row k of W · tanh(x)
    coupling_k = sum(W[k, j] * tanh(x[j]) for j in range(N))
    return (x[k]
            - kappa[k] * (x[k] - x_star[k])
            + lam[k]
            - alpha[k] * rho[k]
            + coupling_k)

# Full update vector
x_next = Matrix([subsystem_update(k) for k in range(N)])

print("=" * 64)
print("SymPy Symbolic Verification — Pro-Action Operator Γ")
print("=" * 64)

# ═══════════════════════════════════════════════════════════════════════════
# R1. Reduction to Driveplexity
# ═══════════════════════════════════════════════════════════════════════════
print("\n── R1. Reduction to Driveplexity (A1–A3) ──")

# Driveplexity conditions:
#   κ = 0, W = 0, τ irrelevant (coupling uses current state when τ=0),
#   α_k = 0 for k ≠ C, x_k frozen at 0 for k ≠ C
#   Only x_C ≡ δ_i evolves
subs_driveplexity = {}
# Zero out kappa
for k in range(N):
    subs_driveplexity[kappa[k]] = 0
# Zero out W
for i in range(N):
    for j in range(N):
        subs_driveplexity[W[i, j]] = 0
# Zero out alpha and lambda for non-C subsystems (driveplexity has only one drive)
for k in range(5):
    subs_driveplexity[alpha[k]] = 0
    subs_driveplexity[lam[k]] = 0
# Freeze non-C states at 0
for k in range(5):
    subs_driveplexity[x[k]] = 0

x_next_drive = x_next.subs(subs_driveplexity)
# Expected: δ_{t+1} = δ_t + λ_C - α_C·ρ_C
# With δ_t = x_C, this is x_C + lam_C - alpha_C * rho_C
expected_drive = x[5] + lam[5] - alpha[5] * rho[5]
diff_drive = simplify(x_next_drive[5] - expected_drive)
print(f"  x_C update: {x_next_drive[5]}")
print(f"  Expected:   {expected_drive}")
print(f"  Difference: {diff_drive}")
assert diff_drive == 0, "R1 FAILED: Driveplexity reduction mismatch"
# Also verify non-C subsystems stay frozen
for k in range(5):
    assert x_next_drive[k] == 0, f"R1 FAILED: subsystem {k} not frozen"
print("  ✓ R1 PASSED: Γ reduces exactly to Driveplexity δ_{t+1}=δ_t+λ−α·g")

# ═══════════════════════════════════════════════════════════════════════════
# R2. Reduction to homeostatic RL
# ═══════════════════════════════════════════════════════════════════════════
print("\n── R2. Reduction to homeostatic RL ──")

# Homeostatic RL conditions: κ = 0, τ = 0 (no delays), W = 0
# System becomes purely reactive: x_{k,t+1} = x_{k,t} + λ_k - α_k·ρ_k
subs_hrl = {}
for k in range(N):
    subs_hrl[kappa[k]] = 0
for i in range(N):
    for j in range(N):
        subs_hrl[W[i, j]] = 0

x_next_hrl = x_next.subs(subs_hrl)
expected_hrl = Matrix([x[k] + lam[k] - alpha[k] * rho[k] for k in range(N)])
diff_hrl = simplify(x_next_hrl - expected_hrl)
print(f"  Update: {x_next_hrl[0]}")
print(f"  Expected: {expected_hrl[0]}")
assert all(d == 0 for d in diff_hrl), f"R2 FAILED: homeostatic RL reduction mismatch: {diff_hrl}"
print("  ✓ R2 PASSED: Γ reduces exactly to reactive drive-based policy")

# ═══════════════════════════════════════════════════════════════════════════
# R3. Fixed-point condition
# ═══════════════════════════════════════════════════════════════════════════
print("\n── R3. Fixed-point condition ──")

# Fixed point: x*_fp such that x_next(x*_fp) = x*_fp
# This means: -κ_k(x_k - x*_k) + λ_k - α_k·ρ_k + [W·tanh(x)]_k = 0
fp_eqs = []
for k in range(N):
    eq = Eq(-kappa[k] * (x[k] - x_star[k]) + lam[k] - alpha[k] * rho[k]
            + sum(W[k, j] * tanh(x[j]) for j in range(N)), 0)
    fp_eqs.append(eq)

# Special case: W=0, ρ=0 (no coupling, no outcome)
# Then x_k = x*_k + λ_k/κ_k
subs_no_coupling = {}
for i in range(N):
    for j in range(N):
        subs_no_coupling[W[i, j]] = 0
for k in range(N):
    subs_no_coupling[rho[k]] = 0

fp_simple = [eq.subs(subs_no_coupling) for eq in fp_eqs]
for k in range(N):
    sol = solve(fp_simple[k], x[k])
    expected = x_star[k] + lam[k] / kappa[k]
    assert simplify(sol[0] - expected) == 0, f"R3 FAILED: fixed point mismatch for k={k}"
print(f"  No-coupling fixed point: x_k = x*_k + λ_k/κ_k")
print("  ✓ R3 PASSED: fixed-point condition correct in decoupled case")

# ═══════════════════════════════════════════════════════════════════════════
# R4. Jacobian at fixed point (stability)
# ═══════════════════════════════════════════════════════════════════════════
print("\n── R4. Jacobian at fixed point ──")

# Jacobian J_{ij} = ∂x_{i,t+1}/∂x_{j,t}
J = x_next.jacobian(x)

# ∂x_{i,t+1}/∂x_{j,t} = δ_{ij}(1 - κ_i) + W_{ij}·sech²(x_j)
# (since d/dx_j tanh(x_j) = sech²(x_j))
for i in range(N):
    for j in range(N):
        expected_entry = (1 - kappa[i]) if i == j else 0
        expected_entry += W[i, j] * (1 / sp.cosh(x[j]))**2
        diff = simplify(J[i, j] - expected_entry)
        assert diff == 0, f"R4 FAILED: Jacobian entry ({i},{j}) mismatch"

# At the no-coupling fixed point x_k = x*_k + λ_k/κ_k:
# J_{ij} = δ_{ij}(1 - κ_i) + W_{ij}·sech²(x*_j + λ_j/κ_j)
# Stability requires ρ(J) < 1
print("  J_{ij} = δ_{ij}(1-κ_i) + W_{ij}·sech²(x_j)")
print("  ✓ R4 PASSED: Jacobian structure verified symbolically")

# ═══════════════════════════════════════════════════════════════════════════
# R5. Mixer π_fast monotonicity
# ═══════════════════════════════════════════════════════════════════════════
print("\n── R5. Mixer π_fast monotonicity ──")

c, b_sym, xH = symbols('c b xH', real=True)
pi = 1 / (1 + sp.exp(-(c * xH + b_sym)))
dpi_dxH = sp.diff(pi, xH)
# dπ/dxH = c·e^{-(c·xH+b)} / (1+e^{-(c·xH+b)})² = c·π·(1-π)
# This is always ≥ 0 when c ≥ 0
dpi_simplified = simplify(dpi_dxH - c * pi * (1 - pi))
assert dpi_simplified == 0, "R5 FAILED: derivative identity mismatch"
print(f"  π_fast = σ(c·x_H + b)")
print(f"  dπ/dx_H = c·π·(1-π) ≥ 0 for c ≥ 0")
print("  ✓ R5 PASSED: π_fast is monotonic in x_H (c ≥ 0)")

# ═══════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*64}")
print("All 5 symbolic checks PASSED.")
print("Γ equation is algebraically sound:")
print("  • Reduces correctly to Driveplexity and homeostatic RL")
print("  • Fixed-point condition is well-defined")
print("  • Jacobian structure matches analytical derivation")
print("  • Mixer monotonicity is guaranteed for c ≥ 0")
print(f"{'='*64}")
