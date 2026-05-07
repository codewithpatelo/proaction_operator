"""
Sensitivity analysis: P1 (elastic return) under ±20% perturbation of κ and λ.
Tests whether the elastic-return half-life is robust to parameter variation.
Outputs: figures/fig5_sensitivity.png
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(42)

# ── Baseline parameters (from Table 1) ────────────────────────────────────────
KAPPAS_BASE = np.array([.10, .10, .15, .12, .08, .10])
LAMBDAS_BASE = np.array([.08, .07, .10, .08, .06, .10])
ALPHAS = np.array([.20, .15, .25, .20, .10, .15])
X_STAR = np.array([.3, .2, .3, .1, .2, .4])
TAU = np.array([1, 1, 1, 1, 1, 5], dtype=int)  # SAM=1, HPA=5

W = np.array([
    [ 0.0,  0.1,  0.0, -0.1,  0.0,  0.0],
    [ 0.1,  0.0,  0.1,  0.0,  0.0,  0.0],
    [ 0.0,  0.1,  0.0,  0.1, -0.1,  0.0],
    [-0.1,  0.0,  0.1,  0.0,  0.1,  0.0],
    [ 0.0,  0.0, -0.1,  0.1,  0.0,  0.1],
    [ 0.0,  0.0,  0.0,  0.0,  0.1,  0.0],
])

PHI_SCALE = 5.0
N_ROUNDS = 30
PERTURBATION_ROUND = 10
PERTURBATION_MAGNITUDE = 0.4

def run_simulation(kappas, lambdas, n_rounds=N_ROUNDS):
    """Run one agent's thermostat dynamics and return x trajectory."""
    buf_len = max(TAU) + 1
    history = [X_STAR.copy() for _ in range(buf_len)]
    x = X_STAR.copy()

    traj = []
    for t in range(n_rounds):
        # Perturbation at round 10: spike hormonal subsystem
        if t == PERTURBATION_ROUND:
            x = x.copy()
            x[2] += PERTURBATION_MAGNITUDE  # H subsystem

        # Delayed coupling
        coupling = np.zeros(6)
        for k in range(6):
            tau_k = TAU[k]
            x_delayed = history[-(tau_k + 1)] if tau_k < len(history) else history[0]
            coupling[k] = np.dot(W[k], np.tanh(PHI_SCALE * x_delayed))

        # Dummy feedback (no real action in pure simulation)
        rho = ALPHAS * 0.1

        x_next = (x
                  - kappas * (x - X_STAR)
                  + lambdas
                  - rho
                  + coupling)
        x_next = np.clip(x_next, -1.0, 2.0)

        history.append(x.copy())
        if len(history) > buf_len:
            history.pop(0)

        x = x_next
        traj.append(x.copy())

    return np.array(traj)

def compute_half_life(traj, subsystem=2):
    """Compute half-life of return to set-point for subsystem k after perturbation."""
    k = subsystem
    x_star_k = X_STAR[k]
    post = traj[PERTURBATION_ROUND:, k]
    peak = abs(post[0] - x_star_k)
    if peak < 1e-6:
        return None
    for i, val in enumerate(post):
        if abs(val - x_star_k) <= peak / 2:
            return i
    return len(post)

# ── Grid sweep ─────────────────────────────────────────────────────────────────
scale_range = np.linspace(0.7, 1.3, 13)  # ±30% to show ±20% clearly
hl_kappa = []
hl_lambda = []

# Vary κ uniformly
for s in scale_range:
    traj = run_simulation(KAPPAS_BASE * s, LAMBDAS_BASE)
    hl = compute_half_life(traj)
    hl_kappa.append(hl if hl is not None else np.nan)

# Vary λ uniformly
for s in scale_range:
    traj = run_simulation(KAPPAS_BASE, LAMBDAS_BASE * s)
    hl = compute_half_life(traj)
    hl_lambda.append(hl if hl is not None else np.nan)

# Baseline
traj_base = run_simulation(KAPPAS_BASE, LAMBDAS_BASE)
hl_base = compute_half_life(traj_base)

# ── Figure ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)

pct = (scale_range - 1) * 100

for ax, hl_vals, param_name, color in [
    (axes[0], hl_kappa,  r'$\kappa$', '#1f77b4'),
    (axes[1], hl_lambda, r'$\lambda$', '#d62728'),
]:
    ax.plot(pct, hl_vals, color=color, lw=2, marker='o', ms=5)
    ax.axvline(0, color='k', ls='--', lw=1, alpha=0.5, label='Baseline')
    ax.axvspan(-20, 20, alpha=0.08, color='grey', label='±20% window')
    if hl_base is not None:
        ax.axhline(hl_base, color='k', ls=':', lw=1, alpha=0.5)
    ax.set_xlabel(f'Perturbation of {param_name} (%)')
    ax.set_ylabel('Half-life $t_{1/2}$ (rounds)')
    ax.set_title(f'Sensitivity to {param_name} scaling')
    ax.legend(fontsize=8)
    ax.set_xlim(-32, 32)

plt.suptitle(
    'P1 elastic return: half-life under uniform parameter scaling\n'
    r'(Subsystem $\mathcal{H}$, perturbation at round 10)',
    fontsize=10
)
plt.tight_layout()
plt.savefig('figures/fig5_sensitivity.png', dpi=150, bbox_inches='tight')
print(f"Saved figures/fig5_sensitivity.png  [baseline half-life = {hl_base} rounds]")
print(f"κ range over ±20%: half-life in {min(hl_kappa[3:10]):.0f}–{max(hl_kappa[3:10]):.0f} rounds")
print(f"λ range over ±20%: half-life in {min(hl_lambda[3:10]):.0f}–{max(hl_lambda[3:10]):.0f} rounds")
