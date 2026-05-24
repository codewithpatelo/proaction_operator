"""
generate_figures.py — Genera figuras preliminares del simulador para el paper NeurIPS.
Requiere: pip install matplotlib numpy
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

with open('simulation.json', 'r') as f:
    data = json.load(f)

agent_a = data['agent1']['history']
agent_b = data['agent2']['history']
rounds = list(range(len(agent_a)))

# Fig 1: Trayectorias de termostatos
fig, axes = plt.subplots(2, 3, figsize=(10, 6), sharex=True)
subs = ['A', 'P', 'H', 'E', 'N', 'C']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
for idx, (ax, sub, col) in enumerate(zip(axes.flat, subs, colors)):
    a_vals = [r['x'][idx] for r in agent_a]
    b_vals = [r['x'][idx] for r in agent_b]
    ax.plot(rounds, a_vals, color=col, lw=1.5, label='Agent A', alpha=0.8)
    ax.plot(rounds, b_vals, color=col, lw=1.5, ls='--', label='Agent B', alpha=0.5)
    ax.axhline(y=0, color='k', ls=':', alpha=0.3)
    ax.set_title(f'$x_{{\\mathcal{{{sub}}}}}$')
    ax.set_ylabel('State')
    if idx == 0:
        ax.legend(fontsize=7)
axes[-1, -1].set_xlabel('Round')
plt.tight_layout()
plt.savefig('figures/fig1_thermostats.png', dpi=150, bbox_inches='tight')
print('Saved figures/fig1_thermostats.png')

# Fig 2: Payoffs acumulados
fig, ax = plt.subplots(figsize=(6, 3.5))
a_payoff = np.cumsum([r['payoff'] for r in agent_a])
b_payoff = np.cumsum([r['payoff'] for r in agent_b])
ax.plot(rounds, a_payoff, label='Agent A', color='#1f77b4')
ax.plot(rounds, b_payoff, label='Agent B', color='#ff7f0e')
ax.set_xlabel('Round')
ax.set_ylabel('Cumulative Payoff')
ax.legend()
ax.set_title('Cumulative payoffs in iterated Prisoner\'s Dilemma')
plt.tight_layout()
plt.savefig('figures/fig2_payoffs.png', dpi=150, bbox_inches='tight')
print('Saved figures/fig2_payoffs.png')

# Fig 3: Elastic return — zoom post-perturbation (round 15)
fig, ax = plt.subplots(figsize=(6, 3.5))
xH_a = [r['x'][2] for r in agent_a]
ax.plot(rounds, xH_a, color='#2ca02c', lw=1.5)
ax.axvline(x=15, color='r', ls='--', alpha=0.5, label='Perturbation (round 15)')
ax.axhline(y=0, color='k', ls=':', alpha=0.3)
ax.set_xlabel('Round')
ax.set_ylabel('$x_{\\mathcal{H}}$ (Hormonal)')
ax.set_title('Elastic return to set-point after perturbation')
ax.legend()
plt.tight_layout()
plt.savefig('figures/fig3_elastic_return.png', dpi=150, bbox_inches='tight')
print('Saved figures/fig3_elastic_return.png')

print('All figures generated.')
