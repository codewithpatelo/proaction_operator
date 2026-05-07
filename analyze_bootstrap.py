"""
Bootstrap CIs for snapshot metrics + recovery/cross-lag comparison figures.
Outputs:
  - bootstrap_results.json  (95% CIs for paper table)
  - figures/fig4_recovery_analysis.png
"""
import json
import math
import random
import sys
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

random.seed(42)
np.random.seed(42)

# ── Load data ─────────────────────────────────────────────────────────────────
d = json.load(open('checkpoints/cells.json', encoding='utf-8'))

by_cond = defaultdict(list)
for v in d.values():
    r = v['result']
    by_cond[r['key']['condition']].append(r)

# ── Bootstrap CI function ──────────────────────────────────────────────────────
def bootstrap_ci(values, n_boot=10000, ci=0.95):
    if len(values) < 2:
        return None, None
    boots = [np.mean(np.random.choice(values, size=len(values), replace=True))
             for _ in range(n_boot)]
    lo = np.percentile(boots, (1 - ci) / 2 * 100)
    hi = np.percentile(boots, (1 + ci) / 2 * 100)
    return lo, hi

# ── Table: cooperation + volatility CIs per condition ─────────────────────────
results = {}
for cond, cells in sorted(by_cond.items()):
    coops = [c['cooperation_rate'] for c in cells]
    vols  = [c['action_volatility']  for c in cells]
    rds   = [c['coop_recovery_delay'] for c in cells if c.get('coop_recovery_delay') is not None]
    cls_  = [c['cross_lag_peak']      for c in cells if c.get('cross_lag_peak')      is not None]

    coop_lo, coop_hi = bootstrap_ci(coops)
    vol_lo,  vol_hi  = bootstrap_ci(vols)
    rd_lo,   rd_hi   = bootstrap_ci(rds)  if rds  else (None, None)
    cl_lo,   cl_hi   = bootstrap_ci(cls_) if cls_ else (None, None)

    results[cond] = {
        'n': len(cells),
        'coop_mean': np.mean(coops), 'coop_sd': np.std(coops),
        'coop_ci95': (coop_lo, coop_hi),
        'vol_mean':  np.mean(vols),  'vol_sd':  np.std(vols),
        'vol_ci95':  (vol_lo,  vol_hi),
        'recovery_n': len(rds),
        'recovery_mean': np.mean(rds) if rds else None,
        'recovery_ci95': (rd_lo, rd_hi),
        'cross_lag_n': len(cls_),
        'cross_lag_mean': np.mean(cls_) if cls_ else None,
        'cross_lag_ci95': (cl_lo, cl_hi),
    }

json.dump(results, open('bootstrap_results.json', 'w'), indent=2, default=str)

# Print summary for paper
print("Bootstrap 95% CIs (10k resamples)\n" + "="*65)
for cond, r in results.items():
    c_lo, c_hi = r['coop_ci95']
    v_lo, v_hi = r['vol_ci95']
    print(f"{cond} (n={r['n']})")
    print(f"  Coop: {r['coop_mean']:.3f} [{c_lo:.3f}, {c_hi:.3f}]  sd={r['coop_sd']:.3f}")
    print(f"  Vol:  {r['vol_mean']:.3f}  [{v_lo:.3f}, {v_hi:.3f}]  sd={r['vol_sd']:.3f}")
    if r['recovery_mean'] is not None:
        rd_lo, rd_hi = r['recovery_ci95']
        print(f"  RecovDelay: {r['recovery_mean']:.2f} [{rd_lo:.2f}, {rd_hi:.2f}]  n={r['recovery_n']}")
    if r['cross_lag_mean'] is not None:
        cl_lo, cl_hi = r['cross_lag_ci95']
        print(f"  CrossLag:   {r['cross_lag_mean']:.2f} [{cl_lo:.2f}, {cl_hi:.2f}]  n={r['cross_lag_n']}")
    print()

# ── Figure 4: Recovery delay distributions across ablations ─────────────────
fg_rd  = [c['coop_recovery_delay'] for c in by_cond['Full-Gamma'] if c.get('coop_recovery_delay') is not None]
noh_rd = [c['coop_recovery_delay'] for c in by_cond['No-H']       if c.get('coop_recovery_delay') is not None]
non_rd = [c['coop_recovery_delay'] for c in by_cond.get('No-N', []) if c.get('coop_recovery_delay') is not None]

fig, ax = plt.subplots(1, 1, figsize=(7, 4.2))

bins = np.arange(0, 26, 2)
ax.hist(fg_rd,  bins=bins, alpha=0.55, color='#1f77b4', label=fr'Full-$\Gamma$ (n={len(fg_rd)}, mean={np.mean(fg_rd):.2f})', density=True)
ax.hist(noh_rd, bins=bins, alpha=0.55, color='#ff7f0e', label=fr'No-$\mathcal{{H}}$ (n={len(noh_rd)}, mean={np.mean(noh_rd):.2f})', density=True)
if non_rd:
    ax.hist(non_rd, bins=bins, alpha=0.55, color='#2ca02c', label=fr'No-$\mathcal{{N}}$ (n={len(non_rd)}, mean={np.mean(non_rd):.2f})', density=True)
ax.axvline(np.mean(fg_rd),  color='#1f77b4', ls='--', lw=1.5)
ax.axvline(np.mean(noh_rd), color='#ff7f0e', ls='--', lw=1.5)
if non_rd:
    ax.axvline(np.mean(non_rd), color='#2ca02c', ls='--', lw=1.5)
ax.set_xlabel('Cooperation recovery delay (rounds post-perturbation)')
ax.set_ylabel('Density')
ax.set_title(r'Recovery delay after forced defection: ablations vs Full-$\Gamma$')
ax.legend(fontsize=9, loc='upper right')

plt.tight_layout()
plt.savefig('figures/fig4_recovery_analysis.png', dpi=150, bbox_inches='tight')
print("Saved figures/fig4_recovery_analysis.png")
