"""Mann-Whitney U tests for No-H and No-N recovery delays vs Full-Gamma.

Replaces the invalid CI-overlap proxy with formal nonparametric tests + Cliff's delta.
"""
import json
from collections import defaultdict
import numpy as np
from scipy.stats import mannwhitneyu

CELLS_PATH = 'checkpoints/cells.json'
OUT_PATH = 'recovery_tests.json'

COMPARISONS = [
    ('Full-Gamma', 'No-H'),
    ('Full-Gamma', 'No-N'),
    ('Full-Gamma', 'ReAct'),
]


def load_recovery_delays(cells_path):
    raw = json.load(open(cells_path, encoding='utf-8'))
    cells = list(raw.values()) if isinstance(raw, dict) else raw
    by_cond = defaultdict(list)
    for c in cells:
        r = c['result']
        cond = c['key']['condition']
        rd = r.get('coop_recovery_delay')
        if rd is None:
            continue
        by_cond[cond].append(float(rd))
    return by_cond


def cliffs_delta(a, b):
    """Cliff's delta effect size. δ in [-1, 1].
    δ = (#(a>b) - #(a<b)) / (n_a * n_b).
    """
    a = np.asarray(a)
    b = np.asarray(b)
    gt = sum(1 for x in a for y in b if x > y)
    lt = sum(1 for x in a for y in b if x < y)
    return (gt - lt) / (len(a) * len(b))


def magnitude(d):
    """Romano et al. 2006 thresholds."""
    ad = abs(d)
    if ad < 0.147:
        return 'negligible'
    if ad < 0.33:
        return 'small'
    if ad < 0.474:
        return 'medium'
    return 'large'


def main():
    by_cond = load_recovery_delays(CELLS_PATH)
    print('Recovery-delay sample sizes:',
          {c: len(v) for c, v in sorted(by_cond.items())})

    results = []
    for cond_a, cond_b in COMPARISONS:
        a = by_cond.get(cond_a, [])
        b = by_cond.get(cond_b, [])
        if not a or not b:
            print(f"  SKIP {cond_a} vs {cond_b}: missing data")
            continue
        u_two, p_two = mannwhitneyu(a, b, alternative='two-sided')
        # Effect direction:
        u_lt, p_lt = mannwhitneyu(a, b, alternative='less')   # a < b
        u_gt, p_gt = mannwhitneyu(a, b, alternative='greater')  # a > b
        delta = cliffs_delta(a, b)
        results.append({
            'comparison': f'{cond_a} vs {cond_b}',
            'n_a': len(a),
            'n_b': len(b),
            'median_a': float(np.median(a)),
            'median_b': float(np.median(b)),
            'mean_a': float(np.mean(a)),
            'mean_b': float(np.mean(b)),
            'u_statistic': float(u_two),
            'p_two_sided': float(p_two),
            'p_a_less_than_b': float(p_lt),
            'p_a_greater_than_b': float(p_gt),
            'cliffs_delta': float(delta),
            'delta_magnitude': magnitude(delta),
        })

    summary = {
        'method': 'Mann-Whitney U (two-sided), Cliff\'s delta effect size with Romano et al. 2006 thresholds',
        'observable': 'coop_recovery_delay (rounds post round-10 perturbation to half-recovery)',
        'comparisons': results,
    }

    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print(f"\nWrote {OUT_PATH}\n")
    header = f"{'comparison':<32} {'n_a':>4} {'n_b':>4} {'med_a':>6} {'med_b':>6} {'U':>9} {'p_2sd':>8} {'δ':>7} {'mag':<10}"
    print(header)
    print('-' * len(header))
    for r in results:
        print(f"{r['comparison']:<32} {r['n_a']:>4} {r['n_b']:>4} "
              f"{r['median_a']:>6.2f} {r['median_b']:>6.2f} "
              f"{r['u_statistic']:>9.1f} {r['p_two_sided']:>8.4f} "
              f"{r['cliffs_delta']:>+7.3f} {r['delta_magnitude']:<10}")


if __name__ == '__main__':
    main()
