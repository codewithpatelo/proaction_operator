"""Holm-Bonferroni-corrected H1-H3 hypothesis tests.

H1: Full-Gamma > ReAct in opponent discrimination (range = max - min over 4 opponents)
H2: Full-Gamma > Random-Gamma
H3: Full-Gamma > Collapse-NC

Statistic: per (provider, seed), opponent range over the four opponents.
Bootstrap test on the difference of mean ranges (10k resamples).
Holm-Bonferroni correction across the 3 comparisons.
"""
import json
from collections import defaultdict
import numpy as np

CELLS_PATH = 'checkpoints/cells.json'
OUT_PATH = 'hypothesis_tests.json'

OPPONENTS = ['TFT', 'GTFT', 'Grim', 'Random']
COMPARISONS = [
    ('H1', 'Full-Gamma', 'ReAct'),
    ('H2', 'Full-Gamma', 'Random-Gamma'),
    ('H3', 'Full-Gamma', 'Collapse-NC'),
]
N_BOOTSTRAP = 10_000
RNG = np.random.default_rng(42)


def load_ranges(cells_path):
    """For each (condition, provider, seed), compute opponent range over 4 opponents."""
    raw = json.load(open(cells_path, encoding='utf-8'))
    cells = list(raw.values()) if isinstance(raw, dict) else raw
    by_cps = defaultdict(dict)  # (cond, prov, seed) -> {opp: coop}
    for c in cells:
        k = c['key']
        r = c['result']
        if r.get('cooperation_rate') is None:
            continue
        cond, prov, seed, opp = k['condition'], k['model'], k['seed'], k['opponent']
        by_cps[(cond, prov, seed)][opp] = r['cooperation_rate']

    ranges = defaultdict(list)  # cond -> list of ranges (over (prov, seed))
    keys = defaultdict(list)
    for (cond, prov, seed), opp2coop in by_cps.items():
        if not all(o in opp2coop for o in OPPONENTS):
            continue
        vals = [opp2coop[o] for o in OPPONENTS]
        rng_val = max(vals) - min(vals)
        ranges[cond].append(rng_val)
        keys[cond].append((prov, seed))
    return ranges, keys


def paired_bootstrap(a, b, n_boot=N_BOOTSTRAP):
    """One-sided test of mean(a) > mean(b) given paired arrays a, b.

    If sample sizes match exactly, treat as paired by index. Otherwise unpaired.
    Returns (delta, ci_low, ci_high, p_one_sided).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) == len(b):
        diffs = a - b
        rng = RNG
        boots = np.empty(n_boot)
        for i in range(n_boot):
            sample = rng.choice(diffs, size=len(diffs), replace=True)
            boots[i] = sample.mean()
        delta = diffs.mean()
        ci_low, ci_high = np.percentile(boots, [2.5, 97.5])
        # one-sided p: P(boot mean <= 0)
        p = float((boots <= 0).mean())
    else:
        rng = RNG
        boots = np.empty(n_boot)
        for i in range(n_boot):
            sa = rng.choice(a, size=len(a), replace=True)
            sb = rng.choice(b, size=len(b), replace=True)
            boots[i] = sa.mean() - sb.mean()
        delta = a.mean() - b.mean()
        ci_low, ci_high = np.percentile(boots, [2.5, 97.5])
        p = float((boots <= 0).mean())
    return float(delta), float(ci_low), float(ci_high), p


def holm_bonferroni(p_values):
    """Apply Holm-Bonferroni correction. Returns adjusted p-values in original order."""
    p_array = np.asarray(p_values, dtype=float)
    n = len(p_array)
    order = np.argsort(p_array)
    adjusted = np.empty(n)
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = (n - rank) * p_array[idx]
        adj = min(adj, 1.0)
        running_max = max(running_max, adj)
        adjusted[idx] = running_max
    return adjusted.tolist()


def align_paired(ranges_a, keys_a, ranges_b, keys_b):
    """Align two condition arrays by (provider, seed). Returns paired (a_vals, b_vals)."""
    map_a = dict(zip(keys_a, ranges_a))
    map_b = dict(zip(keys_b, ranges_b))
    common = sorted(set(map_a) & set(map_b))
    a_vals = [map_a[k] for k in common]
    b_vals = [map_b[k] for k in common]
    return a_vals, b_vals, len(common)


def main():
    ranges, keys = load_ranges(CELLS_PATH)
    print(f"Loaded conditions: { {c: len(v) for c, v in ranges.items()} }")

    raw_results = []
    for tag, cond_a, cond_b in COMPARISONS:
        a, b, n = align_paired(ranges[cond_a], keys[cond_a],
                               ranges[cond_b], keys[cond_b])
        delta, lo, hi, p = paired_bootstrap(a, b)
        raw_results.append({
            'hypothesis': tag,
            'comparison': f'{cond_a} > {cond_b}',
            'n_paired': n,
            'mean_range_a': float(np.mean(a)),
            'mean_range_b': float(np.mean(b)),
            'delta': delta,
            'ci_95': [lo, hi],
            'p_raw': p,
        })

    p_raw = [r['p_raw'] for r in raw_results]
    p_adj = holm_bonferroni(p_raw)
    for r, padj in zip(raw_results, p_adj):
        r['p_holm_adj'] = padj

    summary = {
        'method': 'paired bootstrap (10k resamples), one-sided test on Δ(opponent range), Holm-Bonferroni across 3 comparisons',
        'observable': 'opponent range = max(coop_rate over {TFT, GTFT, Grim, Random}) - min',
        'pairing': 'by (provider, seed)',
        'results': raw_results,
    }

    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print(f"\nWrote {OUT_PATH}\n")
    print(f"{'H':<4} {'compare':<32} {'n':>4} {'Δ(range)':>9} {'95% CI':>20} {'p_raw':>9} {'p_Holm':>9}")
    print('-' * 95)
    for r in raw_results:
        ci = f"[{r['ci_95'][0]:.3f}, {r['ci_95'][1]:.3f}]"
        print(f"{r['hypothesis']:<4} {r['comparison']:<32} {r['n_paired']:>4} "
              f"{r['delta']:>9.3f} {ci:>20} {r['p_raw']:>9.4f} {r['p_holm_adj']:>9.4f}")


if __name__ == '__main__':
    main()
