import json
from collections import defaultdict
import numpy as np

d = json.load(open('checkpoints/cells.json', encoding='utf-8'))
by_cond_opp = defaultdict(lambda: defaultdict(list))
for v in d.values():
    r = v['result']
    cond = r['key']['condition']
    opp = r['key']['opponent']
    by_cond_opp[cond][opp].append(r['cooperation_rate'])

conds = ['Full-Gamma', 'ReAct', 'Random-Gamma', 'No-H', 'HRRL', 'Drive-only', 'Collapse-NC', 'No-E', 'No-N']
opps = ['TFT', 'GTFT', 'Grim', 'Random']

print('Opponent breakdown (mean coop):')
print(f"{'Condition':22s} {'TFT':8s} {'GTFT':8s} {'Grim':8s} {'Random':8s} {'Range':8s}")
print('-' * 70)
for cond in conds:
    vals = []
    for opp in opps:
        data = by_cond_opp[cond].get(opp, [])
        vals.append(np.mean(data) if data else float('nan'))
    rng = max(v for v in vals if not np.isnan(v)) - min(v for v in vals if not np.isnan(v))
    row = f'{cond:22s} ' + ' '.join(f'{v:7.3f}' for v in vals) + f'  {rng:6.3f}'
    print(row)
