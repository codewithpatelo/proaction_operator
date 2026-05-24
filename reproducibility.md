# Reproducibility Guide - Pro-Action Gamma Experiment

This document provides exact commands to reproduce the experiment.

## Quick Start

```bash
# 1. Install dependencies
make install

# 2. Run smoke tests (free, no API calls)
make smoke

# 3. Run calibration (numerical only, ~30s)
make calibration

# 4. Run preflight (1 cell, ~$0.10, ~3min)
make preflight

# 5. Run full benchmark (~$15-20, ~2-3h)
make benchmark-confirm

# 6. Analyze results
make analyze
```

## Environment

Required environment variables (in `.env`):

```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
DEEPSEEK_API_KEY=sk-...
```

## Seeds

- **Calibration**: 42 (hyperparameter selection, excluded from evaluation)
- **Validation**: 999 (held-out for calibration sanity check)
- **Evaluation**: [7, 17, 99, 123, 256, 511, 1024, 2048, 4096, 8192] (10 seeds)

## Docker

For fully containerized reproduction:

```bash
docker build -t proaction-gamma .
docker run --rm -it \
  --env-file .env \
  -v $(pwd)/results:/work/results \
  -v $(pwd)/checkpoints:/work/checkpoints \
  proaction-gamma --all
```

## Outputs

All outputs are written to:

- `checkpoints/cells.json` - completed cell index / cell-level metrics
- `checkpoints/status/benchmark.json` - benchmark status metadata
- `results/results.csv` - raw cell-level benchmark results
- `results/hypotheses.json` - hypothesis-test summary used for paper claims

## Determinism Notes

Only **OpenAI** provides native `seed` parameter. For **Anthropic** and **DeepSeek**, we use `temperature=0` and log verbatim responses. Re-analysis is exact (uses logged responses); regeneration may differ.

## Budget Caps

Hard caps enforced per provider:

- DeepSeek: $4.75 (of $5)
- Anthropic: $18.00 (of $20, below auto-recharge threshold)
- OpenAI: $5.70 (of $6)

Total: $50 global cap.

## Troubleshooting

**Issue**: `checkpoints/` not writable  
**Fix**: `chmod 755 checkpoints` or run from writable directory

**Issue**: API rate limits  
**Fix**: Experiment auto-backoffs; check `exp/budget.py` for current spent

**Issue**: Partial results after crash  
**Fix**: Resume with same command - checkpoints are idempotent

## Citation

If using this experiment:

```bibtex
@article{proaction2026,
  title={Pro-Action: A Formal Gamma Operator for LLM-Agent Self-Regulation},
  author={[Anonymous]},
  year={2026},
  note={Anonymous ICML submission}
}
```
