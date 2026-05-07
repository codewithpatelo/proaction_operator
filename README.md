# The Pro-Action Operator Γ — Supplementary Code

Supplementary material for the NeurIPS 2026 submission:  
**"The Pro-Action Operator: A Scientifically-Informed, Multi-Timescale Self-Regulation Model for LLM-Agents"**

> Anonymous submission. Author identities withheld for double-blind review.

---

## Paper Summary

Contemporary LLM-agent frameworks (ReAct, LangGraph, OpenClaw) address *capability* — what an agent can do — but delegate *activation* to exogenous flow control. This paper proposes Γ, a **Pro-Action operator**: a recursively composable system of six coupled thermostat subsystems (attention, perception, hormonal, emotional, neuropsychological, cognitive) with subsystem-specific delays τ_k inspired by the SAM–HPA biological timescale split.

Each subsystem updates as:

```
x_{k,t+1} = x_{k,t} − κ_k(x_{k,t} − x*_{k,t}) + λ_k − α_k·ρ_k(a_t, e_{t+1}) + [W φ(x_{t−τ_k})]_k
```

Three falsifiable properties are specified a priori (P1 elastic return, P2 Yerkes–Dodson inversion, P3 delayed reappraisal) and tested against a preliminary LLM-agent benchmark using Iterated Prisoner's Dilemma. The paper is submitted as **Concept & Feasibility**: the operator is made formal, computationally verified, and partially empirically instantiated — not fully validated.

---

## Experiment Design

| Dimension | Specification |
|---|---|
| Task | Iterated Prisoner's Dilemma (IPD), 50 rounds |
| Noise | 10% symmetric action noise |
| Perturbation | Forced defection at round 10 (tests P1) |
| Opponents | TFT, Grim, Random, GTFT |
| Providers | OpenAI `gpt-5-nano`, Anthropic `claude-haiku-4-5`, DeepSeek `deepseek-v4-flash` |
| Seeds | 10 evaluation seeds [7, 17, 99, 123, 256, 511, 1024, 2048, 4096, 8192] |
| Unit of analysis | **Cell** = (condition × provider × opponent × seed), 50 rounds |
| Coverage at snapshot | 920 cells: Full-Γ (120), No-H (120), No-E (120), No-N (120), Random-Γ (120), Collapse-NC (120), ReAct (120), Drive-only (40), HRRL (40) |

**Ablation conditions:**

| Condition | What is removed | Purpose |
|---|---|---|
| Full-Γ | — | Reference |
| No-H | Hormonal channel (SAM+HPA) | Tests fast-timescale contribution |
| No-E | Emotional valence | Tests affect as stabilizer/noise |
| No-N | Neuropsychological (slow deliberation) | Tests slow-channel contribution |
| Random-Γ | Biological wiring of W (permuted) | Parameter-count control |
| Collapse-NC | 6-vs-5 subsystem ablation | Tests subsystem-count sensitivity |

**Baselines:**

| Condition | Type | Cells | Description |
|---|---|---|---|
| ReAct | LLM (3 providers) | 120 | Standard ReAct agent without internal state |
| Drive-only | Numerical ($0) | 40 | Single-drive reduction of Γ |
| HRRL | Numerical ($0) | 40 | Keramati-Gutkin 2014 homeostatic RL baseline |

**Pre-committed criteria for No-H adjudication** (before benchmark completion):  
The hormonal channel is considered to contribute distinct dynamic responsiveness if Full-Γ satisfies ≥2 of: (i) shorter cooperation recovery delay; (ii) higher conditional volatility post-perturbation; (iii) larger recovery-delay variance. Null outcome is pre-committed and publishable.

---

## Code Structure

```
ecuacion_proaccion/
├── proaction_paper.tex       # Main paper (NeurIPS 2026 submission)
├── referencias.bib           # Bibliography
├── checklist.tex             # NeurIPS paper checklist
├── neurips_2026.sty          # NeurIPS style file
│
├── exp/
│   ├── runner.py             # Main experiment orchestration (async, checkpointed)
│   ├── watchdog.py           # Background watchdog with auto-respawn
│   ├── prompts.py            # Frozen prompt templates (PROMPT_VERSION=v1.1)
│   ├── llm_clients.py        # Unified provider client (OpenAI, Anthropic, DeepSeek)
│   ├── checkpoint.py         # Idempotent cell checkpointing
│   ├── budget.py             # Hard per-provider budget caps
│   ├── metrics.py            # Episode metric computation
│   ├── conditions_llm.py     # Condition and ablation definitions
│   ├── drive_only_baseline.py # Drive-only numerical baseline
│   ├── hrrl_baseline.py      # HRRL numerical baseline
│   ├── ipd_episode.py        # IPD episode runner
│   ├── analyze.py            # BFI computation and results export
│   └── red_flags.py          # Degenerate-regime detector
│
├── simulator.py              # Numerical Γ simulator (no LLM)
├── smoke_test.py             # Verifies P1 (set-point convergence, half-life)
├── analyze_bootstrap.py      # Bootstrap CIs + Figure 4
├── sensitivity_analysis.py   # Parameter sensitivity (Appendix B / Figure 5)
├── health_check.py           # Dataset integrity audit
│
├── verify_symbolic.py        # SymPy: symbolic master-equation derivation
├── verify_mpmath.py          # mpmath: arbitrary-precision numerical checks
├── verify_z3.py              # Z3 SMT: counterexample search on update-rule properties
├── verify_axioms_z3.py       # Z3 SMT: checks axioms A1–A9 against implementation
├── verify_axioms_graph.py    # Graph-theoretic axiom consistency check
├── verify_axioms_types.py    # Static type checking of Γ subsystem signatures
├── verify_axioms_judge.py    # LLM-assisted axiom plausibility review
│
├── axiom_soundness_report.md # Output of verify_axioms_*.py suite
├── soundness_report.md       # Summary soundness report (all checks)
├── bootstrap_results.json    # Output of analyze_bootstrap.py (CIs in paper Table 3)
│
├── checkpoints/
│   └── cells.json            # Completed cell index (checkpoint store)
└── figures/                  # Generated figures (fig1–fig5)
```

---

## Reproducibility

### Requirements

- Python 3.11+ **or** Docker (recommended for exact environment reproduction)
- API keys for OpenAI, Anthropic, DeepSeek

### Environment

Create a `.env` file in the project root (used by both Docker and local runs):

```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
DEEPSEEK_API_KEY=sk-...
```

---

### Option A — Docker (recommended)

Docker pins the full Python environment and avoids dependency drift. Checkpoints and results are mounted as volumes so they persist across container restarts and remain available on the host.

```bash
# Build the image (one-time)
make docker
# or: docker build -t proaction-gamma .

# Verify numerical implementation (no API calls, free)
docker run --rm proaction-gamma --calibration

# Run one preflight cell (~$0.05, ~3 min)
docker run --rm \
  --env-file .env \
  proaction-gamma --preflight

# Run the full benchmark (idempotent — safe to restart)
docker run --rm \
  --env-file .env \
  -v "$(pwd)/checkpoints:/work/checkpoints" \
  -v "$(pwd)/figures:/work/figures" \
  proaction-gamma --benchmark

# Compute bootstrap CIs and generate figures (after benchmark)
docker run --rm \
  -v "$(pwd)/checkpoints:/work/checkpoints" \
  -v "$(pwd)/figures:/work/figures" \
  proaction-gamma python analyze_bootstrap.py

docker run --rm \
  -v "$(pwd)/checkpoints:/work/checkpoints" \
  -v "$(pwd)/figures:/work/figures" \
  proaction-gamma python sensitivity_analysis.py
```

The container entrypoint is `python -m exp.runner`; pass `--calibration`, `--preflight`, or `--benchmark` as arguments. Override to run any other script by prepending `python`.

**Resuming an interrupted run:** because checkpoints are written to the mounted volume, re-running the same `docker run` command resumes from the last completed cell with no data loss.

---

### Option B — Local (Python 3.11+)

```bash
pip install -r requirements.txt   # or: make install
```

### Reproducing the experiment (local)

```bash
# 1. Verify numerical implementation (no API calls, free)
python smoke_test.py

# 2. Run calibration phase (numerical only, ~30s)
python -m exp.runner --calibration

# 3. Run one preflight cell (~$0.05, ~3 min)
python -m exp.runner --preflight

# 4. Run the full benchmark (budget-capped, idempotent)
python -m exp.runner --benchmark

# 5. Compute bootstrap CIs and generate Figure 4
python analyze_bootstrap.py

# 6. Generate sensitivity analysis (Appendix B)
python sensitivity_analysis.py
```

The runner is **idempotent**: re-running resumes from the last checkpoint. Completed cells are never re-executed.

### Reproducing figures

| Figure | Local command | Docker command |
|---|---|---|
| Fig 1 (thermostats) | `python simulator.py` | `docker run --rm -v "$(pwd)/figures:/work/figures" proaction-gamma python simulator.py` |
| Fig 4 (recovery delay) | `python analyze_bootstrap.py` | `docker run --rm -v "$(pwd)/checkpoints:/work/checkpoints" -v "$(pwd)/figures:/work/figures" proaction-gamma python analyze_bootstrap.py` |
| Fig 5 (sensitivity) | `python sensitivity_analysis.py` | `docker run --rm -v "$(pwd)/figures:/work/figures" proaction-gamma python sensitivity_analysis.py` |

Figs 2 & 3 (payoffs, elastic return zoom) are also generated by `simulator.py` and included in Appendix B of the paper.

### Inference parameters

| Provider | Model | Temperature | Max tokens | Seed |
|---|---|---|---|---|
| OpenAI | `gpt-5-nano` | default | — | per-round integer |
| Anthropic | `claude-haiku-4-5` | 0 | 512 | — (not exposed) |
| DeepSeek | `deepseek-v4-flash` | 0 | 512 | — (deprecated) |

Concurrency: 8 cells in-flight simultaneously. Retries: 4 exponential backoff on transient errors.

### Seeds

- **Calibration seed**: 42 (excluded from evaluation; used for hyperparameter smoke-test)
- **Held-out validation seed**: 999 (calibration sanity check)
- **Evaluation seeds**: `[7, 17, 99, 123, 256, 511, 1024, 2048, 4096, 8192]`

---

## Mathematical Verification

The paper claims computational sanity checks on the master equation (Eq. 2) using four independent tools. All verification scripts are in the repository root and can be run without API keys.

### Verification suite

| Script | Tool | What it checks |
|---|---|---|
| `verify_symbolic.py` | SymPy | Symbolic derivation of the master equation; confirms closed-form update rule is equivalent to step-by-step algebra |
| `verify_mpmath.py` | mpmath (arbitrary precision) | Numerically verifies set-point convergence and half-life under 100-digit arithmetic; rules out floating-point artifacts |
| `verify_z3.py` | Z3 SMT solver | Searches for counterexamples to the update-rule monotonicity and boundedness properties |
| `verify_axioms_z3.py` | Z3 SMT | Checks each axiom A1–A9 against the implementation; verifies no axiom is vacuously satisfied |
| `verify_axioms_graph.py` | NetworkX | Graph-theoretic check: coupling matrix W is consistent with the directed dependency graph implied by the axioms |
| `verify_axioms_types.py` | Python typing | Static type check on subsystem function signatures (ensures composability) |
| `verify_axioms_judge.py` | LLM review | Structured LLM review of each axiom's plausibility; output in `axiom_judge_results.json` |

### Running the verification suite

```bash
# Free — no API keys required (except verify_axioms_judge.py)
python verify_symbolic.py       # SymPy derivation
python verify_mpmath.py         # Arbitrary-precision numerics
python verify_z3.py             # SMT counterexample search
python verify_axioms_z3.py      # Axiom A1–A9 Z3 check
python verify_axioms_graph.py   # Coupling-graph consistency
python verify_axioms_types.py   # Type signature check
```

Pre-computed outputs are in `axiom_soundness_report.md` and `soundness_report.md` for reviewers who do not wish to re-run.

### Calibration

Parameter values in Table 1 of the paper (`κ`, `λ`, `α`, `x*`, `τ`, `W`, `Q`) were assigned provenance categories *before* the LLM benchmark. The calibration procedure is:

1. **Smoke-test calibration** (`smoke_test.py`): parameters are chosen to satisfy three non-divergence criteria in the numerical simulator: (a) `‖x_t − x*‖ < 0.15` after 10 rounds; (b) elastic return half-life `t₁/₂ = 3.2 rounds`; (c) π_fast rises from 0.35 → 0.92 under simulated threat. These criteria were set before any LLM run.

2. **Held-out calibration seed**: seed 42 was used for smoke-test parameter tuning; seed 999 was the held-out validation check. Neither appears in the evaluation seeds.

3. **Evaluation seeds** (`[7, 17, 99, 123, 256, 511, 1024, 2048, 4096, 8192]`): chosen as round numbers with no relationship to calibration.

All calibration constants are frozen in `exp/conditions_llm.py` and in the paper Table 1. The W matrix is available in `exp/conditions_llm.py:FULL_GAMMA_W`.

---

## Prompt Protocol

Each round, the LLM receives a frozen prompt (version-controlled as `PROMPT_VERSION = "v1.1"`) containing the six thermostat values, aggregated cooperation pressure `p(C)`, SAM signal `h_SAM`, and IPD game history. The LLM is explicitly told it may follow or override `p(C)`; the controller modulates rather than overrides.

### Full-Γ system prompt (verbatim)

```
You play an Iterated Prisoner's Dilemma against an opponent. You will play 50 rounds total.

ACTIONS (choose one per round):
- C = Cooperate: both players get the reward if both choose C; you risk exploitation if opponent chooses D.
- D = Defect: you get temptation payoff if opponent cooperates; both get punishment if both defect.

PAYOFFS: CC=(3,3), CD=(0,5), DC=(5,0), DD=(1,1). 10% of actions are flipped by noise.
You cannot see your opponent's identity, only their past actions.

You have access to your own internal regulatory state. Each value is in [0, 1]; the deviation
from your set-point indicates how far each subsystem is from equilibrium. These signals reflect
*your* state, not the opponent's.
```

### Full-Γ user prompt (per-round template, verbatim)

```
Round {t} of 50.
Last 5 rounds (most recent last): {history}

Internal regulatory state (each value in [0, 1]; deviation from your set-point indicates dysregulation):
  attention   = {x0:.2f}  (set-point 0.30; low = scattered focus, high = narrow focus)
  perception  = {x1:.2f}  (set-point 0.20; low = noisy reading, high = clear reading)
  hormonal    = {x2:.2f}  (set-point 0.30; low = under-aroused / calm, high = elevated stress / arousal)
  emotional   = {x3:.2f}  (set-point 0.10; low = neutral affect, high = strong negative valence)
  neuro-fast  = {x4:.2f}  (set-point 0.20; low = passive / reflective, high = ready for fast reactive response)
  cognitive   = {x5:.2f}  (set-point 0.40; low = automatic / quick, high = active deliberation, slow weighing)

Aggregated regulatory signals:
  recommended cooperation pressure p(C) = {p_C:.2f}  (range [0,1]; 0.5 = ambiguous, >0.7 = strong push to C, <0.3 = strong push to D)
  recent acute stress (h_SAM)            = {h_sam:.2f}  (aggregate of hormonal minus emotional, range ≈ [-1, 1]; 0 = baseline, positive = stress spike, negative = below baseline / recovered)

Decide your next action. You may follow or override p(C) based on context.
Respond with exactly one JSON object:
{"action": "C" or "D", "reason": "<one short sentence>"}
```

Placeholder substitutions: `{t}` round number; `{history}` last 5 (action, opponent_action) pairs joined by `|`; `{x0..x5}` the six thermostat values; `{p_C}`, `{h_sam}` the aggregated signals.

### ReAct baseline prompt (verbatim)

```
Round {t} of 50.
Last 5 rounds: {history}
Think step by step about what to do, then respond with JSON:
{"thought": "<your reasoning>", "action": "C" or "D"}
```

ReAct receives the same `{history}` as Full-Γ — the only structural difference is the absence of regulatory-state injection.

### Scrambled-labels control arm

`get_scrambled_prompt` in `exp/prompts.py` permutes semantic labels to neutral identifiers (`attention → metric_A`, `perception → metric_B`, …, `cognitive → metric_F`) while preserving numeric values. This arm tests whether observed behavior is driven by label priming or controller dynamics, and is reserved for post-completion ancillary analysis.

---

## AI Tooling Disclosure

### LLMs in core methodology (policy-execution layer)

| Provider | Model | Role |
|---|---|---|
| OpenAI | `gpt-5-nano` | Agent policy across all LLM conditions |
| Anthropic | `claude-haiku-4-5` | Agent policy across all LLM conditions |
| DeepSeek | `deepseek-v4-flash` | Agent policy across all LLM conditions |

Each model is held fixed across conditions. Provider-level summaries are diagnostics, not claims about provider quality.

### LLMs as authoring and development tools

| Tool | Use case | Review status |
|---|---|---|
| GPT-5.5 + Grammarly | Readability improvements, text refinement | Human reviewed |
| Claude Code | Code assistance, refactoring, experiment monitoring | Human reviewed |
| Windsurf IDE + Cascade | Agentic code development, multi-file refactoring | Human reviewed |
| DeepSeek R4 | Mathematical derivation cross-checks | Human verified |
| Reviewer3 | Simulated peer review | Human verified |
| Elicit | Line of research validation | Human verified |
| Consensus | Hypothesis validation | Human verified |
| Perplexity | Literature research on related work and prior art | Human verified |
| FigureLabs | Figure generation and visual refinement | Human reviewed |

All AI-generated content was reviewed and edited by human authors who assume full responsibility for all scientific claims, citations, experimental design, and code. No AI tool was used to generate core scientific claims without human verification.

This disclosure follows the NeurIPS 2026 LLM usage policy. The use of AI tools for writing and development is declared and does not affect the core scientific methodology, which was designed and verified independently.

### Frozen artifacts

The following files are version-controlled and frozen for reproducibility:

| File | Role |
|---|---|
| `exp/prompts.py` | Prompt templates (`PROMPT_VERSION = "v1.1"`) |
| `exp/llm_clients.py` | Unified provider client with per-provider quirks |
| `exp/runner.py` | Experiment orchestration |
| `exp/conditions_llm.py` | Condition and ablation definitions |

### Determinism per provider

| Provider | Mechanism | Notes |
|---|---|---|
| OpenAI | Native `seed` parameter | `system_fingerprint` logged for drift detection |
| Anthropic | `temperature=0` | No native seed exposed |
| DeepSeek | `temperature=0` | Seed deprecated per API docs |

Full verbatim responses are logged for exact re-analysis even if API regeneration differs.

---

## Checklist for Reviewers

| Item | Location |
|---|---|
| Operator definition (Eq. 1–3) | Paper §3, `simulator.py` |
| Proposition 1 proof sketch | Paper Appendix A |
| Frozen parameter provenance | Paper Table 1, `exp/conditions_llm.py` |
| Coupling matrix W | `exp/conditions_llm.py:FULL_GAMMA_W` |
| Ablation definitions | Paper Table 2, `exp/runner.py:apply_ablation` |
| Evaluation seeds (10 seeds) | Paper §4, `README.md` §Seeds, `exp/runner.py` |
| Calibration procedure & seeds | `README.md` §Calibration, `smoke_test.py` |
| SymPy master-equation check | `verify_symbolic.py` |
| Arbitrary-precision check | `verify_mpmath.py` |
| Z3 SMT counterexample search | `verify_z3.py`, `verify_axioms_z3.py` |
| Axiom A1–A9 verification | `verify_axioms_z3.py`, `axiom_soundness_report.md` |
| Full soundness report | `soundness_report.md` |
| Prompt template | Paper §4.3, `exp/prompts.py:FULL_GAMMA_USER_TEMPLATE` |
| Scrambled-labels control | Paper §4.3, `exp/prompts.py:get_scrambled_prompt` |
| Bootstrap CIs | Paper Table 3, `analyze_bootstrap.py`, `bootstrap_results.json` |
| Pre-committed No-H criteria | Paper §5.1, `README.md` §Experiment Design |
| Sensitivity analysis | Paper Appendix B, `sensitivity_analysis.py` |
| Inference hyperparameters | Paper §4, `exp/budget.py`, `exp/llm_clients.py` |
| Dataset integrity | `health_check.py`, `checkpoints/cells.json` |
