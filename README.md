# The Pro-Action Operator Γ — ICML 2026 Latinx in AI Workshop Supplementary Material

![Pro-Action operator animation](assets/proaction_operator_demo.gif)

This repository contains the anonymous supplementary material for:

**The Pro-Action Operator: The Feasibility of a Bio-Inspired Regulatory Harness for LLM Agents**

The repository is intentionally scoped for review. It includes the paper source, reproducibility code, calibration/smoke checks, prompt templates, benchmark outputs cited in the paper, and the computational verification artifacts referenced in the text. Internal notes, previous venue formats, exploratory analyses, unused figures, and research drafts are excluded through `.gitignore`.

## What the submitted paper claims

The paper introduces the Pro-Action operator Γ as a six-subsystem coupled thermostat used as a **regulatory harness** around LLM policy execution. The implemented benchmark version is deliberately narrow:

- **Delay-free linear coupling** with a hand-designed sparse matrix `W`.
- **Regulatory State Verbalized Interoception (RSVI)**: numerical thermostat state is translated into prompt context without directly prescribing the action.
- **Iterated Prisoner's Dilemma (IPD)** benchmark with 50 rounds, 10% symmetric action noise, four opponent policies, three LLM providers, and fixed evaluation seeds.
- **Main reported comparisons**: Full-Γ, ReAct, HRRL, and Drive-only.

The paper does **not** claim that six subsystems are minimal, that the chosen coupling matrix is unique, that lexical priming is ruled out, that payoff improves, or that the result generalizes to human regulatory behavior. Those are stated as limitations and future work.

## Repository contents for reviewers

```text
ecuacion_proaccion/
├── proaction_paper_icml.tex      # Submitted ICML paper source
├── proaction_paper_icml.pdf      # Compiled paper PDF
├── referencias.bib              # Bibliography
├── icml2025.sty                 # ICML style file
├── icml2025.bst                 # ICML bibliography style
├── README.md                    # This file
├── reproducibility.md           # Command-oriented reproduction guide
├── LICENSE                      # Code license
├── requirements.txt             # Python dependencies
├── Makefile                     # Convenience targets
├── manim_proaction_operator.py  # Optional explanatory animation script
│
├── exp/
│   ├── runner.py                # Main benchmark runner and Γ update
│   ├── prompts.py               # Frozen RSVI and ReAct prompt renderers
│   ├── llm_clients.py           # Provider adapters
│   ├── checkpoint.py            # Idempotent cell checkpointing
│   ├── metrics.py               # Episode metrics
│   ├── drive_only_baseline.py   # Numerical Drive-only baseline
│   ├── hrrl_baseline.py         # Numerical HRRL baseline
│   ├── ipd_episode.py           # IPD opponent policies/utilities
│   ├── analyze.py               # Aggregate analysis utilities
│   └── red_flags.py             # Degenerate-run diagnostics
│
├── smoke_test.py                # Smoke checks cited in the paper
├── simulator.py                 # LLM-free Γ trajectory simulator
├── verify_symbolic.py           # SymPy equation checks
├── verify_mpmath.py             # Arbitrary-precision numerical checks
├── verify_z3.py                 # SMT checks on update properties
├── verify_axioms_z3.py          # SMT coherence checks for A1--A3 commitments
├── axiom_soundness_report.md    # Precomputed axiom check summary
├── soundness_report.md          # Precomputed computational verification summary
│
├── analyze_bootstrap.py         # Bootstrap confidence intervals
├── analyze_hypothesis_tests.py  # Paired bootstrap test for opponent range
├── bootstrap_results.json       # Precomputed bootstrap outputs
│
├── checkpoints/
│   ├── cells.json               # Completed cell index / cell-level metrics
│   └── status/benchmark.json    # Benchmark status metadata
│
├── results/
│   └── results.csv              # Cell-level benchmark results
│
└── figures/
    └── fig1_thermostats.png     # Appendix simulation trace used in the paper
```

## Experiment design

| Dimension | Final submitted specification |
|---|---|
| Task | Iterated Prisoner's Dilemma |
| Rounds | 50 per cell |
| Noise | 10% symmetric action noise |
| Perturbation | Transient load at round 10 on the hormonal/arousal thermostat |
| Opponents | TFT, Grim, Random, GTFT |
| LLM providers | OpenAI `gpt-5-nano`, Anthropic `claude-haiku-4-5`, DeepSeek `deepseek-v4-flash` |
| Evaluation seeds | `7, 17, 99, 123, 256, 511, 1024, 2048, 4096, 8192` |
| Calibration seed | `42`, excluded from evaluation |
| Held-out validation seed | `999`, excluded from evaluation |
| Main reported conditions | Full-Γ, ReAct, HRRL, Drive-only |

The completed benchmark artifact contains 920 cells with balanced opponent coverage. The paper's main table reports the four conditions used for the final argument: Full-Γ, ReAct, HRRL, and Drive-only. Additional planned controls and ablations may appear in code or historical outputs, but they are **not** used as evidence unless explicitly reported in the paper.

## Important limitations reflected in the repository

Several controls were planned but are not treated as completed evidence in the paper:

- **Scrambled-label prompt control**: intended to distinguish numerical regulatory state from lexical label priming. Parsing failures and incomplete usable runs prevented inclusion as evidence.
- **Matched scalar-regulatory prompt control**: intended to test whether a simpler scalar homeostatic signal could reproduce the Full-Γ profile. It was not completed as a matched evidential control.
- **Payoff traces**: per-interaction payoff was not retained, so the paper does not claim cumulative payoff improvement.
- **Timescales and delays**: the broader research roadmap includes distinct regulatory timescales, delayed recovery, and fast/slow arbitration, but the submitted benchmark tests only delay-free linear coupling plus RSVI.
- **Human comparison**: the IPD benchmark is not a human-comparison study; such a study would require longer interactions and genuinely analogous behavioral or physiological measures.

## Quick reproduction

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the LLM-free checks cited in the paper:

```bash
python smoke_test.py
python verify_symbolic.py
python verify_mpmath.py
python verify_z3.py
python verify_axioms_z3.py
```

Run the benchmark pipeline if API keys are available:

```bash
python -m exp.runner --calibration
python -m exp.runner --preflight
python -m exp.runner --benchmark
```

Run analysis on existing results:

```bash
python analyze_bootstrap.py
python analyze_hypothesis_tests.py
```

See `reproducibility.md` for more detailed commands and environment notes.

## API keys

The benchmark requires provider credentials supplied through environment variables or a local `.env` file:

```text
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
DEEPSEEK_API_KEY=...
```

Do not commit `.env` or secret material. The `.gitignore` excludes local secrets and runtime logs.

## Statistical test reported in the paper

The paper reports a paired bootstrap over per-`(provider, seed)` opponent ranges. The reported comparison is Full-Γ > ReAct on opponent range:

```text
Δ = 0.283, 95% CI [0.233, 0.329], n = 30 paired blocks, 10k resamples, one-sided p < 0.001
```

The intent of this test is not to prove architectural necessity. It tests whether removing regulatory-state injection reproduces the same opponent-differentiated range under matched provider/seed blocks.

## Prompt protocol

The canonical prompt implementation is `exp/prompts.py` (`PROMPT_VERSION = "v1.1"`). The paper appendix reproduces the Full-Γ system prompt, the Full-Γ RSVI user prompt, and the ReAct baseline prompt.

RSVI supplies:

- six thermostat values,
- set-points,
- qualitative low/high anchors,
- `p_C` cooperation pressure,
- `h_SAM` acute-arousal contrast,
- recent IPD history.

The prompt explicitly allows the LLM to override `p_C`; Γ modulates policy execution rather than mandating the action.

## License

The code in this repository is distributed under **GNU AGPLv3-or-later** as a strong copyleft license. See `LICENSE`.

## AI Tooling Disclosure

### LLMs in core methodology (policy-execution layer)

| Provider | Model | Role |
|----------|-------|------|
| OpenAI | gpt-5-nano | Agent policy across all LLM conditions |
| Anthropic | claude-haiku-4-5 | Agent policy across all LLM conditions |
| DeepSeek | deepseek-v4-flash | Agent policy across all LLM conditions |

Each model is held fixed across conditions. Provider-level summaries are diagnostics, not claims about provider quality.

### LLMs as authoring and development tools

| Tool | Use case | Review status |
|------|----------|--------------|
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

This disclosure follows the ICML 2025 LLM usage policy. The use of AI tools for writing and development is declared and does not affect the core scientific methodology, which was designed and verified independently.
