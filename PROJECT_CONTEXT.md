# Pro-Action Operator - Project Context Document

## Project Overview

**Objective:** Prepare a NeurIPS Concept & Feasibility paper for the Pro-Action operator $\Gamma$, a multi-timescale self-regulation model for LLM-agents.

**Contribution Type:** Concept & Feasibility (high-risk/high-reward idea with preliminary feasibility evidence, not final validation)

**Submission Target:** NeurIPS 2026

## Current State (May 6, 2026)

### Paper Status
- **File:** `ecuacion_proaccion/proaction_paper.tex`
- **Pages:** 18
- **Build:** Compiles successfully with pdflatex (no critical errors)
- **Last major edits:**
  - Removed "requisite variety" phrase from abstract only (kept Ashby citation in introduction)
  - Added century context to historical section: "Mid-20th century behaviorism..."
  - Rewrote experimental design with narrative structure: "We test this within IPD. The reason is... The setting is..."
  - Added IPD justification: extensive empirical evidence of human behavior in repeated social interaction
  - Removed metacommentary "Following the reproducibility rule..."
  - Grammar pass applied: conclusion in bullet points, future work style with "A matter for further study is..." and "This research was designed with..."

### Experiment Status
- **File:** `ecuacion_proaccion/checkpoints/cells.json`
- **Progress:** 299/920 cells (31.7% coverage)
  - 292 complete (50 rounds)
  - 7 partial
- **Status:** Running in background (checkpoint.json updating)
- **Latest hypothesis check:** `ecuacion_proaccion/hypothesis_results.txt`
  - Full-Gamma: n=120, coop mean=0.720, vol mean=15.867
  - No-H: n=120, coop mean=0.745, vol mean=14.100
  - No-E: n=23, coop mean=0.796, vol mean=13.217
  - Opponent effects: GTFT (0.839), Grim (0.495), Random (0.719), TFT (0.826)

### Completed Tasks
1. Submission framing: Concept & Feasibility category fit
2. Official NeurIPS sources reviewed (checklist, code policy, ethics, handbook)
3. Category-fit check: paper reads as high-risk/high-reward with preliminary evidence
4. Example-paper guide: selective elements from Capsules extracted
5. Data storytelling style: exploratory guided reasoning applied
6. Reference material prepared (TODOs, benchmark guide, AI_TOOLING.md)
7. First MVP paper update with preliminary results and caveats
8. Reviewer TODO integration with status labels
9. Checklist compliance updated for experiments and LLM declaration
10. Grammar and scientific writing pass completed
11. LaTeX build check passed
12. Regressions fixed (abstract, experimental design, historical context)

### Pending Tasks
1. NeurIPS action items:
   - N1: Verify licenses
   - N2: Prepare anonymized code zip
   - N3: Report compute resources
   - N4: Design BFI statistical reporting (deferred to benchmark completion)

## Project Structure

```
ecuacion_proaccion/
├── proaction_paper.tex          # Main paper draft
├── checklist.tex                # NeurIPS checklist
├── referencias.bib              # Bibliography
├── paper_reviews_todo.md        # Reviewer TODOs with status
├── neurips_guidelines_complete.md # Official NeurIPS guidelines
├── AI_TOOLING.md                # AI tooling declaration
├── exp/
│   ├── runner.py                # Benchmark runner
│   ├── checkpoint.py            # Checkpoint management
│   ├── red_flags.py             # Red-flag detection (modified to treat incomplete cells as warnings)
│   └── checkpoints/
│       └── cells.json           # Experiment checkpoint
├── figures/
│   ├── fig1_thermostats.png
│   ├── fig2_payoffs.png
│   └── fig3_elastic_return.png
└── hypothesis_check.py         # Preliminary hypothesis testing
```

## Key Decisions and Constraints

### Writing Style
- **Problem-first voice:** Start with the problem, not the solution
- **Data storytelling:** Exploratory guided reasoning, not robotic templates
- **Historical narrative:** Include century context (e.g., "Mid-20th century behaviorism")
- **No metacommentary:** Don't announce scientific rigor ("Following the reproducibility rule..."), simply be scientific
- **Conclusion style:** Bullet points with "To conclude, the major contributions..."
- **Future work style:** "A matter for further study is..." and "This research was designed with..."


### Experimental Design Decisions
- **Environment:** IPD with 10% noise, 50 rounds, forced-defection perturbation at round 10
- **Rationale:** IPD has extensive empirical evidence of human behavior in repeated social interaction (Axelrod 1984, Nowak 2006, Porcelli 2017)
- **Baselines:** ReAct, LangGraph with memory, Driveplexity/homeostatic-RL reduction, pymdp active-inference
- **Providers:** OpenAI, Anthropic, DeepSeek (held fixed across conditions)
- **Ablations:** Full-$\Gamma$, No-$\mathcal{H}$, No-$\mathcal{E}$, No-$\mathcal{N}$, ReAct-only, Driveplexity

### Benchmark Constraints
- **Final comparisons:** Must use only 50-round cells with balanced coverage
- **Partial cells:** Treated as warnings, not halts (red_flags.py modified)
- **Coverage goal:** 920 total cells (currently 31.7% complete)

## Paper Structure

1. **Abstract:** Problem statement, operator definition, pre-registered properties, contribution type
2. **Introduction:** Problem framing, historical context, scientific question, contributions
3. **Related Work:** LLM-agent orchestration, internal regulation in RL/active inference, multi-timescale biology, affect/interoception, cybernetics
4. **The Pro-Action operator:** Axioms (A1-A9), operator definition, finite delayed parameterization, reductions
5. **Falsifiability protocol:** P1 elastic return, P2 Yerkes-Dodson inverted-U, P3 delayed reappraisal, BFI
6. **Experimental design:** IPD rationale, baselines, providers, parameter provenance, ablations
7. **Preliminary verification:** Computational sanity checks, simulation trajectories
8. **Interim LLM-agent benchmark snapshot:** Coverage, patterns, opponent effects, anomalies
9. **Discussion and limitations:** What $\Gamma$ is not, limitations, broader impacts
10. **Conclusion:** Contributions in bullet points, future work
11. **Appendix:** Finite delayed parameterization argument (deferred)

## Critical Citations

- **axelrod1984evolution** - IPD empirical evidence
- **nowak2006evolutionary** - IPD evolutionary dynamics
- **porcelli2017stress** - SAM-HPA timescale split, Yerkes-Dodson
- **friston2017active** - Active inference
- **keramati2014homeostatic** - Homeostatic RL
- **ashby1956introduction** - Requisite variety (in introduction only)

## Files to Check for Issues

1. **referencias.bib** - Had duplicate mcewen2007physiology entry (fixed)
2. **exp/red_flags.py** - Modified to treat incomplete cells as warnings instead of halts
3. **proaction_paper.tex** - Main paper, verify no regressions after edits
4. **checklist.tex** - NeurIPS checklist, verify alignment with current experiment status

## Next Steps for Continuity

1. Monitor benchmark progress (checkpoints/cells.json)
2. When benchmark reaches higher coverage (e.g., 50%+), consider updating paper with more complete results
3. Complete NeurIPS action items (N1-N4) before submission
4. Verify all citations resolve correctly (run bibtex + pdflatex twice)
5. Final LaTeX build check before submission

## Contact/Context Notes

- **Author identity:** Anonymous for double-blind review
- **Companion work:** Currently under review at regional venue (cited anonymously)
- **Primary contribution:** Modeling, not experimental technique
- **Evidence level:** Feasibility evidence, not final empirical adjudication

---

**Last updated:** May 6, 2026
**Context version:** 1.0
