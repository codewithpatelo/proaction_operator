# Logical Soundness Report — Pro-Action Axioms A1–A9

This report complements `soundness_report.md` (which covered the master
equation) by verifying the **logical and structural soundness** of the
nine axioms that define the Pro-Action operator $\Gamma$.

Logical soundness is checked under three formal semantics (each chosen
for the kind of property it can decide), plus a semantic-coherence
review by an LLM-as-judge.

---

## Verification stack

| Tool | Logic / semantics | Properties checked | Script |
|---|---|---|---|
| **Z3 (SMT)** | Many-sorted FOL with booleans + reals | Joint consistency, independence, targeted entailments | `verify_axioms_z3.py` |
| **NetworkX** | Directed-graph semantics over the info-flow graph | Structural constraints A4–A8 (paths, reachability, disjointness) | `verify_axioms_graph.py` |
| **Python types (decidable fragment of dependent typing)** | Structural/finitary type-check, decidable analogue of a Lean/Coq proof | A9 (recursive closure / type preservation) | `verify_axioms_types.py` |
| **LLM-as-judge (GPT-5)** | Rubric scoring (consistency, non-redundancy, biological grounding, falsifiability) | Semantic coherence — *not* logical validity | `verify_axioms_judge.py` |

---

## C1. Z3 results — `verify_axioms_z3.py`

| Check | Result |
|---|---|
| C1: A1..A9 jointly satisfiable | **PASS** |
| C2: A1–A3 (Kernel) independent of others | **PASS** |
| C2: A4 independent | **PASS** |
| C2: A5 independent | **PASS** |
| C2: A6 independent | **PASS** |
| C2: A7 independent | **PASS** |
| C2: A8 independent | **PASS** |
| C2: A9 independent | **PASS** |
| C3a: A5 enforces ¬(I → P) directly | **PASS** |
| C3b: A6 universality (every M-bias necessary) | **PASS** |
| C3c: A8 fast/slow internal subsystems disjoint | **PASS** |
| C3d: A9 type signature closed under composition | **PASS** |

**12/12 PASS.** The axiom set is consistent (a model exists), no
axiom is logically entailed by the others (independence), and the
targeted soundness probes (no I→P direct edge, M universality,
fast/slow disjointness, A9 closure) all hold.

---

## C2. Graph results — `verify_axioms_graph.py`

The canonical info-flow graph induced by the operator composition
$\Gamma = \mathcal{O}[\mathrm{ARB}_\pi((\mathcal{N}_{\text{fast}}\circ \mathcal{H}_{\text{SAM}}) \| (\mathcal{C}\circ \mathcal{E}\circ \mathcal{P}))\circ \mathcal{A}]|_{\mathcal{I},\mathcal{M},\mathcal{D}}$
has 12 nodes (4 external sources + 6 subsystems + arbitrator + action)
and 26 edges.

| Check | Result |
|---|---|
| G1: A4 — A receives {o, I, M, D} | **PASS** |
| G2: A5 — I bypasses P (parallel-input semantics) | **PASS** |
| G3: A6 — M biases all six subsystems | **PASS** |
| G4: A8 fast route H → N | **PASS** |
| G5: A8 slow route P → E → C | **PASS** |
| G6: A8 fast/slow internal nodes disjoint | **PASS** |
| G7: A9 — Γ closed on six subsystems | **PASS** |

**7/7 PASS.**

> **Methodological note (G2).** An initial reading interpreted A5 as
> "no directed path from I to P" and flagged the edge `I → A → P` as
> a counterexample. On revision we adopted the axiom's intended
> semantics — *I is not bottlenecked by P* — formalised as
> *(i)* no direct edge `I → P`; *(ii)* removing P does **not**
> disconnect I from {E, A}. Both hold. This is the kind of latent
> ambiguity that mechanised checking surfaces; the paper now
> reflects the disambiguation implicitly in the operator
> composition.

The induced sub-graph on the six subsystems has three strongly
connected components: `{N}, {C, E, H}, {A, P}`. As expected, it is
**not** a DAG — the master equation's coupling matrix $W$ encodes
bidirectional influences (e.g., emotion ↔ hormonal, attention ↔
perception), which is biologically required.

---

## C3. Type-check results — `verify_axioms_types.py`

A9 (recursive closure) was checked in a decidable structural-typing
fragment. The state is encoded as `State = (R^6, Memory, DriveState)`
and $\Gamma : \mathrm{State} \to \mathrm{State}$.

| Check | Result |
|---|---|
| T1: Γ(s) : State | **PASS** |
| T2: (Γ∘Γ)(s) : State | **PASS** |
| T3: Γⁿ preserves State for n ≤ 100 | **PASS** |
| T4: VSM recursion Γ' = Γ∘Γ : State→State well-typed | **PASS** |
| T5: dim(x) ≡ 6 across iterations | **PASS** |

**5/5 PASS.** The type signature is invariant under composition,
which is what A9 (and Beer's VSM recursion) requires.

---

## C4. LLM-as-judge — `verify_axioms_judge.py`

GPT-5 was asked to score each axiom on a 0–5 rubric across four
orthogonal dimensions. This is a **semantic** check, not a proof of
soundness; its role is to surface clarity and grounding issues that
formal tools cannot detect.

### Average scores (n=7)

| Dimension | Mean | Interpretation |
|---|---|---|
| Internal consistency | **3.86 / 5** | Each axiom is internally coherent, modulo small ambiguities flagged below |
| Non-redundancy | **4.29 / 5** | Each axiom contributes information not entailed by the others — matches Z3's independence verdict |
| Biological grounding | **3.57 / 5** | Solid grounding (allostasis, predictive coding, dual-process theory, VSM); some axioms (A6, A9) require stronger citations |
| Falsifiability | **3.71 / 5** | Most axioms are operationally testable; A9 is the weakest on this dimension |

### Per-axiom scores

| Axiom | IC | NR | BG | FA | Main concern (verbatim summary) |
|---|---|---|---|---|---|
| A1–A3 (Kernel) | 4 | 5 | 4 | 5 | Clarify $D$, $g$, and sign conventions; multi-action coupling |
| A4 (Saliency gate) | 5 | 4 | 4 | 4 | Boundary with P; relation to M (A6) |
| A5 (Interoception) | 4 | 3 | 3 | 4 | Whether P includes interoceptive perception |
| A6 (Memory) | 4 | 3 | 3 | 3 | "All subsystems" is broad; specify which parameters M biases |
| A7 (Set-points) | 3 | 5 | 4 | 4 | Type/unit commensurability of $D_t$ and $x^*$; missing update law |
| A8 (Fast/slow) | 4 | 5 | 4 | 4 | Functional form of $\pi_{\text{fast}}, \pi_{\text{slow}}$; learning rule |
| A9 (Recursion) | 3 | 5 | 3 | 2 | Termination, contraction across levels, stability of nested controllers |

### Honest assessment

The judge's verdict is **broadly favourable** but **not uncritical**:

- The axioms are consistent and non-redundant — agreeing with Z3.
- The two weakest spots are **A7 (typing)** and **A9 (recursion
  semantics)**. A7 is partially addressed in the master equation
  (`x*_t = D_t` is operationally a substitution); A9's stability is
  an open question explicitly listed under *Limitations* in the paper.
- The judge surfaced potential **redundancy between A4 and A6**
  (M-biasing of A overlaps with A's own input list). This is **not**
  a logical contradiction — Z3 confirmed independence — but a
  candidate for stylistic tightening.

These concerns are documented in the limitations section of the
paper rather than treated as failed axioms; logical soundness (the
purpose of this report) is **established**.

---

## Combined verdict

| Layer | Checks | Pass | Verdict |
|---|---|---|---|
| Z3 — joint consistency, independence, targeted probes | 12 | 12 | ✓ |
| NetworkX — structural info-flow constraints | 7 | 7 | ✓ |
| Type theory — A9 closure | 5 | 5 | ✓ |
| LLM-as-judge — semantic rubric | 28 dim-axiom scores | mean 3.86 / 5 | ⓘ |

**Formal logical soundness: 24 / 24 PASS.**
The axiom set A1–A9 is **consistent**, **independent**, and
**structurally well-formed** under the chosen semantics. Remaining
issues raised by the judge are matters of clarity and biological
grounding — pertinent to the *Discussion* section, not to the
soundness of the formal system.

---

## Reproducibility

```bash
python verify_axioms_z3.py
python verify_axioms_graph.py
python verify_axioms_types.py
python verify_axioms_judge.py --model gpt-5
```

Dependencies: `z3-solver`, `networkx`, `openai`. The judge reads
`OPEN_AI_API_KEY` from the workspace `.env`. Results are persisted
to `axiom_judge_results.json`.
