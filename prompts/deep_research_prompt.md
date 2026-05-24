# Deep Research Prompt: Neurobiological Regulation & LLM Behavior in Iterated Prisoner's Dilemma

## Context

We are writing a scientific paper that operationalizes a **multi-scale regulatory architecture** (the Pro-Action Operator, Γ) for LLM agents in the Iterated Prisoner's Dilemma (IPD). The architecture models the agent as a regulatory system with coupled subsystems (fast SAM-like, slow HPA-like, emotional/reappraisal, cognitive control, memory, and social coupling) rather than as a pure predictor or reward maximizer.

Our goal is to ground the paper's claims in empirical evidence from **two distinct literatures**:
1. **Human neurobiology & cognition** during IPD (or comparable social decision-making under uncertainty and stress)
2. **LLM/agent behavior** in IPD and related game-theoretic settings

---

## Research Questions

### I. Human Neurobiological Subsystems in Iterated Social Dilemmas

Investigate the empirical literature on how the following systems interact during IPD or comparable repeated social games (trust game, ultimatum game, chicken) in humans, with emphasis on **temporal dynamics, causal direction, and bidirectional coupling**:

1. **Amygdala (amígdala)**
   - Role in threat detection, initial appraisal of opponent defection, and bias toward retaliation vs. forgiveness.
   - Does amygdala reactivity predict first-defection responses or long-term cooperative collapse?
   - Evidence of habituation across repeated rounds?

2. **SAM axis (Sympathetic-Adrenomedullary)**
   - Temporal profile: how quickly does sympathetic arousal rise after an opponent's defection?
   - Does SAM activation predict immediate defection (reactive retaliation) or increased risk aversion?
   - Relationship to pupil dilation, heart rate variability (HRV), or skin conductance during IPD.

3. **HPA axis (Hypothalamic-Pituitary-Adrenal)**
   - Slower cortisol dynamics: does cumulative defection history elevate baseline cortisol and shift strategy from cooperation to withdrawal/defection?
   - Evidence for allostatic load (cumulative cost of repeated social stress) in long IPD sessions.
   - Interaction with SAM: does HPA modulate the amplitude or duration of sympathetic responses?

4. **Prefrontal cortex (corteza prefrontal, especially dlPFC, vmPFC, ACC)**
   - Regulatory / top-down control: does PFC activation predict inhibition of retaliation, forgiveness, or strategic switching?
   - Conflict between amygdala-driven retaliation and PFC-driven reappraisal: what experimental evidence exists for this "push-pull" dynamic?
   - Does PFC-mediated regulation show fatigue or depletion across long IPD sessions?

5. **Attention & Perception**
   - What do players attend to in IPD? Opponent's last move, cumulative history, or pattern detection?
   - Evidence for attentional bias toward threat (defection signals) and its modulation by stress or trust.
   - Eye-tracking or EEG studies showing early perceptual discrimination of cooperation vs. defection cues.

6. **Cross-system coupling**
   - Are there studies measuring **multiple systems simultaneously** (e.g., fMRI + HRV + cortisol) in social dilemmas?
   - What is the empirical evidence for **temporal sequencing**: does perception → amygdala → SAM → HPA → PFC regulation → action hold as a causal chain, or are there parallel pathways and feedback loops?
   - Any evidence that **slow allostatic variables** (e.g., cortisol baseline) modulate **fast reactive variables** (e.g., amygdala reactivity to a single defection)?

**Critical framing for the paper**: We are interested in whether human data supports a model where:
- Fast responses are driven by threat-detection + sympathetic arousal
- Slow responses are modulated by cumulative hormonal load + cognitive reappraisal
- The interaction between fast and slow variables predicts strategy transitions (cooperation → defection → recovery)

---

### II. LLM Behavior in Iterated Prisoner's Dilemma

Investigate the empirical literature on LLM agents in IPD and related settings:

1. **Cooperation levels**
   - Do LLMs cooperate more, less, or similarly to humans in IPD?
   - Does cooperation decay over 100 rounds (as in humans) or remain stable?
   - Evidence for conditional cooperation (TFT-like) vs. unconditional cooperation.

2. **Reactivity vs. proactivity**
   - Do LLMs initiate defection, or only retaliate after opponent defection?
   - Is there evidence of "forgiveness" (returning to cooperation after opponent cooperates)?
   - How do LLMs behave against fixed strategies (AllC, AllD, TFT, Grim)?

3. **Strategy identification**
   - Are there studies identifying emergent strategies (TFT, Pavlov, Win-Stay-Lose-Shift, etc.) from LLM action sequences?
   - Do LLMs adapt strategy based on opponent's behavioral history?

4. **Internal state / memory**
   - Do LLMs with explicit state tracking (memory, internal variables) behave differently from prompt-only LLMs?
   - Any evidence that LLMs "hallucinate" opponent strategies or overfit to recent history?

5. **Noise and perturbation**
   - How do LLMs respond to noisy observations (e.g., 10% action flips)?
   - Any studies on stress injection, perturbation, or adversarial framing in IPD with LLMs?

6. **Multi-agent dynamics**
   - Evidence of convergence, oscillation, or collapse in populations of LLM agents.

**Critical framing for the paper**: We are interested in whether LLMs behave as "reactive pattern-matchers" or whether they can sustain coherent strategies under perturbation, noise, and long horizons — and whether adding an internal regulatory state improves behavioral stability.

---

### III. Cross-cutting themes (optional but highly valued)

If the literature permits, also explore:

- **Emotion regulation strategies** (reappraisal vs. suppression) in social dilemmas: do they map onto cooperation maintenance?
- **Interoception** (awareness of internal state) and its role in decision-making under social stress.
- **Computational psychiatry / active inference** models of IPD: do they predict the same multi-system dynamics?
- **Comparative studies** (human vs. algorithm vs. LLM) in the same IPD setup.

---

## Output format

Please structure the response as follows:

1. **Executive Summary** (2-3 paragraphs): the most important, actionable findings for our paper.
2. **Human Neurobiology in IPD**: organized by subsystem, with key papers cited (author, year, key finding, relevance to our model).
3. **LLMs in IPD**: organized by theme, with key papers cited.
4. **Cross-cutting findings**: any integrative or comparative work.
5. **Identified gaps**: what does the literature *not* answer? (These become our opportunity/contribution.)
6. **Recommended citations**: a prioritized list of the 15-20 most relevant papers we should read and cite.

**Tone requirement**: Be precise, cite actual studies, distinguish between established findings and speculative interpretations, and flag methodological limitations. Avoid overstating causality where only correlation exists.
