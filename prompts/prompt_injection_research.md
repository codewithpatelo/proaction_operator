# Research Prompt: Optimal Prompt-Engineering Techniques for Structured State Injection in LLM Decision-Making

## Objective
Identify empirically validated or theoretically grounded prompt-engineering techniques for injecting structured numerical internal-state information into LLM prompts such that the model (a) uses the information non-trivially, (b) is not dominated by semantic priming / lexical anchoring, and (c) retains decision authority rather than following the injected signal blindly.

## Context
We have a multi-subsystem regulatory operator (6 thermostat-like variables in [0,1]) that computes internal state for an LLM agent playing the Iterated Prisoner's Dilemma. The current injection method lists each variable with a semantic label, set-point, and a qualitative low/high interpretation, then asks the model to decide. A scrambled-labels control is used to isolate lexical priming from controller dynamics. We want to know if there are better techniques than this "parameter dump" approach.

## Questions

### 1. Format and Structure
- Does presenting structured numerical state as **JSON, XML, markdown tables, or natural language paragraphs** change how LLMs weight the information in multi-factor decisions?
- Is there evidence that **position within the prompt** (beginning vs. middle vs. end) affects utilization of injected state variables?
- What is known about **information hierarchy**: should actionable aggregated signals (e.g., a scalar recommendation) precede or follow the full decomposed state vector?

### 2. Few-Shot and In-Context Learning
- Are there studies showing that **few-shot exemplars** of state → reasoning → action improve structured-state utilization compared to zero-shot description?
- Does the format of exemplars matter: should the model's own reasoning trace reference the state variables explicitly, or is the mapping implicit?
- What is the optimal number of exemplars before diminishing returns or overfitting to the examples?

### 3. Semantic Priming and Label Effects
- Beyond our own scrambled-labels control, what prompt-engineering literature exists on **label effects** or **framing effects** when injecting numerical state with semantic descriptors (e.g., "hormonal = 0.58 (high arousal)" vs. "metric_C = 0.58")?
- Are there techniques to **decouple semantic content from numerical content**, or to compress semantic labels without losing interpretability?

### 4. Chain-of-Thought and Reasoning Protocols
- The "memory curse" finding (May 2026) shows that Chain-of-Thought can amplify cooperative collapse by causing models to enumerate past defections. What is the evidence on **structured reasoning templates** (step-by-step protocols) versus free CoT when making decisions under injected multi-factor state?
- Are there **contrastive** or **abstractive** reasoning techniques that use state without increasing cognitive load?

### 5. Delta / Change-Based Framing
- Is there evidence that LLMs (or humans, as proxy) overweight **deltas** (changes from prior state) over **levels** (absolute values)?
- Could a delta-first prompt reduce lexical priming while preserving regulatory information?
- Are there precedents in clinical decision support, control systems, or sports analytics where delta-presentation of biomarkers outperforms level-presentation?

### 6. Narrative Compression vs. Precision
- What is the tradeoff between **narrative compression** (one sentence summarizing the regulatory state) and **enumerated precision** (listing all 6 variables)?
- Does compression reduce the risk of semantic priming at the cost of losing subsystem-specific information that the model might have used?

### 7. Cognitive Load and Attention Guidance
- How much injected state can an LLM effectively use before **saturation** or **attention dilution**? Is there a "magic number" for simultaneously active variables?
- Are highlighting techniques (bolding, emojis, markdown headers) empirically shown to increase weighting of specific variables in LLM decisions?

### 8. Comparative Baselines from Adjacent Fields
- How do **clinical decision support systems** present multi-variable patient state to physicians? Is there evidence on format effectiveness?
- What techniques do **reinforcement learning from human feedback (RLHF)** and **constitutional AI** use to inject normative constraints without lexical priming?
- Are there findings from **human-computer interaction** or **cockpit design** on presenting multi-sensor state that transfer to LLM prompting?

## Output Format
For each question, provide:
1. **Key papers** (with citations and direct quotes where possible)
2. **Empirical findings** (effect sizes, conditions, models tested)
3. **Actionable recommendation** for our specific 6-subsystem regulatory state injection
4. **Risk / limitation** of applying the finding to our setting

## Boundaries
- Prioritize empirical studies over theoretical frameworks.
- Distinguish between findings from text-based LLMs and multimodal models.
- Flag studies that test specifically on decision-making games (IPD, ultimatum, trust) as highest relevance.
- Note where findings are preliminary, preprint-only, or have small sample sizes.
