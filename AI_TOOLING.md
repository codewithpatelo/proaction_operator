# AI Tooling Provenance Log

This document records all AI tools used in the preparation of the Pro-Action Γ experiment and paper, per NeurIPS reproducibility requirements.

## LLMs in Core Methodology

These LLMs are the **policy-execution layer** of the agents in the experimental protocol:

| Provider | Model | Role |
|----------|-------|------|
| OpenAI | `gpt-5-nano` | Full-Γ, ablations, ReAct agent policy |
| Anthropic | `claude-haiku-4-5` | Full-Γ, ablations, ReAct agent policy |
| DeepSeek | `deepseek-v4-flash` | Full-Γ, ablations, ReAct agent policy |

**Usage Pattern**: Each model is held fixed across conditions; the experiment isolates the effect of the Γ controller, not provider-level differences. Prompts are frozen and versioned (`PROMPT_VERSION = "v1.1`).

## LLMs as Authoring Tools

These tools assisted in manuscript preparation, code development, and analysis:

| Tool | Use Case | Review Status |
|------|----------|---------------|
| GPT-5.5 | Readability improvements, text refinement | Human reviewed |
| Claude Opus 4.7 | Code assistance, refactoring suggestions, experiment monitoring | Human reviewed |
| Windsurf IDE + Cascade | Agentic code development, multi-file refactoring | Human reviewed |
| DeepSeek R4 | Mathematical derivation cross-checks | Human verified |
| Elicit | Literature search | Human verified |
| Consensus | Hypothesis validation — checking whether claims have underlying empirical evidence | Human verified |
| Perplexity | Literature research on related work and prior art | Human verified |
| Reviewer3 | Self-review of drafts | Human reviewed |

**Declaration**: All AI-generated content was reviewed and edited by human authors who assume full responsibility. No AI tool was used to generate the core scientific claims without human verification.

## Frozen Artifacts

The following are version-controlled and frozen for reproducibility:

- `exp/prompts.py`: Prompt templates with `PROMPT_VERSION`
- `exp/llm_clients.py`: Unified client with per-provider quirks
- `exp/runner.py`: Experiment orchestration
- `exp/conditions_llm.py`: Condition definitions

## Determinism Notes

- **OpenAI**: Uses native `seed` parameter + `system_fingerprint` monitoring
- **Anthropic**: Uses `temperature=0` (no native seed available)
- **DeepSeek**: Uses `temperature=0` (seed deprecated per API docs)

Full verbatim responses are logged for exact re-analysis even if regeneration differs.
