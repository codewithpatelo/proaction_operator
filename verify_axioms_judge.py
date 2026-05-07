"""
LLM-as-judge semantic-coherence evaluation of Pro-Action axioms A1--A9.

The formal tools (Z3, NetworkX, type-check) verify *logical* soundness
under a chosen semantics.  This script complements them with a *semantic*
coherence rubric: an LLM judge scores each axiom and pairwise interaction
on consistency, non-redundancy, biological grounding, and falsifiability.

Reads OPEN_AI_API_KEY from the workspace .env file.
Default model: gpt-5 (override with --model).
"""

import os, json, sys, argparse
from pathlib import Path

# ── load .env ───────────────────────────────────────────────────────────
def load_env():
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
load_env()

API_KEY = os.environ.get("OPEN_AI_API_KEY") or os.environ.get("OPENAI_API_KEY")
if not API_KEY:
    print("ERROR: OPEN_AI_API_KEY not found in .env or environment.")
    sys.exit(1)

# ── axioms ──────────────────────────────────────────────────────────────
AXIOMS = {
    "A1-A3 (Kernel)":
        "Autonomy, Impulse, and Quality Gate: the agent maintains an internal "
        "deficit δ_i that drives action probability p_i(t) = σ(D(δ_i(t))), "
        "reduced by action quality g.",
    "A4 (Saliency as gate)":
        "Not every observation enters the context. An attention operator A "
        "selects õ_t = A(o_t, I_t, D_t, M_t; x_{A,t}).",
    "A5 (Interoception as parallel input)":
        "A body signal I_t feeds into E and A without passing through P.",
    "A6 (Memory as ubiquitous prior)":
        "A state M_t biases all subsystems and updates post-action.",
    "A7 (Drives as set-point generators)":
        "Set-points are not constants: x*_t = D_t.",
    "A8 (Fast/slow parallelism with arbitrator)":
        "Two routes produce action candidates c_fast, c_slow; an arbitrator "
        "assigns precision weights π_fast, π_slow.",
    "A9 (Recursive closure)":
        "Γ is self-applicable: Γ(Γ(·)) preserves type signature, enabling "
        "VSM recursion.",
}

CONTEXT = (
    "These axioms define the Pro-Action operator Γ for LLM-agents: a "
    "multi-timescale self-regulation model with six subsystems "
    "(A=attention, P=perception, H=hormonal, E=emotion, N=narrative, "
    "C=cognition), inspired by SAM (seconds) and HPA (minutes) hormonal "
    "axes, predictive coding, active inference, and Beer's Viable System "
    "Model. The axioms extend the Driveplexity kernel (A1--A3) with "
    "structural constraints A4--A9."
)

RUBRIC = """
Score each axiom on a 0-5 integer scale on these four dimensions:

1. INTERNAL_CONSISTENCY: Is the axiom internally consistent (no
   self-contradiction)?
2. NON_REDUNDANCY: Does it contribute information not already entailed
   by the other axioms?
3. BIOLOGICAL_GROUNDING: Is it consistent with established findings in
   neuroscience / endocrinology / cognitive science (cite the closest
   referent: e.g. predictive coding, salience network, allostasis,
   constructed emotion, dual-process theory, VSM)?
4. FALSIFIABILITY: Could the axiom in principle be falsified by a
   measurable outcome on an LLM-agent or biological subject?

Return ONLY valid JSON:
{
  "axiom": "...",
  "scores": {"INTERNAL_CONSISTENCY": int, "NON_REDUNDANCY": int,
             "BIOLOGICAL_GROUNDING": int, "FALSIFIABILITY": int},
  "rationale": "<<= 60 words>",
  "concerns": "<empty string if none>"
}
"""

# ── OpenAI call ─────────────────────────────────────────────────────────
try:
    from openai import OpenAI
except ImportError:
    print("openai package missing. install: pip install openai")
    sys.exit(1)

def judge(model: str):
    client = OpenAI(api_key=API_KEY)
    out = {}
    for name, statement in AXIOMS.items():
        prompt = (f"{CONTEXT}\n\nAxiom {name}:\n  {statement}\n\n"
                  f"Other axioms (for redundancy/consistency reference):\n"
                  + "\n".join(f"  - {n}: {s}" for n, s in AXIOMS.items() if n != name)
                  + f"\n\nRubric:\n{RUBRIC}")
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system",
                     "content": "You are a rigorous reviewer for a NeurIPS "
                                "submission on multi-timescale self-regulation "
                                "for LLM agents. Be skeptical and concise. "
                                "Return ONLY valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
            )
            content = resp.choices[0].message.content
            data = json.loads(content)
            # some models wrap output as {"reviews":[{...}]} or {"axioms":[{...}]}
            if "scores" not in data:
                for key in ("reviews", "axioms", "results"):
                    if key in data and isinstance(data[key], list) and data[key]:
                        data = data[key][0]
                        break
        except Exception as e:
            data = {"axiom": name, "error": str(e)}
        out[name] = data
        s = data.get("scores", {})
        if s:
            print(f"  {name}: "
                  f"IC={s.get('INTERNAL_CONSISTENCY','?')}, "
                  f"NR={s.get('NON_REDUNDANCY','?')}, "
                  f"BG={s.get('BIOLOGICAL_GROUNDING','?')}, "
                  f"FA={s.get('FALSIFIABILITY','?')}")
            if data.get("concerns"):
                print(f"      concerns: {data['concerns']}")
        else:
            print(f"  {name}: ERROR — {data.get('error', 'no scores')}")
    return out


# ── main ────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-5",
                    help="OpenAI model id (default: gpt-5)")
    ap.add_argument("--out", default="axiom_judge_results.json")
    args = ap.parse_args()

    print(f"Judging axioms with model: {args.model}\n")
    results = judge(args.model)

    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nResults written to {args.out}")

    # aggregate
    n_ok = sum(1 for r in results.values() if "scores" in r)
    print(f"\n{n_ok}/{len(results)} axioms scored successfully")
    if n_ok == len(results):
        avg = {k: 0 for k in ["INTERNAL_CONSISTENCY", "NON_REDUNDANCY",
                              "BIOLOGICAL_GROUNDING", "FALSIFIABILITY"]}
        for r in results.values():
            for k in avg:
                avg[k] += r["scores"].get(k, 0)
        for k in avg:
            avg[k] /= n_ok
        print("\nAverage scores:")
        for k, v in avg.items():
            print(f"  {k}: {v:.2f}/5")


if __name__ == "__main__":
    main()
