"""
Phase 4 (part 1): rule-based case routing for cases built in Phase 3. This is the
deterministic policy the LangGraph Decision Node will call -- kept as a separate,
testable module so the routing logic can be unit-tested and audited independently of
the agent framework.

Design: ESCALATE / REQUEST_MORE_EVIDENCE only -- NO auto-close.

An AUTO_CLOSE tier (score in the bottom 20%, no rule reason, small case) was evaluated
and rejected: the safety check found genuine laundering cases down to the 2.7th score
percentile (min observed: 0.027), meaning NO threshold in this score range was reliably
safe to close without review. Rather than hand-tune a cutoff until the test sample
happened to show zero misses, auto-close was removed entirely, per PRD NFR-6 ("every
autonomous agent decision must be reversible... no irreversible automated action") --
closing a case is effectively irreversible in practice even if technically reversible in
the database, since a closed case is unlikely to be looked at again. Every case therefore
reaches a human reviewer; the agent's job is prioritisation (via score and rationale),
not unattended closure.

  * ESCALATE: high score (top 50% of alerted cases) OR any explainable rule reason fired.
    A rule reason is treated as strong evidence on its own, even at a mid score, because
    rules like structuring have a known high lift (10.1x in the Layer-1 report).
  * REQUEST_MORE_EVIDENCE: everything else -- mid/low score with no rule reason (an
    opaque model-only signal). Still routed to a human, never closed unattended.

Input:  data/processed/cases.json
Output: data/processed/case_decisions.json

Run:
    python src/agent/decision_rules.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

PROC = Path("data/processed")

ESCALATE_PERCENTILE = 0.50


def decide(case: dict, score_percentile: float) -> tuple[str, str]:
    """Returns (decision, rationale) -- rationale is logged for every case."""
    has_reason = len(case["trigger_reasons"]) > 0

    if score_percentile >= ESCALATE_PERCENTILE:
        return "ESCALATE", f"Score in top {(1 - ESCALATE_PERCENTILE):.0%} of alerted cases."
    if has_reason:
        return "ESCALATE", f"Explainable rule reason present: {case['trigger_reasons'][0]}"
    return "REQUEST_MORE_EVIDENCE", (
        "No rule reason, score below escalate threshold -- auto-close is disabled by "
        "design (see module docstring: genuine positives were found even in the lowest "
        "score percentiles during testing). Routed to human review instead."
    )


def main() -> None:
    cases = json.load(open(PROC / "cases.json"))
    scores = np.array([c["max_final_score"] for c in cases])
    percentiles = scores.argsort().argsort() / (len(scores) - 1)  # 0 = lowest score, 1 = highest

    decided = []
    for case, pct in zip(cases, percentiles):
        decision, rationale = decide(case, pct)
        out = dict(case)
        out["score_percentile"] = float(pct)
        out["decision"] = decision
        out["decision_rationale"] = rationale
        decided.append(out)

    with open(PROC / "case_decisions.json", "w") as f:
        json.dump(decided, f, indent=2, default=str)

    print("===== DECISION DISTRIBUTION =====")
    for d in ("ESCALATE", "REQUEST_MORE_EVIDENCE"):
        n = sum(1 for c in decided if c["decision"] == d)
        print(f"  {d}: {n}")

    print("\n===== SAFETY NOTE =====")
    print("  Auto-close is disabled by design: testing found genuine positives even in the")
    print("  bottom score percentiles (min percentile 0.027), so no threshold was found where")
    print("  closing without review was safe. All non-escalated cases route to human review.")

    print("\n===== ESCALATE cases: how much of the real recall do they carry? =====")
    esc = [c for c in decided if c["decision"] == "ESCALATE"]
    total_tp = sum(c["_diagnostic_n_true_positive"] for c in decided)
    esc_tp = sum(c["_diagnostic_n_true_positive"] for c in esc)
    if total_tp:
        print(f"  Escalated cases carry {esc_tp}/{total_tp} ({esc_tp / total_tp:.1%}) of all captured positives.")

    rme = [c for c in decided if c["decision"] == "REQUEST_MORE_EVIDENCE"]
    rme_tp = sum(c["_diagnostic_n_true_positive"] for c in rme)
    print(f"  REQUEST_MORE_EVIDENCE cases still carry {rme_tp}/{total_tp} "
          f"({rme_tp / total_tp:.1%}) of positives -- this is exactly why they are not auto-closed.")

    print(f"\nSaved: {PROC / 'case_decisions.json'}")


if __name__ == "__main__":
    main()