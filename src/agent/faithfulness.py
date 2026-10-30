"""
RAGAS Faithfulness scoring (PRD FR-14): checks whether each claim in a narrative is
semantically supported by the evidence context, catching errors the citation checker
can't -- e.g. a correct transaction ID cited next to a MIS-STATED or over-generalized
claim (case 125's "234 compared to 234" bug, case 25's false monotonic-trend claim).
Citation check verifies existence; this verifies accuracy.

The context given to RAGAS includes the SAME pre-computed summary (build_evidence_summary)
given to the narrative-drafting prompt -- otherwise RAGAS marks a correct total/range as
"unsupported" simply because no single transaction line states the aggregate.

Uses the same Azure deployment as narrative drafting (set via .env), through ragas's
LangchainLLMWrapper so it works with AzureChatOpenAI directly -- no second provider needed.

Run standalone for a quick check:
    python -m src.agent.faithfulness
"""
from __future__ import annotations

from datasets import Dataset
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import Faithfulness

from src.agent.evidence import build_evidence_summary
from src.agent.llm import get_llm

FAITHFULNESS_THRESHOLD = 0.85  # PRD success metric (Section 7): RAGAS faithfulness >= 0.85


def _context_strings(transactions: list[dict]) -> list[str]:
    """One evidence string per transaction, plus the shared pre-computed summary -- the
    same facts given to the narrative prompt, so RAGAS checks the narrative against
    literally what the drafting model saw."""
    out = []
    for t in transactions:
        reason = t["reason"] or "flagged by model anomaly score, no named rule"
        out.append(
            f"Transaction {t['transaction_id']}: {t['timestamp']}, "
            f"{t['src_account']} -> {t['dst_account']}, amount {t['amount']:.2f}, "
            f"{t['payment_type']}, reason: {reason}"
        )
    out.append(build_evidence_summary(transactions))
    return out


def score_faithfulness(narrative: str, transactions: list[dict]) -> float:
    """Returns a 0-1 faithfulness score for one narrative against its evidence."""
    contexts = _context_strings(transactions)
    ds = Dataset.from_dict({
        "question": ["Summarize the suspicious activity in this case."],
        "answer": [narrative],
        "contexts": [contexts],
    })
    llm_wrapper = LangchainLLMWrapper(get_llm(temperature=0.0))
    result = evaluate(ds, metrics=[Faithfulness()], llm=llm_wrapper)
    return float(result["faithfulness"][0])


if __name__ == "__main__":
    import json
    from pathlib import Path

    results = json.load(open(Path("data/processed/case_results.json")))
    for r in results[:3]:
        if r.get("status") not in ("ESCALATED_WITH_VERIFIED_NARRATIVE",
                                    "ESCALATED_NARRATIVE_FAILED_FAITHFULNESS_CHECK"):
            continue
        score = score_faithfulness(r["narrative"], r["transactions"])
        flag = "OK" if score >= FAITHFULNESS_THRESHOLD else "BELOW THRESHOLD -- needs review"
        print(f"case_id={r['case_id']} faithfulness={score:.3f} [{flag}]")