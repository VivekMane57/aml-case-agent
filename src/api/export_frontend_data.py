"""Export agent results as static JSON for the React frontend.

Usage (from the repo root):
    python src/api/export_frontend_data.py
    python src/api/export_frontend_data.py --limit 60      # smaller bundle for a public demo

Writes frontend/public/data/cases.json and frontend/public/data/summary.json.

The `_diagnostic_*` fields hold ground-truth labels. They are deliberately NOT exported,
because a real analyst would not see them (same rule as src/agent/evidence.py).
"""
import argparse
import json
from collections import Counter
from pathlib import Path

RULES = {
    "Structuring": "Structuring",
    "Cross-border fan pattern": "Fan pattern",
    "Amount anomaly": "Amount anomaly",
}
VELOCITY = "Velocity spike"  # computed but excluded from the alert score

STATUS_LABEL = {
    "ESCALATED_WITH_VERIFIED_NARRATIVE": "Summary ready",
    "ROUTED_TO_REVIEW_NO_NARRATIVE": "Needs more evidence",
    "ESCALATED_NARRATIVE_FAILED_FAITHFULNESS_CHECK": "Summary unverified",
    "ESCALATED_NARRATIVE_FAILED_CITATION_CHECK": "Summary unverified",
}


def severity(percentile: float) -> str:
    # Derived from the percentile among alerted cases, because raw scores are all within 0.985-1.0.
    if percentile >= 0.9:
        return "Critical"
    if percentile >= 0.7:
        return "High"
    return "Medium"


def fired_rules(triggers):
    text = " | ".join(triggers)
    return [label for key, label in RULES.items() if key in text]


def build_case(r):
    c = r["case"]
    txs = r["transactions"]
    acct_counts = Counter()
    for t in txs:
        acct_counts[t["src_account"]] += 1
        acct_counts[t["dst_account"]] += 1
    # Most frequent account is the one "under review". Ties (single-transaction cases) fall back to the sender.
    main = txs[0]["src_account"] if txs else str(c["accounts"][0])
    if txs:
        top = acct_counts.most_common()
        best = top[0][1]
        tied = [a for a, n in top if n == best]
        main = tied[0] if len(tied) == 1 else txs[0]["src_account"]

    money_in, money_out = Counter(), Counter()
    out_txs = []
    for t in txs:
        if t["dst_account"] == main:
            direction, cp = "In", t["src_account"]
            money_in[cp] += 1
        else:
            direction, cp = "Out", t["dst_account"]
            money_out[cp] += 1
        out_txs.append({
            "id": t["transaction_id"],
            "date": t["timestamp"][:10],
            "direction": direction,
            "counterparty": cp,
            "amount": round(t["amount"], 2),
            "paymentType": t["payment_type"],
            "flaggedBy": t["reason"] or "Model score only",
        })

    triggers = c["trigger_reasons"]
    pattern = (triggers[0].split(" | ")[0]) if triggers else "Flagged by model score"
    pct = r["score_percentile"]
    return {
        "caseId": r["case_id"],
        "alertId": f"AL-{r['case_id']:04d}",
        "account": main,
        "pattern": pattern,
        "triggers": sorted({x for t in triggers for x in t.split(" | ")}),
        "rulesFired": fired_rules(triggers),
        "velocityFlagged": VELOCITY in " | ".join(triggers),
        "score": round(c["max_final_score"], 5),
        "percentile": round(pct, 4),
        "severity": severity(pct),
        "status": r["status"],
        "statusLabel": STATUS_LABEL[r["status"]],
        "decision": r["decision"],
        "decisionRationale": r["decision_rationale"],
        "narrative": r.get("narrative") or None,
        "citedIds": r.get("cited_transaction_ids") or [],
        "citationAccuracy": r.get("citation_accuracy"),
        "faithfulness": r.get("faithfulness_score"),
        "faithfulnessValid": r.get("faithfulness_valid"),
        "attempts": r.get("narrative_attempts", 0),
        "numAccounts": c["num_accounts"],
        "numTransactions": c["num_transactions"],
        "totalAmount": round(c["total_amount"], 2),
        "timeStart": c["time_start"][:10],
        "timeEnd": c["time_end"][:10],
        "moneyIn": [{"account": a, "count": n} for a, n in money_in.most_common()],
        "moneyOut": [{"account": a, "count": n} for a, n in money_out.most_common()],
        "transactions": out_txs,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/processed/case_results.json")
    ap.add_argument("--out", default="frontend/public/data")
    ap.add_argument("--limit", type=int, default=0, help="keep only the top N cases by percentile")
    args = ap.parse_args()

    results = json.loads(Path(args.src).read_text(encoding="utf-8"))
    cases = [build_case(r) for r in results]
    cases.sort(key=lambda x: (-x["percentile"], -x["score"]))

    escalated = [c for c in cases if c["decision"] == "ESCALATE"]
    verified = [c for c in escalated if c["status"] == "ESCALATED_WITH_VERIFIED_NARRATIVE"]
    summary = {
        "totalCases": len(cases),
        "escalated": len(escalated),
        "verified": len(verified),
        "failedFaithfulness": sum(c["status"] == "ESCALATED_NARRATIVE_FAILED_FAITHFULNESS_CHECK" for c in cases),
        "failedCitation": sum(c["status"] == "ESCALATED_NARRATIVE_FAILED_CITATION_CHECK" for c in cases),
        "noNarrative": sum(c["status"] == "ROUTED_TO_REVIEW_NO_NARRATIVE" for c in cases),
        "meanFaithfulnessVerified": round(sum(c["faithfulness"] or 0 for c in verified) / max(len(verified), 1), 3),
        # "Cases closed by the agent must be zero": the agent only ever outputs these decisions.
        "decisionsSeen": sorted({c["decision"] for c in cases}),
        "casesClosedByAgent": sum(c["decision"] not in ("ESCALATE", "REQUEST_MORE_EVIDENCE") for c in cases),
    }

    if args.limit:
        cases = cases[: args.limit]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "cases.json").write_text(json.dumps(cases, separators=(",", ":")), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"wrote {len(cases)} cases to {out}/cases.json")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
