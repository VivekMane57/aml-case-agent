"""
Loads per-transaction evidence for a case, for use in the LLM prompt. Deliberately
excludes `y` (ground truth) and `typology` -- those are diagnostics only and must never
reach the narrative-drafting prompt, since a real analyst would not have them either.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

PROC = Path("data/processed")

_edges = None
_rules = None


def _load() -> None:
    global _edges, _rules
    if _edges is None:
        _edges = pd.read_parquet(PROC / "edges.parquet")[
            ["edge_id", "src", "dst", "amount", "timestamp", "payment_type"]]
        _rules = pd.read_parquet(PROC / "rule_flags.parquet")[["edge_id", "reason"]]


def get_case_transactions(case: dict) -> list[dict]:
    """Returns one evidence dict per transaction_id in the case -- amount, timestamp,
    accounts, payment type, and the rule-engine reason string. No label, no typology."""
    _load()
    txn_ids = case["transaction_ids"]
    df = _edges[_edges["edge_id"].isin(txn_ids)].merge(_rules, on="edge_id", how="left")
    df = df.sort_values("timestamp")
    return [
        {
            "transaction_id": str(row.edge_id),
            "src_account": str(row.src),
            "dst_account": str(row.dst),
            "amount": float(row.amount),
            "timestamp": str(row.timestamp),
            "payment_type": row.payment_type,
            "reason": row.reason if row.reason and row.reason != "No rule triggered" else None,
        }
        for row in df.itertuples()
    ]


def build_evidence_summary(transactions: list[dict]) -> str:
    """Pre-computed facts (range/total/count/most-recent), shared by the narrative prompt
    and the RAGAS faithfulness context -- so RAGAS can verify the same aggregate claims
    the narrative is allowed to state, instead of marking correct totals/ranges as
    unsupported just because no single transaction line states them."""
    amounts = [t["amount"] for t in transactions]
    by_time = sorted(transactions, key=lambda t: t["timestamp"])
    first, last = by_time[0], by_time[-1]
    return (
        f"Number of transactions: {len(transactions)}. "
        f"Amount range: ${min(amounts):,.2f} to ${max(amounts):,.2f}. "
        f"Total amount: ${sum(amounts):,.2f}. "
        f"Earliest transaction: [{first['transaction_id']}] at {first['timestamp']}. "
        f"Most recent (last) transaction: [{last['transaction_id']}] at {last['timestamp']}."
    )