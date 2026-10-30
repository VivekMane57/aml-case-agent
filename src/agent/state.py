"""Shared state passed between LangGraph nodes. One state instance = one case."""
from __future__ import annotations

from typing import TypedDict


class CaseState(TypedDict, total=False):
    case_id: int
    case: dict                      # raw case object from cases.json (no diagnostics used downstream)
    transactions: list[dict]        # per-txn evidence (no y/typology)
    score_percentile: float
    decision: str                   # "ESCALATE" | "REQUEST_MORE_EVIDENCE"
    decision_rationale: str
    narrative: str
    cited_transaction_ids: list[str]
    citation_valid: bool
    citation_accuracy: float
    invalid_citations: list[str]
    faithfulness_score: float
    faithfulness_valid: bool
    narrative_attempts: int
    status: str                     # final outcome for the audit log