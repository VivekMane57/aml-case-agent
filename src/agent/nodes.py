"""
LangGraph nodes for the AML case investigation agent.

Pipeline: evidence_retrieval -> risk_scoring -> (ESCALATE?) -> narrative_drafting
          -> citation_check -> (invalid? retry) -> faithfulness_check -> (invalid? retry)
          -> finalize

All routing logic is the SAME decide() function already unit-tested in decision_rules.py
-- the agent framework wraps it, it does not reimplement it. Two independent verification
gates run before a narrative is accepted: citation_check (deterministic -- every cited
transaction ID must exist in the evidence) and faithfulness_check (RAGAS -- the narrative's
claims must be semantically supported by the evidence, catching errors like a false trend
claim that cites only valid IDs and so passes citation_check).
"""
from __future__ import annotations

import re
from pathlib import Path

from src.agent.decision_rules import decide
from src.agent.evidence import build_evidence_summary, get_case_transactions
from src.agent.faithfulness import FAITHFULNESS_THRESHOLD, score_faithfulness
from src.agent.llm import get_llm
from src.agent.state import CaseState

PROC = Path("data/processed")
MAX_NARRATIVE_ATTEMPTS = 2

NARRATIVE_SYSTEM_PROMPT = """You are drafting an internal case summary for a bank's AML \
compliance analyst (an initial SAR-style draft, not a final regulatory filing). You will \
be given a list of transactions that were flagged as suspicious, with amounts, timestamps, \
accounts, and any rule-based reason that was triggered.

Rules you MUST follow:
1. Every factual claim (an amount, a count, a pattern) must be traceable to a specific \
transaction in the evidence. When you state a fact, cite the transaction_id(s) it came \
from in square brackets, using the EXACT transaction_id string as it appears in the \
evidence below (e.g. if the evidence shows "[1624680]", cite it as [1624680] -- do NOT \
add any prefix such as "TXN_" that is not already part of the ID). Cite exactly one \
transaction ID per bracket. For multiple transactions use multiple brackets, e.g. \
"[1624680][1629837]" -- do NOT put several IDs in one bracket separated by commas.
2. Do NOT invent any transaction, amount, account, or date not present in the evidence.
3. Do NOT speculate about intent, guilt, or legal conclusions -- describe the observed \
pattern only (e.g. "multiple sub-threshold cash deposits" not "the customer is laundering money").
4. Keep it to 3-5 sentences: what pattern was observed, which accounts/transactions, and \
why it was flagged (reference the rule reasons given, if any).
5. If no rule reason was given for a transaction, say the flag came from the model's \
anomaly score, not a named rule.
6. When a rule reason string contains specific numbers (e.g. "234 txns in 7d vs 59 in \
trailing 30d"), copy BOTH numbers EXACTLY as given, character for character. Never assume \
the two numbers in a comparison are equal, never recompute a number, and never state a \
number that does not appear verbatim in the evidence -- if you are not certain a number is \
correct, omit that detail rather than guess.
7. A "PRE-COMPUTED SUMMARY" block is provided below the transaction list with the exact \
amount range, total, count, earliest transaction, and most recent transaction already \
identified. If you state a range, a total, a count, or refer to "the earliest" or "the \
most recent" transaction, COPY those exact figures/IDs from that block verbatim -- do NOT \
determine these yourself by scanning the list, you will get it wrong.
8. Do NOT describe a sequence as "increasing" or "decreasing" over time unless the values, \
in chronological order, actually move in that direction for every step with no reversals. \
If values go up and down, or reset and repeat, describe it as "fluctuating between X and Y" \
or describe the distinct episodes separately -- do not smooth it into a single trend.
"""


def evidence_retrieval_node(state: CaseState) -> CaseState:
    case = state["case"]
    transactions = get_case_transactions(case)
    return {**state, "transactions": transactions}


def risk_scoring_node(state: CaseState) -> CaseState:
    case = state["case"]
    decision, rationale = decide(case, case["score_percentile"])
    return {
        **state,
        "score_percentile": case["score_percentile"],
        "decision": decision,
        "decision_rationale": rationale,
    }


def route_after_risk_scoring(state: CaseState) -> str:
    return "narrative_drafting" if state["decision"] == "ESCALATE" else "skip_narrative"


def _format_evidence_for_prompt(transactions: list[dict]) -> str:
    lines = []
    for t in transactions:
        reason = t["reason"] or "no rule reason (flagged by model score only)"
        lines.append(
            f"- [{t['transaction_id']}] {t['timestamp']} | {t['src_account']} -> "
            f"{t['dst_account']} | amount={t['amount']:.2f} | {t['payment_type']} | "
            f"reason: {reason}"
        )
    # Pre-computed facts, so the LLM copies instead of (mis)calculating. LLMs reliably
    # get range/total/count wrong when asked to scan a list themselves, even when told
    # explicitly to scan carefully -- offloading the arithmetic removes the failure mode
    # rather than trying to prompt it away. Same summary is also given to RAGAS
    # (faithfulness.py) so correct aggregate claims aren't marked "unsupported".
    summary = build_evidence_summary(transactions)
    return (
        "\n".join(lines)
        + f"\n\nPRE-COMPUTED SUMMARY (copy these figures verbatim if stating a range, "
          f"total, or count): {summary}"
    )


def narrative_drafting_node(state: CaseState) -> CaseState:
    llm = get_llm(temperature=0.0)
    evidence_text = _format_evidence_for_prompt(state["transactions"])
    correction = ""
    if state.get("invalid_citations"):
        correction = (
            f"\n\nYour previous draft cited transaction ID(s) not present in the evidence: "
            f"{state['invalid_citations']}. Rewrite the summary using ONLY the transaction "
            f"IDs listed in the evidence above."
        )
    elif state.get("narrative_attempts", 0) > 0 and not state.get("faithfulness_valid", True):
        correction = (
            f"\n\nYour previous draft contained a claim not fully supported by the evidence "
            f"(e.g. an unverified trend, or a number not copied verbatim from the evidence). "
            f"Re-read every rule above carefully, especially rules 6, 7 and 8, and rewrite "
            f"the summary using ONLY facts directly stated in the evidence and the "
            f"pre-computed summary."
        )
    user_prompt = (
        f"Case involves {state['case']['num_accounts']} account(s) and "
        f"{len(state['transactions'])} flagged transaction(s).\n\nEvidence:\n{evidence_text}"
        f"{correction}"
    )
    response = llm.invoke([
        ("system", NARRATIVE_SYSTEM_PROMPT),
        ("user", user_prompt),
    ])
    attempts = state.get("narrative_attempts", 0) + 1
    return {**state, "narrative": response.content, "narrative_attempts": attempts}


def citation_check_node(state: CaseState) -> CaseState:
    """Hallucination guard, layer 1: every [TXN_ID]-style citation in the narrative must
    exist in the evidence set handed to the model. Deterministic, not an LLM call, so it
    cannot itself hallucinate a pass."""
    valid_ids = {t["transaction_id"] for t in state["transactions"]}
    raw_brackets = re.findall(r"\[([^\[\]]+)\]", state["narrative"])
    cited = {tid.strip() for group in raw_brackets for tid in group.split(",")}
    invalid = sorted(cited - valid_ids)
    cited_valid = cited & valid_ids

    accuracy = len(cited_valid) / len(cited) if cited else 0.0
    return {
        **state,
        "cited_transaction_ids": sorted(cited),
        "citation_valid": len(invalid) == 0 and len(cited) > 0,
        "citation_accuracy": accuracy,
        "invalid_citations": invalid,
    }


def route_after_citation_check(state: CaseState) -> str:
    if state["citation_valid"]:
        return "faithfulness_check"
    if state.get("narrative_attempts", 0) < MAX_NARRATIVE_ATTEMPTS:
        return "retry"
    return "faithfulness_check"  # give up on citations after MAX attempts; still record faithfulness


def faithfulness_check_node(state: CaseState) -> CaseState:
    """Hallucination guard, layer 2 (RAGAS, PRD FR-14): checks whether the narrative's
    claims are actually supported by the evidence -- catches errors citation_check can't,
    like a false trend claim built only from validly-cited transaction IDs."""
    score = score_faithfulness(state["narrative"], state["transactions"])
    return {**state, "faithfulness_score": score, "faithfulness_valid": score >= FAITHFULNESS_THRESHOLD}


def route_after_faithfulness_check(state: CaseState) -> str:
    if not state["citation_valid"]:
        return "done"  # citation already failed after max attempts -- don't retry again here
    if state["faithfulness_valid"]:
        return "done"
    if state.get("narrative_attempts", 0) < MAX_NARRATIVE_ATTEMPTS:
        return "retry"
    return "done"


def finalize_node(state: CaseState) -> CaseState:
    if state["decision"] != "ESCALATE":
        status = "ROUTED_TO_REVIEW_NO_NARRATIVE"
    elif not state["citation_valid"]:
        status = "ESCALATED_NARRATIVE_FAILED_CITATION_CHECK"
    elif not state.get("faithfulness_valid", False):
        status = "ESCALATED_NARRATIVE_FAILED_FAITHFULNESS_CHECK"
    else:
        status = "ESCALATED_WITH_VERIFIED_NARRATIVE"
    return {**state, "status": status}