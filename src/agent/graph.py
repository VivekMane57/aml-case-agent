"""Builds the LangGraph state machine for one case."""
from __future__ import annotations

from langgraph.graph import END, StateGraph

from src.agent.nodes import (
    citation_check_node,
    evidence_retrieval_node,
    faithfulness_check_node,
    finalize_node,
    narrative_drafting_node,
    risk_scoring_node,
    route_after_citation_check,
    route_after_faithfulness_check,
    route_after_risk_scoring,
)
from src.agent.state import CaseState


def build_agent():
    g = StateGraph(CaseState)
    g.add_node("evidence_retrieval", evidence_retrieval_node)
    g.add_node("risk_scoring", risk_scoring_node)
    g.add_node("narrative_drafting", narrative_drafting_node)
    g.add_node("citation_check", citation_check_node)
    g.add_node("faithfulness_check", faithfulness_check_node)
    g.add_node("finalize", finalize_node)

    g.set_entry_point("evidence_retrieval")
    g.add_edge("evidence_retrieval", "risk_scoring")
    g.add_conditional_edges(
        "risk_scoring", route_after_risk_scoring,
        {"narrative_drafting": "narrative_drafting", "skip_narrative": "finalize"},
    )
    g.add_edge("narrative_drafting", "citation_check")
    g.add_conditional_edges(
        "citation_check", route_after_citation_check,
        {"retry": "narrative_drafting", "faithfulness_check": "faithfulness_check"},
    )
    g.add_conditional_edges(
        "faithfulness_check", route_after_faithfulness_check,
        {"retry": "narrative_drafting", "done": "finalize"},
    )
    g.add_edge("finalize", END)
    return g.compile()