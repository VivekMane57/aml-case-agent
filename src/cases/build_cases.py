"""
Phase 3: cluster alerted transactions (TEST split, from alerts.parquet) into investigable
"cases" using connected components over the alert subgraph only -- not the full 1.87M-edge
graph, which would merge everything into one giant component via high-degree accounts.

Case fields are built from data already computed upstream (edges.parquet, rule_flags.parquet,
final_scores.parquet); `typology` and `y` are kept ONLY for diagnostics/quality reporting,
never exposed to the downstream agent as a feature.

Output:
    data/processed/cases.json         -> list of case objects (for the LangGraph agent)
    data/processed/case_quality.json  -> aggregate quality metrics (for the report)

Run:
    python src/cases/build_cases.py
"""
from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import pandas as pd

PROC = Path("data/processed")
MIN_CASE_ALERTS = 1   # keep singleton cases too -- a single large suspicious txn is still a case


def main() -> None:
    alerts = pd.read_parquet(PROC / "alerts.parquet")
    alerts = alerts[alerts["split"] == "test"].copy()

    edges = pd.read_parquet(PROC / "edges.parquet")[
        ["edge_id", "amount", "timestamp", "typology", "payment_type"]]
    rules = pd.read_parquet(PROC / "rule_flags.parquet")[["edge_id", "reason"]]

    df = alerts.merge(edges, on="edge_id").merge(rules, on="edge_id")
    print(f"Alerted test transactions: {len(df):,}")

    # ---- Build the alert-only graph and find connected components ----
    G = nx.Graph()
    for _, row in df.iterrows():
        G.add_edge(row["src"], row["dst"], edge_id=row["edge_id"])

    components = list(nx.connected_components(G))
    print(f"Connected components (cases): {len(components):,}")
    sizes = sorted((len(c) for c in components), reverse=True)
    print(f"Case sizes: largest={sizes[0]}, median={sizes[len(sizes)//2]}, "
          f"singletons={sum(1 for s in sizes if s == 1)}")

    acct_to_case = {acct: i for i, comp in enumerate(components) for acct in comp}
    df["case_id"] = df["src"].map(acct_to_case)

    # ---- Build one case object per component ----
    cases = []
    for case_id, group in df.groupby("case_id"):
        accounts = sorted(set(group["src"]) | set(group["dst"]))
        reasons = sorted(set(r for r in group["reason"] if r and r != "No rule triggered"))
        typologies = group["typology"].value_counts().to_dict()  # diagnostic only
        cases.append({
            "case_id": int(case_id),
            "accounts": accounts,
            "num_accounts": len(accounts),
            "num_transactions": len(group),
            "total_amount": float(group["amount"].sum()),
            "time_start": str(group["timestamp"].min()),
            "time_end": str(group["timestamp"].max()),
            "max_final_score": float(group["final_score"].max()),
            "trigger_reasons": reasons,
            "transaction_ids": group["edge_id"].tolist(),
            # Diagnostics only -- NOT passed to the agent, used only for this quality report:
            "_diagnostic_purity": float(group["y"].mean()),
            "_diagnostic_n_true_positive": int(group["y"].sum()),
            "_diagnostic_typology_breakdown": typologies,
        })

    cases.sort(key=lambda c: c["max_final_score"], reverse=True)
    with open(PROC / "cases.json", "w") as f:
        json.dump(cases, f, indent=2, default=str)
    print(f"\nSaved: {PROC / 'cases.json'} ({len(cases):,} cases)")

    # ---- Quality report ----
    n_cases = len(cases)
    pure_cases = sum(1 for c in cases if c["_diagnostic_purity"] == 1.0)
    mixed_cases = sum(1 for c in cases if 0 < c["_diagnostic_purity"] < 1.0)
    clean_false_cases = sum(1 for c in cases if c["_diagnostic_purity"] == 0.0)
    total_positives_test = int(pd.read_parquet(PROC / "final_scores.parquet")
                                .pipe(lambda d: d[d["split"] == "test"])["y"].sum())
    positives_captured = sum(c["_diagnostic_n_true_positive"] for c in cases)

    quality = {
        "n_cases": n_cases,
        "pure_laundering_cases": pure_cases,
        "mixed_cases": mixed_cases,
        "false_alarm_only_cases": clean_false_cases,
        "total_test_positives": total_positives_test,
        "positives_captured_in_any_case": positives_captured,
        "positive_capture_rate": positives_captured / total_positives_test if total_positives_test else 0,
        "median_case_size_accounts": sizes[len(sizes) // 2],
        "largest_case_accounts": sizes[0],
        "singleton_cases": sum(1 for s in sizes if s == 1),
    }
    with open(PROC / "case_quality.json", "w") as f:
        json.dump(quality, f, indent=2)

    print("\n===== CASE QUALITY =====")
    for k, v in quality.items():
        print(f"  {k}: {v}")

    print("\n--- Top 5 cases by max_final_score ---")
    for c in cases[:5]:
        print(f"  case_id={c['case_id']} accounts={c['num_accounts']} txns={c['num_transactions']} "
              f"amount={c['total_amount']:.0f} purity={c['_diagnostic_purity']:.2f} "
              f"reasons={c['trigger_reasons'][:2]}")


if __name__ == "__main__":
    main()