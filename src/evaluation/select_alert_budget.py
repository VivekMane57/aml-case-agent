"""
Choose an alert budget (top-K by final_score) on VAL only, then report it on TEST once.
Printed as both an absolute K and the recall/precision it buys, so the choice is explicit
and auditable rather than a silently hard-coded number.

Run:
    python src/evaluation/select_alert_budget.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROC = Path("data/processed")

# Candidate budgets to compare on VAL (edit freely; the final choice is made here, once).
CANDIDATE_K = [500, 1000, 1500, 2000, 3000, 5000]


def stats_at_k(df: pd.DataFrame, k: int) -> dict:
    top = df.nlargest(k, "final_score")
    recall = top["y"].sum() / df["y"].sum() if df["y"].sum() else float("nan")
    precision = top["y"].mean()
    return {"K": k, "alerts": k, "caught": int(top["y"].sum()),
            "precision": precision, "recall": recall}


def main() -> None:
    df = pd.read_parquet(PROC / "final_scores.parquet")
    val = df[df["split"] == "val"]
    test = df[df["split"] == "test"]

    print(f"VAL: {len(val):,} txns, {int(val['y'].sum())} positives")
    print(f"{'K':>6}{'precision':>12}{'recall':>10}{'caught':>9}")
    for k in CANDIDATE_K:
        s = stats_at_k(val, k)
        print(f"{s['K']:>6}{s['precision']:>12.4f}{s['recall']:>10.4f}{s['caught']:>9}")

    # ---- Chosen budget: edit this after looking at the table above ----
    CHOSEN_K = 2000
    print(f"\nChosen K = {CHOSEN_K} (edit CHOSEN_K in this file after reviewing the VAL table)")

    print(f"\nTEST (report only, chosen once): {len(test):,} txns, {int(test['y'].sum())} positives")
    s = stats_at_k(test, CHOSEN_K)
    print(f"  K={s['K']}  precision={s['precision']:.4f}  recall={s['recall']:.4f}  caught={s['caught']}")

    alerts = df[df["split"].isin(["val", "test"])].copy()
    alerts["is_alert"] = False
    for split_name in ("val", "test"):
        m = alerts["split"] == split_name
        thresh = alerts.loc[m, "final_score"].nlargest(CHOSEN_K).min()
        alerts.loc[m, "is_alert"] = alerts.loc[m, "final_score"] >= thresh
    alerts[alerts["is_alert"]].to_parquet(PROC / "alerts.parquet", index=False)
    print(f"\nSaved: {PROC / 'alerts.parquet'} ({int(alerts['is_alert'].sum()):,} alerted transactions)")


if __name__ == "__main__":
    main()