"""
Compare XGBoost vs one or more GNN score files on the TEST split, using saved scores only.
Prints: overall metrics (+ rank-average blend), equal-budget union check, and per-typology
recall within a fixed alert budget.

Run:
    python src/evaluation/compare_models.py gnn_scores_idrop40.parquet gnn_scores_noid.parquet
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

PROC = Path("data/processed")
K_LIST = (100, 500, 1000, 2000)
BUDGETS = (500, 1000, 2000)
TYPO_BUDGET = 2000


def p_at_k(y: np.ndarray, s: np.ndarray, k: int) -> float:
    return float(y[np.argsort(-s)[:k]].mean())


def report(name: str, y: np.ndarray, s: np.ndarray) -> None:
    line = f"{name:<8} ROC-AUC={roc_auc_score(y, s):.4f} PR-AUC={average_precision_score(y, s):.4f}"
    for k in K_LIST:
        line += f" P@{k}={p_at_k(y, s, k):.3f}"
    print(line)


def load_scores(fname: str, col: str, extra: list[str] | None = None) -> pd.DataFrame:
    df = pd.read_parquet(PROC / fname)
    score_col = "score" if "score" in df.columns else [c for c in df.columns if "score" in c][0]
    keep = ["edge_id", score_col] + (extra or [])
    return df[keep].rename(columns={score_col: col})


def main() -> None:
    gfiles = sys.argv[1:] or ["gnn_scores.parquet"]
    xgb = load_scores("baseline_scores.parquet", "xgb", extra=["y", "split"])
    edges = pd.read_parquet(PROC / "edges.parquet")[["edge_id", "typology"]]

    for gfile in gfiles:
        print("=" * 78)
        print(f"GNN scores file: {gfile}")
        print("=" * 78)
        gnn = load_scores(gfile, "gnn")
        df = xgb.merge(gnn, on="edge_id").merge(edges, on="edge_id", how="left")
        test = df[df["split"] == "test"].reset_index(drop=True)
        y = test["y"].to_numpy()
        sx, sg = test["xgb"].to_numpy(), test["gnn"].to_numpy()
        print(f"TEST: {len(test):,} txns, {int(y.sum())} positives\n")

        test["blend"] = 0.5 * test["xgb"].rank(pct=True) + 0.5 * test["gnn"].rank(pct=True)

        print("--- Overall ---")
        report("xgb", y, sx)
        report("gnn", y, sg)
        report("blend", y, test["blend"].to_numpy())

        print("\n--- Equal-budget check: union(top-K of each) vs XGB alone at the same alert count ---")
        for k in BUDGETS:
            top_x = np.argsort(-sx)[:k]
            top_g = np.argsort(-sg)[:k]
            union = np.union1d(top_x, top_g)
            xgb_same = np.argsort(-sx)[:len(union)]
            print(f"K={k}: union alerts={len(union)} caught={int(y[union].sum())} "
                  f"| XGB alone with {len(union)} alerts caught={int(y[xgb_same].sum())}")

        print(f"\n--- Per-typology recall inside the top-{TYPO_BUDGET} alerts (positives only) ---")
        in_x = np.zeros(len(test), bool)
        in_x[np.argsort(-sx)[:TYPO_BUDGET]] = True
        in_g = np.zeros(len(test), bool)
        in_g[np.argsort(-sg)[:TYPO_BUDGET]] = True
        pos = test[test["y"] == 1].copy()
        pos["xgb_hit"] = in_x[pos.index.to_numpy()]
        pos["gnn_hit"] = in_g[pos.index.to_numpy()]
        table = pos.groupby("typology").agg(n=("y", "size"), xgb_recall=("xgb_hit", "mean"),
                                            gnn_recall=("gnn_hit", "mean"))
        print(table.sort_values("n", ascending=False).round(3).to_string())
        print()


if __name__ == "__main__":
    main()