"""
Evaluate XGBoost vs GNN on TEST edges split by whether the accounts were seen in TRAIN.
XGBoost never uses account IDs; the GNN does (learned per-account embeddings), so a
gap on unseen accounts indicates identity memorisation rather than structural learning.

Run:
    python src/evaluation/unseen_accounts.py                              # gnn_scores.parquet
    python src/evaluation/unseen_accounts.py gnn_scores_idrop.parquet     # a tagged GNN run
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

PROC = Path("data/processed")


def r_precision(y: np.ndarray, s: np.ndarray) -> float:
    k = int(y.sum())
    return float(y[np.argsort(-s)[:k]].mean()) if k else float("nan")


def main() -> None:
    gfile = sys.argv[1] if len(sys.argv) > 1 else "gnn_scores.parquet"
    print(f"GNN scores file: {gfile}\n")

    meta = pd.read_parquet(PROC / "rule_flags.parquet")[["edge_id", "src", "dst", "split"]]
    x = pd.read_parquet(PROC / "baseline_scores.parquet")[["edge_id", "y", "score"]].rename(
        columns={"score": "xgb"})
    g = pd.read_parquet(PROC / gfile)
    gcol = "score" if "score" in g.columns else [c for c in g.columns if "score" in c][0]
    g = g[["edge_id", gcol]].rename(columns={gcol: "gnn"})

    # Build the seen-account set from the full metadata table BEFORE merging with scores:
    # gnn_scores*.parquet covers only val/test edges, so an inner merge would drop every
    # train row and make every test account look "unseen".
    tr = meta[meta["split"] == "train"]
    train_accts = set(tr["src"]) | set(tr["dst"])
    df = meta.merge(x, on="edge_id").merge(g, on="edge_id")

    test = df[df["split"] == "test"].copy()
    test["src_seen"] = test["src"].isin(train_accts)
    test["dst_seen"] = test["dst"].isin(train_accts)
    both_seen = test["src_seen"] & test["dst_seen"]
    both_unseen = ~test["src_seen"] & ~test["dst_seen"]

    subsets = {
        "all test": np.ones(len(test), bool),
        "both accounts seen in train": both_seen.to_numpy(),
        "at least one account unseen": (~both_seen).to_numpy(),
        "both accounts unseen": both_unseen.to_numpy(),
    }

    print(f"{'subset':<30}{'n':>9}{'pos':>7}   model  ROC-AUC  PR-AUC  R-prec")
    for name, m in subsets.items():
        sub = test[m]
        y = sub["y"].to_numpy()
        n, pos = len(sub), int(y.sum())
        if pos < 5 or pos == n:
            print(f"{name:<30}{n:>9,}{pos:>7}   (too few positives to evaluate)")
            continue
        for model in ("xgb", "gnn"):
            s = sub[model].to_numpy()
            print(f"{name:<30}{n:>9,}{pos:>7}   {model:<5}  "
                  f"{roc_auc_score(y, s):.4f}  {average_precision_score(y, s):.4f}  "
                  f"{r_precision(y, s):.4f}")


if __name__ == "__main__":
    main()