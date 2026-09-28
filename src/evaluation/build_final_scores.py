"""
Freeze the final scoring rule and write data/processed/final_scores.parquet (val + test).

Rule (fixed a priori, not tuned):
  * XGBoost is the primary scorer.
  * If BOTH accounts of a transaction appear in the TRAIN split:
        final_score = 0.5 * pct_rank(xgb) + 0.5 * pct_rank(gnn_idrop40)
    otherwise:
        final_score = pct_rank(xgb)
Percentile ranks are computed within each split, so final_score is a within-batch ranking
score, not a calibrated probability. A production version would need a fixed reference
distribution for the ranks.

Run:
    python src/evaluation/build_final_scores.py
    python src/evaluation/build_final_scores.py gnn_scores_idrop40.parquet
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

PROC = Path("data/processed")


def main() -> None:
    gfile = sys.argv[1] if len(sys.argv) > 1 else "gnn_scores_idrop40.parquet"
    print(f"GNN scores file: {gfile}\n")

    meta = pd.read_parquet(PROC / "rule_flags.parquet")[["edge_id", "src", "dst", "split"]]
    x = pd.read_parquet(PROC / "baseline_scores.parquet")[["edge_id", "y", "score"]].rename(
        columns={"score": "xgb_score"})
    g = pd.read_parquet(PROC / gfile)
    gcol = "score" if "score" in g.columns else [c for c in g.columns if "score" in c][0]
    g = g[["edge_id", gcol]].rename(columns={gcol: "gnn_score"})

    # Seen-account set from the full metadata table (train split), before any merge.
    tr = meta[meta["split"] == "train"]
    train_accts = set(tr["src"]) | set(tr["dst"])

    parts = []
    for split_name in ("val", "test"):
        df = (meta[meta["split"] == split_name].merge(x, on="edge_id").merge(g, on="edge_id")
              .reset_index(drop=True))
        seen = (df["src"].isin(train_accts) & df["dst"].isin(train_accts)).to_numpy()
        rx = df["xgb_score"].rank(pct=True).to_numpy()
        rg = df["gnn_score"].rank(pct=True).to_numpy()
        df["both_seen"] = seen
        df["final_score"] = np.where(seen, 0.5 * rx + 0.5 * rg, rx)
        df["scoring_path"] = np.where(seen, "xgb+gnn", "xgb_only")
        parts.append(df)

        y = df["y"].to_numpy()
        print(f"{split_name.upper()}: {len(df):,} txns | both_seen={seen.mean():.1%} | "
              f"PR-AUC xgb={average_precision_score(y, rx):.4f} "
              f"final={average_precision_score(y, df['final_score']):.4f}")

    out = pd.concat(parts, ignore_index=True)[
        ["edge_id", "src", "dst", "split", "y", "xgb_score", "gnn_score",
         "both_seen", "scoring_path", "final_score"]]
    out.to_parquet(PROC / "final_scores.parquet", index=False)
    print(f"\nSaved: {PROC / 'final_scores.parquet'} ({len(out):,} rows)")


if __name__ == "__main__":
    main()