"""
Is the XGB + GNN(idrop40) blend genuinely better than XGB alone (paired bootstrap), and does
gating the blend to accounts seen in train avoid the cold-start penalty?

Blend weights are fixed a priori (0.5/0.5 on percentile ranks), NOT tuned here.
"Seen" always means: the account appears in the TRAIN split (same definition on val and test).

Run:
    python src/evaluation/blend_eval.py gnn_scores_idrop40.parquet          # TEST split
    python src/evaluation/blend_eval.py gnn_scores_idrop40.parquet val      # VAL split (robustness check)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

PROC = Path("data/processed")
N_BOOT = 200
SEED = 0


def main() -> None:
    gfile = sys.argv[1] if len(sys.argv) > 1 else "gnn_scores_idrop40.parquet"
    split_name = sys.argv[2] if len(sys.argv) > 2 else "test"
    if split_name not in ("val", "test"):
        raise SystemExit("split must be 'val' or 'test' (GNN scores only cover val/test)")
    print(f"GNN scores file: {gfile} | split: {split_name}\n")

    meta = pd.read_parquet(PROC / "rule_flags.parquet")[["edge_id", "src", "dst", "split"]]
    x = pd.read_parquet(PROC / "baseline_scores.parquet")[["edge_id", "y", "score"]].rename(
        columns={"score": "xgb"})
    g = pd.read_parquet(PROC / gfile)
    gcol = "score" if "score" in g.columns else [c for c in g.columns if "score" in c][0]
    g = g[["edge_id", gcol]].rename(columns={gcol: "gnn"})

    # Seen-account set comes from the full metadata table BEFORE merging with scores
    # (GNN scores cover only val/test edges, so merging first would drop all train rows).
    tr = meta[meta["split"] == "train"]
    train_accts = set(tr["src"]) | set(tr["dst"])

    df = (meta[meta["split"] == split_name].merge(x, on="edge_id").merge(g, on="edge_id")
          .reset_index(drop=True))

    seen = (df["src"].isin(train_accts) & df["dst"].isin(train_accts)).to_numpy()
    y = df["y"].to_numpy()
    rx = df["xgb"].rank(pct=True).to_numpy()
    rg = df["gnn"].rank(pct=True).to_numpy()
    blend = 0.5 * rx + 0.5 * rg
    gated = np.where(seen, blend, rx)          # unseen accounts fall back to XGBoost alone
    variants = {"xgb": rx, "blend": blend, "gated_blend": gated}

    print(f"{split_name.upper()}: {len(y):,} txns, {int(y.sum())} positives | "
          f"both-seen share: {seen.mean():.1%} | positives seen/unseen: "
          f"{int(y[seen].sum())}/{int(y[~seen].sum())}\n")

    def safe_ap(yy: np.ndarray, ss: np.ndarray) -> float:
        return average_precision_score(yy, ss) if yy.sum() > 0 else float("nan")

    print("Point estimates (PR-AUC):")
    for name, s in variants.items():
        print(f"  {name:<12} all={safe_ap(y, s):.4f}"
              f"  seen-only={safe_ap(y[seen], s[seen]):.4f}"
              f"  unseen-only={safe_ap(y[~seen], s[~seen]):.4f}")

    rng = np.random.default_rng(SEED)
    n = len(y)
    diffs = {"blend": [], "gated_blend": []}
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        if yb.sum() == 0:
            continue
        base = average_precision_score(yb, rx[idx])
        for k in diffs:
            diffs[k].append(average_precision_score(yb, variants[k][idx]) - base)

    print(f"\nPaired bootstrap ({N_BOOT} resamples, row-level): PR-AUC gain over XGB alone")
    for k, d in diffs.items():
        lo, hi = np.percentile(d, [2.5, 97.5])
        verdict = "CI excludes 0" if lo > 0 else "CI includes 0 -> not distinguishable from noise"
        print(f"  {k:<12} mean={np.mean(d):+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  ({verdict})")
    print("\nNote: row-level bootstrap ignores correlation between transactions of the same "
          "account, so the true uncertainty is wider than these intervals suggest.")


if __name__ == "__main__":
    main()