"""
Builds time-aware account-level features -- strictly from PAST transactions (no leakage).

Input:
    data/processed/edges.parquet   (from build_graph.py)

Output:
    data/processed/edge_features.parquet   -> edges.parquet + new engineered columns (src_*, dst_*)

Every feature is computed strictly from data BEFORE this transaction (rolling closed='left',
expanding().shift(1)), so no future information leaks across the train/val/test split.

Run:
    python src/features/build_features.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROC = Path("data/processed")
WINDOWS = {"7d": "7D", "30d": "30D"}


def make_long(edges: pd.DataFrame) -> pd.DataFrame:
    """Each edge appears twice: once as 'out' for the sender, once as 'in' for the receiver."""
    out = pd.DataFrame({
        "account": edges["src"].to_numpy(),
        "counterparty": edges["dst"].to_numpy(),
        "timestamp": edges["timestamp"].to_numpy(),
        "amount": edges["amount"].to_numpy(),
        "edge_id": edges["edge_id"].to_numpy(),
        "direction": "out",
    })
    inn = pd.DataFrame({
        "account": edges["dst"].to_numpy(),
        "counterparty": edges["src"].to_numpy(),
        "timestamp": edges["timestamp"].to_numpy(),
        "amount": edges["amount"].to_numpy(),
        "edge_id": edges["edge_id"].to_numpy(),
        "direction": "in",
    })
    long_df = pd.concat([out, inn], ignore_index=True)
    long_df = long_df.sort_values(["account", "timestamp"], kind="mergesort").reset_index(drop=True)
    return long_df


def add_rolling(long_df: pd.DataFrame) -> pd.DataFrame:
    idx = long_df.set_index("timestamp")

    for name, offset in WINDOWS.items():
        roll = idx.groupby("account", sort=False)["amount"].rolling(offset, closed="left")
        long_df[f"txn_count_{name}"] = roll.count().to_numpy()
        long_df[f"amount_sum_{name}"] = roll.sum().to_numpy()
    long_df[[f"amount_sum_{n}" for n in WINDOWS]] = long_df[[f"amount_sum_{n}" for n in WINDOWS]].fillna(0.0)
    long_df[[f"txn_count_{n}" for n in WINDOWS]] = long_df[[f"txn_count_{n}" for n in WINDOWS]].fillna(0.0)

    # All-history mean/std STRICTLY before this txn (for z-score) -> shift(1) + expanding
    g = long_df.groupby("account", sort=False)["amount"]
    shifted = g.shift(1)
    long_df["hist_amount_mean"] = shifted.groupby(long_df["account"]).expanding().mean().reset_index(level=0, drop=True)
    long_df["hist_amount_std"] = shifted.groupby(long_df["account"]).expanding().std().reset_index(level=0, drop=True)

    # Time since this account's previous transaction (seconds)
    prev_ts = long_df.groupby("account", sort=False)["timestamp"].shift(1)
    long_df["seconds_since_prev_txn"] = (long_df["timestamp"] - prev_ts).dt.total_seconds()

    # Is this a new counterparty, or a repeat? (fan-out / structuring signal)
    pair_seen_before = long_df.duplicated(subset=["account", "counterparty"], keep="first")
    long_df["is_new_counterparty"] = (~pair_seen_before).astype(float)
    roll_new = long_df.set_index("timestamp").groupby("account", sort=False)["is_new_counterparty"] \
        .rolling("30D", closed="left").sum()
    long_df["new_counterparties_30d"] = roll_new.fillna(0.0).to_numpy()

    return long_df


def main() -> None:
    edges = pd.read_parquet(PROC / "edges.parquet")
    edges["timestamp"] = pd.to_datetime(edges["timestamp"])

    print("Building long format (each edge x 2 directions)...")
    long_df = make_long(edges)
    print(f"Long rows: {len(long_df):,}")

    print("Computing rolling features (may take a couple of minutes on large data)...")
    long_df = add_rolling(long_df)

    denom = long_df["hist_amount_std"].replace(0, np.nan)
    long_df["amount_zscore"] = ((long_df["amount"] - long_df["hist_amount_mean"]) / denom).fillna(0.0)

    feat_cols = [
        "txn_count_7d", "amount_sum_7d", "txn_count_30d", "amount_sum_30d",
        "seconds_since_prev_txn", "new_counterparties_30d", "amount_zscore",
    ]

    src_feats = long_df.loc[long_df["direction"] == "out", ["edge_id", *feat_cols]]
    src_feats = src_feats.rename(columns={c: f"src_{c}" for c in feat_cols})

    dst_feats = long_df.loc[long_df["direction"] == "in", ["edge_id", *feat_cols]]
    dst_feats = dst_feats.rename(columns={c: f"dst_{c}" for c in feat_cols})

    out = edges.merge(src_feats, on="edge_id", how="left").merge(dst_feats, on="edge_id", how="left")
    out.to_parquet(PROC / "edge_features.parquet", index=False)

    print("\n===== FEATURE SUMMARY =====")
    print(f"Rows: {len(out):,} | New feature columns: {len(feat_cols) * 2}")

    print("\n--- Leakage sanity check ---")
    first_ever = long_df.sort_values("timestamp").drop_duplicates("account", keep="first")
    print("Every account's FIRST-EVER transaction -> txn_count_30d should be 0:")
    print(f"  mean={first_ever['txn_count_30d'].mean():.4f}  max={first_ever['txn_count_30d'].max():.4f}")

    print("\n--- amount_zscore by label (src side, laundering signal check) ---")
    print(out.groupby("y")["src_amount_zscore"].agg(["mean", "std", "count"]).to_string())

    print(f"\nSaved: {PROC / 'edge_features.parquet'}")


if __name__ == "__main__":
    main()