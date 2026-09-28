"""
Layer 3 (GNN) - Step 1: Build a PyG graph object from the engineered edge features.

Structural graph (data.edge_index / data.time) is used ONLY for message passing
(who talks to whom, and when). Supervision edges (edge_label_index / edge_label /
edge_label_time) are a separate table with the classification target and are what
LinkNeighborLoader actually samples around, with temporal sampling constraining
neighbors to edges that happened at or before each supervision edge's own timestamp --
this mirrors the closed='left' discipline already used for the tabular features, so a
test-split prediction can never "see" a future transaction.

Edge-feature preprocessing for the neural net (trees did not need any of this):
  1. NaN handling: *_seconds_since_prev_txn is NaN for an account's first-ever
     transaction. NaN is replaced by 0 AFTER scaling (= the train mean) and an explicit
     "no previous txn" indicator column is added, so the information is not lost.
  2. Heavy tails: amounts, sums, counts, gaps and z-scores span many orders of magnitude
     (z-scores reach 1000+). A signed log1p transform is applied before scaling so a
     handful of extreme rows cannot dominate training.
  3. payment_type_code is one-hot encoded (it is a category, not an ordered quantity).
  4. The StandardScaler is fitted on the TRAIN split only.

Input:
    data/processed/edge_features.parquet
    data/processed/rule_flags.parquet

Output:
    artifacts/graph_data.pt          -> dict with the PyG Data object + supervision tensors
    artifacts/account_to_idx.pkl     -> account id -> node index mapping

Run:
    python src/models/gnn/build_graph.py
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data

PROC = Path("data/processed")
ARTIFACTS = Path("artifacts")

RULE_COLS = ["r_structuring", "r_high_velocity", "r_amount_anomaly", "r_fan_pattern", "rule_score"]

# Heavy-tailed continuous columns -> signed log1p before scaling.
LOG_COLS = [
    "amount",
    "src_txn_count_7d", "src_amount_sum_7d", "src_txn_count_30d", "src_amount_sum_30d",
    "src_seconds_since_prev_txn", "src_new_counterparties_30d", "src_amount_zscore",
    "dst_txn_count_7d", "dst_amount_sum_7d", "dst_txn_count_30d", "dst_amount_sum_30d",
    "dst_seconds_since_prev_txn", "dst_new_counterparties_30d", "dst_amount_zscore",
]
# Already well-behaved (or binary) columns -> scaled as-is.
PLAIN_COLS = ["log_amount", "hour", "dayofweek", "same_currency", "cross_border"] + RULE_COLS
NAN_COLS = ["src_seconds_since_prev_txn", "dst_seconds_since_prev_txn"]


def signed_log1p(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x))


def build_edge_matrix(df: pd.DataFrame, train_mask: np.ndarray):
    """Returns (matrix float32, column names). Scaler is fitted on train rows only."""
    missing_flags = np.stack([df[c].isna().to_numpy() for c in NAN_COLS], axis=1).astype(np.float32)
    missing_names = [f"{c}__missing" for c in NAN_COLS]

    log_part = signed_log1p(df[LOG_COLS].to_numpy(dtype=np.float64))   # NaN stays NaN
    plain_part = df[PLAIN_COLS].to_numpy(dtype=np.float64)
    cont = np.concatenate([log_part, plain_part], axis=1)
    cont_names = [f"log1p({c})" for c in LOG_COLS] + PLAIN_COLS

    scaler = StandardScaler()          # ignores NaN when fitting
    scaler.fit(cont[train_mask])
    cont = scaler.transform(cont)
    cont = np.nan_to_num(cont, nan=0.0)  # NaN -> train mean; the __missing flag keeps the info

    onehot = pd.get_dummies(df["payment_type_code"].astype(int), prefix="pt", dtype=np.float32)
    matrix = np.concatenate([cont.astype(np.float32), missing_flags, onehot.to_numpy()], axis=1)
    names = cont_names + missing_names + list(onehot.columns)

    assert np.isfinite(matrix).all(), "edge feature matrix still contains NaN/inf"
    return matrix, names


def main() -> None:
    feats = pd.read_parquet(PROC / "edge_features.parquet")
    rules = pd.read_parquet(PROC / "rule_flags.parquet")[["edge_id", *RULE_COLS]]
    df = feats.merge(rules, on="edge_id", how="left")

    for c in ["same_currency", "cross_border"]:
        df[c] = df[c].astype(int)
    for c in RULE_COLS:
        df[c] = df[c].fillna(0)
        if c != "rule_score":
            df[c] = df[c].astype(int)

    df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)

    # ---- Node index mapping (accounts -> contiguous integer ids) ----
    accounts = pd.unique(pd.concat([df["src"], df["dst"]], ignore_index=True))
    acct_to_idx = {a: i for i, a in enumerate(accounts)}
    num_nodes = len(accounts)
    print(f"Nodes (unique accounts): {num_nodes:,}")

    src_idx = df["src"].map(acct_to_idx).to_numpy()
    dst_idx = df["dst"].map(acct_to_idx).to_numpy()

    # ---- Structural graph for message passing: bidirectional; the only payload is the
    # timestamp, which gates temporal sampling ----
    edge_index = torch.tensor(
        np.concatenate([np.stack([src_idx, dst_idx]), np.stack([dst_idx, src_idx])], axis=1),
        dtype=torch.long,
    )
    timestamps = df["timestamp"].astype("datetime64[s]").astype("int64").to_numpy()  # unix seconds
    edge_time = torch.tensor(np.concatenate([timestamps, timestamps]), dtype=torch.long)

    # ---- Supervision edges: the actual classification targets ----
    train_mask = (df["split"] == "train").to_numpy()
    edge_attr_np, feature_names = build_edge_matrix(df, train_mask)

    edge_label_index = torch.tensor(np.stack([src_idx, dst_idx]), dtype=torch.long)
    edge_label = torch.tensor(df["y"].to_numpy(), dtype=torch.float)
    edge_label_time = torch.tensor(timestamps, dtype=torch.long)
    edge_attr = torch.tensor(edge_attr_np, dtype=torch.float)

    # Node features: a learnable embedding table lives in the model itself, so `x` is None.
    data = Data(edge_index=edge_index, time=edge_time, num_nodes=num_nodes)

    payload = {
        "data": data,
        "edge_label_index": edge_label_index,
        "edge_label": edge_label,
        "edge_label_time": edge_label_time,
        "edge_attr": edge_attr,
        "split": df["split"].to_numpy(),
        "edge_id": df["edge_id"].to_numpy(),
        "num_nodes": num_nodes,
        "feature_cols": feature_names,
    }

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    torch.save(payload, ARTIFACTS / "graph_data.pt")
    with open(ARTIFACTS / "account_to_idx.pkl", "wb") as f:
        pickle.dump(acct_to_idx, f)

    print(f"Structural edges (bidirectional): {edge_index.shape[1]:,}")
    print(f"Supervision edges: {edge_label_index.shape[1]:,}")
    print(f"Edge feature dim: {edge_attr.shape[1]} (NaN-free: {bool(torch.isfinite(edge_attr).all())})")
    print(f"Positive rate overall: {edge_label.mean().item():.4%}")
    for split_name in ["train", "val", "test"]:
        m = df["split"] == split_name
        print(f"  {split_name}: {m.sum():,} edges, {df.loc[m, 'y'].sum():,} positives")
    print(f"\nSaved: {ARTIFACTS / 'graph_data.pt'}")


if __name__ == "__main__":
    main()