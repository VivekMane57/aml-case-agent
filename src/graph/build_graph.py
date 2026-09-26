"""
Sample transactions se graph tables banata hai (nodes + edges) aur basic graph EDA print karta hai.

Input:
    data/processed/saml_sample.parquet
    data/processed/core_accounts.parquet

Output:
    data/processed/edges.parquet   -> ek row = ek transaction (edge), src/dst = node ids
    data/processed/nodes.parquet   -> ek row = ek account (node)

Split chronological hai (70% train / 15% val / 15% test by time), random nahi.

WARNING (leakage):
    - edges["typology"] aur nodes["group"] label se bane hain -> sirf analysis ke liye,
      kabhi model feature mat banana.

Run:
    python src/graph/build_graph.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

PROC = Path("data/processed")
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15  # baaki test


def component_sizes(src: np.ndarray, dst: np.ndarray, n_nodes: int) -> np.ndarray:
    """Weakly connected components ke sizes (sirf un nodes ke jo in edges me aaye)."""
    adj = coo_matrix((np.ones(len(src), dtype=np.int8), (src, dst)), shape=(n_nodes, n_nodes))
    _, labels = connected_components(adj, directed=True, connection="weak")
    touched = np.unique(np.concatenate([src, dst]))
    sizes = np.bincount(labels[touched])
    return np.sort(sizes[sizes > 0])[::-1]


def main() -> None:
    df = pd.read_parquet(PROC / "saml_sample.parquet")
    core = pd.read_parquet(PROC / "core_accounts.parquet")
    df = df.sort_values(["timestamp", "tx_id"]).reset_index(drop=True)
    n_edges = len(df)

    # ---- Node ids: accounts -> 0..N-1
    senders = df["Sender_account"].to_numpy()
    receivers = df["Receiver_account"].to_numpy()
    accounts, inverse = np.unique(np.concatenate([senders, receivers]), return_inverse=True)
    src, dst = inverse[:n_edges], inverse[n_edges:]
    n_nodes = len(accounts)

    nodes = pd.DataFrame({"node_id": np.arange(n_nodes), "account": accounts})
    nodes = nodes.merge(core, on="account", how="left")
    nodes["group"] = nodes["group"].astype("object").fillna("neighbor")  # label-derived: sirf EDA

    # ---- Chronological split
    t_train = df["timestamp"].iloc[int(n_edges * TRAIN_FRAC)]
    t_val = df["timestamp"].iloc[int(n_edges * (TRAIN_FRAC + VAL_FRAC))]
    split = np.where(df["timestamp"] <= t_train, "train",
                     np.where(df["timestamp"] <= t_val, "val", "test"))

    # ---- Edge table (row-local features; velocity/z-score wale features next step me)
    edges = pd.DataFrame({
        "edge_id": np.arange(n_edges),
        "tx_id": df["tx_id"].to_numpy(),
        "src": src,
        "dst": dst,
        "timestamp": df["timestamp"].to_numpy(),
        "amount": df["Amount"].to_numpy(),
        "log_amount": np.log1p(df["Amount"].to_numpy()),
        "hour": df["timestamp"].dt.hour.to_numpy(),
        "dayofweek": df["timestamp"].dt.dayofweek.to_numpy(),
        "same_currency": (df["Payment_currency"].astype(str) == df["Received_currency"].astype(str)).to_numpy(),
        "cross_border": (df["Sender_bank_location"].astype(str) != df["Receiver_bank_location"].astype(str)).to_numpy(),
        "payment_type": df["Payment_type"].astype(str).to_numpy(),
        "payment_type_code": df["Payment_type"].astype("category").cat.codes.to_numpy(),
        "y": df["Is_laundering"].astype("int8").to_numpy(),
        "typology": df["Laundering_type"].astype(str).to_numpy(),  # label leak: sirf analysis
        "split": split,
    })

    PROC.mkdir(parents=True, exist_ok=True)
    edges.to_parquet(PROC / "edges.parquet", index=False)
    nodes.to_parquet(PROC / "nodes.parquet", index=False)

    # ================= EDA =================
    print("===== GRAPH SUMMARY =====")
    print(f"Nodes: {n_nodes:,} | Edges: {n_edges:,}")
    print(f"Time range: {df['timestamp'].min()} -> {df['timestamp'].max()}")
    print(f"Split cutoffs: train <= {t_train} | val <= {t_val} | test > {t_val}")

    print("\n--- Split-wise labels (har split me positives hone chahiye) ---")
    by_split = edges.groupby("split")["y"].agg(edges="size", laundering="sum", rate="mean")
    print(by_split.loc[["train", "val", "test"]].to_string())

    print("\n--- Typology x split (sirf laundering edges) ---")
    bad = edges[edges["y"] == 1]
    ct = pd.crosstab(bad["typology"], bad["split"])
    ct = ct.reindex(columns=["train", "val", "test"], fill_value=0)
    print(ct.to_string())

    print("\n--- Degree (poore sample pe, sirf EDA) ---")
    out_deg = np.bincount(src, minlength=n_nodes)
    in_deg = np.bincount(dst, minlength=n_nodes)
    deg = out_deg + in_deg
    print("Total degree percentiles:",
          {p: int(np.percentile(deg, p)) for p in (50, 90, 99)}, "| max:", int(deg.max()))
    nodes["degree"] = deg
    print("Median degree by group:")
    print(nodes.groupby("group")["degree"].median().to_string())

    print("\n--- Connectivity ---")
    sizes = component_sizes(src, dst, n_nodes)
    print(f"Weakly connected components (poora graph): {len(sizes):,} | giant component: {sizes[0]:,} nodes")

    bad_mask = edges["y"].to_numpy() == 1
    l_sizes = component_sizes(src[bad_mask], dst[bad_mask], n_nodes)
    print(f"Sirf laundering edges ke components (natural 'cases'): {len(l_sizes):,}")
    print(f"  size median={int(np.median(l_sizes))}, p90={int(np.percentile(l_sizes, 90))}, max={int(l_sizes[0])}")

    print(f"\nSaved: {PROC / 'edges.parquet'}")
    print(f"Saved: {PROC / 'nodes.parquet'}")


if __name__ == "__main__":
    main()