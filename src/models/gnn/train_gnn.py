"""
Layer 3: Train GraphSAGE edge classifier with TEMPORAL neighbor sampling.

Temporal sampling (time_attr="time" on the structural graph, edge_label_time on the
supervision edges) means: when predicting a transaction at time t, the sampler only
walks along structural edges with time <= t. This is what actually prevents the GNN
from "seeing" future transactions of an account when scoring a past one -- the same
guarantee the tabular pipeline already had via closed='left' rolling windows.
Requires pyg-lib >= 0.4 (edge-level temporal sampling).

Training-loop notes:
  * Row alignment uses batch.input_id (position of each seed edge inside the edge set
    handed to the loader), so features/labels stay matched even with shuffle=True.
  * Each epoch uses ALL train positives + a fresh random fraction of train negatives
    (--neg-frac). Neighbor sampling is the expensive part, and 99.5% of edges are
    negatives, so this makes an epoch minutes instead of hours.
  * Per-epoch validation uses a fixed random subset of val edges (--val-frac) for early
    stopping. The final numbers are computed on the FULL val and test splits.
  * ID dropout (--id-drop): random nodes get a shared "unknown" embedding in training so
    the model cannot rely on memorised account IDs. Accounts absent from the train
    supervision edges are treated as unknown at inference.
  * --tag writes gnn_best_<tag>.pt / gnn_scores_<tag>.parquet so earlier runs are kept.

Run:
    python src/models/gnn/train_gnn.py --smoke
    python src/models/gnn/train_gnn.py --tag idrop
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.loader import LinkNeighborLoader

from model import EdgeClassifier

ARTIFACTS = Path("artifacts")
PROC = Path("data/processed")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def precision_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    if k <= 0 or k > len(scores):
        return float("nan")
    order = np.argsort(-scores)[:k]
    return float(y_true[order].mean())


def make_loader(payload, idx: np.ndarray, batch_size: int, shuffle: bool, fanout: list[int],
                num_workers: int = 0):
    """Loader over the supervision edges at positions `idx` of the full edge table.
    batch.input_id then holds each seed's position within `idx`."""
    idx_t = torch.from_numpy(idx).long()
    return LinkNeighborLoader(
        data=payload["data"],
        num_neighbors=fanout,
        edge_label_index=payload["edge_label_index"][:, idx_t],
        edge_label=payload["edge_label"][idx_t],
        edge_label_time=payload["edge_label_time"][idx_t],
        time_attr="time",
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )


def run_batch(model, batch, e_attr: torch.Tensor) -> torch.Tensor:
    return model(
        batch.n_id.to(DEVICE),
        batch.edge_index.to(DEVICE),
        batch.edge_label_index[0].to(DEVICE),
        batch.edge_label_index[1].to(DEVICE),
        e_attr.to(DEVICE),
    )


@torch.no_grad()
def predict(model, payload, idx: np.ndarray, fanout, batch_size: int, num_workers: int = 0):
    """Returns (labels, scores) aligned with `idx` order."""
    model.eval()
    idx_t = torch.from_numpy(idx).long()
    attr_sel = payload["edge_attr"][idx_t]
    scores = np.zeros(len(idx), dtype=np.float32)
    loader = make_loader(payload, idx, batch_size, False, fanout, num_workers)
    for batch in loader:
        pos = batch.input_id.cpu()
        logits = run_batch(model, batch, attr_sel[pos])
        scores[pos.numpy()] = torch.sigmoid(logits).cpu().numpy()
    labels = payload["edge_label"][idx_t].numpy()
    return labels, scores


def sample_epoch_idx(train_idx: np.ndarray, y_all: np.ndarray, neg_frac: float, rng, max_edges: int | None):
    pos = train_idx[y_all[train_idx] == 1]
    neg = train_idx[y_all[train_idx] == 0]
    n_neg = int(len(neg) * neg_frac)
    if max_edges is not None:                       # smoke-test cap
        pos = rng.choice(pos, size=min(len(pos), max_edges // 10), replace=False)
        n_neg = min(n_neg, max(max_edges - len(pos), 0))
    neg_s = rng.choice(neg, size=n_neg, replace=False)
    return np.concatenate([pos, neg_s]), len(neg_s) / max(len(pos), 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true", help="tiny run to verify the pipeline and time it")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--fanout", type=str, default="10,5", help="neighbors per hop, e.g. 10,5")
    p.add_argument("--neg-frac", type=float, default=0.10, help="fraction of train negatives per epoch")
    p.add_argument("--val-frac", type=float, default=0.20, help="fraction of val edges for per-epoch early stopping")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--id-drop", type=float, default=0.5,
                   help="prob. of replacing a node's ID embedding with the shared unknown vector in training")
    p.add_argument("--tag", type=str, default="",
                   help="suffix for output files (e.g. 'idrop') so earlier results are not overwritten")
    args = p.parse_args()

    sfx = f"_{args.tag}" if args.tag else ""
    ckpt_path = ARTIFACTS / f"gnn_best{sfx}.pt"
    scores_path = PROC / f"gnn_scores{sfx}.parquet"

    fanout = [int(x) for x in args.fanout.split(",")]
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    payload = torch.load(ARTIFACTS / "graph_data.pt", weights_only=False)
    split = payload["split"]
    y_all = payload["edge_label"].numpy()
    train_idx = np.flatnonzero(split == "train")
    val_idx_full = np.flatnonzero(split == "val")
    test_idx_full = np.flatnonzero(split == "test")

    val_sub = np.sort(rng.choice(val_idx_full, size=int(len(val_idx_full) * args.val_frac), replace=False))
    max_train_edges = None
    epochs = args.epochs
    if args.smoke:
        max_train_edges, epochs = 6000, 1
        val_sub = np.sort(rng.choice(val_idx_full, size=min(4000, len(val_idx_full)), replace=False))

    print(f"Device: {DEVICE} | fanout={fanout} | batch={args.batch_size} | neg_frac={args.neg_frac} "
          f"| id_drop={args.id_drop}")
    model = EdgeClassifier(num_nodes=payload["num_nodes"], edge_feat_dim=payload["edge_attr"].shape[1],
                           id_drop=args.id_drop).to(DEVICE)

    # "seen" = accounts appearing in TRAIN supervision edges (same definition as unseen_accounts.py)
    train_nodes = torch.unique(payload["edge_label_index"][:, torch.from_numpy(train_idx).long()])
    model.encoder.seen[train_nodes.to(DEVICE)] = True
    print(f"Seen accounts (train): {len(train_nodes):,} of {payload['num_nodes']:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    best_val_ap, bad_epochs = -1.0, 0
    for epoch in range(1, epochs + 1):
        epoch_idx, neg_pos_ratio = sample_epoch_idx(train_idx, y_all, args.neg_frac, rng, max_train_edges)
        # weight matches the *sampled* class ratio, not the raw 1:190 one
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(neg_pos_ratio, dtype=torch.float, device=DEVICE))
        attr_sel = payload["edge_attr"][torch.from_numpy(epoch_idx).long()]
        loader = make_loader(payload, epoch_idx, args.batch_size, True, fanout, args.num_workers)

        model.train()
        total_loss, n_batches, t0 = 0.0, 0, time.time()
        n_total_batches = int(np.ceil(len(epoch_idx) / args.batch_size))
        for batch in loader:
            pos = batch.input_id.cpu()
            logits = run_batch(model, batch, attr_sel[pos])
            loss = criterion(logits, batch.edge_label.to(DEVICE))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
            if n_batches % 20 == 0 or n_batches == 1:
                per = (time.time() - t0) / n_batches
                print(f"  epoch {epoch} batch {n_batches}/{n_total_batches} "
                      f"loss={total_loss / n_batches:.4f} ({per:.2f}s/batch)")
        train_secs = time.time() - t0

        y_val, s_val = predict(model, payload, val_sub, fanout, args.batch_size, args.num_workers)
        val_auc, val_ap = roc_auc_score(y_val, s_val), average_precision_score(y_val, s_val)
        print(f"Epoch {epoch:02d} | train_loss={total_loss / n_batches:.4f} | "
              f"val(subset) ROC-AUC={val_auc:.4f} PR-AUC={val_ap:.4f} | {train_secs / 60:.1f} min")

        if args.smoke:
            per_batch = train_secs / n_batches
            n_pos_train = int((y_all[train_idx] == 1).sum())
            full_batches = int(np.ceil((len(train_idx) * args.neg_frac + n_pos_train) / args.batch_size))
            print(f"\n[smoke] {per_batch:.2f}s/batch -> one FULL epoch ~ {per_batch * full_batches / 60:.1f} min "
                  f"({full_batches} batches at neg_frac={args.neg_frac})")
            print("[smoke] pipeline OK -- run without --smoke for real training")
            return

        if val_ap > best_val_ap:
            best_val_ap, bad_epochs = val_ap, 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stopping at epoch {epoch} (best val PR-AUC={best_val_ap:.4f})")
                break

    print("\n===== FINAL GNN EVALUATION (best checkpoint, FULL splits) =====")
    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    rows = []
    for name, idx in [("val", val_idx_full), ("test", test_idx_full)]:
        y_true, scores = predict(model, payload, idx, fanout, args.batch_size, args.num_workers)
        auc, ap, base = roc_auc_score(y_true, scores), average_precision_score(y_true, scores), y_true.mean()
        print(f"\n--- {name.upper()} split ({len(idx):,} txns, {int(y_true.sum())} positives, base rate {base:.4%}) ---")
        print(f"  ROC-AUC: {auc:.4f}  PR-AUC: {ap:.4f}")
        for k in (50, 100, 500, 1000):
            pk = precision_at_k(y_true, scores, k)
            if not np.isnan(pk):
                print(f"  Precision@{k}: {pk:.4f}  (lift={pk / base:.1f}x base rate)")
        rows.append(pd.DataFrame({"edge_id": payload["edge_id"][idx], "y": y_true.astype(int),
                                  "split": name, "score": scores}))

    pd.concat(rows).to_parquet(scores_path, index=False)
    print(f"\nSaved model: {ckpt_path}")
    print(f"Saved scores: {scores_path}")


if __name__ == "__main__":
    main()