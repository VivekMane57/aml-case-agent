"""
Layer 2 baseline: XGBoost tabular classifier on engineered features + rule-engine flags.

Input:
    data/processed/edge_features.parquet
    data/processed/rule_flags.parquet

Output:
    artifacts/baseline_xgb.json              -> trained model
    data/processed/baseline_scores.parquet   -> edge_id, y, split, score (predicted probability)

Uses the same chronological train/val/test split as everything upstream: trains on train,
picks nothing on test (only reports it), uses val for early stopping.

Run:
    python src/models/tabular/train_baseline.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score

PROC = Path("data/processed")
ARTIFACTS = Path("artifacts")

# Rule-engine outputs are included as features here (not as fixed heuristic thresholds) --
# a learned model can combine even a weak/noisy rule (e.g. r_high_velocity) with other
# signals non-linearly, so it is not excluded the way it was from the standalone rule score.
RULE_COLS = ["r_structuring", "r_high_velocity", "r_amount_anomaly", "r_fan_pattern", "rule_score"]

FEATURE_COLS = [
    "amount", "log_amount", "hour", "dayofweek",
    "same_currency", "cross_border", "payment_type_code",
    "src_txn_count_7d", "src_amount_sum_7d", "src_txn_count_30d", "src_amount_sum_30d",
    "src_seconds_since_prev_txn", "src_new_counterparties_30d", "src_amount_zscore",
    "dst_txn_count_7d", "dst_amount_sum_7d", "dst_txn_count_30d", "dst_amount_sum_30d",
    "dst_seconds_since_prev_txn", "dst_new_counterparties_30d", "dst_amount_zscore",
] + RULE_COLS


def precision_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    if k <= 0 or k > len(scores):
        return float("nan")
    order = np.argsort(-scores)[:k]
    return float(y_true[order].mean())


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

    X = df[FEATURE_COLS]
    y = df["y"].to_numpy()
    split = df["split"].to_numpy()

    train_mask = split == "train"
    val_mask = split == "val"
    test_mask = split == "test"

    n_pos = max((y[train_mask] == 1).sum(), 1)
    n_neg = (y[train_mask] == 0).sum()
    scale_pos_weight = n_neg / n_pos

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        tree_method="hist",
        early_stopping_rounds=30,
        n_jobs=-1,
    )

    print("Training XGBoost baseline...")
    model.fit(
        X[train_mask], y[train_mask],
        eval_set=[(X[val_mask], y[val_mask])],
        verbose=False,
    )
    print(f"Best iteration: {model.best_iteration}")

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    model.save_model(ARTIFACTS / "baseline_xgb.json")

    scores = model.predict_proba(X)[:, 1]
    out = df[["edge_id", "y", "split"]].copy()
    out["score"] = scores
    out.to_parquet(PROC / "baseline_scores.parquet", index=False)

    print("\n===== BASELINE MODEL EVALUATION =====")
    for name, mask in [("val", val_mask), ("test", test_mask)]:
        yt, st = y[mask], scores[mask]
        auc = roc_auc_score(yt, st)
        ap = average_precision_score(yt, st)
        base_rate = yt.mean()
        print(f"\n--- {name.upper()} split ({mask.sum():,} txns, {int(yt.sum())} positives, "
              f"base rate {base_rate:.4%}) ---")
        print(f"  ROC-AUC: {auc:.4f}  PR-AUC (avg precision): {ap:.4f}")
        for k in (50, 100, 500, 1000):
            p = precision_at_k(yt, st, k)
            if not np.isnan(p):
                print(f"  Precision@{k}: {p:.4f}  (lift={p / base_rate:.1f}x base rate)")

    print("\n--- Feature importance (top 15, gain-based) ---")
    importance = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    print(importance.head(15).to_string())

    print(f"\nSaved model: {ARTIFACTS / 'baseline_xgb.json'}")
    print(f"Saved scores: {PROC / 'baseline_scores.parquet'}")


if __name__ == "__main__":
    main()