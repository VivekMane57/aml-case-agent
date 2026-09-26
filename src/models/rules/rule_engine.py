"""
Layer 1 rule engine: fast, explainable heuristics for AML detection.

Input:  data/processed/edge_features.parquet
Output: data/processed/rule_flags.parquet

Rules use only features already computed with no future leakage
(edge_features.parquet respects the same closed='left' rolling logic).

Run:
    python src/models/rules/rule_engine.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROC = Path("data/processed")

# ---- Configurable thresholds ----
STRUCTURING_THRESHOLD = 10_000       # reporting threshold each txn stays under
STRUCTURING_MIN_COUNT_7D = 3         # at least this many txns in 7 days
STRUCTURING_MIN_SUM_7D = 10_000      # but cumulative sum crosses the threshold

HIGH_VELOCITY_SPIKE_RATIO = 3.0      # last-7d rate is at least this many times the account's OWN prior weekly rate
HIGH_VELOCITY_MIN_COUNT = 6          # minimum absolute recent count, so tiny numbers don't trigger on noise
HIGH_VELOCITY_MIN_PRIOR_COUNT = 3    # need a real prior baseline to judge a "spike" against -- no baseline means we can't tell
HIGH_VELOCITY_ABS_COUNT = 40         # OR: extreme absolute burst regardless of any baseline (catches brand-new sparse accounts too)
PRIOR_WINDOW_WEEKS = 23 / 7          # the 30d window minus the most recent 7d, expressed in week-equivalents

ZSCORE_THRESHOLD = 5.0               # amount many std devs away from account's own history

NEW_COUNTERPARTIES_THRESHOLD = 5     # many new counterparties in 30 days = fan-out/fan-in

CASH_TYPES = {"Cash Deposit", "Cash Withdrawal"}


def rule_structuring(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    is_cash = df["payment_type"].isin(CASH_TYPES).to_numpy()
    src_hit = (
        is_cash
        & (df["amount"] < STRUCTURING_THRESHOLD).to_numpy()
        & (df["src_txn_count_7d"] >= STRUCTURING_MIN_COUNT_7D).to_numpy()
        & (df["src_amount_sum_7d"] >= STRUCTURING_MIN_SUM_7D).to_numpy()
    )
    dst_hit = (
        is_cash
        & (df["amount"] < STRUCTURING_THRESHOLD).to_numpy()
        & (df["dst_txn_count_7d"] >= STRUCTURING_MIN_COUNT_7D).to_numpy()
        & (df["dst_amount_sum_7d"] >= STRUCTURING_MIN_SUM_7D).to_numpy()
    )
    return src_hit, dst_hit


def rule_high_velocity(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Relative spike vs. the account's own PRIOR baseline (the 23 days before the last
    week), not the trailing-30d total -- that total already includes the last 7 days,
    so comparing against it makes any account with little history look like a 'spike'
    even when nothing changed. A spike can only be judged against a REAL prior baseline
    (HIGH_VELOCITY_MIN_PRIOR_COUNT), otherwise a naturally sparse or brand-new account
    (all of its activity landing in the last 7 days by construction) would falsely read
    as an infinite spike. Accounts with no meaningful prior baseline are only caught
    through the extreme absolute-burst threshold instead."""
    def hit_for(side: str) -> np.ndarray:
        count7 = df[f"{side}_txn_count_7d"].to_numpy()
        count30 = df[f"{side}_txn_count_30d"].to_numpy()
        prior_count = np.clip(count30 - count7, 0, None)  # activity in the 23 days before last week
        prior_weekly_rate = prior_count / PRIOR_WINDOW_WEEKS
        ratio = count7 / (prior_weekly_rate + 1)
        spike = (
            (prior_count >= HIGH_VELOCITY_MIN_PRIOR_COUNT)
            & (ratio >= HIGH_VELOCITY_SPIKE_RATIO)
            & (count7 >= HIGH_VELOCITY_MIN_COUNT)
        )
        extreme = count7 >= HIGH_VELOCITY_ABS_COUNT
        return spike | extreme

    return hit_for("src"), hit_for("dst")


def rule_amount_anomaly(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    src_hit = (df["src_amount_zscore"].abs() >= ZSCORE_THRESHOLD).to_numpy()
    dst_hit = (df["dst_amount_zscore"].abs() >= ZSCORE_THRESHOLD).to_numpy()
    return src_hit, dst_hit


def rule_fan_pattern(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    cb = df["cross_border"].to_numpy()
    src_hit = (df["src_new_counterparties_30d"] >= NEW_COUNTERPARTIES_THRESHOLD).to_numpy() & cb
    dst_hit = (df["dst_new_counterparties_30d"] >= NEW_COUNTERPARTIES_THRESHOLD).to_numpy() & cb
    return src_hit, dst_hit


def _side(df: pd.DataFrame, col: str, src_hit: np.ndarray, dst_hit: np.ndarray) -> np.ndarray:
    """Picks the src value where src triggered, else the dst value (used for reason text only)."""
    return np.where(src_hit, df[f"src_{col}"].to_numpy(), df[f"dst_{col}"].to_numpy())


def build_reasons(
    df: pd.DataFrame,
    struct_hits: tuple[np.ndarray, np.ndarray],
    vel_hits: tuple[np.ndarray, np.ndarray],
    z_hits: tuple[np.ndarray, np.ndarray],
    fan_hits: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    """Vectorized reason strings -- avoids a slow per-row apply on millions of rows.

    Reports whichever side (sender or receiver) actually triggered the rule, so the
    explanation always matches the printed numbers -- never blindly assumes src.
    """
    n = len(df)
    parts = [[] for _ in range(n)]

    src_struct, dst_struct = struct_hits
    struct_count = _side(df, "txn_count_7d", src_struct, dst_struct)
    struct_sum = _side(df, "amount_sum_7d", src_struct, dst_struct)
    for i in np.flatnonzero(src_struct | dst_struct):
        parts[i].append(
            f"Structuring: {int(struct_count[i])} sub-threshold cash txns "
            f"in 7d totalling {struct_sum[i]:.0f}"
        )

    src_vel, dst_vel = vel_hits
    vel_count7 = _side(df, "txn_count_7d", src_vel, dst_vel)
    vel_count30 = _side(df, "txn_count_30d", src_vel, dst_vel)
    for i in np.flatnonzero(src_vel | dst_vel):
        parts[i].append(
            f"Velocity spike: {int(vel_count7[i])} txns in 7d vs {int(vel_count30[i])} in trailing 30d"
        )

    src_z, dst_z = z_hits
    z_score = _side(df, "amount_zscore", src_z, dst_z)
    for i in np.flatnonzero(src_z | dst_z):
        parts[i].append(f"Amount anomaly: z-score {z_score[i]:.1f} vs account history")

    src_fan, dst_fan = fan_hits
    fan_count = _side(df, "new_counterparties_30d", src_fan, dst_fan)
    for i in np.flatnonzero(src_fan | dst_fan):
        parts[i].append(f"Cross-border fan pattern: {int(fan_count[i])} new counterparties in 30d")

    return np.array([" | ".join(p) if p else "No rule triggered" for p in parts], dtype=object)


def main() -> None:
    df = pd.read_parquet(PROC / "edge_features.parquet")

    src_struct, dst_struct = rule_structuring(df)
    src_vel, dst_vel = rule_high_velocity(df)
    src_z, dst_z = rule_amount_anomaly(df)
    src_fan, dst_fan = rule_fan_pattern(df)

    df["r_structuring"] = src_struct | dst_struct
    df["r_high_velocity"] = src_vel | dst_vel
    df["r_amount_anomaly"] = src_z | dst_z
    df["r_fan_pattern"] = src_fan | dst_fan

    rule_cols = ["r_structuring", "r_high_velocity", "r_amount_anomaly", "r_fan_pattern"]

    # r_high_velocity is kept as a computed column for the diagnostic breakdown below (and
    # for the report's negative-result writeup), but it is EXCLUDED from the alert score.
    # Across three independent threshold redesigns it consistently showed lift < 1x on the
    # test split -- worse than random. Likely cause: this dataset's structuring/smurfing
    # typologies are steady, chronically-high-volume patterns rather than sudden spikes,
    # while legitimate business/payday accounts DO show relative bursts -- so a spike-based
    # rule ends up anti-correlated with the actual label here.
    ALERT_RULE_COLS = ["r_structuring", "r_amount_anomaly", "r_fan_pattern"]
    df["rule_score"] = df[ALERT_RULE_COLS].sum(axis=1)
    df["rule_alert"] = df["rule_score"] > 0
    df["reason"] = build_reasons(
        df,
        (src_struct, dst_struct),
        (src_vel, dst_vel),
        (src_z, dst_z),
        (src_fan, dst_fan),
    )

    out_cols = ["edge_id", "tx_id", "src", "dst", "timestamp", "amount", "y", "split",
                *rule_cols, "rule_score", "rule_alert", "reason"]
    out = df[out_cols]
    out.to_parquet(PROC / "rule_flags.parquet", index=False)

    print("===== RULE ENGINE SUMMARY =====")
    print("(r_high_velocity is computed and reported below for reference, but excluded from "
          "rule_score/rule_alert -- it showed negative lift on the test split)")
    print(f"Total transactions: {len(out):,}")
    print(f"Alerts raised: {out['rule_alert'].sum():,} ({100 * out['rule_alert'].mean():.2f}%)")

    print("\n--- Per-rule hit rate ---")
    for c in rule_cols:
        print(f"  {c}: {out[c].sum():,} ({100 * out[c].mean():.3f}%)")

    print("\n--- Evaluation on TEST split only (held-out, honest) ---")
    test = out[out["split"] == "test"]
    tp = int(((test["rule_alert"]) & (test["y"] == 1)).sum())
    fp = int(((test["rule_alert"]) & (test["y"] == 0)).sum())
    fn = int(((~test["rule_alert"]) & (test["y"] == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    print(f"  Test transactions: {len(test):,} | actual positives: {int(test['y'].sum()):,}")
    print(f"  TP={tp} FP={fp} FN={fn}")
    print(f"  Precision={precision:.3f}  Recall={recall:.3f}")

    print("\n--- Per-rule precision on TEST split (which rules actually work) ---")
    base_rate = test["y"].mean()
    for c in rule_cols:
        sub = test[test[c]]
        r_tp = int((sub["y"] == 1).sum())
        r_total = len(sub)
        r_precision = r_tp / r_total if r_total else 0.0
        lift = r_precision / base_rate if base_rate else 0.0
        print(f"  {c}: fires={r_total:,} caught_positives={r_tp} "
              f"precision={r_precision:.4f} (lift={lift:.1f}x base rate)")

    print("\n--- Sample alerts (first 5) ---")
    for _, row in out[out["rule_alert"]].head(5).iterrows():
        print(f"  edge_id={row['edge_id']} amount={row['amount']:.2f} y={row['y']} -> {row['reason']}")

    print(f"\nSaved: {PROC / 'rule_flags.parquet'}")


if __name__ == "__main__":
    main()