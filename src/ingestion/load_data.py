"""
SAML-D ka graph-friendly sample banata hai (random rows nahi, accounts ke hisaab se).

Steps:
1. Pass 1: poori CSV chunks me padh ke laundering accounts aur saare accounts nikaalo.
2. Background normal accounts ka random sample lo.
3. Pass 2: jin rows me sender YA receiver "core" account hai (laundering + background),
   wo rakh lo. Isse core accounts ki poori history milti hai.

Outputs:
    data/processed/saml_sample.parquet    -> transactions (tx_id = original CSV row number)
    data/processed/core_accounts.parquet  -> core accounts aur unka group (laundering/background)

Run:
    python src/ingestion/load_data.py
    python src/ingestion/load_data.py --n-background 50000 --seed 7
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")

CATEGORICAL_COLS = [
    "Payment_currency",
    "Received_currency",
    "Sender_bank_location",
    "Receiver_bank_location",
    "Payment_type",
    "Laundering_type",
]


def find_csv(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    files = sorted(RAW_DIR.rglob("*.csv"))
    if not files:
        raise FileNotFoundError(f"'{RAW_DIR.resolve()}' me koi CSV nahi mili.")
    return files[0]


def scan_accounts(csv_path: Path, chunk_size: int):
    """Pass 1: sirf 3 columns padh ke account universe aur laundering accounts nikaalo."""
    all_accounts: set[int] = set()
    laundering_accounts: set[int] = set()
    n_rows = 0
    n_laundering = 0

    cols = ["Sender_account", "Receiver_account", "Is_laundering"]
    for i, chunk in enumerate(pd.read_csv(csv_path, usecols=cols, chunksize=chunk_size), start=1):
        n_rows += len(chunk)
        all_accounts.update(chunk["Sender_account"].unique().tolist())
        all_accounts.update(chunk["Receiver_account"].unique().tolist())

        bad = chunk[chunk["Is_laundering"] == 1]
        n_laundering += len(bad)
        laundering_accounts.update(bad["Sender_account"].unique().tolist())
        laundering_accounts.update(bad["Receiver_account"].unique().tolist())
        print(f"  [pass 1] chunk {i}: {n_rows:,} rows scanned")

    return all_accounts, laundering_accounts, n_rows, n_laundering


def extract_rows(csv_path: Path, core_accounts: np.ndarray, chunk_size: int) -> pd.DataFrame:
    """Pass 2: jin rows me sender ya receiver core account hai, wo rakho."""
    parts = []
    kept = 0
    for i, chunk in enumerate(pd.read_csv(csv_path, chunksize=chunk_size), start=1):
        mask = chunk["Sender_account"].isin(core_accounts) | chunk["Receiver_account"].isin(core_accounts)
        sel = chunk[mask]  # index = original CSV row number (chunks me continue hota hai)
        kept += len(sel)
        parts.append(sel)
        print(f"  [pass 2] chunk {i}: {kept:,} rows kept so far")

    df = pd.concat(parts)
    df.index.name = "tx_id"
    return df.reset_index()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=None, help="CSV path (default: data/raw me jo mile)")
    parser.add_argument("--n-background", type=int, default=20_000,
                        help="random background normal accounts ki sankhya")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk-size", type=int, default=1_000_000,
                        help="MemoryError aaye to isko kam kar (jaise 300000)")
    args = parser.parse_args()

    csv_path = find_csv(args.csv)
    print(f"File: {csv_path} ({csv_path.stat().st_size / 1e6:.0f} MB)")

    # ---- Pass 1
    print("\nPass 1: accounts scan ho rahe hain...")
    all_accounts, laundering_accounts, n_rows, n_laundering = scan_accounts(csv_path, args.chunk_size)
    print(f"\nTotal rows: {n_rows:,} | laundering rows: {n_laundering:,} "
          f"({100 * n_laundering / n_rows:.4f}%)")
    print(f"Unique accounts: {len(all_accounts):,} | laundering accounts: {len(laundering_accounts):,}")

    # ---- Background sample
    rng = np.random.default_rng(args.seed)
    pool = np.array(sorted(all_accounts - laundering_accounts), dtype=np.int64)
    n_bg = min(args.n_background, len(pool))
    background = rng.choice(pool, size=n_bg, replace=False)

    core = np.array(sorted(laundering_accounts), dtype=np.int64)
    core = np.concatenate([core, background])

    # ---- Pass 2
    print("\nPass 2: rows extract ho rahe hain...")
    df = extract_rows(csv_path, core, args.chunk_size)

    df["timestamp"] = pd.to_datetime(df["Date"] + " " + df["Time"])
    df = df.sort_values(["timestamp", "tx_id"]).reset_index(drop=True)
    for col in CATEGORICAL_COLS:
        df[col] = df[col].astype("category")

    core_df = pd.DataFrame({
        "account": core,
        "group": ["laundering"] * len(laundering_accounts) + ["background"] * n_bg,
    })

    # ---- Save
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_DIR / "saml_sample.parquet", index=False)
    core_df.to_parquet(OUT_DIR / "core_accounts.parquet", index=False)

    # ---- Summary
    n_sample_accounts = pd.unique(pd.concat([df["Sender_account"], df["Receiver_account"]])).size
    print("\n===== SAMPLE SUMMARY =====")
    print(f"Rows kept: {len(df):,} ({100 * len(df) / n_rows:.2f}% of full data)")
    print(f"Accounts in sample (nodes): {n_sample_accounts:,}")
    print(f"Core accounts: {len(core):,} (laundering={len(laundering_accounts):,}, background={n_bg:,})")
    print(f"Laundering rows: {int(df['Is_laundering'].sum()):,} "
          f"({100 * df['Is_laundering'].mean():.3f}% of sample)")
    print(f"Time range: {df['timestamp'].min()} -> {df['timestamp'].max()}")
    print("\nSuspicious typologies in sample:")
    print(df.loc[df["Is_laundering"] == 1, "Laundering_type"].value_counts().to_string())
    print(f"\nSaved: {OUT_DIR / 'saml_sample.parquet'}")
    print(f"Saved: {OUT_DIR / 'core_accounts.parquet'}")


if __name__ == "__main__":
    main()