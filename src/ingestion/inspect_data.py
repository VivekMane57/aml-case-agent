from pathlib import Path
import pandas as pd

RAW_DIR = Path("data/raw")

# folder me jo bhi CSV hai wo pick karega
csv_path = next(RAW_DIR.glob("*.csv"))
print(f"File: {csv_path.name}")

# sirf pehli 200k rows, taaki laptop pe heavy na ho
df = pd.read_csv(csv_path, nrows=200_000)

print("\nShape:", df.shape)
print("\nColumns:", list(df.columns))
print("\nDtypes:\n", df.dtypes)
print("\nHead:\n", df.head())
print("\nMissing values:\n", df.isna().sum())

# label column ka naam dataset me dekh ke adjust kar sakta hai
for col in df.columns:
    if "launder" in col.lower():
        print(f"\nValue counts for {col}:\n", df[col].value_counts())