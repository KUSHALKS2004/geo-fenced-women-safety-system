import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "data" / "cleaned_district_ipc_2001_2012_v2.csv"

df = pd.read_csv(INPUT_FILE)

print("=" * 70)
print("SAFEHER-AI - INVESTIGATE DUPLICATE RECORD")
print("=" * 70)

# Find duplicate State-District-Year
duplicates = df[
    df.duplicated(
        subset=["STATE/UT", "DISTRICT", "YEAR"],
        keep=False
    )
].sort_values(["STATE/UT", "DISTRICT", "YEAR"])

print("\nDuplicate records:\n")

print(duplicates.to_string(index=False))

print("\n" + "-" * 70)
print("ROW DETAILS")
print("-" * 70)

for index, row in duplicates.iterrows():
    print(f"\nDataFrame index: {index}")
    print(row.to_string())

print("\n" + "=" * 70)
print("END")
print("=" * 70)