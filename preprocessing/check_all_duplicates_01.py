import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "data" / "cleaned_district_ipc_2001_2012_v2.csv"

df = pd.read_csv(INPUT_FILE)

print("=" * 70)
print("SAFEHER-AI - ALL DUPLICATE STATE-DISTRICT-YEAR CHECK")
print("=" * 70)

key_columns = ["STATE/UT", "DISTRICT", "YEAR"]

duplicates = df[
    df.duplicated(
        subset=key_columns,
        keep=False
    )
].sort_values(key_columns)

print(f"\nTotal duplicate rows: {len(duplicates)}")

if len(duplicates) > 0:

    print("\nDuplicate groups:\n")

    groups = (
        duplicates
        .groupby(key_columns)
        .size()
        .reset_index(name="RECORD_COUNT")
    )

    print(groups.to_string(index=False))

    print("\n" + "-" * 70)
    print("NUMBER OF DUPLICATE GROUPS")
    print("-" * 70)

    print(len(groups))

else:

    print("\nNo duplicate State-District-Year groups found.")


print("\n" + "=" * 70)
print("END")
print("=" * 70)