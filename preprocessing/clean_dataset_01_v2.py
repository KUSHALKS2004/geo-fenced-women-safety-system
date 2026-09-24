import pandas as pd
from pathlib import Path

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "data" / "dstrIPC_1.csv"
OUTPUT_FILE = BASE_DIR / "data" / "cleaned_district_ipc_2001_2012_v2.csv"


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("SAFEHER-AI - DATASET 01 CLEANING V2")
print("=" * 70)

df = pd.read_csv(INPUT_FILE)

print(f"\nOriginal rows    : {len(df)}")
print(f"Original columns : {len(df.columns)}")


# ============================================================
# BASIC CLEANING
# ============================================================

df.columns = df.columns.str.strip()

df["STATE/UT"] = df["STATE/UT"].astype(str).str.strip()
df["DISTRICT"] = df["DISTRICT"].astype(str).str.strip()


# ============================================================
# FIND AGGREGATE DISTRICT RECORDS
# ============================================================

print("\n" + "-" * 70)
print("CHECKING AGGREGATE DISTRICT RECORDS")
print("-" * 70)

district_upper = df["DISTRICT"].str.upper()

# Records whose district name ends with TOTAL
aggregate_mask = district_upper.str.contains(r"\bTOTAL$", regex=True, na=False)

aggregate_rows = df[aggregate_mask].copy()

print(f"\nAggregate candidate rows found: {len(aggregate_rows)}")

if len(aggregate_rows) > 0:

    print("\nAggregate district names:")
    
    summary = (
        aggregate_rows
        .groupby(["STATE/UT", "DISTRICT"])
        .size()
        .reset_index(name="ROW_COUNT")
        .sort_values(["STATE/UT", "DISTRICT"])
    )

    print(summary.to_string(index=False))

else:
    print("No aggregate district records found.")


# ============================================================
# REMOVE ONLY CONFIRMED AGGREGATE RECORDS
# ============================================================

df_clean = df[~aggregate_mask].copy()

print("\n" + "-" * 70)
print("AFTER AGGREGATE REMOVAL")
print("-" * 70)

print(f"Rows before removal : {len(df)}")
print(f"Rows after removal  : {len(df_clean)}")
print(f"Rows removed        : {len(df) - len(df_clean)}")


# ============================================================
# CHECK DUPLICATE STATE-DISTRICT-YEAR RECORDS
# ============================================================

print("\n" + "-" * 70)
print("CHECKING DUPLICATE STATE-DISTRICT-YEAR RECORDS")
print("-" * 70)

key_columns = ["STATE/UT", "DISTRICT", "YEAR"]

duplicates = df_clean[
    df_clean.duplicated(
        subset=key_columns,
        keep=False
    )
].sort_values(key_columns)

if len(duplicates) == 0:

    print("\nNo duplicate State-District-Year records found.")

else:

    print(f"\nDuplicate rows found: {len(duplicates)}")
    print("\nDuplicate records:")

    print(
        duplicates[
            key_columns + ["TOTAL IPC CRIMES"]
        ].to_string(index=False)
    )


# ============================================================
# CHECK EXACT DUPLICATE ROWS
# ============================================================

exact_duplicates = df_clean[
    df_clean.duplicated(keep=False)
]

print("\nExact duplicate rows:", len(exact_duplicates))


# ============================================================
# CHECK MISSING VALUES
# ============================================================

print("\n" + "-" * 70)
print("CHECKING MISSING VALUES")
print("-" * 70)

missing = df_clean.isnull().sum()

missing = missing[missing > 0]

if len(missing) == 0:
    print("\nNo missing values found.")
else:
    print(missing)


# ============================================================
# BASIC DATASET SUMMARY
# ============================================================

print("\n" + "-" * 70)
print("FINAL DATASET SUMMARY")
print("-" * 70)

print(f"Rows               : {len(df_clean)}")
print(f"Columns            : {len(df_clean.columns)}")
print(f"States/UTs         : {df_clean['STATE/UT'].nunique()}")
print(f"Districts          : {df_clean['DISTRICT'].nunique()}")
print(f"Years              : {df_clean['YEAR'].min()} - {df_clean['YEAR'].max()}")


# ============================================================
# SAVE
# ============================================================

df_clean.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 70)
print("CLEANING V2 COMPLETED")
print("=" * 70)

print(f"\nSaved file:")
print(OUTPUT_FILE)

print("\nNext step:")
print("Review the terminal output before starting EDA/K-Means.")