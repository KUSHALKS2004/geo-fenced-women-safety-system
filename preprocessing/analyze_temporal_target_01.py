import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "feature_engineering_01"
    / "dataset_01_engineered.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "feature_engineering_01"
    / "temporal_target_analysis"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("TEMPORAL TARGET ANALYSIS - DATASET 01")
print("=" * 70)

df = pd.read_csv(INPUT_FILE)

print(f"\nInput file: {INPUT_FILE}")
print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")


# ============================================================
# BASIC CHECK
# ============================================================

required_columns = [
    "STATE/UT",
    "DISTRICT",
    "YEAR",
    "WOMEN_CRIME_TOTAL",
    "WOMEN_CRIME_SHARE",
    "TOTAL IPC CRIMES",
]

missing = [col for col in required_columns if col not in df.columns]

if missing:
    print("\nERROR: Missing required columns:")
    for col in missing:
        print(f"  - {col}")
    raise SystemExit(1)


# ============================================================
# DATA TYPES
# ============================================================

df["YEAR"] = pd.to_numeric(df["YEAR"], errors="coerce")

numeric_columns = [
    "WOMEN_CRIME_TOTAL",
    "WOMEN_CRIME_SHARE",
    "TOTAL IPC CRIMES",
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")


# ============================================================
# CHECK DUPLICATE STATE-DISTRICT-YEAR KEYS
# ============================================================

KEY_COLUMNS = ["STATE/UT", "DISTRICT", "YEAR"]

duplicate_counts = (
    df.groupby(KEY_COLUMNS)
    .size()
    .reset_index(name="COUNT")
)

duplicate_keys = duplicate_counts[
    duplicate_counts["COUNT"] > 1
].copy()

print("\n" + "=" * 70)
print("DUPLICATE TEMPORAL KEYS")
print("=" * 70)

if duplicate_keys.empty:
    print("No duplicate STATE/UT + DISTRICT + YEAR keys found.")

else:
    print(
        f"Duplicate key groups found: "
        f"{len(duplicate_keys)}"
    )

    print("\nThese records cannot be used directly for an unambiguous")
    print("year-to-year temporal pair.")

    print("\nDuplicate groups:")

    for _, row in duplicate_keys.iterrows():
        print(
            f"  {row['STATE/UT']} | "
            f"{row['DISTRICT']} | "
            f"{int(row['YEAR'])} | "
            f"{int(row['COUNT'])} records"
        )


# ============================================================
# REMOVE ONLY AMBIGUOUS KEYS FOR TEMPORAL MATCHING
# ============================================================

duplicate_key_set = set(
    tuple(row[col] for col in KEY_COLUMNS)
    for _, row in duplicate_keys.iterrows()
)

df["TEMPORAL_KEY"] = list(
    zip(
        df["STATE/UT"],
        df["DISTRICT"],
        df["YEAR"]
    )
)

temporal_df = df[
    ~df["TEMPORAL_KEY"].isin(duplicate_key_set)
].copy()

print("\nRows available for temporal matching:")
print(f"  Original rows: {len(df)}")
print(f"  Excluded ambiguous rows: {len(df) - len(temporal_df)}")
print(f"  Usable rows: {len(temporal_df)}")


# ============================================================
# CREATE NEXT-YEAR DATA
# ============================================================

current = temporal_df.copy()

future = temporal_df[
    [
        "STATE/UT",
        "DISTRICT",
        "YEAR",
        "WOMEN_CRIME_TOTAL",
        "WOMEN_CRIME_SHARE",
        "TOTAL IPC CRIMES",
    ]
].copy()

future["YEAR"] = future["YEAR"] - 1

future = future.rename(
    columns={
        "WOMEN_CRIME_TOTAL": "NEXT_YEAR_WOMEN_CRIME_TOTAL",
        "WOMEN_CRIME_SHARE": "NEXT_YEAR_WOMEN_CRIME_SHARE",
        "TOTAL IPC CRIMES": "NEXT_YEAR_TOTAL_IPC_CRIMES",
    }
)


# ============================================================
# MERGE CURRENT YEAR WITH NEXT YEAR
# ============================================================

temporal_pairs = current.merge(
    future,
    on=["STATE/UT", "DISTRICT", "YEAR"],
    how="inner",
)

# Ensure it really is a one-year transition
temporal_pairs["NEXT_YEAR"] = temporal_pairs["YEAR"] + 1


# ============================================================
# CALCULATE TARGET CHANGES
# ============================================================

temporal_pairs["WOMEN_CRIME_CHANGE"] = (
    temporal_pairs["NEXT_YEAR_WOMEN_CRIME_TOTAL"]
    - temporal_pairs["WOMEN_CRIME_TOTAL"]
)

temporal_pairs["WOMEN_CRIME_CHANGE_PCT"] = np.where(
    temporal_pairs["WOMEN_CRIME_TOTAL"] > 0,
    (
        temporal_pairs["WOMEN_CRIME_CHANGE"]
        / temporal_pairs["WOMEN_CRIME_TOTAL"]
    ) * 100,
    np.nan,
)

temporal_pairs["WOMEN_SHARE_CHANGE"] = (
    temporal_pairs["NEXT_YEAR_WOMEN_CRIME_SHARE"]
    - temporal_pairs["WOMEN_CRIME_SHARE"]
)

temporal_pairs["TOTAL_IPC_CHANGE"] = (
    temporal_pairs["NEXT_YEAR_TOTAL_IPC_CRIMES"]
    - temporal_pairs["TOTAL IPC CRIMES"]
)


# ============================================================
# SORT
# ============================================================

temporal_pairs = temporal_pairs.sort_values(
    ["YEAR", "STATE/UT", "DISTRICT"]
).reset_index(drop=True)


# ============================================================
# BASIC RESULTS
# ============================================================

print("\n" + "=" * 70)
print("TEMPORAL PAIR RESULTS")
print("=" * 70)

print(f"\nValid current-year → next-year pairs: {len(temporal_pairs)}")

if len(temporal_pairs) > 0:

    print(
        f"Current years: "
        f"{temporal_pairs['YEAR'].min()} "
        f"to "
        f"{temporal_pairs['YEAR'].max()}"
    )

    print(
        f"Target years: "
        f"{temporal_pairs['NEXT_YEAR'].min()} "
        f"to "
        f"{temporal_pairs['NEXT_YEAR'].max()}"
    )

    print(
        f"Unique reporting units: "
        f"{temporal_pairs[['STATE/UT', 'DISTRICT']].drop_duplicates().shape[0]}"
    )


# ============================================================
# PAIRS PER TRANSITION
# ============================================================

print("\n" + "=" * 70)
print("PAIRS BY YEAR")
print("=" * 70)

pairs_by_year = (
    temporal_pairs
    .groupby(["YEAR", "NEXT_YEAR"])
    .size()
    .reset_index(name="PAIR_COUNT")
)

print(pairs_by_year.to_string(index=False))


# ============================================================
# TARGET 1: NEXT-YEAR WOMEN CRIME TOTAL
# ============================================================

print("\n" + "=" * 70)
print("TARGET 1: NEXT-YEAR WOMEN CRIME TOTAL")
print("=" * 70)

print(
    temporal_pairs["NEXT_YEAR_WOMEN_CRIME_TOTAL"]
    .describe()
    .to_string()
)


# ============================================================
# TARGET 2: NEXT-YEAR WOMEN CRIME SHARE
# ============================================================

print("\n" + "=" * 70)
print("TARGET 2: NEXT-YEAR WOMEN CRIME SHARE")
print("=" * 70)

print(
    temporal_pairs["NEXT_YEAR_WOMEN_CRIME_SHARE"]
    .describe()
    .to_string()
)


# ============================================================
# TARGET 3: CHANGE IN WOMEN CRIME
# ============================================================

print("\n" + "=" * 70)
print("TARGET 3: WOMEN CRIME CHANGE")
print("=" * 70)

print(
    temporal_pairs["WOMEN_CRIME_CHANGE"]
    .describe()
    .to_string()
)


# ============================================================
# TARGET 4: PERCENTAGE CHANGE
# ============================================================

print("\n" + "=" * 70)
print("TARGET 4: WOMEN CRIME PERCENTAGE CHANGE")
print("=" * 70)

print(
    temporal_pairs["WOMEN_CRIME_CHANGE_PCT"]
    .describe()
    .to_string()
)


# ============================================================
# TARGET 5: CHANGE IN WOMEN CRIME SHARE
# ============================================================

print("\n" + "=" * 70)
print("TARGET 5: WOMEN CRIME SHARE CHANGE")
print("=" * 70)

print(
    temporal_pairs["WOMEN_SHARE_CHANGE"]
    .describe()
    .to_string()
)


# ============================================================
# ZERO TARGET CHECK
# ============================================================

print("\n" + "=" * 70)
print("ZERO VALUES IN TARGETS")
print("=" * 70)

target_columns = [
    "NEXT_YEAR_WOMEN_CRIME_TOTAL",
    "NEXT_YEAR_WOMEN_CRIME_SHARE",
    "WOMEN_CRIME_CHANGE",
    "WOMEN_CRIME_CHANGE_PCT",
    "WOMEN_SHARE_CHANGE",
]

for col in target_columns:

    zero_count = (temporal_pairs[col] == 0).sum()

    print(
        f"{col}: "
        f"{zero_count} zeros "
        f"({zero_count / len(temporal_pairs) * 100:.2f}%)"
    )


# ============================================================
# NEGATIVE / POSITIVE CHANGE
# ============================================================

print("\n" + "=" * 70)
print("DIRECTION OF WOMEN CRIME CHANGE")
print("=" * 70)

increase = (
    temporal_pairs["WOMEN_CRIME_CHANGE"] > 0
).sum()

decrease = (
    temporal_pairs["WOMEN_CRIME_CHANGE"] < 0
).sum()

no_change = (
    temporal_pairs["WOMEN_CRIME_CHANGE"] == 0
).sum()

total = len(temporal_pairs)

print(
    f"Increase:  {increase:,} "
    f"({increase / total * 100:.2f}%)"
)

print(
    f"Decrease:  {decrease:,} "
    f"({decrease / total * 100:.2f}%)"
)

print(
    f"No change: {no_change:,} "
    f"({no_change / total * 100:.2f}%)"
)


# ============================================================
# EXTREME CHANGES
# ============================================================

print("\n" + "=" * 70)
print("LARGEST INCREASES")
print("=" * 70)

largest_increases = (
    temporal_pairs
    .sort_values("WOMEN_CRIME_CHANGE", ascending=False)
    [
        [
            "STATE/UT",
            "DISTRICT",
            "YEAR",
            "NEXT_YEAR",
            "WOMEN_CRIME_TOTAL",
            "NEXT_YEAR_WOMEN_CRIME_TOTAL",
            "WOMEN_CRIME_CHANGE",
        ]
    ]
    .head(20)
)

print(largest_increases.to_string(index=False))


print("\n" + "=" * 70)
print("LARGEST DECREASES")
print("=" * 70)

largest_decreases = (
    temporal_pairs
    .sort_values("WOMEN_CRIME_CHANGE", ascending=True)
    [
        [
            "STATE/UT",
            "DISTRICT",
            "YEAR",
            "NEXT_YEAR",
            "WOMEN_CRIME_TOTAL",
            "NEXT_YEAR_WOMEN_CRIME_TOTAL",
            "WOMEN_CRIME_CHANGE",
        ]
    ]
    .head(20)
)

print(largest_decreases.to_string(index=False))


# ============================================================
# SAVE TEMPORAL DATASET
# ============================================================

output_file = (
    OUTPUT_DIR
    / "dataset_01_temporal_pairs.csv"
)

temporal_pairs.to_csv(
    output_file,
    index=False
)


# ============================================================
# SAVE PAIR COUNTS
# ============================================================

pairs_file = (
    OUTPUT_DIR
    / "temporal_pairs_by_year.csv"
)

pairs_by_year.to_csv(
    pairs_file,
    index=False
)


# ============================================================
# SAVE DUPLICATE KEYS
# ============================================================

duplicates_file = (
    OUTPUT_DIR
    / "ambiguous_temporal_keys.csv"
)

duplicate_keys.to_csv(
    duplicates_file,
    index=False
)


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("TEMPORAL TARGET ANALYSIS COMPLETE")
print("=" * 70)

print("\nFiles created:")

print(f"1. {output_file}")
print(f"2. {pairs_file}")
print(f"3. {duplicates_file}")

print("\nNext step:")
print("Review the target distributions before building the MLP.")