import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "master_2001_2014"
    / "master_harmonized_ipc_2001_2014.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "master_2001_2014"
    / "temporal_model"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 80)
print("TEMPORAL DATASET CREATION")
print("2001–2014")
print("=" * 80)

df = pd.read_csv(INPUT_FILE)

print("\nMaster dataset shape:", df.shape)


# ============================================================
# BASIC CLEANING
# ============================================================

df["YEAR"] = pd.to_numeric(
    df["YEAR"],
    errors="coerce"
)

df["TOTAL_IPC"] = pd.to_numeric(
    df["TOTAL_IPC"],
    errors="coerce"
)

df["WOMEN_CRIME_CORE_TOTAL"] = pd.to_numeric(
    df["WOMEN_CRIME_CORE_TOTAL"],
    errors="coerce"
)


# ============================================================
# NORMALIZE REPORTING-UNIT NAMES
# ============================================================

def normalize_text(value):

    if pd.isna(value):
        return ""

    value = str(value).strip().upper()

    # Standardize punctuation
    value = value.replace(".", "")
    value = value.replace(",", "")
    value = value.replace("-", " ")

    # Standardize multiple spaces
    value = " ".join(value.split())

    return value


df["STATE_NORMALIZED"] = (
    df["STATE_UT"]
    .apply(normalize_text)
)

df["DISTRICT_NORMALIZED"] = (
    df["DISTRICT"]
    .apply(normalize_text)
)


# ============================================================
# STATE FAMILY
#
# Andhra Pradesh and Telangana require special handling
# because Telangana was separately represented in 2014.
#
# We do NOT merge arbitrary states.
# ============================================================

def get_state_family(state):

    if state in [
        "ANDHRA PRADESH",
        "TELANGANA"
    ]:
        return "ANDHRA_PRADESH_TELANGANA_FAMILY"

    return state


df["STATE_FAMILY"] = (
    df["STATE_NORMALIZED"]
    .apply(get_state_family)
)


# ============================================================
# CANONICAL REPORTING UNIT
# ============================================================

df["REPORTING_UNIT_KEY"] = (
    df["STATE_FAMILY"]
    + "||"
    + df["DISTRICT_NORMALIZED"]
)


# ============================================================
# CREATE WOMEN CRIME SHARE
# ============================================================

df["WOMEN_CRIME_SHARE"] = np.where(
    df["TOTAL_IPC"] > 0,
    df["WOMEN_CRIME_CORE_TOTAL"]
    / df["TOTAL_IPC"],
    np.nan
)


# ============================================================
# ZERO IPC RECORDS
# ============================================================

zero_total = df[
    df["TOTAL_IPC"] == 0
]

print("\nRecords with TOTAL_IPC = 0:")
print(len(zero_total))

if len(zero_total) > 0:

    print(
        zero_total[
            [
                "STATE_UT",
                "DISTRICT",
                "YEAR"
            ]
        ].to_string(index=False)
    )


# ============================================================
# SORT
# ============================================================

df = df.sort_values(
    [
        "REPORTING_UNIT_KEY",
        "YEAR"
    ]
).reset_index(
    drop=True
)


# ============================================================
# CHECK REPORTING-UNIT/YEAR DUPLICATES
# ============================================================

duplicate_keys = (
    df.duplicated(
        subset=[
            "REPORTING_UNIT_KEY",
            "YEAR"
        ],
        keep=False
    )
)

print("\n" + "=" * 80)
print("CANONICAL KEY CHECK")
print("=" * 80)

print(
    "Duplicate canonical reporting-unit/year records:",
    duplicate_keys.sum()
)

if duplicate_keys.sum() > 0:

    print(
        "\nDuplicate canonical records:"
    )

    print(
        df.loc[
            duplicate_keys,
            [
                "STATE_UT",
                "DISTRICT",
                "YEAR",
                "REPORTING_UNIT_KEY",
                "TOTAL_IPC"
            ]
        ]
        .sort_values(
            [
                "YEAR",
                "REPORTING_UNIT_KEY"
            ]
        )
        .to_string(index=False)
    )


# ============================================================
# CREATE NEXT-YEAR TARGET
# ============================================================

df["TARGET_YEAR"] = (
    df["YEAR"] + 1
)


# ============================================================
# SHIFT NEXT YEAR WITHIN CANONICAL UNIT
# ============================================================

df["NEXT_YEAR_WOMEN_CRIME_SHARE"] = (
    df
    .groupby("REPORTING_UNIT_KEY")[
        "WOMEN_CRIME_SHARE"
    ]
    .shift(-1)
)

df["NEXT_YEAR_TOTAL_IPC"] = (
    df
    .groupby("REPORTING_UNIT_KEY")[
        "TOTAL_IPC"
    ]
    .shift(-1)
)

df["NEXT_YEAR_WOMEN_CRIME_TOTAL"] = (
    df
    .groupby("REPORTING_UNIT_KEY")[
        "WOMEN_CRIME_CORE_TOTAL"
    ]
    .shift(-1)
)

df["ACTUAL_NEXT_YEAR"] = (
    df
    .groupby("REPORTING_UNIT_KEY")[
        "YEAR"
    ]
    .shift(-1)
)


# ============================================================
# VALID TEMPORAL PAIR
# ============================================================

df["VALID_TEMPORAL_PAIR"] = (
    df["ACTUAL_NEXT_YEAR"]
    == df["TARGET_YEAR"]
)


# ============================================================
# DIAGNOSTIC PAIR COUNTS BEFORE FILTERING
# ============================================================

print("\n" + "=" * 80)
print("RAW TEMPORAL MATCHING")
print("=" * 80)

raw_pairs = df[
    df["VALID_TEMPORAL_PAIR"]
].copy()

raw_pairs_summary = (
    raw_pairs
    .groupby(
        [
            "YEAR",
            "TARGET_YEAR"
        ]
    )
    .size()
    .reset_index(
        name="RAW_PAIR_COUNT"
    )
)

print(
    raw_pairs_summary.to_string(
        index=False
    )
)


# ============================================================
# KEEP VALID YEAR PAIRS
# ============================================================

temporal_df = df[
    df["VALID_TEMPORAL_PAIR"]
].copy()


# ============================================================
# REMOVE UNDEFINED TARGETS
# ============================================================

temporal_df = temporal_df[
    temporal_df[
        "NEXT_YEAR_WOMEN_CRIME_SHARE"
    ].notna()
].copy()


# ============================================================
# REMOVE ZERO IPC CURRENT-YEAR RECORDS
# ============================================================

temporal_df = temporal_df[
    temporal_df["TOTAL_IPC"] > 0
].copy()


# ============================================================
# REMOVE ZERO IPC TARGET-YEAR RECORDS
# ============================================================

temporal_df = temporal_df[
    temporal_df["NEXT_YEAR_TOTAL_IPC"] > 0
].copy()


# ============================================================
# DROP INTERNAL CHECK COLUMNS
# ============================================================

temporal_df = temporal_df.drop(
    columns=[
        "ACTUAL_NEXT_YEAR",
        "VALID_TEMPORAL_PAIR"
    ]
)


# ============================================================
# TEMPORAL SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("FINAL TEMPORAL PAIR SUMMARY")
print("=" * 80)

print(
    "\nTotal valid temporal pairs:",
    len(temporal_df)
)

print(
    "Unique reporting units:",
    temporal_df[
        "REPORTING_UNIT_KEY"
    ].nunique()
)

print(
    "Current year range:",
    temporal_df["YEAR"].min(),
    "to",
    temporal_df["YEAR"].max()
)

print(
    "Target year range:",
    temporal_df["TARGET_YEAR"].min(),
    "to",
    temporal_df["TARGET_YEAR"].max()
)


# ============================================================
# PAIRS BY YEAR
# ============================================================

pairs_by_year = (
    temporal_df
    .groupby(
        [
            "YEAR",
            "TARGET_YEAR"
        ]
    )
    .size()
    .reset_index(
        name="PAIR_COUNT"
    )
)

print("\nPairs by year:")

print(
    pairs_by_year.to_string(
        index=False
    )
)


# ============================================================
# TARGET STATISTICS
# ============================================================

print("\n" + "=" * 80)
print("TARGET STATISTICS")
print("=" * 80)

target = (
    temporal_df[
        "NEXT_YEAR_WOMEN_CRIME_SHARE"
    ]
)

print(
    target.describe()
)

print(
    "\nTarget minimum:",
    target.min()
)

print(
    "Target maximum:",
    target.max()
)

out_of_range = (
    (target < 0)
    |
    (target > 1)
).sum()

print(
    "Target values outside [0,1]:",
    out_of_range
)


# ============================================================
# DUPLICATE FINAL TEMPORAL KEYS
# ============================================================

duplicate_pairs = (
    temporal_df
    .duplicated(
        subset=[
            "REPORTING_UNIT_KEY",
            "YEAR",
            "TARGET_YEAR"
        ],
        keep=False
    )
)

print(
    "\nDuplicate temporal pairs:",
    duplicate_pairs.sum()
)


# ============================================================
# SAVE TEMPORAL DATASET
# ============================================================

output_file = (
    OUTPUT_DIR
    / "temporal_dataset_2001_2014.csv"
)

temporal_df.to_csv(
    output_file,
    index=False
)


# ============================================================
# SAVE PAIR SUMMARY
# ============================================================

pairs_file = (
    OUTPUT_DIR
    / "temporal_pair_summary_2001_2014.csv"
)

pairs_by_year.to_csv(
    pairs_file,
    index=False
)


# ============================================================
# SAVE TARGET SUMMARY
# ============================================================

target_summary = (
    target
    .describe()
    .to_frame()
)

target_summary_file = (
    OUTPUT_DIR
    / "target_summary_2001_2014.csv"
)

target_summary.to_csv(
    target_summary_file
)


# ============================================================
# SAVE RAW MATCH SUMMARY
# ============================================================

raw_pairs_file = (
    OUTPUT_DIR
    / "raw_temporal_match_summary_2001_2014.csv"
)

raw_pairs_summary.to_csv(
    raw_pairs_file,
    index=False
)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 80)
print("TEMPORAL DATASET CREATED")
print("=" * 80)

print("\nSaved to:")
print(output_file)

print("\nPair summary:")
print(pairs_file)

print("\nTarget summary:")
print(target_summary_file)

print("\nRaw matching summary:")
print(raw_pairs_file)

print("\n" + "=" * 80)
print("TEMPORAL DATASET PREPARATION COMPLETED")
print("=" * 80)