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
    / "temporal_model"
    / "temporal_dataset_2001_2014.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "master_2001_2014"
    / "temporal_model"
    / "validation"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_FILE)

print("=" * 80)
print("TEMPORAL DATASET VALIDATION + EDA")
print("2001–2014")
print("=" * 80)

print("\nDataset shape:")
print(df.shape)


# ============================================================
# 1. COLUMN CHECK
# ============================================================

print("\n" + "=" * 80)
print("1. COLUMN CHECK")
print("=" * 80)

print("\nColumns:")
for i, col in enumerate(df.columns, start=1):
    print(f"{i:02d}. {col}")

print("\nTotal columns:", len(df.columns))


# ============================================================
# 2. BASIC INFORMATION
# ============================================================

print("\n" + "=" * 80)
print("2. BASIC INFORMATION")
print("=" * 80)

print("\nData types:")
print(df.dtypes)

print("\nMemory usage:")
print(df.memory_usage(deep=True).sum())


# ============================================================
# 3. MISSING VALUES
# ============================================================

print("\n" + "=" * 80)
print("3. MISSING VALUE CHECK")
print("=" * 80)

missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)

if len(missing) == 0:
    print("\nNo missing values found.")
else:
    print("\nColumns with missing values:")
    print(missing)


# ============================================================
# 4. EXACT DUPLICATES
# ============================================================

print("\n" + "=" * 80)
print("4. EXACT DUPLICATE CHECK")
print("=" * 80)

exact_duplicates = df.duplicated().sum()

print("\nExact duplicate rows:", exact_duplicates)


# ============================================================
# 5. TEMPORAL PAIR DUPLICATES
# ============================================================

print("\n" + "=" * 80)
print("5. TEMPORAL PAIR DUPLICATE CHECK")
print("=" * 80)

pair_columns = [
    "REPORTING_UNIT_KEY",
    "YEAR",
    "TARGET_YEAR"
]

if all(col in df.columns for col in pair_columns):

    pair_duplicates = df.duplicated(
        subset=pair_columns,
        keep=False
    ).sum()

    print("\nDuplicate temporal pair records:", pair_duplicates)

    if pair_duplicates > 0:
        print("\nDuplicate temporal pairs:")
        print(
            df[
                df.duplicated(
                    subset=pair_columns,
                    keep=False
                )
            ][pair_columns]
            .sort_values(pair_columns)
            .to_string(index=False)
        )


# ============================================================
# 6. YEAR RANGE
# ============================================================

print("\n" + "=" * 80)
print("6. YEAR RANGE CHECK")
print("=" * 80)

print("\nCurrent YEAR:")
print("Minimum:", df["YEAR"].min())
print("Maximum:", df["YEAR"].max())

print("\nTarget YEAR:")
print("Minimum:", df["TARGET_YEAR"].min())
print("Maximum:", df["TARGET_YEAR"].max())


# ============================================================
# 7. YEAR RELATIONSHIP CHECK
# ============================================================

print("\n" + "=" * 80)
print("7. TEMPORAL RELATIONSHIP CHECK")
print("=" * 80)

invalid_year_pairs = df[
    df["TARGET_YEAR"] != df["YEAR"] + 1
]

print(
    "\nRecords where TARGET_YEAR != YEAR + 1:",
    len(invalid_year_pairs)
)

if len(invalid_year_pairs) > 0:
    print("\nInvalid records:")
    print(
        invalid_year_pairs[
            ["STATE_UT", "DISTRICT", "YEAR", "TARGET_YEAR"]
        ].head(20).to_string(index=False)
    )


# ============================================================
# 8. PAIRS BY YEAR
# ============================================================

print("\n" + "=" * 80)
print("8. TEMPORAL PAIRS BY YEAR")
print("=" * 80)

pair_summary = (
    df.groupby(["YEAR", "TARGET_YEAR"])
    .size()
    .reset_index(name="PAIR_COUNT")
)

print("\n")
print(pair_summary.to_string(index=False))

pair_summary.to_csv(
    OUTPUT_DIR / "validation_pairs_by_year.csv",
    index=False
)


# ============================================================
# 9. ZERO TOTAL IPC CHECK
# ============================================================

print("\n" + "=" * 80)
print("9. ZERO TOTAL IPC CHECK")
print("=" * 80)

zero_current = (df["TOTAL_IPC"] == 0).sum()

print("\nCurrent-year TOTAL_IPC = 0:", zero_current)

if "NEXT_YEAR_TOTAL_IPC" in df.columns:
    zero_target = (df["NEXT_YEAR_TOTAL_IPC"] == 0).sum()
    print("Target-year TOTAL_IPC = 0:", zero_target)


# ============================================================
# 10. TARGET VALIDATION
# ============================================================

TARGET = "NEXT_YEAR_WOMEN_CRIME_SHARE"

print("\n" + "=" * 80)
print("10. TARGET VALIDATION")
print("=" * 80)

print("\nTarget column:", TARGET)

print("\nTarget statistics:")
print(df[TARGET].describe())

target_min = df[TARGET].min()
target_max = df[TARGET].max()

print("\nTarget minimum:", target_min)
print("Target maximum:", target_max)

outside_target = df[
    (df[TARGET] < 0) |
    (df[TARGET] > 1)
]

print(
    "\nTarget values outside [0,1]:",
    len(outside_target)
)

if len(outside_target) > 0:
    print(outside_target[[TARGET]].head(20))


# ============================================================
# 11. TARGET DISTRIBUTION
# ============================================================

print("\n" + "=" * 80)
print("11. TARGET DISTRIBUTION")
print("=" * 80)

target_bins = pd.cut(
    df[TARGET],
    bins=[
        -np.inf,
        0.02,
        0.04,
        0.06,
        0.08,
        0.10,
        0.15,
        0.20,
        0.30,
        0.50,
        np.inf
    ]
)

target_distribution = (
    target_bins
    .value_counts(sort=False)
    .reset_index()
)

target_distribution.columns = [
    "TARGET_RANGE",
    "COUNT"
]

target_distribution["PERCENTAGE"] = (
    target_distribution["COUNT"]
    / len(df)
    * 100
)

print(
    target_distribution.to_string(index=False)
)

target_distribution.to_csv(
    OUTPUT_DIR / "target_distribution.csv",
    index=False
)


# ============================================================
# 12. TARGET BY YEAR
# ============================================================

print("\n" + "=" * 80)
print("12. TARGET STATISTICS BY CURRENT YEAR")
print("=" * 80)

target_by_year = (
    df.groupby("YEAR")[TARGET]
    .agg(
        COUNT="count",
        MEAN="mean",
        MEDIAN="median",
        STD="std",
        MIN="min",
        MAX="max"
    )
    .reset_index()
)

print(
    target_by_year.to_string(index=False)
)

target_by_year.to_csv(
    OUTPUT_DIR / "target_statistics_by_year.csv",
    index=False
)


# ============================================================
# 13. REPORTING UNIT COVERAGE
# ============================================================

print("\n" + "=" * 80)
print("13. REPORTING UNIT COVERAGE")
print("=" * 80)

print(
    "\nUnique reporting units:",
    df["REPORTING_UNIT_KEY"].nunique()
)

unit_pair_counts = (
    df.groupby("REPORTING_UNIT_KEY")
    .size()
    .describe()
)

print("\nTemporal pair count per reporting unit:")
print(unit_pair_counts)


# ============================================================
# 14. STATE / UT COVERAGE
# ============================================================

print("\n" + "=" * 80)
print("14. STATE / UT COVERAGE")
print("=" * 80)

state_counts = (
    df.groupby("STATE_UT")
    .size()
    .sort_values(ascending=False)
)

print("\nNumber of temporal pairs by State/UT:")
print(state_counts.to_string())

state_counts.reset_index(
    name="PAIR_COUNT"
).to_csv(
    OUTPUT_DIR / "state_pair_counts.csv",
    index=False
)


# ============================================================
# 15. NUMERIC FEATURE CORRELATION
# ============================================================

print("\n" + "=" * 80)
print("15. TARGET CORRELATION")
print("=" * 80)

numeric_columns = df.select_dtypes(
    include=[np.number]
).columns

correlation = (
    df[numeric_columns]
    .corr()[TARGET]
    .sort_values(
        ascending=False
    )
)

print("\nCorrelation with target:")
print(correlation.to_string())

correlation.to_csv(
    OUTPUT_DIR / "target_correlations.csv",
    header=["CORRELATION"]
)


# ============================================================
# 16. POTENTIAL DATA LEAKAGE CHECK
# ============================================================

print("\n" + "=" * 80)
print("16. POTENTIAL DATA LEAKAGE CHECK")
print("=" * 80)

leakage_keywords = [
    "NEXT_YEAR",
    "TARGET",
    "FUTURE"
]

potential_leakage = []

for col in df.columns:

    upper_col = col.upper()

    if any(
        keyword in upper_col
        for keyword in leakage_keywords
    ):
        potential_leakage.append(col)

print("\nColumns containing target/future keywords:")

for col in potential_leakage:
    print("-", col)


# ============================================================
# 17. NUMERIC SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("17. NUMERIC FEATURE SUMMARY")
print("=" * 80)

numeric_summary = (
    df.select_dtypes(include=[np.number])
    .describe()
    .T
)

print(
    numeric_summary.to_string()
)

numeric_summary.to_csv(
    OUTPUT_DIR / "numeric_feature_summary.csv"
)


# ============================================================
# 18. FINAL VALIDATION SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("FINAL VALIDATION SUMMARY")
print("=" * 80)

checks = {
    "Dataset rows": len(df),
    "Dataset columns": len(df.columns),
    "Missing values": int(df.isnull().sum().sum()),
    "Exact duplicates": int(exact_duplicates),
    "Invalid year relationships": len(invalid_year_pairs),
    "Target outside [0,1]": len(outside_target),
    "Unique reporting units": df["REPORTING_UNIT_KEY"].nunique(),
    "Minimum target": target_min,
    "Maximum target": target_max,
}

for key, value in checks.items():
    print(f"{key}: {value}")


# ============================================================
# SAVE VALIDATED COPY
# ============================================================

validated_file = (
    OUTPUT_DIR
    / "validated_temporal_dataset_2001_2014.csv"
)

df.to_csv(
    validated_file,
    index=False
)

print("\n" + "=" * 80)
print("VALIDATION COMPLETED")
print("=" * 80)

print("\nValidation outputs saved to:")
print(OUTPUT_DIR)

print("\nValidated dataset saved to:")
print(validated_file)