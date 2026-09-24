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
    / "temporal_target_analysis"
    / "dataset_01_temporal_pairs.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "feature_engineering_01"
    / "mlp_dataset_01"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("PREPARING MLP DATASET - DATASET 01")
print("=" * 70)

df = pd.read_csv(INPUT_FILE)

print(f"\nInput rows: {len(df)}")
print(f"Input columns: {len(df.columns)}")


# ============================================================
# TARGET
# ============================================================

TARGET = "NEXT_YEAR_WOMEN_CRIME_SHARE"

if TARGET not in df.columns:
    raise ValueError(
        f"Target column '{TARGET}' not found."
    )


# ============================================================
# FEATURES
# ============================================================
#
# These are CURRENT-YEAR features.
#
# The model will use year t information
# to predict year t+1 women-crime share.
#
# We deliberately do NOT use:
# - NEXT_YEAR_* columns
# - WOMEN_CRIME_CHANGE
# - WOMEN_CRIME_CHANGE_PCT
# - WOMEN_SHARE_CHANGE
# - RECORD_ID
# - DISTRICT
# - STATE/UT
#
# K-Means cluster label is also NOT used as an MLP input.
# K-Means remains an independent unsupervised component.
# ============================================================

FEATURES = [
    "WOMEN_CRIME_TOTAL",
    "WOMEN_VIOLENCE",
    "RAPE_RATE_PROXY",
    "KIDNAPPING_RATE_PROXY",
    "WOMEN_CRIME_SHARE",
    "VIOLENT_CRIME",
    "SERIOUS_VIOLENT_CRIME",
    "PROPERTY_CRIME",
    "ECONOMIC_OFFENCES",
    "PUBLIC_ORDER_CRIME",
    "VIOLENT_CRIME_SHARE",
    "PROPERTY_CRIME_SHARE",
    "MURDER_SHARE",
    "ROBBERY_SHARE",
    "TOTAL IPC CRIMES",
    "YEAR_NORMALIZED",
]


# ============================================================
# CHECK FEATURES
# ============================================================

missing_features = [
    col for col in FEATURES
    if col not in df.columns
]

if missing_features:
    print("\nERROR: Missing features:")

    for col in missing_features:
        print(f"  - {col}")

    raise SystemExit(1)


# ============================================================
# SELECT DATA
# ============================================================

mlp_df = df[
    [
        "STATE/UT",
        "DISTRICT",
        "YEAR",
        "NEXT_YEAR",
        *FEATURES,
        TARGET,
    ]
].copy()


# ============================================================
# NUMERIC CONVERSION
# ============================================================

for col in FEATURES + [TARGET]:

    mlp_df[col] = pd.to_numeric(
        mlp_df[col],
        errors="coerce"
    )


# ============================================================
# CHECK MISSING / INFINITE VALUES
# ============================================================

print("\n" + "=" * 70)
print("DATA QUALITY CHECK")
print("=" * 70)

missing_count = mlp_df[
    FEATURES + [TARGET]
].isna().sum().sum()

infinite_count = np.isinf(
    mlp_df[FEATURES + [TARGET]].to_numpy()
).sum()

print(f"Missing numeric values: {missing_count}")
print(f"Infinite numeric values: {infinite_count}")


# ============================================================
# REMOVE INVALID ROWS
# ============================================================

before = len(mlp_df)

mlp_df = mlp_df.replace(
    [np.inf, -np.inf],
    np.nan
)

mlp_df = mlp_df.dropna(
    subset=FEATURES + [TARGET]
).copy()

after = len(mlp_df)

print(f"Rows before cleaning: {before}")
print(f"Rows after cleaning:  {after}")
print(f"Rows removed:         {before - after}")


# ============================================================
# SORT CHRONOLOGICALLY
# ============================================================

mlp_df = mlp_df.sort_values(
    ["YEAR", "STATE/UT", "DISTRICT"]
).reset_index(drop=True)


# ============================================================
# TEMPORAL SPLIT
# ============================================================
#
# Current year → next year
#
# TRAIN:
# 2001 → 2002
# ...
# 2008 → 2009
#
# VALIDATION:
# 2009 → 2010
# 2010 → 2011
#
# TEST:
# 2011 → 2012
#
# This prevents future years from entering training.
# ============================================================

train_df = mlp_df[
    mlp_df["YEAR"] <= 2008
].copy()

validation_df = mlp_df[
    mlp_df["YEAR"].isin([2009, 2010])
].copy()

test_df = mlp_df[
    mlp_df["YEAR"] == 2011
].copy()


# ============================================================
# SPLIT SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("CHRONOLOGICAL SPLIT")
print("=" * 70)

print("\nTRAIN")
print(
    f"Years: {train_df['YEAR'].min()} → "
    f"{train_df['NEXT_YEAR'].max()}"
)
print(f"Rows: {len(train_df):,}")

print("\nVALIDATION")
print(
    f"Years: {validation_df['YEAR'].min()} → "
    f"{validation_df['NEXT_YEAR'].max()}"
)
print(f"Rows: {len(validation_df):,}")

print("\nTEST")
print(
    f"Years: {test_df['YEAR'].min()} → "
    f"{test_df['NEXT_YEAR'].max()}"
)
print(f"Rows: {len(test_df):,}")


# ============================================================
# VERIFY NO YEAR OVERLAP
# ============================================================

train_years = set(train_df["YEAR"])
validation_years = set(validation_df["YEAR"])
test_years = set(test_df["YEAR"])

print("\n" + "=" * 70)
print("YEAR OVERLAP CHECK")
print("=" * 70)

print(
    "Train ∩ Validation:",
    train_years.intersection(validation_years)
)

print(
    "Train ∩ Test:",
    train_years.intersection(test_years)
)

print(
    "Validation ∩ Test:",
    validation_years.intersection(test_years)
)


# ============================================================
# TARGET DISTRIBUTION BY SPLIT
# ============================================================

print("\n" + "=" * 70)
print("TARGET DISTRIBUTION")
print("=" * 70)

for name, subset in [
    ("TRAIN", train_df),
    ("VALIDATION", validation_df),
    ("TEST", test_df),
]:

    print(f"\n{name}")

    print(
        subset[TARGET]
        .describe()
        .to_string()
    )


# ============================================================
# FEATURE SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FEATURE SUMMARY")
print("=" * 70)

print(f"\nNumber of features: {len(FEATURES)}")

for i, feature in enumerate(FEATURES, start=1):
    print(f"{i:2}. {feature}")


# ============================================================
# SAVE COMPLETE MLP DATASET
# ============================================================

full_output = (
    OUTPUT_DIR
    / "dataset_01_mlp_ready.csv"
)

mlp_df.to_csv(
    full_output,
    index=False
)


# ============================================================
# SAVE TRAIN / VALIDATION / TEST
# ============================================================

train_output = (
    OUTPUT_DIR
    / "train_dataset_01.csv"
)

validation_output = (
    OUTPUT_DIR
    / "validation_dataset_01.csv"
)

test_output = (
    OUTPUT_DIR
    / "test_dataset_01.csv"
)

train_df.to_csv(
    train_output,
    index=False
)

validation_df.to_csv(
    validation_output,
    index=False
)

test_df.to_csv(
    test_output,
    index=False
)


# ============================================================
# SAVE FEATURE LIST
# ============================================================

feature_list = pd.DataFrame(
    {
        "FEATURE_NUMBER": range(1, len(FEATURES) + 1),
        "FEATURE": FEATURES,
    }
)

feature_list_output = (
    OUTPUT_DIR
    / "mlp_feature_list.csv"
)

feature_list.to_csv(
    feature_list_output,
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("MLP DATASET PREPARATION COMPLETE")
print("=" * 70)

print("\nFiles created:")

print(f"1. {full_output}")
print(f"2. {train_output}")
print(f"3. {validation_output}")
print(f"4. {test_output}")
print(f"5. {feature_list_output}")

print("\nTarget:")
print(f"  {TARGET}")

print("\nModeling approach:")
print("  Current year features → next-year women-crime share")

print("\nImportant:")
print("  MLP has NOT been trained yet.")
print("  No random train/test split was used.")
print("  Future years are kept out of training.")