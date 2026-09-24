import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.preprocessing import StandardScaler


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "master_2001_2014"
    / "temporal_model"
    / "validation"
    / "validated_temporal_dataset_2001_2014.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "master_2001_2014"
    / "temporal_model"
    / "mlp"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_FILE)

print("=" * 80)
print("MLP DATASET PREPARATION")
print("SafeHer-AI | 2001–2014")
print("=" * 80)

print("\nInput shape:", df.shape)


# ============================================================
# TARGET
# ============================================================

TARGET = "NEXT_YEAR_WOMEN_CRIME_SHARE"


# ============================================================
# FEATURES
# ============================================================
# IMPORTANT:
# Only current-year information is used.
#
# NEXT_YEAR_* columns are deliberately excluded to prevent
# future-data leakage.
#
# STATE/DISTRICT text fields are also excluded at this stage.
# GIS/location encoding will be handled separately later.

FEATURES = [
    "MURDER",
    "ATTEMPT_MURDER",
    "CULPABLE_HOMICIDE",
    "RAPE",
    "CUSTODIAL_RAPE",
    "KIDNAPPING_ABDUCTION",
    "DACOITY",
    "ROBBERY",
    "BURGLARY",
    "THEFT",
    "AUTO_THEFT",
    "RIOTS",
    "CRIMINAL_BREACH_TRUST",
    "CHEATING",
    "ARSON",
    "DOWRY_DEATHS",
    "ASSAULT_WOMEN",
    "INSULT_WOMEN",
    "CRUELTY_WOMEN",
    "IMPORTATION_GIRLS",
    "DEATH_BY_NEGLIGENCE",
    "OTHER_IPC",
    "TOTAL_IPC",
    "WOMEN_CRIME_CORE_TOTAL",
    "VIOLENT_CRIME",
    "PROPERTY_CRIME",
    "ECONOMIC_OFFENCES",
    "PUBLIC_ORDER_CRIME",
    "WOMEN_CRIME_SHARE",
    "YEAR",
]


# ============================================================
# FEATURE CHECK
# ============================================================

print("\n" + "=" * 80)
print("FEATURE CHECK")
print("=" * 80)

missing_features = [
    col for col in FEATURES
    if col not in df.columns
]

if missing_features:
    print("\nMissing required features:")
    for col in missing_features:
        print("-", col)

    raise ValueError(
        "Required MLP features are missing from the dataset."
    )

print("\nNumber of features:", len(FEATURES))

print("\nFeatures:")

for i, col in enumerate(FEATURES, start=1):
    print(f"{i:02d}. {col}")


# ============================================================
# LEAKAGE CHECK
# ============================================================

print("\n" + "=" * 80)
print("LEAKAGE CHECK")
print("=" * 80)

future_columns = [
    col for col in FEATURES
    if "NEXT_YEAR" in col.upper()
]

if future_columns:
    print("\nERROR: Future columns detected:")
    print(future_columns)

    raise ValueError(
        "Future-data leakage detected in feature set."
    )

print("\nNo NEXT_YEAR columns are used as input features.")


# ============================================================
# MODEL DATAFRAME
# ============================================================

model_df = df[
    FEATURES + [
        TARGET,
        "TARGET_YEAR",
        "STATE_UT",
        "DISTRICT",
        "REPORTING_UNIT_KEY",
    ]
].copy()


# ============================================================
# FINAL MISSING VALUE CHECK
# ============================================================

print("\n" + "=" * 80)
print("MODEL DATA QUALITY CHECK")
print("=" * 80)

feature_missing = model_df[FEATURES].isnull().sum().sum()
target_missing = model_df[TARGET].isnull().sum()

print("\nMissing feature values:", feature_missing)
print("Missing target values:", target_missing)

if feature_missing > 0 or target_missing > 0:
    raise ValueError(
        "Missing values detected in model data."
    )


# ============================================================
# CHRONOLOGICAL SPLIT
# ============================================================
#
# Current YEAR determines the split.
#
# TRAIN:
#   2001–2008
#
# VALIDATION:
#   2009–2010
#
# TEST:
#   2011–2013
#
# Therefore:
#
# TRAIN targets -> 2002–2009
# VALID targets -> 2010–2011
# TEST targets  -> 2012–2014
#
# No future current-year records enter an earlier split.

train_df = model_df[
    model_df["YEAR"] <= 2008
].copy()

validation_df = model_df[
    (model_df["YEAR"] >= 2009) &
    (model_df["YEAR"] <= 2010)
].copy()

test_df = model_df[
    model_df["YEAR"] >= 2011
].copy()


# ============================================================
# SPLIT SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("CHRONOLOGICAL SPLIT")
print("=" * 80)

print("\nTRAIN")
print("Current year:", train_df["YEAR"].min(), "to", train_df["YEAR"].max())
print("Target year:", train_df["TARGET_YEAR"].min(), "to", train_df["TARGET_YEAR"].max())
print("Rows:", len(train_df))

print("\nVALIDATION")
print(
    "Current year:",
    validation_df["YEAR"].min(),
    "to",
    validation_df["YEAR"].max()
)
print(
    "Target year:",
    validation_df["TARGET_YEAR"].min(),
    "to",
    validation_df["TARGET_YEAR"].max()
)
print("Rows:", len(validation_df))

print("\nTEST")
print("Current year:", test_df["YEAR"].min(), "to", test_df["YEAR"].max())
print("Target year:", test_df["TARGET_YEAR"].min(), "to", test_df["TARGET_YEAR"].max())
print("Rows:", len(test_df))


# ============================================================
# CHECK SPLIT YEAR OVERLAP
# ============================================================

train_years = set(train_df["YEAR"])
validation_years = set(validation_df["YEAR"])
test_years = set(test_df["YEAR"])

print("\n" + "=" * 80)
print("SPLIT OVERLAP CHECK")
print("=" * 80)

print("Train ∩ Validation:", train_years.intersection(validation_years))
print("Train ∩ Test:", train_years.intersection(test_years))
print("Validation ∩ Test:", validation_years.intersection(test_years))


if (
    train_years.intersection(validation_years)
    or train_years.intersection(test_years)
    or validation_years.intersection(test_years)
):
    raise ValueError(
        "Temporal split overlap detected."
    )

print("\nNo current-year overlap between splits.")


# ============================================================
# CHECK TARGET-YEAR OVERLAP
# ============================================================

train_target_years = set(train_df["TARGET_YEAR"])
validation_target_years = set(validation_df["TARGET_YEAR"])
test_target_years = set(test_df["TARGET_YEAR"])

print("\nTarget-year overlap:")
print(
    "Train ∩ Validation:",
    train_target_years.intersection(validation_target_years)
)

print(
    "Train ∩ Test:",
    train_target_years.intersection(test_target_years)
)

print(
    "Validation ∩ Test:",
    validation_target_years.intersection(test_target_years)
)


# ============================================================
# X / y
# ============================================================

X_train = train_df[FEATURES].copy()
y_train = train_df[TARGET].copy()

X_validation = validation_df[FEATURES].copy()
y_validation = validation_df[TARGET].copy()

X_test = test_df[FEATURES].copy()
y_test = test_df[TARGET].copy()


# ============================================================
# FEATURE TRANSFORMATION
# ============================================================
#
# Crime counts are highly right-skewed.
#
# log1p is applied to non-year numeric features.
#
# YEAR is kept as a numerical temporal feature.
#
# IMPORTANT:
# The transformation parameters are learned ONLY from training data.

COUNT_FEATURES = [
    col for col in FEATURES
    if col != "YEAR"
]


# Apply log1p to crime-count/share features.
#
# Shares are between 0 and 1. Applying log1p to them is not
# necessary and would distort their interpretation.
#
# Therefore:
# - count features -> log1p
# - WOMEN_CRIME_SHARE -> unchanged
# - YEAR -> unchanged

LOG_FEATURES = [
    col for col in COUNT_FEATURES
    if col != "WOMEN_CRIME_SHARE"
]


for col in LOG_FEATURES:

    X_train[col] = np.log1p(X_train[col])
    X_validation[col] = np.log1p(X_validation[col])
    X_test[col] = np.log1p(X_test[col])


# ============================================================
# STANDARDIZATION
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_validation_scaled = scaler.transform(
    X_validation
)

X_test_scaled = scaler.transform(
    X_test
)


# ============================================================
# CONVERT BACK TO DATAFRAMES
# ============================================================

X_train_scaled = pd.DataFrame(
    X_train_scaled,
    columns=FEATURES,
    index=X_train.index
)

X_validation_scaled = pd.DataFrame(
    X_validation_scaled,
    columns=FEATURES,
    index=X_validation.index
)

X_test_scaled = pd.DataFrame(
    X_test_scaled,
    columns=FEATURES,
    index=X_test.index
)


# ============================================================
# SCALE CHECK
# ============================================================

print("\n" + "=" * 80)
print("SCALING CHECK")
print("=" * 80)

print("\nTraining means after scaling:")
print(
    X_train_scaled.mean()
    .round(6)
    .to_string()
)

print("\nTraining standard deviations after scaling:")
print(
    X_train_scaled.std()
    .round(6)
    .to_string()
)


# ============================================================
# TARGET SUMMARY BY SPLIT
# ============================================================

print("\n" + "=" * 80)
print("TARGET SUMMARY BY SPLIT")
print("=" * 80)

print("\nTRAIN TARGET")
print(y_train.describe())

print("\nVALIDATION TARGET")
print(y_validation.describe())

print("\nTEST TARGET")
print(y_test.describe())


# ============================================================
# SAVE RAW SPLITS
# ============================================================

train_output = train_df.copy()
validation_output = validation_df.copy()
test_output = test_df.copy()

train_output.to_csv(
    OUTPUT_DIR / "train_temporal_2001_2008.csv",
    index=False
)

validation_output.to_csv(
    OUTPUT_DIR / "validation_temporal_2009_2010.csv",
    index=False
)

test_output.to_csv(
    OUTPUT_DIR / "test_temporal_2011_2013.csv",
    index=False
)


# ============================================================
# SAVE SCALED FEATURES + TARGET
# ============================================================

train_scaled_output = X_train_scaled.copy()
train_scaled_output[TARGET] = y_train.values
train_scaled_output["YEAR"] = train_df["YEAR"].values
train_scaled_output["TARGET_YEAR"] = train_df["TARGET_YEAR"].values

validation_scaled_output = X_validation_scaled.copy()
validation_scaled_output[TARGET] = y_validation.values
validation_scaled_output["YEAR"] = validation_df["YEAR"].values
validation_scaled_output["TARGET_YEAR"] = validation_df["TARGET_YEAR"].values

test_scaled_output = X_test_scaled.copy()
test_scaled_output[TARGET] = y_test.values
test_scaled_output["YEAR"] = test_df["YEAR"].values
test_scaled_output["TARGET_YEAR"] = test_df["TARGET_YEAR"].values


train_scaled_output.to_csv(
    OUTPUT_DIR / "train_scaled_2001_2008.csv",
    index=False
)

validation_scaled_output.to_csv(
    OUTPUT_DIR / "validation_scaled_2009_2010.csv",
    index=False
)

test_scaled_output.to_csv(
    OUTPUT_DIR / "test_scaled_2011_2013.csv",
    index=False
)


# ============================================================
# SAVE FEATURE LIST
# ============================================================

feature_list = pd.DataFrame({
    "FEATURE_NUMBER": range(1, len(FEATURES) + 1),
    "FEATURE": FEATURES,
    "TRANSFORMATION": [
        "log1p + StandardScaler"
        if col in LOG_FEATURES
        else "StandardScaler"
        for col in FEATURES
    ]
})

feature_list.to_csv(
    OUTPUT_DIR / "mlp_feature_list.csv",
    index=False
)


# ============================================================
# SAVE SPLIT SUMMARY
# ============================================================

split_summary = pd.DataFrame({
    "SPLIT": [
        "TRAIN",
        "VALIDATION",
        "TEST"
    ],
    "CURRENT_YEAR_MIN": [
        train_df["YEAR"].min(),
        validation_df["YEAR"].min(),
        test_df["YEAR"].min()
    ],
    "CURRENT_YEAR_MAX": [
        train_df["YEAR"].max(),
        validation_df["YEAR"].max(),
        test_df["YEAR"].max()
    ],
    "TARGET_YEAR_MIN": [
        train_df["TARGET_YEAR"].min(),
        validation_df["TARGET_YEAR"].min(),
        test_df["TARGET_YEAR"].min()
    ],
    "TARGET_YEAR_MAX": [
        train_df["TARGET_YEAR"].max(),
        validation_df["TARGET_YEAR"].max(),
        test_df["TARGET_YEAR"].max()
    ],
    "ROWS": [
        len(train_df),
        len(validation_df),
        len(test_df)
    ]
})

split_summary.to_csv(
    OUTPUT_DIR / "mlp_split_summary.csv",
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("MLP DATASET PREPARATION COMPLETED")
print("=" * 80)

print("\nFeature count:", len(FEATURES))

print("\nRows:")
print("Train:", len(train_df))
print("Validation:", len(validation_df))
print("Test:", len(test_df))

print("\nSaved files:")
print(
    OUTPUT_DIR / "train_temporal_2001_2008.csv"
)
print(
    OUTPUT_DIR / "validation_temporal_2009_2010.csv"
)
print(
    OUTPUT_DIR / "test_temporal_2011_2013.csv"
)
print(
    OUTPUT_DIR / "train_scaled_2001_2008.csv"
)
print(
    OUTPUT_DIR / "validation_scaled_2009_2010.csv"
)
print(
    OUTPUT_DIR / "test_scaled_2011_2013.csv"
)
print(
    OUTPUT_DIR / "mlp_feature_list.csv"
)
print(
    OUTPUT_DIR / "mlp_split_summary.csv"
)

print("\nNext step:")
print("Train the MLP neural network.")