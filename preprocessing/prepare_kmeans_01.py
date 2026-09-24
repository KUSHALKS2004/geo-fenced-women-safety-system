import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.preprocessing import StandardScaler


# ============================================================
# SAFEHER-AI
# DATASET 01 - K-MEANS PREPARATION
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
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SCALED_FILE = (
    OUTPUT_DIR
    / "kmeans_scaled_features.csv"
)

FEATURE_FILE = (
    OUTPUT_DIR
    / "kmeans_selected_features.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("SAFEHER-AI - K-MEANS DATA PREPARATION")
print("=" * 70)

df = pd.read_csv(INPUT_FILE)

print("\nInput dataset:")
print(f"Rows    : {len(df)}")
print(f"Columns : {len(df.columns)}")


# ============================================================
# SELECT FEATURES
# ============================================================

print("\n" + "-" * 70)
print("1. SELECTING K-MEANS FEATURES")
print("-" * 70)


# We deliberately exclude:
#
# TOTAL IPC CRIMES
# YEAR_NORMALIZED
#
# TOTAL IPC CRIMES is an aggregate of crime categories.
# YEAR_NORMALIZED is excluded from the first clustering model
# so clusters represent crime patterns rather than time periods.


selected_features = [
    "WOMEN_CRIME_TOTAL",
    "WOMEN_VIOLENCE",
    "RAPE_RATE_PROXY",
    "KIDNAPPING_RATE_PROXY",

    "VIOLENT_CRIME",
    "SERIOUS_VIOLENT_CRIME",

    "PROPERTY_CRIME",

    "ECONOMIC_OFFENCES",
    "PUBLIC_ORDER_CRIME",

    "VIOLENT_CRIME_SHARE",
    "PROPERTY_CRIME_SHARE",

    "MURDER_SHARE",
    "ROBBERY_SHARE"
]


print("\nSelected features:")

for feature in selected_features:
    print(" -", feature)


# Check that every feature exists

missing_features = [
    feature
    for feature in selected_features
    if feature not in df.columns
]

if missing_features:

    print("\nERROR: Missing features:")
    
    for feature in missing_features:
        print(" -", feature)

    raise ValueError(
        "One or more selected features are missing."
    )


# ============================================================
# CREATE FEATURE MATRIX
# ============================================================

X = df[selected_features].copy()


# ============================================================
# DATA QUALITY CHECK
# ============================================================

print("\n" + "-" * 70)
print("2. DATA QUALITY CHECK")
print("-" * 70)

print("\nMissing values:")
print(X.isna().sum().sum())

print("\nInfinite values:")
print(np.isinf(X).sum().sum())


# ============================================================
# LOG TRANSFORMATION
# ============================================================

print("\n" + "-" * 70)
print("3. LOG TRANSFORMATION")
print("-" * 70)


# Count-based features are highly right-skewed.
# log1p reduces the influence of extremely large values.
#
# Ratio/share features are already bounded and are therefore
# kept unchanged.


count_features = [
    "WOMEN_CRIME_TOTAL",
    "WOMEN_VIOLENCE",
    "VIOLENT_CRIME",
    "SERIOUS_VIOLENT_CRIME",
    "PROPERTY_CRIME",
    "ECONOMIC_OFFENCES",
    "PUBLIC_ORDER_CRIME"
]


for feature in count_features:

    X[feature] = np.log1p(X[feature])


print("\nLog-transformed features:")

for feature in count_features:
    print(" -", feature)


# ============================================================
# CHECK AFTER TRANSFORMATION
# ============================================================

print("\n" + "-" * 70)
print("4. TRANSFORMED FEATURE STATISTICS")
print("-" * 70)

print(
    X.describe().T[
        [
            "count",
            "mean",
            "std",
            "min",
            "50%",
            "max"
        ]
    ].to_string()
)


# ============================================================
# STANDARDIZATION
# ============================================================

print("\n" + "-" * 70)
print("5. STANDARD SCALING")
print("-" * 70)


scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

X_scaled = pd.DataFrame(
    X_scaled,
    columns=selected_features
)


print("\nScaling completed.")

print("\nScaled feature means:")

print(
    X_scaled.mean()
    .round(6)
    .to_string()
)


print("\nScaled feature standard deviations:")

print(
    X_scaled.std(ddof=0)
    .round(6)
    .to_string()
)


# ============================================================
# SAVE SELECTED FEATURES
# ============================================================

selected_output = df[
    [
        "RECORD_ID",
        "STATE/UT",
        "DISTRICT",
        "YEAR"
    ]
    + selected_features
].copy()


selected_output.to_csv(
    FEATURE_FILE,
    index=False
)


# ============================================================
# SAVE SCALED DATA
# ============================================================

scaled_output = df[
    [
        "RECORD_ID",
        "STATE/UT",
        "DISTRICT",
        "YEAR"
    ]
].copy()


for feature in selected_features:
    scaled_output[feature] = X_scaled[feature]


scaled_output.to_csv(
    SCALED_FILE,
    index=False
)


# ============================================================
# FINAL CHECK
# ============================================================

print("\n" + "-" * 70)
print("6. FINAL K-MEANS MATRIX")
print("-" * 70)

print(
    f"\nRows    : {X_scaled.shape[0]}"
)

print(
    f"Features: {X_scaled.shape[1]}"
)


print("\nAny NaN in scaled matrix:")
print(X_scaled.isna().sum().sum())


print("\nAny infinite values in scaled matrix:")
print(np.isinf(X_scaled).sum().sum())


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("K-MEANS PREPARATION COMPLETED")
print("=" * 70)

print("\nSelected feature file:")
print(FEATURE_FILE)

print("\nScaled feature file:")
print(SCALED_FILE)

print("\nIMPORTANT:")
print("K-Means has NOT been executed yet.")

print("\nNext step:")
print("Evaluate K values using clustering metrics.")