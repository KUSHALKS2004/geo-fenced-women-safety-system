import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = (
    BASE_DIR
    / "models"
    / "dataset_01_mlp"
)

PREDICTIONS_FILE = (
    MODEL_DIR
    / "test_predictions_dataset_01.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "feature_engineering_01"
    / "mlp_dataset_01"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD
# ============================================================

print("=" * 70)
print("MLP ERROR ANALYSIS - DATASET 01")
print("=" * 70)

df = pd.read_csv(PREDICTIONS_FILE)

print(f"\nTest records: {len(df)}")


# ============================================================
# COLUMN NAMES
# ============================================================

TARGET = "NEXT_YEAR_WOMEN_CRIME_SHARE"

PREDICTION = (
    "PREDICTED_NEXT_YEAR_WOMEN_CRIME_SHARE"
)


# ============================================================
# CHECK
# ============================================================

required = [
    "STATE/UT",
    "DISTRICT",
    "YEAR",
    "NEXT_YEAR",
    TARGET,
    PREDICTION,
]

missing = [
    col for col in required
    if col not in df.columns
]

if missing:
    print("\nERROR: Missing columns:")

    for col in missing:
        print(f"  - {col}")

    raise SystemExit(1)


# ============================================================
# ERROR CALCULATIONS
# ============================================================

df["ERROR"] = (
    df[PREDICTION]
    - df[TARGET]
)

df["ABSOLUTE_ERROR"] = (
    abs(df["ERROR"])
)

df["SQUARED_ERROR"] = (
    df["ERROR"] ** 2
)

df["DIRECTION"] = np.where(
    df["ERROR"] > 0,
    "OVERPREDICTED",
    np.where(
        df["ERROR"] < 0,
        "UNDERPREDICTED",
        "EXACT"
    )
)


# ============================================================
# BASIC METRICS
# ============================================================

mae = mean_absolute_error(
    df[TARGET],
    df[PREDICTION]
)

rmse = np.sqrt(
    mean_squared_error(
        df[TARGET],
        df[PREDICTION]
    )
)

r2 = r2_score(
    df[TARGET],
    df[PREDICTION]
)

print("\n" + "=" * 70)
print("TEST METRICS")
print("=" * 70)

print(f"MAE:  {mae:.6f}")
print(f"RMSE: {rmse:.6f}")
print(f"R²:   {r2:.6f}")


# ============================================================
# PREDICTION RANGE
# ============================================================

print("\n" + "=" * 70)
print("PREDICTION RANGE")
print("=" * 70)

print(
    f"Actual minimum:     "
    f"{df[TARGET].min():.6f}"
)

print(
    f"Actual maximum:     "
    f"{df[TARGET].max():.6f}"
)

print(
    f"Predicted minimum:  "
    f"{df[PREDICTION].min():.6f}"
)

print(
    f"Predicted maximum:  "
    f"{df[PREDICTION].max():.6f}"
)


# ============================================================
# CHECK INVALID PREDICTIONS
# ============================================================

below_zero = (
    df[PREDICTION] < 0
).sum()

above_one = (
    df[PREDICTION] > 1
).sum()

print("\n" + "=" * 70)
print("PREDICTION BOUNDARY CHECK")
print("=" * 70)

print(
    f"Predictions < 0: "
    f"{below_zero}"
)

print(
    f"Predictions > 1: "
    f"{above_one}"
)


# ============================================================
# ERROR DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("ERROR DISTRIBUTION")
print("=" * 70)

print(
    df["ABSOLUTE_ERROR"]
    .describe()
    .to_string()
)


# ============================================================
# ERROR THRESHOLDS
# ============================================================

print("\n" + "=" * 70)
print("ERROR THRESHOLD COUNTS")
print("=" * 70)

thresholds = [
    0.01,
    0.02,
    0.03,
    0.05,
    0.10,
]

for threshold in thresholds:

    count = (
        df["ABSOLUTE_ERROR"] <= threshold
    ).sum()

    percentage = (
        count / len(df) * 100
    )

    print(
        f"Absolute error <= {threshold:.2f}: "
        f"{count:,} "
        f"({percentage:.2f}%)"
    )


# ============================================================
# OVER / UNDER PREDICTION
# ============================================================

print("\n" + "=" * 70)
print("PREDICTION DIRECTION")
print("=" * 70)

direction_counts = (
    df["DIRECTION"]
    .value_counts()
)

for direction, count in direction_counts.items():

    print(
        f"{direction}: "
        f"{count:,} "
        f"({count / len(df) * 100:.2f}%)"
    )


# ============================================================
# ERROR BY YEAR
# ============================================================

print("\n" + "=" * 70)
print("ERROR BY CURRENT YEAR")
print("=" * 70)

error_by_year = (
    df.groupby("YEAR")
    .agg(
        RECORDS=(TARGET, "size"),
        MAE=("ABSOLUTE_ERROR", "mean"),
        RMSE=("SQUARED_ERROR",
              lambda x: np.sqrt(x.mean())),
        ACTUAL_MEAN=(TARGET, "mean"),
        PREDICTED_MEAN=(PREDICTION, "mean"),
    )
    .reset_index()
)

print(
    error_by_year
    .to_string(index=False)
)


# ============================================================
# LARGEST ABSOLUTE ERRORS
# ============================================================

print("\n" + "=" * 70)
print("LARGEST ABSOLUTE ERRORS")
print("=" * 70)

largest_errors = (
    df.sort_values(
        "ABSOLUTE_ERROR",
        ascending=False
    )
    [
        [
            "STATE/UT",
            "DISTRICT",
            "YEAR",
            "NEXT_YEAR",
            TARGET,
            PREDICTION,
            "ERROR",
            "ABSOLUTE_ERROR",
        ]
    ]
    .head(20)
)

print(
    largest_errors
    .to_string(index=False)
)


# ============================================================
# SMALLEST ERRORS
# ============================================================

print("\n" + "=" * 70)
print("SMALLEST ABSOLUTE ERRORS")
print("=" * 70)

smallest_errors = (
    df.sort_values(
        "ABSOLUTE_ERROR",
        ascending=True
    )
    [
        [
            "STATE/UT",
            "DISTRICT",
            "YEAR",
            "NEXT_YEAR",
            TARGET,
            PREDICTION,
            "ERROR",
            "ABSOLUTE_ERROR",
        ]
    ]
    .head(20)
)

print(
    smallest_errors
    .to_string(index=False)
)


# ============================================================
# SAVE FULL ERROR ANALYSIS
# ============================================================

full_output = (
    OUTPUT_DIR
    / "mlp_test_error_analysis.csv"
)

df.to_csv(
    full_output,
    index=False
)


# ============================================================
# SAVE YEAR ANALYSIS
# ============================================================

year_output = (
    OUTPUT_DIR
    / "mlp_error_by_year.csv"
)

error_by_year.to_csv(
    year_output,
    index=False
)


# ============================================================
# SAVE EXTREME ERRORS
# ============================================================

extreme_output = (
    OUTPUT_DIR
    / "mlp_largest_errors.csv"
)

largest_errors.to_csv(
    extreme_output,
    index=False
)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("MLP ERROR ANALYSIS COMPLETE")
print("=" * 70)

print("\nFiles created:")

print(f"1. {full_output}")
print(f"2. {year_output}")
print(f"3. {extreme_output}")

print("\nNext step:")
print(
    "Use the error analysis to determine "
    "whether risk-band construction is justified."
)