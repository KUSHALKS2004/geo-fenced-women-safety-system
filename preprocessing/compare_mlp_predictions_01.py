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

ORIGINAL_FILE = (
    BASE_DIR
    / "models"
    / "dataset_01_mlp"
    / "test_predictions_dataset_01.csv"
)

BOUNDED_FILE = (
    BASE_DIR
    / "models"
    / "dataset_01_bounded_mlp"
    / "test_predictions_bounded_dataset_01.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "models"
    / "dataset_01_mlp_comparison"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD
# ============================================================

original = pd.read_csv(
    ORIGINAL_FILE
)

bounded = pd.read_csv(
    BOUNDED_FILE
)

TARGET = "NEXT_YEAR_WOMEN_CRIME_SHARE"

ORIGINAL_PREDICTION = (
    "PREDICTED_NEXT_YEAR_WOMEN_CRIME_SHARE"
)

BOUNDED_PREDICTION = (
    "BOUNDED_PREDICTED_NEXT_YEAR_WOMEN_CRIME_SHARE"
)


# ============================================================
# CLIP ORIGINAL PREDICTIONS
# ============================================================

original[
    "CLIPPED_PREDICTION"
] = np.clip(
    original[ORIGINAL_PREDICTION],
    0,
    1
)


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate(
    name,
    actual,
    predicted
):

    mae = mean_absolute_error(
        actual,
        predicted
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted
        )
    )

    r2 = r2_score(
        actual,
        predicted
    )

    print(f"\n{name}")

    print(f"MAE:  {mae:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"R²:   {r2:.6f}")

    return mae, rmse, r2


# ============================================================
# ORIGINAL
# ============================================================

print("=" * 70)
print("DATASET 01 MLP MODEL COMPARISON")
print("=" * 70)

original_metrics = evaluate(
    "ORIGINAL MLP",
    original[TARGET],
    original[ORIGINAL_PREDICTION]
)


# ============================================================
# CLIPPED ORIGINAL
# ============================================================

clipped_metrics = evaluate(
    "ORIGINAL MLP + OUTPUT CLIPPING",
    original[TARGET],
    original["CLIPPED_PREDICTION"]
)


# ============================================================
# BOUNDED
# ============================================================

bounded_metrics = evaluate(
    "BOUNDED MLP",
    bounded[TARGET],
    bounded[BOUNDED_PREDICTION]
)


# ============================================================
# BOUNDARY CHECK
# ============================================================

print("\n" + "=" * 70)
print("BOUNDARY CHECK")
print("=" * 70)

print(
    "Original predictions < 0:",
    (original[ORIGINAL_PREDICTION] < 0).sum()
)

print(
    "Original predictions > 1:",
    (original[ORIGINAL_PREDICTION] > 1).sum()
)

print(
    "Clipped predictions < 0:",
    (original["CLIPPED_PREDICTION"] < 0).sum()
)

print(
    "Clipped predictions > 1:",
    (original["CLIPPED_PREDICTION"] > 1).sum()
)


# ============================================================
# COMPARISON TABLE
# ============================================================

comparison = pd.DataFrame(
    {
        "MODEL": [
            "Original MLP",
            "Original MLP + Clipping",
            "Bounded MLP",
        ],
        "MAE": [
            original_metrics[0],
            clipped_metrics[0],
            bounded_metrics[0],
        ],
        "RMSE": [
            original_metrics[1],
            clipped_metrics[1],
            bounded_metrics[1],
        ],
        "R2": [
            original_metrics[2],
            clipped_metrics[2],
            bounded_metrics[2],
        ],
    }
)


# ============================================================
# PRINT
# ============================================================

print("\n" + "=" * 70)
print("FINAL COMPARISON")
print("=" * 70)

print(
    comparison.to_string(
        index=False
    )
)


# ============================================================
# SAVE
# ============================================================

comparison_file = (
    OUTPUT_DIR
    / "dataset_01_mlp_model_comparison.csv"
)

comparison.to_csv(
    comparison_file,
    index=False
)


# ============================================================
# SAVE CLIPPED PREDICTIONS
# ============================================================

clipped_output = (
    OUTPUT_DIR
    / "dataset_01_clipped_predictions.csv"
)

clipped_results = original[
    [
        "STATE/UT",
        "DISTRICT",
        "YEAR",
        "NEXT_YEAR",
        TARGET,
        ORIGINAL_PREDICTION,
    ]
].copy()

clipped_results[
    "CLIPPED_PREDICTION"
] = original[
    "CLIPPED_PREDICTION"
]

clipped_results[
    "CLIPPED_ABSOLUTE_ERROR"
] = abs(
    clipped_results[TARGET]
    -
    clipped_results["CLIPPED_PREDICTION"]
)

clipped_results.to_csv(
    clipped_output,
    index=False
)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("COMPARISON COMPLETE")
print("=" * 70)

print(f"\nSaved:")
print(comparison_file)
print(clipped_output)

print("\nNext:")
print(
    "Use the comparison to select the Dataset 01 "
    "prediction output before constructing risk bands."
)