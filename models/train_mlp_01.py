import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = (
    BASE_DIR
    / "data"
    / "feature_engineering_01"
    / "mlp_dataset_01"
)

MODEL_DIR = (
    BASE_DIR
    / "models"
    / "dataset_01_mlp"
)

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# FILES
# ============================================================

TRAIN_FILE = DATA_DIR / "train_dataset_01.csv"
VALIDATION_FILE = DATA_DIR / "validation_dataset_01.csv"
TEST_FILE = DATA_DIR / "test_dataset_01.csv"


# ============================================================
# FEATURES AND TARGET
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

TARGET = "NEXT_YEAR_WOMEN_CRIME_SHARE"


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("MLP TRAINING - DATASET 01")
print("=" * 70)

train_df = pd.read_csv(TRAIN_FILE)
validation_df = pd.read_csv(VALIDATION_FILE)
test_df = pd.read_csv(TEST_FILE)

print("\nDataset sizes:")
print(f"Train:      {len(train_df):,}")
print(f"Validation: {len(validation_df):,}")
print(f"Test:       {len(test_df):,}")


# ============================================================
# CREATE X / y
# ============================================================

X_train = train_df[FEATURES].copy()
y_train = train_df[TARGET].copy()

X_validation = validation_df[FEATURES].copy()
y_validation = validation_df[TARGET].copy()

X_test = test_df[FEATURES].copy()
y_test = test_df[TARGET].copy()


# ============================================================
# FEATURE SCALING
# ============================================================
#
# IMPORTANT:
# StandardScaler is fitted ONLY on training data.
#
# Validation and test data are transformed using
# the training scaler.
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
# VERIFY SCALING
# ============================================================

print("\n" + "=" * 70)
print("SCALING CHECK")
print("=" * 70)

print(
    f"Training scaled mean: "
    f"{X_train_scaled.mean():.6f}"
)

print(
    f"Training scaled std:  "
    f"{X_train_scaled.std():.6f}"
)


# ============================================================
# BASELINE
# ============================================================
#
# Predict the training-set mean for every validation/test row.
# This gives us a simple benchmark.
# ============================================================

baseline_prediction = np.full(
    len(y_test),
    y_train.mean()
)

baseline_mae = mean_absolute_error(
    y_test,
    baseline_prediction
)

baseline_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        baseline_prediction
    )
)

baseline_r2 = r2_score(
    y_test,
    baseline_prediction
)

print("\n" + "=" * 70)
print("BASELINE - TEST SET")
print("=" * 70)

print(f"Baseline MAE:  {baseline_mae:.6f}")
print(f"Baseline RMSE: {baseline_rmse:.6f}")
print(f"Baseline R²:   {baseline_r2:.6f}")


# ============================================================
# MLP MODEL
# ============================================================

print("\n" + "=" * 70)
print("BUILDING MLP")
print("=" * 70)

model = MLPRegressor(
    hidden_layer_sizes=(64, 32, 16),
    activation="relu",
    solver="adam",
    alpha=0.0001,
    batch_size=64,
    learning_rate_init=0.001,
    max_iter=500,
    random_state=42,
    early_stopping=False,
    verbose=True
)


# ============================================================
# TRAIN
# ============================================================

print("\nTraining MLP...")

model.fit(
    X_train_scaled,
    y_train
)

print("\nTraining complete.")


# ============================================================
# PREDICTIONS
# ============================================================

train_predictions = model.predict(
    X_train_scaled
)

validation_predictions = model.predict(
    X_validation_scaled
)

test_predictions = model.predict(
    X_test_scaled
)


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate_model(
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
# MODEL EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("MLP PERFORMANCE")
print("=" * 70)

train_metrics = evaluate_model(
    "TRAIN",
    y_train,
    train_predictions
)

validation_metrics = evaluate_model(
    "VALIDATION",
    y_validation,
    validation_predictions
)

test_metrics = evaluate_model(
    "TEST",
    y_test,
    test_predictions
)


# ============================================================
# COMPARE WITH BASELINE
# ============================================================

print("\n" + "=" * 70)
print("MLP VS BASELINE")
print("=" * 70)

print(
    f"\nBaseline Test MAE: "
    f"{baseline_mae:.6f}"
)

print(
    f"MLP Test MAE:      "
    f"{test_metrics[0]:.6f}"
)

print(
    f"\nBaseline Test RMSE: "
    f"{baseline_rmse:.6f}"
)

print(
    f"MLP Test RMSE:      "
    f"{test_metrics[1]:.6f}"
)

print(
    f"\nBaseline Test R²: "
    f"{baseline_r2:.6f}"
)

print(
    f"MLP Test R²:     "
    f"{test_metrics[2]:.6f}"
)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

results = test_df[
    [
        "STATE/UT",
        "DISTRICT",
        "YEAR",
        "NEXT_YEAR",
        TARGET,
    ]
].copy()

results["PREDICTED_NEXT_YEAR_WOMEN_CRIME_SHARE"] = (
    test_predictions
)

results["ABSOLUTE_ERROR"] = (
    abs(
        results[TARGET]
        - results[
            "PREDICTED_NEXT_YEAR_WOMEN_CRIME_SHARE"
        ]
    )
)

results_file = (
    MODEL_DIR
    / "test_predictions_dataset_01.csv"
)

results.to_csv(
    results_file,
    index=False
)


# ============================================================
# SAVE METRICS
# ============================================================

metrics = pd.DataFrame(
    {
        "MODEL": [
            "Baseline",
            "MLP",
        ],
        "MAE": [
            baseline_mae,
            test_metrics[0],
        ],
        "RMSE": [
            baseline_rmse,
            test_metrics[1],
        ],
        "R2": [
            baseline_r2,
            test_metrics[2],
        ],
    }
)

metrics_file = (
    MODEL_DIR
    / "model_metrics_dataset_01.csv"
)

metrics.to_csv(
    metrics_file,
    index=False
)


# ============================================================
# SAVE MODEL
# ============================================================

model_file = (
    MODEL_DIR
    / "mlp_model_dataset_01.joblib"
)

scaler_file = (
    MODEL_DIR
    / "mlp_scaler_dataset_01.joblib"
)

joblib.dump(
    model,
    model_file
)

joblib.dump(
    scaler,
    scaler_file
)


# ============================================================
# SAVE FEATURE LIST
# ============================================================

feature_file = (
    MODEL_DIR
    / "model_features_dataset_01.csv"
)

pd.DataFrame(
    {
        "FEATURE_NUMBER": range(
            1,
            len(FEATURES) + 1
        ),
        "FEATURE": FEATURES,
    }
).to_csv(
    feature_file,
    index=False
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("MLP TRAINING COMPLETE")
print("=" * 70)

print("\nFiles created:")

print(f"1. {results_file}")
print(f"2. {metrics_file}")
print(f"3. {model_file}")
print(f"4. {scaler_file}")
print(f"5. {feature_file}")

print("\nTarget:")
print(TARGET)

print("\nThe model predicts:")
print("Next-year women-crime share")

print("\nNext step:")
print("Analyze MLP predictions and errors before creating risk bands.")