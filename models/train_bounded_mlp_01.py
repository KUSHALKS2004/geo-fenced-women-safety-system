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
    / "dataset_01_bounded_mlp"
)

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# DATA FILES
# ============================================================

TRAIN_FILE = DATA_DIR / "train_dataset_01.csv"
VALIDATION_FILE = DATA_DIR / "validation_dataset_01.csv"
TEST_FILE = DATA_DIR / "test_dataset_01.csv"


# ============================================================
# FEATURES
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
# LOGIT / SIGMOID FUNCTIONS
# ============================================================

def safe_logit(y):
    """
    Convert a proportion in [0,1] to log-odds.

    Values are clipped slightly away from exactly 0 and 1
    so that logit remains numerically stable.
    """

    y = np.asarray(y, dtype=float)

    eps = 1e-6

    y = np.clip(
        y,
        eps,
        1 - eps
    )

    return np.log(
        y / (1 - y)
    )


def sigmoid(x):
    """
    Convert unrestricted model output back into [0,1].
    """

    x = np.asarray(x, dtype=float)

    # Prevent numerical overflow
    x = np.clip(
        x,
        -50,
        50
    )

    return 1 / (
        1 + np.exp(-x)
    )


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("BOUNDED MLP TRAINING - DATASET 01")
print("=" * 70)

train_df = pd.read_csv(TRAIN_FILE)
validation_df = pd.read_csv(VALIDATION_FILE)
test_df = pd.read_csv(TEST_FILE)

print("\nDataset sizes:")
print(f"Train:      {len(train_df):,}")
print(f"Validation: {len(validation_df):,}")
print(f"Test:       {len(test_df):,}")


# ============================================================
# X / y
# ============================================================

X_train = train_df[FEATURES].copy()
X_validation = validation_df[FEATURES].copy()
X_test = test_df[FEATURES].copy()

y_train = train_df[TARGET].copy()
y_validation = validation_df[TARGET].copy()
y_test = test_df[TARGET].copy()


# ============================================================
# SCALE FEATURES
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_validation_scaled = scaler.transform(
    X_validation
)

X_test_scaled = scaler.transform(
    X_test
)


# ============================================================
# TRANSFORM TARGET
# ============================================================

y_train_logit = safe_logit(
    y_train.values
)


# ============================================================
# BUILD MODEL
# ============================================================

print("\n" + "=" * 70)
print("BUILDING BOUNDED MLP")
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

print("\nTraining bounded MLP...")

model.fit(
    X_train_scaled,
    y_train_logit
)

print("\nTraining complete.")


# ============================================================
# RAW LOGIT PREDICTIONS
# ============================================================

train_raw = model.predict(
    X_train_scaled
)

validation_raw = model.predict(
    X_validation_scaled
)

test_raw = model.predict(
    X_test_scaled
)


# ============================================================
# CONVERT BACK TO SHARE
# ============================================================

train_predictions = sigmoid(
    train_raw
)

validation_predictions = sigmoid(
    validation_raw
)

test_predictions = sigmoid(
    test_raw
)


# ============================================================
# METRIC FUNCTION
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
# PERFORMANCE
# ============================================================

print("\n" + "=" * 70)
print("BOUNDED MLP PERFORMANCE")
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
# BOUNDARY CHECK
# ============================================================

print("\n" + "=" * 70)
print("PREDICTION BOUNDARY CHECK")
print("=" * 70)

below_zero = (
    test_predictions < 0
).sum()

above_one = (
    test_predictions > 1
).sum()

print(
    f"Predictions < 0: {below_zero}"
)

print(
    f"Predictions > 1: {above_one}"
)

print(
    f"Minimum prediction: "
    f"{test_predictions.min():.6f}"
)

print(
    f"Maximum prediction: "
    f"{test_predictions.max():.6f}"
)


# ============================================================
# COMPARE WITH ORIGINAL MLP
# ============================================================

ORIGINAL_METRICS_FILE = (
    BASE_DIR
    / "models"
    / "dataset_01_mlp"
    / "model_metrics_dataset_01.csv"
)

if ORIGINAL_METRICS_FILE.exists():

    original_metrics = pd.read_csv(
        ORIGINAL_METRICS_FILE
    )

    original_test = original_metrics[
        original_metrics["MODEL"] == "MLP"
    ]

    if len(original_test) > 0:

        original_mae = float(
            original_test["MAE"].iloc[0]
        )

        original_rmse = float(
            original_test["RMSE"].iloc[0]
        )

        original_r2 = float(
            original_test["R2"].iloc[0]
        )

        print("\n" + "=" * 70)
        print("ORIGINAL MLP VS BOUNDED MLP")
        print("=" * 70)

        print("\nOriginal MLP:")
        print(f"MAE:  {original_mae:.6f}")
        print(f"RMSE: {original_rmse:.6f}")
        print(f"R²:   {original_r2:.6f}")

        print("\nBounded MLP:")
        print(f"MAE:  {test_metrics[0]:.6f}")
        print(f"RMSE: {test_metrics[1]:.6f}")
        print(f"R²:   {test_metrics[2]:.6f}")


# ============================================================
# SAVE TEST PREDICTIONS
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

results[
    "BOUNDED_PREDICTED_NEXT_YEAR_WOMEN_CRIME_SHARE"
] = test_predictions

results["ABSOLUTE_ERROR"] = (
    abs(
        results[TARGET]
        -
        results[
            "BOUNDED_PREDICTED_NEXT_YEAR_WOMEN_CRIME_SHARE"
        ]
    )
)

results_file = (
    MODEL_DIR
    / "test_predictions_bounded_dataset_01.csv"
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
            "Bounded MLP"
        ],
        "MAE": [
            test_metrics[0]
        ],
        "RMSE": [
            test_metrics[1]
        ],
        "R2": [
            test_metrics[2]
        ],
        "PREDICTIONS_BELOW_ZERO": [
            below_zero
        ],
        "PREDICTIONS_ABOVE_ONE": [
            above_one
        ],
    }
)

metrics_file = (
    MODEL_DIR
    / "bounded_model_metrics_dataset_01.csv"
)

metrics.to_csv(
    metrics_file,
    index=False
)


# ============================================================
# SAVE MODEL + SCALER
# ============================================================

model_file = (
    MODEL_DIR
    / "bounded_mlp_model_dataset_01.joblib"
)

scaler_file = (
    MODEL_DIR
    / "bounded_mlp_scaler_dataset_01.joblib"
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
    / "bounded_mlp_features_dataset_01.csv"
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
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("BOUNDED MLP TRAINING COMPLETE")
print("=" * 70)

print("\nFiles created:")

print(f"1. {results_file}")
print(f"2. {metrics_file}")
print(f"3. {model_file}")
print(f"4. {scaler_file}")
print(f"5. {feature_file}")

print("\nTarget:")
print(TARGET)

print("\nOutput constraint:")
print("Predicted women-crime share is constrained to [0, 1].")

print("\nNext step:")
print("Compare the bounded MLP with the original MLP.")