import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = (
    BASE_DIR
    / "data"
    / "master_2001_2014"
    / "temporal_model"
    / "mlp"
)

OUTPUT_DIR = (
    BASE_DIR
    / "models"
    / "dataset_01_02_03_mlp"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# FILES
# ============================================================

TRAIN_FILE = DATA_DIR / "train_scaled_2001_2008.csv"
VAL_FILE = DATA_DIR / "validation_scaled_2009_2010.csv"
TEST_FILE = DATA_DIR / "test_scaled_2011_2013.csv"


TARGET = "NEXT_YEAR_WOMEN_CRIME_SHARE"


# ============================================================
# LOAD DATA
# ============================================================

train = pd.read_csv(TRAIN_FILE)
validation = pd.read_csv(VAL_FILE)
test = pd.read_csv(TEST_FILE)

print("=" * 80)
print("MLP REGRESSION MODEL")
print("SafeHer-AI | 2001–2014")
print("=" * 80)

print("\nTrain shape:", train.shape)
print("Validation shape:", validation.shape)
print("Test shape:", test.shape)


# ============================================================
# FEATURES
# ============================================================

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
# X / y
# ============================================================

X_train = train[FEATURES]
y_train = train[TARGET]

X_val = validation[FEATURES]
y_val = validation[TARGET]

X_test = test[FEATURES]
y_test = test[TARGET]


# ============================================================
# MODEL
# ============================================================

print("\n" + "=" * 80)
print("MODEL CONFIGURATION")
print("=" * 80)

model = MLPRegressor(
    hidden_layer_sizes=(64, 32, 16),
    activation="relu",
    solver="adam",
    alpha=0.0001,
    batch_size=64,
    learning_rate_init=0.001,
    max_iter=500,
    random_state=42,
    early_stopping=False
)

print("\nArchitecture:")
print("Input layer :", len(FEATURES))
print("Hidden 1    : 64 neurons")
print("Hidden 2    : 32 neurons")
print("Hidden 3    : 16 neurons")
print("Output      : 1")
print("Activation  : ReLU")
print("Optimizer   : Adam")


# ============================================================
# TRAIN
# ============================================================

print("\n" + "=" * 80)
print("TRAINING MLP")
print("=" * 80)

model.fit(
    X_train,
    y_train
)

print("\nTraining completed.")
print("Iterations:", model.n_iter_)
print("Final loss:", model.loss_)


# ============================================================
# PREDICTIONS
# ============================================================

train_pred = model.predict(X_train)
val_pred = model.predict(X_val)
test_pred = model.predict(X_test)


# ============================================================
# METRICS FUNCTION
# ============================================================

def calculate_metrics(y_true, y_pred):

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred
        )
    )

    r2 = r2_score(
        y_true,
        y_pred
    )

    return mae, rmse, r2


# ============================================================
# METRICS
# ============================================================

train_mae, train_rmse, train_r2 = calculate_metrics(
    y_train,
    train_pred
)

val_mae, val_rmse, val_r2 = calculate_metrics(
    y_val,
    val_pred
)

test_mae, test_rmse, test_r2 = calculate_metrics(
    y_test,
    test_pred
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 80)
print("MODEL PERFORMANCE")
print("=" * 80)

print("\nTRAIN")
print("MAE  :", round(train_mae, 6))
print("RMSE :", round(train_rmse, 6))
print("R²   :", round(train_r2, 6))

print("\nVALIDATION")
print("MAE  :", round(val_mae, 6))
print("RMSE :", round(val_rmse, 6))
print("R²   :", round(val_r2, 6))

print("\nTEST")
print("MAE  :", round(test_mae, 6))
print("RMSE :", round(test_rmse, 6))
print("R²   :", round(test_r2, 6))


# ============================================================
# PREDICTION RANGE
# ============================================================

print("\n" + "=" * 80)
print("PREDICTION RANGE")
print("=" * 80)

print("\nTrain prediction:")
print("Min:", train_pred.min())
print("Max:", train_pred.max())

print("\nValidation prediction:")
print("Min:", val_pred.min())
print("Max:", val_pred.max())

print("\nTest prediction:")
print("Min:", test_pred.min())
print("Max:", test_pred.max())


# ============================================================
# CLIPPED PREDICTIONS
# ============================================================

# Women-crime share must lie between 0 and 1.

test_pred_clipped = np.clip(
    test_pred,
    0,
    1
)

val_pred_clipped = np.clip(
    val_pred,
    0,
    1
)

train_pred_clipped = np.clip(
    train_pred,
    0,
    1
)


# ============================================================
# CLIPPED METRICS
# ============================================================

train_clipped_metrics = calculate_metrics(
    y_train,
    train_pred_clipped
)

val_clipped_metrics = calculate_metrics(
    y_val,
    val_pred_clipped
)

test_clipped_metrics = calculate_metrics(
    y_test,
    test_pred_clipped
)

print("\n" + "=" * 80)
print("CLIPPED MODEL PERFORMANCE")
print("=" * 80)

print("\nTRAIN")
print("MAE  :", round(train_clipped_metrics[0], 6))
print("RMSE :", round(train_clipped_metrics[1], 6))
print("R²   :", round(train_clipped_metrics[2], 6))

print("\nVALIDATION")
print("MAE  :", round(val_clipped_metrics[0], 6))
print("RMSE :", round(val_clipped_metrics[1], 6))
print("R²   :", round(val_clipped_metrics[2], 6))

print("\nTEST")
print("MAE  :", round(test_clipped_metrics[0], 6))
print("RMSE :", round(test_clipped_metrics[1], 6))
print("R²   :", round(test_clipped_metrics[2], 6))


# ============================================================
# SAVE METRICS
# ============================================================

metrics = pd.DataFrame({
    "SPLIT": [
        "TRAIN",
        "VALIDATION",
        "TEST",
        "TRAIN_CLIPPED",
        "VALIDATION_CLIPPED",
        "TEST_CLIPPED"
    ],
    "MAE": [
        train_mae,
        val_mae,
        test_mae,
        train_clipped_metrics[0],
        val_clipped_metrics[0],
        test_clipped_metrics[0]
    ],
    "RMSE": [
        train_rmse,
        val_rmse,
        test_rmse,
        train_clipped_metrics[1],
        val_clipped_metrics[1],
        test_clipped_metrics[1]
    ],
    "R2": [
        train_r2,
        val_r2,
        test_r2,
        train_clipped_metrics[2],
        val_clipped_metrics[2],
        test_clipped_metrics[2]
    ]
})

metrics.to_csv(
    OUTPUT_DIR / "mlp_performance_metrics.csv",
    index=False
)


# ============================================================
# SAVE TEST PREDICTIONS
# ============================================================

prediction_output = test[
    ["YEAR", "TARGET_YEAR"]
].copy()

prediction_output["ACTUAL_WOMEN_CRIME_SHARE"] = y_test.values
prediction_output["PREDICTED_WOMEN_CRIME_SHARE"] = test_pred
prediction_output["PREDICTED_WOMEN_CRIME_SHARE_CLIPPED"] = (
    test_pred_clipped
)

prediction_output["ABSOLUTE_ERROR"] = np.abs(
    prediction_output["ACTUAL_WOMEN_CRIME_SHARE"]
    -
    prediction_output["PREDICTED_WOMEN_CRIME_SHARE_CLIPPED"]
)

prediction_output.to_csv(
    OUTPUT_DIR / "mlp_test_predictions.csv",
    index=False
)


# ============================================================
# SAVE MODEL
# ============================================================

model_file = (
    OUTPUT_DIR
    / "mlp_women_crime_share_2001_2014.joblib"
)

joblib.dump(
    model,
    model_file
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("MLP TRAINING COMPLETED")
print("=" * 80)

print("\nModel saved to:")
print(model_file)

print("\nMetrics saved to:")
print(
    OUTPUT_DIR / "mlp_performance_metrics.csv"
)

print("\nTest predictions saved to:")
print(
    OUTPUT_DIR / "mlp_test_predictions.csv"
)

print("\nNext step:")
print("Risk-band calibration and evaluation.")