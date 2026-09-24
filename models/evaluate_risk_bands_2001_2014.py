import pandas as pd
import numpy as np
from pathlib import Path
import joblib

from sklearn.metrics import confusion_matrix


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

MODEL_DIR = (
    BASE_DIR
    / "models"
    / "dataset_01_02_03_mlp"
)

OUTPUT_DIR = (
    BASE_DIR
    / "models"
    / "dataset_01_02_03_risk_bands"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# FILES
# ============================================================

TRAIN_FILE = DATA_DIR / "train_scaled_2001_2008.csv"
VAL_FILE = DATA_DIR / "validation_scaled_2009_2010.csv"
TEST_FILE = DATA_DIR / "test_scaled_2011_2013.csv"

MODEL_FILE = (
    MODEL_DIR
    / "mlp_women_crime_share_2001_2014.joblib"
)

TARGET = "NEXT_YEAR_WOMEN_CRIME_SHARE"


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
# LOAD
# ============================================================

print("=" * 80)
print("RISK-BAND CALIBRATION AND EVALUATION - FIXED")
print("SafeHer-AI | 2001–2014")
print("=" * 80)

train = pd.read_csv(TRAIN_FILE)
validation = pd.read_csv(VAL_FILE)
test = pd.read_csv(TEST_FILE)

model = joblib.load(MODEL_FILE)

print("\nTrain:", train.shape)
print("Validation:", validation.shape)
print("Test:", test.shape)


# ============================================================
# CHECK EXACT FEATURES
# ============================================================

print("\n" + "=" * 80)
print("FEATURE CHECK")
print("=" * 80)

for name, data in [
    ("TRAIN", train),
    ("VALIDATION", validation),
    ("TEST", test)
]:

    missing = [
        col for col in FEATURES
        if col not in data.columns
    ]

    if missing:
        raise ValueError(
            f"{name} is missing features: {missing}"
        )

print("\nAll 30 MLP features are present.")


# ============================================================
# CHECK TARGET
# ============================================================

for name, data in [
    ("TRAIN", train),
    ("VALIDATION", validation),
    ("TEST", test)
]:

    if TARGET not in data.columns:
        raise ValueError(
            f"{TARGET} missing from {name}"
        )


# ============================================================
# EXACT TRAINING INPUTS
# ============================================================
#
# IMPORTANT:
# These are already scaled by the original preparation script.
#
# DO NOT fit another scaler here.
#

X_train = train[FEATURES].copy()
X_val = validation[FEATURES].copy()
X_test = test[FEATURES].copy()

y_train = train[TARGET].copy()
y_val = validation[TARGET].copy()
y_test = test[TARGET].copy()


# ============================================================
# PREDICTIONS
# ============================================================

print("\n" + "=" * 80)
print("GENERATING PREDICTIONS")
print("=" * 80)

train_raw_prediction = model.predict(X_train)
val_raw_prediction = model.predict(X_val)
test_raw_prediction = model.predict(X_test)


# ============================================================
# VERIFY AGAINST ORIGINAL MLP RANGE
# ============================================================

print("\nRaw prediction ranges:")

print(
    "Train:",
    train_raw_prediction.min(),
    "to",
    train_raw_prediction.max()
)

print(
    "Validation:",
    val_raw_prediction.min(),
    "to",
    val_raw_prediction.max()
)

print(
    "Test:",
    test_raw_prediction.min(),
    "to",
    test_raw_prediction.max()
)


# ============================================================
# CLIP TO VALID SHARE RANGE
# ============================================================

train_prediction = np.clip(
    train_raw_prediction,
    0,
    1
)

val_prediction = np.clip(
    val_raw_prediction,
    0,
    1
)

test_prediction = np.clip(
    test_raw_prediction,
    0,
    1
)


# ============================================================
# PREDICTION SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("CLIPPED PREDICTION SUMMARY")
print("=" * 80)

for name, prediction in [
    ("TRAIN", train_prediction),
    ("VALIDATION", val_prediction),
    ("TEST", test_prediction)
]:

    print(f"\n{name}")

    print(
        "Min:",
        round(float(prediction.min()), 6)
    )

    print(
        "Mean:",
        round(float(prediction.mean()), 6)
    )

    print(
        "Median:",
        round(float(np.median(prediction)), 6)
    )

    print(
        "Max:",
        round(float(prediction.max()), 6)
    )


# ============================================================
# THRESHOLDS
# ============================================================
#
# Thresholds are calculated ONLY from training targets.
#
# 33rd percentile -> LOW/MEDIUM
# 66th percentile -> MEDIUM/HIGH
#

low_medium_threshold = y_train.quantile(0.33)

medium_high_threshold = y_train.quantile(0.66)

print("\n" + "=" * 80)
print("TRAINING-ONLY RISK THRESHOLDS")
print("=" * 80)

print(
    "\nLOW / MEDIUM:",
    round(float(low_medium_threshold), 6)
)

print(
    "MEDIUM / HIGH:",
    round(float(medium_high_threshold), 6)
)


# ============================================================
# BAND FUNCTION
# ============================================================

def assign_risk_band(values):

    return np.select(
        [
            values <= low_medium_threshold,
            values <= medium_high_threshold
        ],
        [
            "LOW",
            "MEDIUM"
        ],
        default="HIGH"
    )


# ============================================================
# ACTUAL BANDS
# ============================================================

train_actual = assign_risk_band(
    y_train.values
)

val_actual = assign_risk_band(
    y_val.values
)

test_actual = assign_risk_band(
    y_test.values
)


# ============================================================
# PREDICTED BANDS
# ============================================================

train_predicted = assign_risk_band(
    train_prediction
)

val_predicted = assign_risk_band(
    val_prediction
)

test_predicted = assign_risk_band(
    test_prediction
)


# ============================================================
# DISTRIBUTION FUNCTION
# ============================================================

LABELS = [
    "LOW",
    "MEDIUM",
    "HIGH"
]


def distribution(actual, predicted):

    actual_counts = (
        pd.Series(actual)
        .value_counts()
        .reindex(
            LABELS,
            fill_value=0
        )
    )

    predicted_counts = (
        pd.Series(predicted)
        .value_counts()
        .reindex(
            LABELS,
            fill_value=0
        )
    )

    return actual_counts, predicted_counts


# ============================================================
# PRINT DISTRIBUTIONS
# ============================================================

print("\n" + "=" * 80)
print("RISK-BAND DISTRIBUTIONS")
print("=" * 80)

for name, actual, predicted in [
    ("TRAIN", train_actual, train_predicted),
    ("VALIDATION", val_actual, val_predicted),
    ("TEST", test_actual, test_predicted)
]:

    actual_counts, predicted_counts = distribution(
        actual,
        predicted
    )

    print("\n" + "-" * 60)
    print(name)
    print("-" * 60)

    print("\nActual:")
    print(actual_counts)

    print("\nPredicted:")
    print(predicted_counts)


# ============================================================
# CONFUSION MATRIX
# ============================================================

def evaluate(
    name,
    actual,
    predicted
):

    matrix = confusion_matrix(
        actual,
        predicted,
        labels=LABELS
    )

    agreement = (
        np.trace(matrix)
        /
        matrix.sum()
        *
        100
    )

    print("\n" + "=" * 80)
    print(name)
    print("=" * 80)

    matrix_df = pd.DataFrame(
        matrix,
        index=[
            "Actual LOW",
            "Actual MEDIUM",
            "Actual HIGH"
        ],
        columns=[
            "Pred LOW",
            "Pred MEDIUM",
            "Pred HIGH"
        ]
    )

    print("\nConfusion matrix:")
    print(matrix_df)

    print(
        "\nExact risk-band agreement:",
        round(float(agreement), 2),
        "%"
    )

    return matrix, agreement


train_matrix, train_agreement = evaluate(
    "TRAIN RISK-BAND EVALUATION",
    train_actual,
    train_predicted
)

val_matrix, val_agreement = evaluate(
    "VALIDATION RISK-BAND EVALUATION",
    val_actual,
    val_predicted
)

test_matrix, test_agreement = evaluate(
    "TEST RISK-BAND EVALUATION",
    test_actual,
    test_predicted
)


# ============================================================
# TEST OUTPUT
# ============================================================

test_output = test[
    [
        "YEAR",
        "TARGET_YEAR"
    ]
].copy()

test_output[
    "ACTUAL_WOMEN_CRIME_SHARE"
] = y_test.values

test_output[
    "PREDICTED_WOMEN_CRIME_SHARE"
] = test_prediction

test_output[
    "ACTUAL_RISK_BAND"
] = test_actual

test_output[
    "PREDICTED_RISK_BAND"
] = test_predicted

test_output[
    "ABSOLUTE_SHARE_ERROR"
] = np.abs(
    y_test.values
    -
    test_prediction
)

test_output.to_csv(
    OUTPUT_DIR
    / "test_risk_band_predictions.csv",
    index=False
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

test_matrix_df = pd.DataFrame(
    test_matrix,
    index=[
        "Actual LOW",
        "Actual MEDIUM",
        "Actual HIGH"
    ],
    columns=[
        "Pred LOW",
        "Pred MEDIUM",
        "Pred HIGH"
    ]
)

test_matrix_df.to_csv(
    OUTPUT_DIR
    / "test_risk_band_confusion_matrix.csv"
)


# ============================================================
# THRESHOLDS
# ============================================================

threshold_df = pd.DataFrame({
    "THRESHOLD": [
        "LOW_MEDIUM",
        "MEDIUM_HIGH"
    ],
    "VALUE": [
        low_medium_threshold,
        medium_high_threshold
    ],
    "SOURCE": [
        "TRAINING TARGET 33rd PERCENTILE",
        "TRAINING TARGET 66th PERCENTILE"
    ]
})

threshold_df.to_csv(
    OUTPUT_DIR
    / "risk_band_thresholds.csv",
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

summary = pd.DataFrame({
    "SPLIT": [
        "TRAIN",
        "VALIDATION",
        "TEST"
    ],
    "ROWS": [
        len(train),
        len(validation),
        len(test)
    ],
    "EXACT_BAND_AGREEMENT_PERCENT": [
        train_agreement,
        val_agreement,
        test_agreement
    ]
})

summary.to_csv(
    OUTPUT_DIR
    / "risk_band_summary.csv",
    index=False
)


# ============================================================
# COMPLETION
# ============================================================

print("\n" + "=" * 80)
print("FIXED RISK-BAND EVALUATION COMPLETED")
print("=" * 80)

print("\nThresholds:")
print(
    "LOW    <= ",
    round(float(low_medium_threshold), 6)
)

print(
    "MEDIUM > ",
    round(float(low_medium_threshold), 6),
    "and <= ",
    round(float(medium_high_threshold), 6)
)

print(
    "HIGH   > ",
    round(float(medium_high_threshold), 6)
)

print(
    "\nTEST EXACT BAND AGREEMENT:",
    round(float(test_agreement), 2),
    "%"
)

print("\nSaved to:")
print(OUTPUT_DIR)