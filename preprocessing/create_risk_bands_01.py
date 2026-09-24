import pandas as pd
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

TRAIN_FILE = (
    BASE_DIR
    / "data"
    / "feature_engineering_01"
    / "mlp_dataset_01"
    / "train_dataset_01.csv"
)

TEST_FILE = (
    BASE_DIR
    / "data"
    / "feature_engineering_01"
    / "mlp_dataset_01"
    / "test_dataset_01.csv"
)

PREDICTION_FILE = (
    BASE_DIR
    / "models"
    / "dataset_01_mlp_comparison"
    / "dataset_01_clipped_predictions.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "models"
    / "dataset_01_risk_bands"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# COLUMN NAMES
# ============================================================

TARGET = "NEXT_YEAR_WOMEN_CRIME_SHARE"

KEY_COLUMNS = [
    "STATE/UT",
    "DISTRICT",
    "YEAR",
    "NEXT_YEAR"
]


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("DATASET 1 - RISK BAND CONSTRUCTION")
print("=" * 70)

print("\nLoading training data...")
train = pd.read_csv(TRAIN_FILE)

print("Loading test data...")
test = pd.read_csv(TEST_FILE)

print("Loading clipped MLP predictions...")
predictions = pd.read_csv(PREDICTION_FILE)


print("\nTraining shape:", train.shape)
print("Test shape:", test.shape)
print("Prediction shape:", predictions.shape)


# ============================================================
# DISPLAY PREDICTION COLUMNS
# ============================================================

print("\nPrediction file columns:")
print(predictions.columns.tolist())


# ============================================================
# IDENTIFY PREDICTION COLUMN
# ============================================================

possible_prediction_columns = [
    "CLIPPED_PREDICTION",
    "clipped_prediction",
    "PREDICTED_WOMEN_CRIME_SHARE",
    "PREDICTION",
    "prediction"
]

prediction_column = None

for column in possible_prediction_columns:
    if column in predictions.columns:
        prediction_column = column
        break


if prediction_column is None:
    raise ValueError(
        "\nCould not find the clipped prediction column.\n"
        "Please check the prediction file columns printed above."
    )


print("\nUsing prediction column:", prediction_column)


# ============================================================
# TRAINING-ONLY THRESHOLDS
# ============================================================

print("\n" + "=" * 70)
print("CALCULATING TRAINING-ONLY RISK THRESHOLDS")
print("=" * 70)

train_target = pd.to_numeric(
    train[TARGET],
    errors="coerce"
).dropna()


LOW_MEDIUM_THRESHOLD = train_target.quantile(1 / 3)
MEDIUM_HIGH_THRESHOLD = train_target.quantile(2 / 3)


print("\nTraining target statistics:")
print("Count :", len(train_target))
print("Mean  :", train_target.mean())
print("Median:", train_target.median())
print("Min   :", train_target.min())
print("Max   :", train_target.max())

print("\nRisk thresholds:")
print(
    "Low / Medium threshold :",
    LOW_MEDIUM_THRESHOLD
)

print(
    "Medium / High threshold:",
    MEDIUM_HIGH_THRESHOLD
)


# ============================================================
# RISK BAND FUNCTION
# ============================================================

def assign_risk_band(value):

    if pd.isna(value):
        return "UNKNOWN"

    if value < LOW_MEDIUM_THRESHOLD:
        return "LOW"

    elif value < MEDIUM_HIGH_THRESHOLD:
        return "MEDIUM"

    else:
        return "HIGH"


# ============================================================
# PREPARE TEST PREDICTIONS
# ============================================================

print("\n" + "=" * 70)
print("MERGING TEST DATA WITH MODEL PREDICTIONS")
print("=" * 70)


# Make sure key columns exist
missing_test_keys = [
    col for col in KEY_COLUMNS
    if col not in test.columns
]

missing_prediction_keys = [
    col for col in KEY_COLUMNS
    if col not in predictions.columns
]

if missing_test_keys:
    raise ValueError(
        f"Missing key columns in test dataset: {missing_test_keys}"
    )

if missing_prediction_keys:
    raise ValueError(
        f"Missing key columns in prediction file: {missing_prediction_keys}"
    )


# Keep only required prediction columns
prediction_subset = predictions[
    KEY_COLUMNS + [prediction_column]
].copy()


# Rename prediction column
prediction_subset = prediction_subset.rename(
    columns={
        prediction_column: "PREDICTED_WOMEN_CRIME_SHARE"
    }
)


# Merge
result = test.merge(
    prediction_subset,
    on=KEY_COLUMNS,
    how="inner",
    validate="one_to_one"
)


print("\nMerged test records:", len(result))


if len(result) != len(test):

    print(
        "\nWARNING:"
        f" {len(test) - len(result)} test records "
        "did not match prediction records."
    )

else:

    print("All test records matched predictions.")


# ============================================================
# CREATE ACTUAL AND PREDICTED RISK BANDS
# ============================================================

result["ACTUAL_RISK_BAND"] = result[TARGET].apply(
    assign_risk_band
)

result["PREDICTED_RISK_BAND"] = result[
    "PREDICTED_WOMEN_CRIME_SHARE"
].apply(
    assign_risk_band
)


# ============================================================
# BAND AGREEMENT
# ============================================================

result["RISK_BAND_MATCH"] = (
    result["ACTUAL_RISK_BAND"]
    == result["PREDICTED_RISK_BAND"]
)


agreement = result["RISK_BAND_MATCH"].mean() * 100


# ============================================================
# BAND COUNTS
# ============================================================

actual_counts = (
    result["ACTUAL_RISK_BAND"]
    .value_counts()
    .reindex(
        ["LOW", "MEDIUM", "HIGH"],
        fill_value=0
    )
)

predicted_counts = (
    result["PREDICTED_RISK_BAND"]
    .value_counts()
    .reindex(
        ["LOW", "MEDIUM", "HIGH"],
        fill_value=0
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

confusion_matrix = pd.crosstab(
    result["ACTUAL_RISK_BAND"],
    result["PREDICTED_RISK_BAND"],
    rownames=["ACTUAL"],
    colnames=["PREDICTED"],
    dropna=False
)

confusion_matrix = confusion_matrix.reindex(
    index=["LOW", "MEDIUM", "HIGH"],
    columns=["LOW", "MEDIUM", "HIGH"],
    fill_value=0
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 70)
print("RISK BAND RESULTS")
print("=" * 70)

print("\nThresholds:")
print(
    f"LOW < {LOW_MEDIUM_THRESHOLD:.6f}"
)

print(
    f"MEDIUM: {LOW_MEDIUM_THRESHOLD:.6f} "
    f"to < {MEDIUM_HIGH_THRESHOLD:.6f}"
)

print(
    f"HIGH >= {MEDIUM_HIGH_THRESHOLD:.6f}"
)


print("\nActual test risk-band distribution:")
print(actual_counts)

print("\nPredicted test risk-band distribution:")
print(predicted_counts)


print("\nActual vs Predicted risk-band table:")
print(confusion_matrix)


print(
    f"\nRisk-band agreement: {agreement:.2f}%"
)


# ============================================================
# SAVE THRESHOLDS
# ============================================================

thresholds = pd.DataFrame({
    "THRESHOLD_NAME": [
        "LOW_MEDIUM",
        "MEDIUM_HIGH"
    ],
    "VALUE": [
        LOW_MEDIUM_THRESHOLD,
        MEDIUM_HIGH_THRESHOLD
    ],
    "SOURCE": [
        "Training target - 33.33 percentile",
        "Training target - 66.67 percentile"
    ]
})


thresholds_file = (
    OUTPUT_DIR
    / "risk_band_thresholds_dataset_01.csv"
)

thresholds.to_csv(
    thresholds_file,
    index=False
)


# ============================================================
# SAVE TEST RISK BANDS
# ============================================================

risk_band_file = (
    OUTPUT_DIR
    / "dataset_01_test_risk_bands.csv"
)

result.to_csv(
    risk_band_file,
    index=False
)


# ============================================================
# SAVE CONFUSION MATRIX
# ============================================================

confusion_file = (
    OUTPUT_DIR
    / "dataset_01_risk_band_confusion_matrix.csv"
)

confusion_matrix.to_csv(
    confusion_file
)


# ============================================================
# SAVE SUMMARY
# ============================================================

summary = pd.DataFrame({
    "METRIC": [
        "Test records",
        "Low-medium threshold",
        "Medium-high threshold",
        "Actual low records",
        "Actual medium records",
        "Actual high records",
        "Predicted low records",
        "Predicted medium records",
        "Predicted high records",
        "Risk-band agreement percentage"
    ],
    "VALUE": [
        len(result),
        LOW_MEDIUM_THRESHOLD,
        MEDIUM_HIGH_THRESHOLD,
        actual_counts["LOW"],
        actual_counts["MEDIUM"],
        actual_counts["HIGH"],
        predicted_counts["LOW"],
        predicted_counts["MEDIUM"],
        predicted_counts["HIGH"],
        agreement
    ]
})


summary_file = (
    OUTPUT_DIR
    / "dataset_01_risk_band_summary.csv"
)

summary.to_csv(
    summary_file,
    index=False
)


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("FILES SAVED")
print("=" * 70)

print("\nOutput folder:")
print(OUTPUT_DIR)

print("\nCreated files:")
print("1.", thresholds_file.name)
print("2.", risk_band_file.name)
print("3.", confusion_file.name)
print("4.", summary_file.name)

print("\n" + "=" * 70)
print("DATASET 1 RISK-BAND STEP COMPLETED")
print("=" * 70)