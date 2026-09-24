import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# SAFEHER-AI
# DATASET 01 - FEATURE ENGINEERING
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "cleaned_district_ipc_2001_2012_v2.csv"
)

OUTPUT_DIR = BASE_DIR / "data" / "feature_engineering_01"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "dataset_01_engineered.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("SAFEHER-AI - DATASET 01 FEATURE ENGINEERING")
print("=" * 70)

df = pd.read_csv(INPUT_FILE)

print("\nInput dataset:")
print(f"Rows    : {len(df)}")
print(f"Columns : {len(df.columns)}")


# ============================================================
# 1. CREATE UNIQUE RECORD ID
# ============================================================

print("\n" + "-" * 70)
print("1. CREATING RECORD ID")
print("-" * 70)

df["RECORD_ID"] = (
    df["STATE/UT"].astype(str).str.replace(" ", "_")
    + "_"
    + df["DISTRICT"].astype(str).str.replace(" ", "_")
    + "_"
    + df["YEAR"].astype(str)
)

# Add occurrence number for the one repeated
# State-District-Year combination.
df["RECORD_OCCURRENCE"] = (
    df.groupby(
        ["STATE/UT", "DISTRICT", "YEAR"]
    ).cumcount() + 1
)

df["RECORD_ID"] = (
    df["RECORD_ID"]
    + "_"
    + df["RECORD_OCCURRENCE"].astype(str)
)

print("Unique RECORD_ID created.")

print(
    "\nDuplicate State-District-Year groups still present:"
)

duplicate_groups = (
    df[
        df.duplicated(
            ["STATE/UT", "DISTRICT", "YEAR"],
            keep=False
        )
    ]
    .groupby(
        ["STATE/UT", "DISTRICT", "YEAR"]
    )
    .size()
)

if len(duplicate_groups) == 0:
    print("None")
else:
    print(duplicate_groups)


# ============================================================
# 2. WOMEN-SAFETY FEATURES
# ============================================================

print("\n" + "-" * 70)
print("2. WOMEN-SAFETY FEATURES")
print("-" * 70)


# We use the parent categories here rather than adding
# parent + child categories together.

df["WOMEN_CRIME_TOTAL"] = (
    df["RAPE"]
    + df["KIDNAPPING & ABDUCTION"]
    + df["DOWRY DEATHS"]
    + df[
        "ASSAULT ON WOMEN WITH INTENT TO OUTRAGE HER MODESTY"
    ]
    + df["INSULT TO MODESTY OF WOMEN"]
    + df["CRUELTY BY HUSBAND OR HIS RELATIVES"]
    + df["IMPORTATION OF GIRLS FROM FOREIGN COUNTRIES"]
)


df["WOMEN_VIOLENCE"] = (
    df["RAPE"]
    + df["KIDNAPPING & ABDUCTION"]
    + df["ASSAULT ON WOMEN WITH INTENT TO OUTRAGE HER MODESTY"]
    + df["DOWRY DEATHS"]
    + df["CRUELTY BY HUSBAND OR HIS RELATIVES"]
)


df["RAPE_RATE_PROXY"] = (
    df["RAPE"] /
    (df["TOTAL IPC CRIMES"] + 1)
)


df["KIDNAPPING_RATE_PROXY"] = (
    df["KIDNAPPING & ABDUCTION"] /
    (df["TOTAL IPC CRIMES"] + 1)
)


df["WOMEN_CRIME_SHARE"] = (
    df["WOMEN_CRIME_TOTAL"] /
    (df["TOTAL IPC CRIMES"] + 1)
)


print("Created:")
print(" - WOMEN_CRIME_TOTAL")
print(" - WOMEN_VIOLENCE")
print(" - RAPE_RATE_PROXY")
print(" - KIDNAPPING_RATE_PROXY")
print(" - WOMEN_CRIME_SHARE")


# ============================================================
# 3. VIOLENT CRIME FEATURES
# ============================================================

print("\n" + "-" * 70)
print("3. VIOLENT CRIME FEATURES")
print("-" * 70)


df["VIOLENT_CRIME"] = (
    df["MURDER"]
    + df["ATTEMPT TO MURDER"]
    + df[
        "CULPABLE HOMICIDE NOT AMOUNTING TO MURDER"
    ]
    + df["ROBBERY"]
    + df["RIOTS"]
    + df["HURT/GREVIOUS HURT"]
)


df["SERIOUS_VIOLENT_CRIME"] = (
    df["MURDER"]
    + df["ATTEMPT TO MURDER"]
    + df[
        "CULPABLE HOMICIDE NOT AMOUNTING TO MURDER"
    ]
    + df["ROBBERY"]
)


print("Created:")
print(" - VIOLENT_CRIME")
print(" - SERIOUS_VIOLENT_CRIME")


# ============================================================
# 4. PROPERTY CRIME FEATURES
# ============================================================

print("\n" + "-" * 70)
print("4. PROPERTY CRIME FEATURES")
print("-" * 70)


# THEFT already contains AUTO THEFT + OTHER THEFT.
# Therefore we use THEFT once.

df["PROPERTY_CRIME"] = (
    df["THEFT"]
    + df["BURGLARY"]
    + df["ROBBERY"]
    + df["DACOITY"]
)


df["THEFT_SHARE"] = (
    df["THEFT"] /
    (df["TOTAL IPC CRIMES"] + 1)
)


df["BURGLARY_SHARE"] = (
    df["BURGLARY"] /
    (df["TOTAL IPC CRIMES"] + 1)
)


print("Created:")
print(" - PROPERTY_CRIME")
print(" - THEFT_SHARE")
print(" - BURGLARY_SHARE")


# ============================================================
# 5. SOCIAL / ECONOMIC OFFENCE FEATURES
# ============================================================

print("\n" + "-" * 70)
print("5. SOCIAL / ECONOMIC OFFENCE FEATURES")
print("-" * 70)


df["ECONOMIC_OFFENCES"] = (
    df["CRIMINAL BREACH OF TRUST"]
    + df["CHEATING"]
    + df["COUNTERFIETING"]
)


df["PUBLIC_ORDER_CRIME"] = (
    df["RIOTS"]
    + df["ARSON"]
)


print("Created:")
print(" - ECONOMIC_OFFENCES")
print(" - PUBLIC_ORDER_CRIME")


# ============================================================
# 6. CRIME DENSITY-STYLE FEATURES
# ============================================================

print("\n" + "-" * 70)
print("6. CRIME SHARE FEATURES")
print("-" * 70)


df["VIOLENT_CRIME_SHARE"] = (
    df["VIOLENT_CRIME"] /
    (df["TOTAL IPC CRIMES"] + 1)
)


df["PROPERTY_CRIME_SHARE"] = (
    df["PROPERTY_CRIME"] /
    (df["TOTAL IPC CRIMES"] + 1)
)


df["MURDER_SHARE"] = (
    df["MURDER"] /
    (df["TOTAL IPC CRIMES"] + 1)
)


df["ROBBERY_SHARE"] = (
    df["ROBBERY"] /
    (df["TOTAL IPC CRIMES"] + 1)
)


print("Created:")
print(" - VIOLENT_CRIME_SHARE")
print(" - PROPERTY_CRIME_SHARE")
print(" - MURDER_SHARE")
print(" - ROBBERY_SHARE")


# ============================================================
# 7. TEMPORAL FEATURES
# ============================================================

print("\n" + "-" * 70)
print("7. TEMPORAL FEATURES")
print("-" * 70)


# Normalize year to a 0-1 range for future modelling.

df["YEAR_NORMALIZED"] = (
    (df["YEAR"] - df["YEAR"].min())
    /
    (df["YEAR"].max() - df["YEAR"].min())
)


print("Created:")
print(" - YEAR_NORMALIZED")


# ============================================================
# 8. FEATURE SUMMARY
# ============================================================

engineered_features = [
    "WOMEN_CRIME_TOTAL",
    "WOMEN_VIOLENCE",
    "RAPE_RATE_PROXY",
    "KIDNAPPING_RATE_PROXY",
    "WOMEN_CRIME_SHARE",
    "VIOLENT_CRIME",
    "SERIOUS_VIOLENT_CRIME",
    "PROPERTY_CRIME",
    "THEFT_SHARE",
    "BURGLARY_SHARE",
    "ECONOMIC_OFFENCES",
    "PUBLIC_ORDER_CRIME",
    "VIOLENT_CRIME_SHARE",
    "PROPERTY_CRIME_SHARE",
    "MURDER_SHARE",
    "ROBBERY_SHARE",
    "YEAR_NORMALIZED"
]


print("\n" + "-" * 70)
print("8. ENGINEERED FEATURES")
print("-" * 70)

for feature in engineered_features:
    print(f" - {feature}")


# ============================================================
# 9. CHECK FOR INVALID VALUES
# ============================================================

print("\n" + "-" * 70)
print("9. DATA QUALITY CHECK")
print("-" * 70)


numeric_engineered = df[
    engineered_features
].select_dtypes(
    include=np.number
)


print(
    "\nNaN values:",
    numeric_engineered.isna().sum().sum()
)


print(
    "Infinite values:",
    np.isinf(numeric_engineered).sum().sum()
)


# ============================================================
# 10. DESCRIPTIVE STATISTICS
# ============================================================

print("\n" + "-" * 70)
print("10. ENGINEERED FEATURE STATISTICS")
print("-" * 70)


feature_stats = (
    df[engineered_features]
    .describe()
    .T
)

print(
    feature_stats[
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


feature_stats.to_csv(
    OUTPUT_DIR / "engineered_feature_statistics.csv"
)


# ============================================================
# 11. CORRELATION CHECK
# ============================================================

print("\n" + "-" * 70)
print("11. CORRELATION WITH TOTAL IPC CRIMES")
print("-" * 70)


correlation = (
    df[
        engineered_features
        + ["TOTAL IPC CRIMES"]
    ]
    .corr()["TOTAL IPC CRIMES"]
    .sort_values(ascending=False)
)


print(correlation.to_string())


correlation.to_csv(
    OUTPUT_DIR / "engineered_feature_correlations.csv"
)


# ============================================================
# 12. SAVE ENGINEERED DATASET
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# 13. SAVE K-MEANS CANDIDATE FEATURES
# ============================================================

# IMPORTANT:
# We are NOT running K-Means yet.
# This file is only a candidate feature matrix.

kmeans_candidates = [
    "WOMEN_CRIME_TOTAL",
    "WOMEN_VIOLENCE",
    "RAPE_RATE_PROXY",
    "KIDNAPPING_RATE_PROXY",
    "VIOLENT_CRIME",
    "PROPERTY_CRIME",
    "ECONOMIC_OFFENCES",
    "PUBLIC_ORDER_CRIME",
    "VIOLENT_CRIME_SHARE",
    "PROPERTY_CRIME_SHARE",
    "MURDER_SHARE",
    "ROBBERY_SHARE",
    "YEAR_NORMALIZED"
]


kmeans_dataset = df[
    [
        "RECORD_ID",
        "STATE/UT",
        "DISTRICT",
        "YEAR"
    ]
    + kmeans_candidates
].copy()


kmeans_file = (
    OUTPUT_DIR
    / "kmeans_candidate_features.csv"
)


kmeans_dataset.to_csv(
    kmeans_file,
    index=False
)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("FEATURE ENGINEERING COMPLETED")
print("=" * 70)

print("\nEngineered dataset:")
print(OUTPUT_FILE)

print("\nK-Means candidate dataset:")
print(kmeans_file)

print("\nRows:")
print(len(df))

print("\nOriginal columns:")
print(33)

print("\nFinal columns:")
print(len(df.columns))

print("\nIMPORTANT:")
print("K-Means has NOT been executed yet.")

print("\nNext step:")
print("Review engineered features → scale features → select K → K-Means")