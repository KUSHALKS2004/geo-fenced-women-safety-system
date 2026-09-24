import pandas as pd
from pathlib import Path


# ============================================================
# SAFEHER-AI
# Geo-Fenced Emergency Alert and Safety Assistance System
# for Women
#
# DATASET 01
# District-wise IPC Crime Data
# 2001-2012
# ============================================================


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_PATH = BASE_DIR / "data" / "dstrIPC_1.csv"

OUTPUT_PATH = BASE_DIR / "data" / "cleaned_district_ipc_2001_2012.csv"


# ============================================================
# 2. LOAD DATA
# ============================================================

print("=" * 70)
print("SAFEHER-AI")
print("DATASET 01 - CLEANING AND EDA")
print("=" * 70)

print("\nLoading dataset...")

df = pd.read_csv(INPUT_PATH)

print("Dataset loaded successfully.")


# ============================================================
# 3. ORIGINAL DATASET INFORMATION
# ============================================================

print("\n" + "-" * 70)
print("ORIGINAL DATASET")
print("-" * 70)

print("Rows    :", df.shape[0])
print("Columns :", df.shape[1])


# ============================================================
# 4. CHECK TOTAL ROWS
# ============================================================

print("\n" + "-" * 70)
print("AGGREGATE / TOTAL ROW CHECK")
print("-" * 70)

total_rows = df[
    df["DISTRICT"].astype(str).str.strip().str.upper() == "TOTAL"
]

print("TOTAL rows found:", len(total_rows))

if len(total_rows) > 0:
    print("\nExamples:")
    print(total_rows[["STATE/UT", "DISTRICT", "YEAR"]].head(20))


# ============================================================
# 5. REMOVE TOTAL ROWS
# ============================================================

df_clean = df[
    df["DISTRICT"].astype(str).str.strip().str.upper() != "TOTAL"
].copy()

print("\nAfter removing TOTAL rows:")
print("Rows:", len(df_clean))


# ============================================================
# 6. CHECK DUPLICATE STATE-DISTRICT-YEAR RECORDS
# ============================================================

print("\n" + "-" * 70)
print("DUPLICATE STATE-DISTRICT-YEAR CHECK")
print("-" * 70)

duplicate_key = df_clean.duplicated(
    subset=["STATE/UT", "DISTRICT", "YEAR"]
).sum()

print(
    "Duplicate State-District-Year records:",
    duplicate_key
)


# ============================================================
# 7. CHECK MISSING VALUES
# ============================================================

print("\n" + "-" * 70)
print("MISSING VALUES AFTER CLEANING")
print("-" * 70)

missing = df_clean.isnull().sum()

if missing.sum() == 0:
    print("No missing values found.")
else:
    print(missing[missing > 0])


# ============================================================
# 8. DATA TYPES
# ============================================================

print("\n" + "-" * 70)
print("DATA TYPES")
print("-" * 70)

print(df_clean.dtypes)


# ============================================================
# 9. YEAR DISTRIBUTION
# ============================================================

print("\n" + "-" * 70)
print("RECORDS PER YEAR")
print("-" * 70)

print(
    df_clean["YEAR"]
    .value_counts()
    .sort_index()
)


# ============================================================
# 10. STATE DISTRIBUTION
# ============================================================

print("\n" + "-" * 70)
print("RECORDS PER STATE / UT")
print("-" * 70)

print(
    df_clean["STATE/UT"]
    .value_counts()
)


# ============================================================
# 11. DISTRICT COUNT
# ============================================================

print("\n" + "-" * 70)
print("DISTRICT INFORMATION")
print("-" * 70)

print(
    "Unique district names:",
    df_clean["DISTRICT"].nunique()
)


# ============================================================
# 12. IMPORTANT WOMEN-SAFETY FEATURES
# ============================================================

women_features = [
    "RAPE",
    "KIDNAPPING & ABDUCTION",
    "KIDNAPPING AND ABDUCTION OF WOMEN AND GIRLS",
    "DOWRY DEATHS",
    "ASSAULT ON WOMEN WITH INTENT TO OUTRAGE HER MODESTY",
    "INSULT TO MODESTY OF WOMEN",
    "CRUELTY BY HUSBAND OR HIS RELATIVES",
    "IMPORTATION OF GIRLS FROM FOREIGN COUNTRIES"
]


print("\n" + "-" * 70)
print("WOMEN-SAFETY FEATURES")
print("-" * 70)

for feature in women_features:

    if feature in df_clean.columns:

        print(
            f"{feature}: "
            f"{df_clean[feature].sum():,.0f}"
        )


# ============================================================
# 13. GENERAL CRIME FEATURES
# ============================================================

general_features = [
    "MURDER",
    "ATTEMPT TO MURDER",
    "ROBBERY",
    "BURGLARY",
    "THEFT",
    "RIOTS",
    "ARSON",
    "HURT/GREVIOUS HURT",
    "CAUSING DEATH BY NEGLIGENCE",
    "TOTAL IPC CRIMES"
]


print("\n" + "-" * 70)
print("GENERAL CRIME FEATURES")
print("-" * 70)

for feature in general_features:

    if feature in df_clean.columns:

        print(
            f"{feature}: "
            f"{df_clean[feature].sum():,.0f}"
        )


# ============================================================
# 14. DESCRIPTIVE STATISTICS
# ============================================================

print("\n" + "-" * 70)
print("DESCRIPTIVE STATISTICS")
print("-" * 70)

print(
    df_clean[
        women_features + general_features
    ].describe().T
)


# ============================================================
# 15. TOP DISTRICTS BY TOTAL IPC CRIMES
# ============================================================

print("\n" + "-" * 70)
print("TOP 20 DISTRICT-YEAR RECORDS BY TOTAL IPC CRIMES")
print("-" * 70)

top_crime_records = df_clean.sort_values(
    by="TOTAL IPC CRIMES",
    ascending=False
)[
    [
        "STATE/UT",
        "DISTRICT",
        "YEAR",
        "TOTAL IPC CRIMES"
    ]
].head(20)

print(top_crime_records.to_string(index=False))


# ============================================================
# 16. TOP DISTRICT-YEAR RECORDS FOR RAPE
# ============================================================

print("\n" + "-" * 70)
print("TOP 20 DISTRICT-YEAR RECORDS BY RAPE")
print("-" * 70)

top_rape = df_clean.sort_values(
    by="RAPE",
    ascending=False
)[
    [
        "STATE/UT",
        "DISTRICT",
        "YEAR",
        "RAPE"
    ]
].head(20)

print(top_rape.to_string(index=False))


# ============================================================
# 17. TOP DISTRICT-YEAR RECORDS FOR KIDNAPPING
# ============================================================

print("\n" + "-" * 70)
print("TOP 20 DISTRICT-YEAR RECORDS BY KIDNAPPING & ABDUCTION")
print("-" * 70)

top_kidnapping = df_clean.sort_values(
    by="KIDNAPPING & ABDUCTION",
    ascending=False
)[
    [
        "STATE/UT",
        "DISTRICT",
        "YEAR",
        "KIDNAPPING & ABDUCTION"
    ]
].head(20)

print(top_kidnapping.to_string(index=False))


# ============================================================
# 18. CHECK COMPONENT RELATIONSHIPS
# ============================================================

print("\n" + "-" * 70)
print("CATEGORY RELATIONSHIP CHECKS")
print("-" * 70)

# Rape relationship
if all(
    col in df_clean.columns
    for col in ["RAPE", "CUSTODIAL RAPE", "OTHER RAPE"]
):

    rape_difference = (
        df_clean["RAPE"]
        - (
            df_clean["CUSTODIAL RAPE"]
            + df_clean["OTHER RAPE"]
        )
    )

    print(
        "RAPE - (CUSTODIAL RAPE + OTHER RAPE)"
    )

    print(
        rape_difference.describe()
    )


# Kidnapping relationship
if all(
    col in df_clean.columns
    for col in [
        "KIDNAPPING & ABDUCTION",
        "KIDNAPPING AND ABDUCTION OF WOMEN AND GIRLS",
        "KIDNAPPING AND ABDUCTION OF OTHERS"
    ]
):

    kidnapping_difference = (
        df_clean["KIDNAPPING & ABDUCTION"]
        - (
            df_clean[
                "KIDNAPPING AND ABDUCTION OF WOMEN AND GIRLS"
            ]
            +
            df_clean[
                "KIDNAPPING AND ABDUCTION OF OTHERS"
            ]
        )
    )

    print(
        "\nKIDNAPPING & ABDUCTION - "
        "(WOMEN/GIRLS + OTHERS)"
    )

    print(
        kidnapping_difference.describe()
    )


# ============================================================
# 19. SAVE CLEANED DATASET
# ============================================================

df_clean.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\n" + "-" * 70)
print("CLEANED DATASET SAVED")
print("-" * 70)

print(OUTPUT_PATH)


# ============================================================
# 20. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("DATASET 01 CLEANING AND EDA COMPLETE")
print("=" * 70)

print("\nOriginal rows :", len(df))
print("Cleaned rows  :", len(df_clean))
print("Removed rows  :", len(df) - len(df_clean))

print("\nNext stage:")
print("1. Visual EDA")
print("2. Feature selection")
print("3. Feature engineering")
print("4. K-Means clustering")
print("5. Cluster evaluation")