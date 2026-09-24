import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data"

DATASET_01 = (
    DATA_DIR
    / "cleaned_district_ipc_2001_2012_v2.csv"
)

DATASET_02_03 = (
    DATA_DIR
    / "dataset_02_03_2013_2014"
    / "feature_engineering"
    / "engineered_dataset_02_03_2013_2014.csv"
)

OUTPUT_DIR = (
    DATA_DIR
    / "master_2001_2014"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD DATASETS
# ============================================================

print("=" * 80)
print("MASTER DATASET PREPARATION")
print("2001–2014")
print("=" * 80)

df1 = pd.read_csv(DATASET_01)
df23 = pd.read_csv(DATASET_02_03)

print("\nDataset 1 shape:", df1.shape)
print("Dataset 2 + 3 shape:", df23.shape)


# ============================================================
# STANDARDIZE BASIC COLUMN NAMES
# ============================================================

if "STATE/UT" in df1.columns:
    df1 = df1.rename(
        columns={
            "STATE/UT": "STATE_UT"
        }
    )

if "States/UTs" in df23.columns:
    df23 = df23.rename(
        columns={
            "States/UTs": "STATE_UT"
        }
    )

if "State/UT" in df23.columns:
    df23 = df23.rename(
        columns={
            "State/UT": "STATE_UT"
        }
    )


# ============================================================
# COMMON ANALYTICAL COLUMNS
#
# HURT IS INTENTIONALLY EXCLUDED.
#
# Dataset 1 (2001–2012) does not contain a directly
# comparable standalone HURT field.
#
# We do NOT create artificial HURT=0 values.
# ============================================================

common_columns = [
    "STATE_UT",
    "DISTRICT",
    "YEAR",

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
    "PUBLIC_ORDER_CRIME"
]


# ============================================================
# CHECK DATASET 1 COMPATIBILITY
# ============================================================

print("\n" + "=" * 80)
print("CHECKING DATASET 1 COMPATIBILITY")
print("=" * 80)

missing_dataset1 = [
    col
    for col in common_columns
    if col not in df1.columns
]

missing_dataset23 = [
    col
    for col in common_columns
    if col not in df23.columns
]

print("\nMissing common columns in Dataset 1:")
print(missing_dataset1)

print("\nMissing common columns in Dataset 2 + 3:")
print(missing_dataset23)


# ============================================================
# DATASET 1 COLUMN MAPPING
# ============================================================

dataset1_mapping = {

    "MURDER":
        "MURDER",

    "ATTEMPT TO MURDER":
        "ATTEMPT_MURDER",

    "CULPABLE HOMICIDE NOT AMOUNTING TO MURDER":
        "CULPABLE_HOMICIDE",

    "RAPE":
        "RAPE",

    "CUSTODIAL RAPE":
        "CUSTODIAL_RAPE",

    "KIDNAPPING & ABDUCTION":
        "KIDNAPPING_ABDUCTION",

    "DACOITY":
        "DACOITY",

    "ROBBERY":
        "ROBBERY",

    "BURGLARY":
        "BURGLARY",

    "THEFT":
        "THEFT",

    "AUTO THEFT":
        "AUTO_THEFT",

    "RIOTS":
        "RIOTS",

    "CRIMINAL BREACH OF TRUST":
        "CRIMINAL_BREACH_TRUST",

    "CHEATING":
        "CHEATING",

    "ARSON":
        "ARSON",

    "DOWRY DEATHS":
        "DOWRY_DEATHS",

    "ASSAULT ON WOMEN WITH INTENT TO OUTRAGE HER MODESTY":
        "ASSAULT_WOMEN",

    "INSULT TO MODESTY OF WOMEN":
        "INSULT_WOMEN",

    "CRUELTY BY HUSBAND OR HIS RELATIVES":
        "CRUELTY_WOMEN",

    "IMPORTATION OF GIRLS FROM FOREIGN COUNTRIES":
        "IMPORTATION_GIRLS",

    "CAUSING DEATH BY NEGLIGENCE":
        "DEATH_BY_NEGLIGENCE",

    "OTHER IPC CRIMES":
        "OTHER_IPC",

    "TOTAL IPC CRIMES":
        "TOTAL_IPC"
}


# ============================================================
# FIND DATASET 1 COLUMN NAMES
# ============================================================

print("\n" + "=" * 80)
print("DATASET 1 COLUMN CHECK")
print("=" * 80)

for source_col, target_col in dataset1_mapping.items():

    if source_col in df1.columns:

        print(
            f"{source_col} -> {target_col}"
        )

    else:

        print(
            f"NOT FOUND: {source_col}"
        )


# ============================================================
# RENAME DATASET 1
# ============================================================

df1 = df1.rename(
    columns=dataset1_mapping
)


# ============================================================
# CREATE WOMEN CRIME CORE TOTAL
# ============================================================

if "WOMEN_CRIME_CORE_TOTAL" not in df1.columns:

    women_components = [
        "RAPE",
        "CUSTODIAL_RAPE",
        "DOWRY_DEATHS",
        "ASSAULT_WOMEN",
        "INSULT_WOMEN",
        "CRUELTY_WOMEN",
        "IMPORTATION_GIRLS"
    ]

    existing = [
        col
        for col in women_components
        if col in df1.columns
    ]

    df1["WOMEN_CRIME_CORE_TOTAL"] = (
        df1[existing]
        .sum(axis=1)
    )


# ============================================================
# CREATE VIOLENT CRIME
#
# HURT is intentionally NOT included because it is not
# available as a directly comparable field in Dataset 1.
# ============================================================

if "VIOLENT_CRIME" not in df1.columns:

    violent_components = [
        "MURDER",
        "ATTEMPT_MURDER",
        "CULPABLE_HOMICIDE",
        "RAPE",
        "KIDNAPPING_ABDUCTION",
        "DACOITY",
        "ROBBERY",
        "RIOTS"
    ]

    existing = [
        col
        for col in violent_components
        if col in df1.columns
    ]

    df1["VIOLENT_CRIME"] = (
        df1[existing]
        .sum(axis=1)
    )


# ============================================================
# CREATE PROPERTY CRIME
# ============================================================

if "PROPERTY_CRIME" not in df1.columns:

    property_components = [
        "ROBBERY",
        "BURGLARY",
        "THEFT",
        "AUTO_THEFT",
        "ARSON"
    ]

    existing = [
        col
        for col in property_components
        if col in df1.columns
    ]

    df1["PROPERTY_CRIME"] = (
        df1[existing]
        .sum(axis=1)
    )


# ============================================================
# CREATE ECONOMIC OFFENCES
# ============================================================

if "ECONOMIC_OFFENCES" not in df1.columns:

    economic_components = [
        "CRIMINAL_BREACH_TRUST",
        "CHEATING"
    ]

    existing = [
        col
        for col in economic_components
        if col in df1.columns
    ]

    df1["ECONOMIC_OFFENCES"] = (
        df1[existing]
        .sum(axis=1)
    )


# ============================================================
# CREATE PUBLIC ORDER CRIME
# ============================================================

if "PUBLIC_ORDER_CRIME" not in df1.columns:

    public_order_components = [
        "RIOTS"
    ]

    existing = [
        col
        for col in public_order_components
        if col in df1.columns
    ]

    df1["PUBLIC_ORDER_CRIME"] = (
        df1[existing]
        .sum(axis=1)
    )


# ============================================================
# KEEP COMMON COLUMNS ONLY
# ============================================================

df1_common = df1[
    [
        col
        for col in common_columns
        if col in df1.columns
    ]
].copy()

df23_common = df23[
    [
        col
        for col in common_columns
        if col in df23.columns
    ]
].copy()


# ============================================================
# FINAL COMMON COLUMN CHECK
# ============================================================

print("\n" + "=" * 80)
print("FINAL COMMON COLUMN CHECK")
print("=" * 80)

missing_after_mapping = [
    col
    for col in common_columns
    if col not in df1_common.columns
    or col not in df23_common.columns
]

print(
    "Missing after harmonization:",
    missing_after_mapping
)

if missing_after_mapping:

    raise ValueError(
        "Common analytical schema is incomplete. "
        "Review Dataset 1 column mapping before continuing."
    )


# ============================================================
# NUMERIC CONVERSION
# ============================================================

numeric_columns = [
    col
    for col in common_columns
    if col not in [
        "STATE_UT",
        "DISTRICT"
    ]
]

for col in numeric_columns:

    df1_common[col] = pd.to_numeric(
        df1_common[col],
        errors="coerce"
    )

    df23_common[col] = pd.to_numeric(
        df23_common[col],
        errors="coerce"
    )


# ============================================================
# ADD SOURCE DATASET
# ============================================================

df1_common["SOURCE_DATASET"] = (
    "DATASET_01_2001_2012"
)

df23_common["SOURCE_DATASET"] = (
    "DATASET_02_03_2013_2014"
)


# ============================================================
# COMBINE
# ============================================================

master = pd.concat(
    [
        df1_common,
        df23_common
    ],
    ignore_index=True
)


# ============================================================
# SORT
# ============================================================

master = master.sort_values(
    [
        "YEAR",
        "STATE_UT",
        "DISTRICT"
    ]
).reset_index(
    drop=True
)


# ============================================================
# MASTER DATASET QUALITY CHECK
# ============================================================

print("\n" + "=" * 80)
print("MASTER DATASET QUALITY CHECK")
print("=" * 80)

print("\nShape:")
print(master.shape)

print("\nYears:")
print(
    master["YEAR"]
    .value_counts()
    .sort_index()
)

print("\nSources:")
print(
    master["SOURCE_DATASET"]
    .value_counts()
)

print("\nStates/UTs:")
print(
    master["STATE_UT"]
    .nunique()
)

print("\nReporting units:")
print(
    master["DISTRICT"]
    .nunique()
)


# ============================================================
# MISSING VALUES
# ============================================================

print("\nMissing values:")

missing = (
    master
    .isna()
    .sum()
)

missing = missing[
    missing > 0
]

if len(missing) == 0:

    print("None")

else:

    print(missing)


# ============================================================
# EXACT DUPLICATES
# ============================================================

print(
    "\nExact duplicate rows:",
    master.duplicated().sum()
)


# ============================================================
# KEY DUPLICATES
# ============================================================

key_duplicates = (
    master
    .duplicated(
        subset=[
            "STATE_UT",
            "DISTRICT",
            "YEAR"
        ],
        keep=False
    )
)

print(
    "Duplicate STATE/DISTRICT/YEAR records:",
    key_duplicates.sum()
)

if key_duplicates.sum() > 0:

    print("\nDuplicate keys:")

    print(
        master.loc[
            key_duplicates,
            [
                "STATE_UT",
                "DISTRICT",
                "YEAR",
                "TOTAL_IPC",
                "SOURCE_DATASET"
            ]
        ]
        .sort_values(
            [
                "YEAR",
                "STATE_UT",
                "DISTRICT"
            ]
        )
        .to_string(index=False)
    )


# ============================================================
# YEAR COVERAGE
# ============================================================

print("\n" + "=" * 80)
print("YEAR COVERAGE")
print("=" * 80)

year_summary = (
    master
    .groupby("YEAR")
    .agg(
        RECORDS=("YEAR", "size"),
        STATES=("STATE_UT", "nunique"),
        REPORTING_UNITS=("DISTRICT", "nunique"),
        TOTAL_IPC=("TOTAL_IPC", "sum"),
        WOMEN_CRIME=("WOMEN_CRIME_CORE_TOTAL", "sum")
    )
    .reset_index()
)

print(
    year_summary.to_string(index=False)
)


# ============================================================
# YEAR CONTINUITY CHECK
# ============================================================

print("\n" + "=" * 80)
print("YEAR CONTINUITY CHECK")
print("=" * 80)

years = sorted(
    master["YEAR"]
    .dropna()
    .unique()
)

print("Minimum year:", min(years))
print("Maximum year:", max(years))

expected_years = list(
    range(
        int(min(years)),
        int(max(years)) + 1
    )
)

missing_years = [
    year
    for year in expected_years
    if year not in years
]

if len(missing_years) == 0:

    print(
        "All years from",
        min(years),
        "to",
        max(years),
        "are present."
    )

else:

    print(
        "Missing years:",
        missing_years
    )


# ============================================================
# SAVE MASTER DATASET
# ============================================================

output_file = (
    OUTPUT_DIR
    / "master_harmonized_ipc_2001_2014.csv"
)

master.to_csv(
    output_file,
    index=False
)


# ============================================================
# SAVE YEAR SUMMARY
# ============================================================

year_summary_file = (
    OUTPUT_DIR
    / "year_summary_2001_2014.csv"
)

year_summary.to_csv(
    year_summary_file,
    index=False
)


# ============================================================
# SAVE COLUMN LIST
# ============================================================

column_summary = pd.DataFrame({
    "COLUMN_NAME": master.columns
})

column_summary.to_csv(
    OUTPUT_DIR
    / "master_column_list_2001_2014.csv",
    index=False
)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 80)
print("MASTER DATASET CREATED")
print("=" * 80)

print("\nSaved to:")
print(output_file)

print("\nYear summary saved to:")
print(year_summary_file)

print("\nColumn list saved to:")
print(
    OUTPUT_DIR
    / "master_column_list_2001_2014.csv"
)

print("\n" + "=" * 80)
print("MASTER DATASET PREPARATION COMPLETED")
print("=" * 80)