import pandas as pd
from pathlib import Path


# ============================================================
# DATASET 2 + DATASET 3
# 2013 + 2014 HARMONIZATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data"

FILE_2013 = DATA_DIR / "dstrIPC_2013.csv"
FILE_2014 = DATA_DIR / "dstrIPC_1_2014.csv"

OUTPUT_DIR = DATA_DIR / "dataset_02_03_2013_2014"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "harmonized_ipc_2013_2014.csv"


# ============================================================
# LOAD
# ============================================================

print("=" * 80)
print("DATASET 2 + DATASET 3")
print("2013 + 2014 HARMONIZATION")
print("=" * 80)

print("\nLoading 2013...")
df13 = pd.read_csv(FILE_2013)

print("2013 shape:", df13.shape)

print("\nLoading 2014...")
df14 = pd.read_csv(FILE_2014)

print("2014 shape:", df14.shape)


# ============================================================
# STANDARDIZE BASIC COLUMN NAMES
# ============================================================

df13 = df13.rename(
    columns={
        "STATE/UT": "STATE_UT",
        "DISTRICT": "DISTRICT",
        "YEAR": "YEAR"
    }
)

df14 = df14.rename(
    columns={
        "States/UTs": "STATE_UT",
        "District": "DISTRICT",
        "Year": "YEAR"
    }
)


# ============================================================
# CHECK YEARS
# ============================================================

print("\n2013 years:", df13["YEAR"].unique())
print("2014 years:", df14["YEAR"].unique())


# ============================================================
# FIND AGGREGATE RECORDS
# ============================================================

print("\n" + "=" * 80)
print("CHECKING AGGREGATE RECORDS")
print("=" * 80)


def find_aggregate_records(df, dataset_name):

    district_text = (
        df["DISTRICT"]
        .astype(str)
        .str.upper()
    )

    state_text = (
        df["STATE_UT"]
        .astype(str)
        .str.upper()
    )

    mask = (
        district_text.str.contains("TOTAL", na=False)
        |
        state_text.str.contains("TOTAL", na=False)
    )

    candidates = df.loc[
        mask,
        ["STATE_UT", "DISTRICT", "YEAR"]
    ].copy()

    print(f"\n{dataset_name} aggregate candidates:", len(candidates))

    if len(candidates) > 0:
        print(candidates.to_string(index=False))

    return mask


aggregate13 = find_aggregate_records(
    df13,
    "2013"
)

aggregate14 = find_aggregate_records(
    df14,
    "2014"
)


# ============================================================
# REMOVE ONLY AGGREGATE RECORDS
# ============================================================

before13 = len(df13)
before14 = len(df14)

df13 = df13.loc[~aggregate13].copy()
df14 = df14.loc[~aggregate14].copy()

print("\nRecords removed:")
print("2013:", before13 - len(df13))
print("2014:", before14 - len(df14))

print("\nRemaining:")
print("2013:", len(df13))
print("2014:", len(df14))


# ============================================================
# COMMON FEATURE EXTRACTION
# ============================================================

print("\n" + "=" * 80)
print("CREATING COMMON FEATURES")
print("=" * 80)


# ------------------------------------------------------------
# 2013
# ------------------------------------------------------------

df13_common = pd.DataFrame()

df13_common["STATE_UT"] = df13["STATE_UT"]
df13_common["DISTRICT"] = df13["DISTRICT"]
df13_common["YEAR"] = df13["YEAR"]

df13_common["MURDER"] = df13["MURDER"]

df13_common["ATTEMPT_MURDER"] = (
    df13["ATTEMPT TO MURDER"]
)

df13_common["CULPABLE_HOMICIDE"] = (
    df13[
        "CULPABLE HOMICIDE NOT AMOUNTING TO MURDER"
    ]
)

df13_common["RAPE"] = df13["RAPE"]

df13_common["CUSTODIAL_RAPE"] = (
    df13["CUSTODIAL RAPE"]
)

df13_common["KIDNAPPING_ABDUCTION"] = (
    df13["KIDNAPPING & ABDUCTION"]
)

df13_common["DACOITY"] = df13["DACOITY"]

df13_common["ROBBERY"] = df13["ROBBERY"]

df13_common["BURGLARY"] = df13["BURGLARY"]

df13_common["THEFT"] = df13["THEFT"]

df13_common["AUTO_THEFT"] = df13["AUTO THEFT"]

df13_common["RIOTS"] = df13["RIOTS"]

df13_common["CRIMINAL_BREACH_TRUST"] = (
    df13["CRIMINAL BREACH OF TRUST"]
)

df13_common["CHEATING"] = df13["CHEATING"]

df13_common["ARSON"] = df13["ARSON"]

df13_common["HURT"] = df13["HURT/GREVIOUS HURT"]

df13_common["DOWRY_DEATHS"] = (
    df13["DOWRY DEATHS"]
)

df13_common["ASSAULT_WOMEN"] = (
    df13[
        "ASSAULT ON WOMEN WITH INTENT TO OUTRAGE HER MODESTY"
    ]
)

df13_common["INSULT_WOMEN"] = (
    df13["INSULT TO MODESTY OF WOMEN"]
)

df13_common["CRUELTY_WOMEN"] = (
    df13[
        "CRUELTY BY HUSBAND OR HIS RELATIVES"
    ]
)

df13_common["IMPORTATION_GIRLS"] = (
    df13[
        "IMPORTATION OF GIRLS FROM FOREIGN COUNTRIES"
    ]
)

df13_common["DEATH_BY_NEGLIGENCE"] = (
    df13["CAUSING DEATH BY NEGLIGENCE"]
)

df13_common["OTHER_IPC"] = (
    df13["OTHER IPC CRIMES"]
)

df13_common["TOTAL_IPC"] = (
    df13["TOTAL IPC CRIMES"]
)


# ------------------------------------------------------------
# 2014
# ------------------------------------------------------------

df14_common = pd.DataFrame()

df14_common["STATE_UT"] = df14["STATE_UT"]
df14_common["DISTRICT"] = df14["DISTRICT"]
df14_common["YEAR"] = df14["YEAR"]

df14_common["MURDER"] = df14["Murder"]

df14_common["ATTEMPT_MURDER"] = (
    df14["Attempt to commit Murder"]
)

df14_common["CULPABLE_HOMICIDE"] = (
    df14[
        "Culpable Homicide not amounting to Murder"
    ]
)

df14_common["RAPE"] = df14["Rape"]

df14_common["CUSTODIAL_RAPE"] = (
    df14["Custodial Rape"]
)

df14_common["KIDNAPPING_ABDUCTION"] = (
    df14["Kidnapping & Abduction_Total"]
)

df14_common["DACOITY"] = df14["Dacoity"]

df14_common["ROBBERY"] = df14["Robbery"]

# Use the aggregate burglary field rather than
# its subcomponents.
df14_common["BURGLARY"] = (
    df14["Criminal Trespass/Burglary"]
)

df14_common["THEFT"] = df14["Theft"]

df14_common["AUTO_THEFT"] = df14["Auto Theft"]

df14_common["RIOTS"] = df14["Riots"]

df14_common["CRIMINAL_BREACH_TRUST"] = (
    df14["Criminal Breach of Trust"]
)

df14_common["CHEATING"] = df14["Cheating"]

df14_common["ARSON"] = df14["Arson"]

df14_common["HURT"] = (
    df14["Grievous Hurt"] + df14["Hurt"]
)

df14_common["DOWRY_DEATHS"] = (
    df14["Dowry Deaths"]
)

df14_common["ASSAULT_WOMEN"] = (
    df14[
        "Assault on Women with intent to outrage her Modesty"
    ]
)

df14_common["INSULT_WOMEN"] = (
    df14["Insult to the Modesty of Women"]
)

df14_common["CRUELTY_WOMEN"] = (
    df14[
        "Cruelty by Husband or his Relatives"
    ]
)

df14_common["IMPORTATION_GIRLS"] = (
    df14[
        "Importation of Girls from Foreign Country"
    ]
)

df14_common["DEATH_BY_NEGLIGENCE"] = (
    df14["Causing Death by Negligence"]
)

df14_common["OTHER_IPC"] = (
    df14["Other IPC crimes"]
)

df14_common["TOTAL_IPC"] = (
    df14["Total Cognizable IPC crimes"]
)


# ============================================================
# COMMON WOMEN-CRIME CORE
# ============================================================

print("\nCreating comparable women-crime core...")

"""
Important:

2013 contains:
KIDNAPPING AND ABDUCTION OF WOMEN AND GIRLS

2014 changes the kidnapping classification into
multiple detailed categories.

Therefore we do NOT force the 2013 women/girls
kidnapping field into the 2014 field.

Instead, for the common 2013-2014 model we use
the women-related categories that are directly
comparable in both years.
"""

WOMEN_CORE_COLUMNS = [
    "RAPE",
    "DOWRY_DEATHS",
    "ASSAULT_WOMEN",
    "INSULT_WOMEN",
    "CRUELTY_WOMEN",
    "IMPORTATION_GIRLS"
]


df13_common["WOMEN_CRIME_CORE_TOTAL"] = (
    df13_common[WOMEN_CORE_COLUMNS]
    .sum(axis=1)
)

df14_common["WOMEN_CRIME_CORE_TOTAL"] = (
    df14_common[WOMEN_CORE_COLUMNS]
    .sum(axis=1)
)


# ============================================================
# COMMON VIOLENT CRIME
# ============================================================

VIOLENT_COLUMNS = [
    "MURDER",
    "ATTEMPT_MURDER",
    "CULPABLE_HOMICIDE",
    "RAPE",
    "KIDNAPPING_ABDUCTION",
    "DACOITY",
    "ROBBERY",
    "ASSAULT_WOMEN"
]

df13_common["VIOLENT_CRIME"] = (
    df13_common[VIOLENT_COLUMNS]
    .sum(axis=1)
)

df14_common["VIOLENT_CRIME"] = (
    df14_common[VIOLENT_COLUMNS]
    .sum(axis=1)
)


# ============================================================
# PROPERTY CRIME
# ============================================================

PROPERTY_COLUMNS = [
    "DACOITY",
    "ROBBERY",
    "BURGLARY",
    "THEFT",
    "AUTO_THEFT"
]

df13_common["PROPERTY_CRIME"] = (
    df13_common[PROPERTY_COLUMNS]
    .sum(axis=1)
)

df14_common["PROPERTY_CRIME"] = (
    df14_common[PROPERTY_COLUMNS]
    .sum(axis=1)
)


# ============================================================
# ECONOMIC / PUBLIC ORDER FEATURES
# ============================================================

ECONOMIC_COLUMNS = [
    "CRIMINAL_BREACH_TRUST",
    "CHEATING"
]

PUBLIC_ORDER_COLUMNS = [
    "RIOTS",
    "ARSON"
]

df13_common["ECONOMIC_OFFENCES"] = (
    df13_common[ECONOMIC_COLUMNS]
    .sum(axis=1)
)

df14_common["ECONOMIC_OFFENCES"] = (
    df14_common[ECONOMIC_COLUMNS]
    .sum(axis=1)
)

df13_common["PUBLIC_ORDER_CRIME"] = (
    df13_common[PUBLIC_ORDER_COLUMNS]
    .sum(axis=1)
)

df14_common["PUBLIC_ORDER_CRIME"] = (
    df14_common[PUBLIC_ORDER_COLUMNS]
    .sum(axis=1)
)


# ============================================================
# SHARES / PROXIES
# ============================================================

for df in [df13_common, df14_common]:

    df["WOMEN_CRIME_SHARE"] = (
        df["WOMEN_CRIME_CORE_TOTAL"]
        / df["TOTAL_IPC"].replace(0, pd.NA)
    )

    df["VIOLENT_CRIME_SHARE"] = (
        df["VIOLENT_CRIME"]
        / df["TOTAL_IPC"].replace(0, pd.NA)
    )

    df["PROPERTY_CRIME_SHARE"] = (
        df["PROPERTY_CRIME"]
        / df["TOTAL_IPC"].replace(0, pd.NA)
    )

    df["RAPE_SHARE"] = (
        df["RAPE"]
        / df["TOTAL_IPC"].replace(0, pd.NA)
    )

    df["KIDNAPPING_SHARE"] = (
        df["KIDNAPPING_ABDUCTION"]
        / df["TOTAL_IPC"].replace(0, pd.NA)
    )


# ============================================================
# SOURCE DATASET
# ============================================================

df13_common["SOURCE_DATASET"] = "DATASET_02_2013"
df14_common["SOURCE_DATASET"] = "DATASET_03_2014"


# ============================================================
# COMBINE
# ============================================================

print("\n" + "=" * 80)
print("COMBINING 2013 + 2014")
print("=" * 80)

combined = pd.concat(
    [
        df13_common,
        df14_common
    ],
    ignore_index=True
)


# ============================================================
# NUMERIC CLEANUP
# ============================================================

numeric_columns = combined.select_dtypes(
    include="number"
).columns

combined[numeric_columns] = (
    combined[numeric_columns]
    .apply(pd.to_numeric, errors="coerce")
)


# ============================================================
# CHECK INVALID VALUES
# ============================================================

print("\nCombined shape:", combined.shape)

print("\nYears:")
print(
    combined["YEAR"]
    .value_counts()
    .sort_index()
)

print("\nMissing values:")
missing = combined.isnull().sum()
print(
    missing[missing > 0]
)

print("\nInfinite values:")

import numpy as np

inf_count = np.isinf(
    combined[numeric_columns]
).sum().sum()

print(inf_count)


# ============================================================
# DUPLICATE KEY CHECK
# ============================================================

duplicate_keys = combined[
    combined.duplicated(
        subset=[
            "STATE_UT",
            "DISTRICT",
            "YEAR"
        ],
        keep=False
    )
].sort_values(
    [
        "STATE_UT",
        "DISTRICT",
        "YEAR"
    ]
)

print("\nDuplicate STATE/DISTRICT/YEAR keys:")
print(len(duplicate_keys))

if len(duplicate_keys) > 0:
    print(
        duplicate_keys[
            [
                "STATE_UT",
                "DISTRICT",
                "YEAR",
                "SOURCE_DATASET"
            ]
        ].to_string(index=False)
    )


# ============================================================
# SAVE
# ============================================================

combined.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("HARMONIZATION COMPLETED")
print("=" * 80)

print("\n2013 records:", len(df13_common))
print("2014 records:", len(df14_common))
print("Combined records:", len(combined))

print("\nCommon columns:")
print(len(combined.columns))

print("\nOutput:")
print(OUTPUT_FILE)

print("\nOriginal files were NOT modified.")

print("\n" + "=" * 80)
print("DATASET 2 + DATASET 3 READY")
print("=" * 80)