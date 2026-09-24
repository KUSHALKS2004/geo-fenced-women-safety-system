import pandas as pd
from pathlib import Path


# ============================================================
# PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

FILE = (
    BASE_DIR
    / "data"
    / "dataset_02_03_2013_2014"
    / "harmonized_ipc_2013_2014.csv"
)


# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(FILE)


# ============================================================
# FIND ZERO TOTAL IPC
# ============================================================

zero_total = df[
    df["TOTAL_IPC"] == 0
].copy()


print("=" * 80)
print("ZERO TOTAL IPC INVESTIGATION")
print("=" * 80)

print("\nTotal records with TOTAL_IPC = 0:")
print(len(zero_total))


# ============================================================
# DISPLAY
# ============================================================

columns = [
    "STATE_UT",
    "DISTRICT",
    "YEAR",
    "TOTAL_IPC",
    "WOMEN_CRIME_CORE_TOTAL",
    "VIOLENT_CRIME",
    "PROPERTY_CRIME",
    "RAPE",
    "KIDNAPPING_ABDUCTION",
    "SOURCE_DATASET"
]

print("\nRecords:")
print(
    zero_total[columns].to_string(index=False)
)


# ============================================================
# CHECK WHETHER ALL CRIME COUNTS ARE ZERO
# ============================================================

crime_columns = [
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
    "HURT",
    "DOWRY_DEATHS",
    "ASSAULT_WOMEN",
    "INSULT_WOMEN",
    "CRUELTY_WOMEN",
    "IMPORTATION_GIRLS",
    "DEATH_BY_NEGLIGENCE",
    "OTHER_IPC"
]

zero_total["SUM_COMMON_CRIMES"] = (
    zero_total[crime_columns].sum(axis=1)
)

print("\nCommon crime count sum:")
print(
    zero_total[
        [
            "STATE_UT",
            "DISTRICT",
            "YEAR",
            "TOTAL_IPC",
            "SUM_COMMON_CRIMES"
        ]
    ].to_string(index=False)
)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 80)
print("INVESTIGATION COMPLETED")
print("=" * 80)