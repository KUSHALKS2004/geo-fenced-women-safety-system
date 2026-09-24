import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# SAFEHER-AI
# DATASET 01 - EXPLORATORY DATA ANALYSIS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "data" / "cleaned_district_ipc_2001_2012_v2.csv"

OUTPUT_DIR = BASE_DIR / "data" / "eda_dataset_01"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("SAFEHER-AI - DATASET 01 EDA")
print("=" * 70)

df = pd.read_csv(INPUT_FILE)

print("\nDataset loaded successfully.")

print(f"Rows    : {len(df)}")
print(f"Columns : {len(df.columns)}")


# ============================================================
# 1. DATASET INFORMATION
# ============================================================

print("\n" + "-" * 70)
print("1. DATASET INFORMATION")
print("-" * 70)

print("\nColumn names:")

for i, column in enumerate(df.columns, start=1):
    print(f"{i:2}. {column}")


print("\nData types:")

print(df.dtypes)


# ============================================================
# 2. YEARS
# ============================================================

print("\n" + "-" * 70)
print("2. YEAR DISTRIBUTION")
print("-" * 70)

year_counts = df["YEAR"].value_counts().sort_index()

print("\nRecords per year:")
print(year_counts.to_string())


# ============================================================
# YEARLY TOTAL IPC CRIMES
# ============================================================

yearly_crime = (
    df.groupby("YEAR")["TOTAL IPC CRIMES"]
    .sum()
    .sort_index()
)

print("\nTotal IPC crimes by year:")
print(yearly_crime.to_string())


# Plot

plt.figure(figsize=(10, 5))

plt.plot(
    yearly_crime.index,
    yearly_crime.values,
    marker="o"
)

plt.title("Total IPC Crimes by Year")
plt.xlabel("Year")
plt.ylabel("Total IPC Crimes")
plt.grid(True)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "01_total_ipc_by_year.png",
    dpi=300
)

plt.show()


# ============================================================
# 3. STATE/UT ANALYSIS
# ============================================================

print("\n" + "-" * 70)
print("3. STATE/UT ANALYSIS")
print("-" * 70)

state_counts = df["STATE/UT"].value_counts()

print("\nNumber of records by State/UT:")
print(state_counts.to_string())


# Total crimes by State/UT

state_crime = (
    df.groupby("STATE/UT")["TOTAL IPC CRIMES"]
    .sum()
    .sort_values(ascending=False)
)

print("\nTop 15 States/UTs by total IPC crimes:")

print(
    state_crime.head(15).to_string()
)


# Plot top 15

plt.figure(figsize=(12, 6))

state_crime.head(15).sort_values().plot(
    kind="barh"
)

plt.title("Top 15 States/UTs by Total IPC Crimes")
plt.xlabel("Total IPC Crimes")
plt.ylabel("State/UT")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "02_top_states_ipc_crimes.png",
    dpi=300
)

plt.show()


# ============================================================
# 4. DISTRICT ANALYSIS
# ============================================================

print("\n" + "-" * 70)
print("4. DISTRICT ANALYSIS")
print("-" * 70)

district_counts = df["DISTRICT"].value_counts()

print(f"\nUnique district/reporting-unit names: {df['DISTRICT'].nunique()}")


district_crime = (
    df.groupby(["STATE/UT", "DISTRICT"])["TOTAL IPC CRIMES"]
    .sum()
    .sort_values(ascending=False)
)

print("\nTop 20 district/reporting units by total IPC crimes:")

print(
    district_crime.head(20).to_string()
)


# ============================================================
# 5. WOMEN-RELATED CRIMES
# ============================================================

print("\n" + "-" * 70)
print("5. WOMEN-RELATED CRIME ANALYSIS")
print("-" * 70)


women_columns = [
    "RAPE",
    "CUSTODIAL RAPE",
    "OTHER RAPE",
    "KIDNAPPING & ABDUCTION",
    "KIDNAPPING AND ABDUCTION OF WOMEN AND GIRLS",
    "DOWRY DEATHS",
    "ASSAULT ON WOMEN WITH INTENT TO OUTRAGE HER MODESTY",
    "INSULT TO MODESTY OF WOMEN",
    "CRUELTY BY HUSBAND OR HIS RELATIVES",
    "IMPORTATION OF GIRLS FROM FOREIGN COUNTRIES"
]


print("\nTotal women-related crimes:")

women_totals = df[women_columns].sum().sort_values(ascending=False)

print(women_totals.to_string())


# Plot

plt.figure(figsize=(12, 7))

women_totals.sort_values().plot(
    kind="barh"
)

plt.title("Women-Related Crime Categories")
plt.xlabel("Total Reported Cases")
plt.ylabel("Crime Category")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "03_women_related_crimes.png",
    dpi=300
)

plt.show()


# ============================================================
# 6. GENERAL CRIME CATEGORIES
# ============================================================

print("\n" + "-" * 70)
print("6. GENERAL CRIME CATEGORY ANALYSIS")
print("-" * 70)


general_columns = [
    "MURDER",
    "ATTEMPT TO MURDER",
    "CULPABLE HOMICIDE NOT AMOUNTING TO MURDER",
    "DACOITY",
    "PREPARATION AND ASSEMBLY FOR DACOITY",
    "ROBBERY",
    "BURGLARY",
    "THEFT",
    "AUTO THEFT",
    "OTHER THEFT",
    "RIOTS",
    "CRIMINAL BREACH OF TRUST",
    "CHEATING",
    "COUNTERFIETING",
    "ARSON",
    "HURT/GREVIOUS HURT",
    "CAUSING DEATH BY NEGLIGENCE",
    "OTHER IPC CRIMES"
]


general_totals = (
    df[general_columns]
    .sum()
    .sort_values(ascending=False)
)

print("\nGeneral crime category totals:")

print(general_totals.to_string())


# ============================================================
# 7. DESCRIPTIVE STATISTICS
# ============================================================

print("\n" + "-" * 70)
print("7. DESCRIPTIVE STATISTICS")
print("-" * 70)

numeric_columns = df.select_dtypes(
    include=np.number
).columns

statistics = df[numeric_columns].describe().T

print(
    statistics[
        ["count", "mean", "std", "min", "50%", "max"]
    ].to_string()
)


statistics.to_csv(
    OUTPUT_DIR / "descriptive_statistics.csv"
)


# ============================================================
# 8. CORRELATION ANALYSIS
# ============================================================

print("\n" + "-" * 70)
print("8. CORRELATION WITH TOTAL IPC CRIMES")
print("-" * 70)

correlations = (
    df[numeric_columns]
    .corr()["TOTAL IPC CRIMES"]
    .sort_values(ascending=False)
)

print(
    correlations.to_string()
)


correlations.to_csv(
    OUTPUT_DIR / "correlation_with_total_ipc.csv"
)


# ============================================================
# 9. HIGHLY SKEWED FEATURES
# ============================================================

print("\n" + "-" * 70)
print("9. SKEWNESS")
print("-" * 70)

skewness = (
    df[numeric_columns]
    .skew()
    .sort_values(ascending=False)
)

print(skewness.to_string())

skewness.to_csv(
    OUTPUT_DIR / "feature_skewness.csv"
)


# ============================================================
# 10. ZERO-HEAVY FEATURES
# ============================================================

print("\n" + "-" * 70)
print("10. ZERO VALUE ANALYSIS")
print("-" * 70)

zero_percentage = (
    (df[numeric_columns] == 0).mean() * 100
).sort_values(ascending=False)

print("\nPercentage of zero values:")
print(zero_percentage.to_string())

zero_percentage.to_csv(
    OUTPUT_DIR / "zero_percentage.csv"
)


# ============================================================
# 11. TOP DISTRICT-YEAR RECORDS
# ============================================================

print("\n" + "-" * 70)
print("11. TOP DISTRICT-YEAR RECORDS")
print("-" * 70)

top_records = (
    df[
        [
            "STATE/UT",
            "DISTRICT",
            "YEAR",
            "TOTAL IPC CRIMES",
            "RAPE",
            "KIDNAPPING & ABDUCTION",
            "DOWRY DEATHS",
            "ASSAULT ON WOMEN WITH INTENT TO OUTRAGE HER MODESTY",
            "CRUELTY BY HUSBAND OR HIS RELATIVES"
        ]
    ]
    .sort_values(
        "TOTAL IPC CRIMES",
        ascending=False
    )
    .head(20)
)

print(
    top_records.to_string(index=False)
)


top_records.to_csv(
    OUTPUT_DIR / "top_20_district_year_records.csv",
    index=False
)


# ============================================================
# 12. WOMEN CRIME FEATURE
# ============================================================

print("\n" + "-" * 70)
print("12. WOMEN CRIME FEATURE")
print("-" * 70)

df["WOMEN_CRIME_TOTAL"] = df[women_columns].sum(axis=1)

print(
    df["WOMEN_CRIME_TOTAL"].describe().to_string()
)


# ============================================================
# SAVE EDA-READY COPY
# ============================================================

eda_file = OUTPUT_DIR / "dataset_01_eda_ready.csv"

df.to_csv(
    eda_file,
    index=False
)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("EDA COMPLETED")
print("=" * 70)

print("\nEDA output directory:")
print(OUTPUT_DIR)

print("\nGenerated files:")

for file in sorted(OUTPUT_DIR.iterdir()):
    print(" -", file.name)

print("\nNext stage:")
print("Feature engineering → K-Means preparation")