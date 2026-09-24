import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "dataset_02_03_2013_2014"
    / "harmonized_ipc_2013_2014.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "dataset_02_03_2013_2014"
    / "eda"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 80)
print("EDA — DATASET 2 + DATASET 3")
print("2013–2014")
print("=" * 80)

df = pd.read_csv(INPUT_FILE)

print("\nDataset shape:", df.shape)


# ============================================================
# BASIC INFORMATION
# ============================================================

print("\n" + "=" * 80)
print("1. BASIC INFORMATION")
print("=" * 80)

print("\nYears:")
print(df["YEAR"].value_counts().sort_index())

print("\nStates/UTs:")
print(df["STATE_UT"].nunique())

print("\nReporting units:")
print(df["DISTRICT"].nunique())

print("\nSource datasets:")
print(df["SOURCE_DATASET"].value_counts())


# ============================================================
# MISSING VALUES
# ============================================================

print("\n" + "=" * 80)
print("2. MISSING VALUES")
print("=" * 80)

missing = df.isnull().sum()

missing = missing[missing > 0]

if len(missing) == 0:
    print("No missing values.")
else:
    print(missing)


# ============================================================
# ZERO TOTAL IPC
# ============================================================

print("\n" + "=" * 80)
print("3. ZERO TOTAL IPC RECORDS")
print("=" * 80)

zero_total = df[df["TOTAL_IPC"] == 0]

print("Records with TOTAL_IPC = 0:", len(zero_total))

if len(zero_total) > 0:
    print(
        zero_total[
            [
                "STATE_UT",
                "DISTRICT",
                "YEAR",
                "SOURCE_DATASET"
            ]
        ].to_string(index=False)
    )


# ============================================================
# DESCRIPTIVE STATISTICS
# ============================================================

print("\n" + "=" * 80)
print("4. DESCRIPTIVE STATISTICS")
print("=" * 80)

analysis_columns = [
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
    "OTHER_IPC",
    "TOTAL_IPC",
    "WOMEN_CRIME_CORE_TOTAL",
    "VIOLENT_CRIME",
    "PROPERTY_CRIME",
    "ECONOMIC_OFFENCES",
    "PUBLIC_ORDER_CRIME"
]

stats = df[analysis_columns].describe().T

print(stats.to_string())


# ============================================================
# YEARLY TOTALS
# ============================================================

print("\n" + "=" * 80)
print("5. YEARLY TOTALS")
print("=" * 80)

yearly_totals = (
    df.groupby("YEAR")[
        [
            "TOTAL_IPC",
            "WOMEN_CRIME_CORE_TOTAL",
            "VIOLENT_CRIME",
            "PROPERTY_CRIME",
            "ECONOMIC_OFFENCES",
            "PUBLIC_ORDER_CRIME"
        ]
    ]
    .sum()
    .reset_index()
)

print(yearly_totals.to_string(index=False))


# ============================================================
# YEARLY AVERAGES
# ============================================================

print("\n" + "=" * 80)
print("6. YEARLY AVERAGES PER REPORTING UNIT")
print("=" * 80)

yearly_mean = (
    df.groupby("YEAR")[
        [
            "TOTAL_IPC",
            "WOMEN_CRIME_CORE_TOTAL",
            "VIOLENT_CRIME",
            "PROPERTY_CRIME",
            "ECONOMIC_OFFENCES",
            "PUBLIC_ORDER_CRIME"
        ]
    ]
    .mean()
    .reset_index()
)

print(yearly_mean.to_string(index=False))


# ============================================================
# WOMEN-CRIME ANALYSIS
# ============================================================

print("\n" + "=" * 80)
print("7. WOMEN-RELATED CRIME ANALYSIS")
print("=" * 80)

women_columns = [
    "RAPE",
    "CUSTODIAL_RAPE",
    "DOWRY_DEATHS",
    "ASSAULT_WOMEN",
    "INSULT_WOMEN",
    "CRUELTY_WOMEN",
    "IMPORTATION_GIRLS"
]

women_totals = (
    df[women_columns]
    .sum()
    .sort_values(ascending=False)
)

print("\nTotal counts:")
print(women_totals)


# ============================================================
# WOMEN CRIME BY YEAR
# ============================================================

women_by_year = (
    df.groupby("YEAR")[women_columns]
    .sum()
)

print("\nWomen-related crimes by year:")
print(women_by_year.to_string())


# ============================================================
# TOP STATES BY TOTAL IPC
# ============================================================

print("\n" + "=" * 80)
print("8. TOP STATES/UTs BY TOTAL IPC")
print("=" * 80)

state_totals = (
    df.groupby("STATE_UT")[
        [
            "TOTAL_IPC",
            "WOMEN_CRIME_CORE_TOTAL",
            "VIOLENT_CRIME"
        ]
    ]
    .sum()
    .sort_values(
        "TOTAL_IPC",
        ascending=False
    )
)

print(
    state_totals.head(20).to_string()
)


# ============================================================
# TOP REPORTING UNITS
# ============================================================

print("\n" + "=" * 80)
print("9. TOP REPORTING UNITS")
print("=" * 80)

district_totals = (
    df.groupby(
        [
            "STATE_UT",
            "DISTRICT"
        ]
    )[
        [
            "TOTAL_IPC",
            "WOMEN_CRIME_CORE_TOTAL",
            "VIOLENT_CRIME"
        ]
    ]
    .sum()
    .sort_values(
        "TOTAL_IPC",
        ascending=False
    )
)

print(
    district_totals.head(20).to_string()
)


# ============================================================
# CORRELATION ANALYSIS
# ============================================================

print("\n" + "=" * 80)
print("10. CORRELATION WITH TOTAL IPC")
print("=" * 80)

correlation_columns = [
    "MURDER",
    "ATTEMPT_MURDER",
    "CULPABLE_HOMICIDE",
    "RAPE",
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
    "OTHER_IPC",
    "WOMEN_CRIME_CORE_TOTAL",
    "VIOLENT_CRIME",
    "PROPERTY_CRIME",
    "ECONOMIC_OFFENCES",
    "PUBLIC_ORDER_CRIME"
]

correlations = (
    df[correlation_columns + ["TOTAL_IPC"]]
    .corr()["TOTAL_IPC"]
    .drop("TOTAL_IPC")
    .sort_values(ascending=False)
)

print(correlations)


# ============================================================
# ZERO PERCENTAGES
# ============================================================

print("\n" + "=" * 80)
print("11. ZERO VALUE PERCENTAGES")
print("=" * 80)

zero_percentages = (
    df[analysis_columns]
    .eq(0)
    .mean()
    * 100
)

zero_percentages = (
    zero_percentages
    .sort_values(ascending=False)
)

print(zero_percentages.to_string())


# ============================================================
# SKEWNESS
# ============================================================

print("\n" + "=" * 80)
print("12. SKEWNESS")
print("=" * 80)

skewness = (
    df[analysis_columns]
    .skew()
    .sort_values(ascending=False)
)

print(skewness.to_string())


# ============================================================
# TOP 20 RECORDS
# ============================================================

print("\n" + "=" * 80)
print("13. TOP 20 REPORTING UNIT-YEAR RECORDS")
print("=" * 80)

top_records = (
    df.sort_values(
        "TOTAL_IPC",
        ascending=False
    )
    [
        [
            "STATE_UT",
            "DISTRICT",
            "YEAR",
            "TOTAL_IPC",
            "WOMEN_CRIME_CORE_TOTAL",
            "VIOLENT_CRIME"
        ]
    ]
    .head(20)
)

print(top_records.to_string(index=False))


# ============================================================
# 2013 VS 2014 CHANGE
# ============================================================

print("\n" + "=" * 80)
print("14. 2013 VS 2014 OVERALL COMPARISON")
print("=" * 80)

comparison_columns = [
    "TOTAL_IPC",
    "WOMEN_CRIME_CORE_TOTAL",
    "VIOLENT_CRIME",
    "PROPERTY_CRIME",
    "ECONOMIC_OFFENCES",
    "PUBLIC_ORDER_CRIME"
]

comparison = (
    df.groupby("YEAR")[comparison_columns]
    .sum()
)

print(comparison.to_string())

if 2013 in comparison.index and 2014 in comparison.index:

    print("\nPercentage change from 2013 to 2014:")

    pct_change = (
        (
            comparison.loc[2014]
            - comparison.loc[2013]
        )
        /
        comparison.loc[2013].replace(0, np.nan)
    ) * 100

    print(pct_change.to_string())


# ============================================================
# SAVE OUTPUTS
# ============================================================

print("\n" + "=" * 80)
print("SAVING EDA OUTPUTS")
print("=" * 80)


stats.to_csv(
    OUTPUT_DIR / "descriptive_statistics_02_03.csv"
)

yearly_totals.to_csv(
    OUTPUT_DIR / "yearly_totals_02_03.csv",
    index=False
)

yearly_mean.to_csv(
    OUTPUT_DIR / "yearly_means_02_03.csv",
    index=False
)

women_totals.to_csv(
    OUTPUT_DIR / "women_crime_totals_02_03.csv"
)

women_by_year.to_csv(
    OUTPUT_DIR / "women_crime_by_year_02_03.csv"
)

state_totals.to_csv(
    OUTPUT_DIR / "state_totals_02_03.csv"
)

district_totals.to_csv(
    OUTPUT_DIR / "reporting_unit_totals_02_03.csv"
)

correlations.to_csv(
    OUTPUT_DIR / "correlations_with_total_ipc_02_03.csv"
)

zero_percentages.to_csv(
    OUTPUT_DIR / "zero_percentages_02_03.csv"
)

skewness.to_csv(
    OUTPUT_DIR / "skewness_02_03.csv"
)

top_records.to_csv(
    OUTPUT_DIR / "top_records_02_03.csv",
    index=False
)

comparison.to_csv(
    OUTPUT_DIR / "2013_vs_2014_comparison_02_03.csv"
)


# ============================================================
# FINAL
# ============================================================

print("\nEDA files saved to:")

print(OUTPUT_DIR)

print("\n" + "=" * 80)
print("EDA COMPLETED")
print("=" * 80)