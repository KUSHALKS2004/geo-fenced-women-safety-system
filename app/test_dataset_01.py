import pandas as pd
from pathlib import Path


# ============================================================
# GEO-FENCED EMERGENCY ALERT AND SAFETY ASSISTANCE SYSTEM
# FOR WOMEN
#
# DATASET 01 - DISTRICT-WISE IPC CRIMES
# TESTING: 2001-2012
# ============================================================


# ============================================================
# 1. DATASET PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "dstrIPC_1.csv"


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("=" * 70)
print("DATASET 01 - DISTRICT-WISE IPC CRIMES")
print("GEO-FENCED EMERGENCY ALERT AND SAFETY ASSISTANCE SYSTEM")
print("=" * 70)

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully.")


# ============================================================
# 3. BASIC DATASET INFORMATION
# ============================================================

print("\n" + "-" * 70)
print("DATASET SHAPE")
print("-" * 70)

print("Number of rows    :", df.shape[0])
print("Number of columns :", df.shape[1])


# ============================================================
# 4. COLUMN NAMES
# ============================================================

print("\n" + "-" * 70)
print("COLUMN NAMES")
print("-" * 70)

for number, column in enumerate(df.columns, start=1):
    print(f"{number:2}. {column}")


# ============================================================
# 5. FIRST FIVE RECORDS
# ============================================================

print("\n" + "-" * 70)
print("FIRST 5 RECORDS")
print("-" * 70)

print(df.head())


# ============================================================
# 6. LAST FIVE RECORDS
# ============================================================

print("\n" + "-" * 70)
print("LAST 5 RECORDS")
print("-" * 70)

print(df.tail())


# ============================================================
# 7. DATA TYPES
# ============================================================

print("\n" + "-" * 70)
print("DATA TYPES")
print("-" * 70)

print(df.dtypes)


# ============================================================
# 8. MISSING VALUES
# ============================================================

print("\n" + "-" * 70)
print("MISSING VALUES")
print("-" * 70)

missing_values = df.isnull().sum()

if missing_values.sum() == 0:
    print("No missing values found.")
else:
    print(missing_values[missing_values > 0])


# ============================================================
# 9. DUPLICATE RECORDS
# ============================================================

print("\n" + "-" * 70)
print("DUPLICATE RECORDS")
print("-" * 70)

duplicate_count = df.duplicated().sum()

print("Duplicate rows:", duplicate_count)


# ============================================================
# 10. YEARS
# ============================================================

print("\n" + "-" * 70)
print("YEARS PRESENT IN DATASET")
print("-" * 70)

years = sorted(df["YEAR"].dropna().unique())

print(years)

print("Number of years:", len(years))


# ============================================================
# 11. STATES / UNION TERRITORIES
# ============================================================

print("\n" + "-" * 70)
print("STATES / UNION TERRITORIES")
print("-" * 70)

states = sorted(df["STATE/UT"].dropna().unique())

print("Number of States/UTs:", len(states))

for state in states:
    print(state)


# ============================================================
# 12. DISTRICTS
# ============================================================

print("\n" + "-" * 70)
print("DISTRICT INFORMATION")
print("-" * 70)

print("Number of unique districts:", df["DISTRICT"].nunique())


# ============================================================
# 13. RECORDS PER YEAR
# ============================================================

print("\n" + "-" * 70)
print("NUMBER OF RECORDS PER YEAR")
print("-" * 70)

records_per_year = df["YEAR"].value_counts().sort_index()

print(records_per_year)


# ============================================================
# 14. RECORDS PER STATE / UT
# ============================================================

print("\n" + "-" * 70)
print("TOP 20 STATES/UTs BY NUMBER OF RECORDS")
print("-" * 70)

records_per_state = (
    df["STATE/UT"]
    .value_counts()
    .head(20)
)

print(records_per_state)


# ============================================================
# 15. NUMERIC COLUMNS
# ============================================================

print("\n" + "-" * 70)
print("NUMERIC COLUMNS")
print("-" * 70)

numeric_columns = df.select_dtypes(include="number").columns

print("Number of numeric columns:", len(numeric_columns))

for column in numeric_columns:
    print(column)


# ============================================================
# 16. BASIC STATISTICAL SUMMARY
# ============================================================

print("\n" + "-" * 70)
print("STATISTICAL SUMMARY")
print("-" * 70)

print(df.describe())


# ============================================================
# 17. TOTAL IPC CRIME
# ============================================================

if "TOTAL IPC CRIMES" in df.columns:

    print("\n" + "-" * 70)
    print("TOTAL IPC CRIMES")
    print("-" * 70)

    print("Total reported IPC crimes:",
          df["TOTAL IPC CRIMES"].sum())

    print("Average IPC crimes per record:",
          df["TOTAL IPC CRIMES"].mean())

    print("Maximum IPC crimes in a record:",
          df["TOTAL IPC CRIMES"].max())


# ============================================================
# 18. WOMEN-SAFETY RELATED VARIABLES
# ============================================================

print("\n" + "-" * 70)
print("WOMEN-SAFETY RELATED COLUMNS")
print("-" * 70)

women_keywords = [
    "RAPE",
    "KIDNAPPING",
    "WOMEN",
    "DOWRY",
    "ASSAULT ON WOMEN",
    "INSULT TO MODESTY",
    "CRUELTY"
]

women_columns = []

for column in df.columns:

    column_upper = column.upper()

    for keyword in women_keywords:

        if keyword in column_upper:

            women_columns.append(column)
            break


for column in women_columns:
    print(column)


# ============================================================
# 19. WOMEN-RELATED CRIME SUMMARY
# ============================================================

print("\n" + "-" * 70)
print("WOMEN-RELATED CRIME TOTALS")
print("-" * 70)

for column in women_columns:

    if pd.api.types.is_numeric_dtype(df[column]):

        print(
            f"{column}: {df[column].sum():,.0f}"
        )


# ============================================================
# 20. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("INITIAL DATASET TESTING COMPLETE")
print("=" * 70)

print("\nNext stage:")
print("1. Exploratory Data Analysis")
print("2. Feature selection")
print("3. Feature engineering")
print("4. K-Means clustering")
print("5. Cluster evaluation")