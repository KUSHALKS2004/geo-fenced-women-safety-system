import pandas as pd
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATASET_02 = BASE_DIR / "data" / "dstrIPC_2013.csv"
DATASET_03 = BASE_DIR / "data" / "dstrIPC_1_2014.csv"


# ============================================================
# FUNCTION
# ============================================================

def inspect_dataset(file_path, dataset_name):

    print("\n")
    print("=" * 80)
    print(f"{dataset_name}")
    print("=" * 80)

    df = pd.read_csv(file_path)

    print("\nFILE:")
    print(file_path)

    print("\nSHAPE:")
    print(df.shape)

    print("\nCOLUMNS:")
    for i, column in enumerate(df.columns, start=1):
        print(f"{i:02d}. {column}")

    print("\nDATA TYPES:")
    print(df.dtypes)

    print("\nMISSING VALUES:")
    missing = df.isnull().sum()
    print(missing[missing > 0])

    if missing.sum() == 0:
        print("No missing values.")

    print("\nEXACT DUPLICATE ROWS:")
    print(df.duplicated().sum())

    print("\nYEAR VALUES:")
    if "YEAR" in df.columns:
        print(df["YEAR"].unique())

    print("\nSTATE/UT INFORMATION:")
    if "STATE/UT" in df.columns:
        print("Unique states/UTs:", df["STATE/UT"].nunique())
        print(sorted(df["STATE/UT"].dropna().unique()))

    print("\nDISTRICT INFORMATION:")
    if "DISTRICT" in df.columns:
        print("Unique district/reporting units:", df["DISTRICT"].nunique())

        print("\nFirst 30 reporting units:")
        print(
            df["DISTRICT"]
            .dropna()
            .drop_duplicates()
            .sort_values()
            .head(30)
            .to_string(index=False)
        )

    print("\nSAMPLE RECORDS:")
    print(df.head(5).to_string())

    print("\nNUMERIC COLUMNS:")
    numeric_columns = df.select_dtypes(
        include="number"
    ).columns.tolist()

    print("Number of numeric columns:", len(numeric_columns))
    print(numeric_columns)

    print("\nTOTAL COLUMNS CONTAINING 'TOTAL':")
    total_columns = [
        col for col in df.columns
        if "TOTAL" in col.upper()
    ]

    print(total_columns)

    # --------------------------------------------------------
    # TOTAL RECORDS
    # --------------------------------------------------------

    if "DISTRICT" in df.columns:

        total_records = df[
            df["DISTRICT"]
            .astype(str)
            .str.upper()
            .str.contains("TOTAL", na=False)
        ]

        print("\nRECORDS WITH 'TOTAL' IN DISTRICT:")
        print(len(total_records))

        if len(total_records) > 0:
            print(
                total_records[
                    ["STATE/UT", "DISTRICT"]
                ]
                .drop_duplicates()
                .to_string(index=False)
            )

    return df


# ============================================================
# LOAD + INSPECT BOTH
# ============================================================

df_2013 = inspect_dataset(
    DATASET_02,
    "DATASET 2 — 2013"
)

df_2014 = inspect_dataset(
    DATASET_03,
    "DATASET 3 — 2014"
)


# ============================================================
# COLUMN COMPARISON
# ============================================================

print("\n")
print("=" * 80)
print("2013 vs 2014 COLUMN COMPARISON")
print("=" * 80)

columns_2013 = set(df_2013.columns)
columns_2014 = set(df_2014.columns)

only_2013 = sorted(columns_2013 - columns_2014)
only_2014 = sorted(columns_2014 - columns_2013)
common = sorted(columns_2013 & columns_2014)

print("\nCOMMON COLUMNS:")
print("Count:", len(common))
for col in common:
    print(" -", col)

print("\nONLY IN 2013:")
print("Count:", len(only_2013))
for col in only_2013:
    print(" -", col)

print("\nONLY IN 2014:")
print("Count:", len(only_2014))
for col in only_2014:
    print(" -", col)


# ============================================================
# STATE COMPARISON
# ============================================================

if "STATE/UT" in df_2013.columns and "STATE/UT" in df_2014.columns:

    states_2013 = set(
        df_2013["STATE/UT"].dropna().unique()
    )

    states_2014 = set(
        df_2014["STATE/UT"].dropna().unique()
    )

    print("\n")
    print("=" * 80)
    print("STATE/UT COMPARISON")
    print("=" * 80)

    print("\nStates/UTs in both:")
    print(len(states_2013 & states_2014))

    print("\nOnly in 2013:")
    print(sorted(states_2013 - states_2014))

    print("\nOnly in 2014:")
    print(sorted(states_2014 - states_2013))


# ============================================================
# FINAL
# ============================================================

print("\n")
print("=" * 80)
print("INSPECTION COMPLETED")
print("=" * 80)

print("\nNo files were modified.")
print("No cleaning was performed.")
print("No records were deleted.")

print("\nNext step:")
print("Use the inspection results to create the common")
print("2013–2014 preprocessing pipeline.")