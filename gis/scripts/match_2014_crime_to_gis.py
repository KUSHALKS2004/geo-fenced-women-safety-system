import pandas as pd
import geopandas as gpd
from pathlib import Path
import re


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

CRIME_FILE = (
    BASE_DIR
    / "data"
    / "master_2001_2014"
    / "master_harmonized_ipc_2001_2014.csv"
)

GIS_FILE = (
    BASE_DIR
    / "gis"
    / "data"
    / "District_NWIC.geojson"
)

OUTPUT_DIR = BASE_DIR / "gis" / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(value):
    if pd.isna(value):
        return ""

    value = str(value).upper().strip()

    # Replace punctuation with spaces
    value = re.sub(r"[^A-Z0-9]+", " ", value)

    # Remove extra spaces
    value = re.sub(r"\s+", " ", value).strip()

    return value


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading crime data...")

crime = pd.read_csv(CRIME_FILE)

# Only 2014
crime_2014 = crime[crime["YEAR"] == 2014].copy()

print("2014 crime records:", len(crime_2014))


print("\nLoading GIS data...")

gis = gpd.read_file(GIS_FILE)

print("GIS records:", len(gis))


# ============================================================
# NORMALIZED KEYS
# ============================================================

crime_2014["STATE_KEY"] = crime_2014["STATE_UT"].apply(normalize_text)
crime_2014["DISTRICT_KEY"] = crime_2014["DISTRICT"].apply(normalize_text)

gis["STATE_KEY"] = gis["state_name"].apply(normalize_text)
gis["DISTRICT_KEY"] = gis["district"].apply(normalize_text)


# ============================================================
# EXACT STATE + DISTRICT MATCH
# ============================================================

gis_keys = set(
    zip(
        gis["STATE_KEY"],
        gis["DISTRICT_KEY"]
    )
)

crime_2014["GIS_EXACT_MATCH"] = crime_2014.apply(
    lambda row:
        (row["STATE_KEY"], row["DISTRICT_KEY"]) in gis_keys,
    axis=1
)


# ============================================================
# MATCH STATISTICS
# ============================================================

matched = crime_2014["GIS_EXACT_MATCH"].sum()
unmatched = len(crime_2014) - matched

print("\n==============================================")
print("GIS MATCHING SUMMARY")
print("==============================================")

print("2014 crime records :", len(crime_2014))
print("Exact GIS matches  :", matched)
print("Unmatched records  :", unmatched)

print(
    "Match percentage   :",
    round((matched / len(crime_2014)) * 100, 2),
    "%"
)


# ============================================================
# SHOW MATCHED RECORDS
# ============================================================

matched_df = crime_2014[
    crime_2014["GIS_EXACT_MATCH"]
].copy()

matched_df[
    [
        "STATE_UT",
        "DISTRICT",
        "YEAR",
        "TOTAL_IPC"
    ]
].to_csv(
    OUTPUT_DIR / "gis_matched_2014.csv",
    index=False
)


# ============================================================
# SHOW UNMATCHED RECORDS
# ============================================================

unmatched_df = crime_2014[
    ~crime_2014["GIS_EXACT_MATCH"]
].copy()

unmatched_df[
    [
        "STATE_UT",
        "DISTRICT",
        "YEAR",
        "TOTAL_IPC"
    ]
].to_csv(
    OUTPUT_DIR / "gis_unmatched_2014.csv",
    index=False
)


# ============================================================
# STATE-WISE SUMMARY
# ============================================================

state_summary = (
    crime_2014
    .groupby("STATE_UT")
    .agg(
        TOTAL_RECORDS=("DISTRICT", "count"),
        GIS_MATCHED=("GIS_EXACT_MATCH", "sum")
    )
    .reset_index()
)

state_summary["GIS_UNMATCHED"] = (
    state_summary["TOTAL_RECORDS"]
    - state_summary["GIS_MATCHED"]
)

state_summary["MATCH_PERCENT"] = (
    state_summary["GIS_MATCHED"]
    / state_summary["TOTAL_RECORDS"]
    * 100
)

state_summary = state_summary.sort_values(
    "MATCH_PERCENT"
)

state_summary.to_csv(
    OUTPUT_DIR / "gis_match_state_summary_2014.csv",
    index=False
)


# ============================================================
# DISPLAY UNMATCHED RECORDS
# ============================================================

print("\n==============================================")
print("UNMATCHED 2014 REPORTING UNITS")
print("==============================================")

if len(unmatched_df) > 0:

    print(
        unmatched_df[
            [
                "STATE_UT",
                "DISTRICT",
                "TOTAL_IPC"
            ]
        ].to_string(index=False)
    )

else:
    print("No unmatched records.")


# ============================================================
# SAVE BASIC MATCHING DATASET
# ============================================================

crime_2014[
    [
        "STATE_UT",
        "DISTRICT",
        "YEAR",
        "TOTAL_IPC",
        "GIS_EXACT_MATCH"
    ]
].to_csv(
    OUTPUT_DIR / "gis_matching_diagnostic_2014.csv",
    index=False
)


print("\n==============================================")
print("FILES SAVED")
print("==============================================")

print(
    OUTPUT_DIR / "gis_matching_diagnostic_2014.csv"
)

print(
    OUTPUT_DIR / "gis_matched_2014.csv"
)

print(
    OUTPUT_DIR / "gis_unmatched_2014.csv"
)

print(
    OUTPUT_DIR / "gis_match_state_summary_2014.csv"
)

print("\nGIS matching diagnostic completed.")