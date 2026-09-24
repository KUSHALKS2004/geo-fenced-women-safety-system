import pandas as pd
import geopandas as gpd
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

RISK_FILE = (
    BASE_DIR
    / "gis"
    / "outputs"
    / "predicted_2014_risk_by_reporting_unit.csv"
)

GIS_FILE = (
    BASE_DIR
    / "gis"
    / "data"
    / "District_NWIC.geojson"
)

OUTPUT_DIR = (
    BASE_DIR
    / "gis"
    / "outputs"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD RISK DATA
# ============================================================

print("\nLoading prediction/risk data...")

risk = pd.read_csv(
    RISK_FILE
)

print(
    "Prediction records:",
    len(risk)
)


# Only records that have an exact GIS match
risk_matched = risk[
    risk["GIS_EXACT_MATCH"] == True
].copy()

print(
    "GIS-matched records:",
    len(risk_matched)
)


# ============================================================
# LOAD GIS
# ============================================================

print("\nLoading district GIS...")

gis = gpd.read_file(
    GIS_FILE
)
# ============================================================
# CONVERT GIS DATA TO WGS84 FOR WEB MAPPING
# ============================================================

print("Original GIS CRS:", gis.crs)

if gis.crs is None:
    raise ValueError(
        "GIS dataset has no CRS information."
    )

gis = gis.to_crs(epsg=4326)

print("Converted GIS CRS:", gis.crs)
print(
    "GIS polygons:",
    len(gis)
)


# ============================================================
# NORMALIZED JOIN KEYS
# ============================================================

# Risk dataset already contains normalized keys
risk_matched["STATE_KEY"] = (
    risk_matched["STATE_KEY"]
    .astype(str)
)

risk_matched["DISTRICT_KEY"] = (
    risk_matched["DISTRICT_KEY"]
    .astype(str)
)


# GIS GeoJSON contains the original fields:
# state_name and district.
# Recreate normalized keys here.

def normalize_text(value):

    if pd.isna(value):
        return ""

    value = str(value).upper().strip()

    import re

    value = re.sub(
        r"[^A-Z0-9]+",
        " ",
        value
    )

    value = re.sub(
        r"\s+",
        " ",
        value
    ).strip()

    return value


def state_family(value):

    value = normalize_text(value)

    if value in {
        "ANDHRA PRADESH",
        "TELANGANA"
    }:
        return "ANDHRA PRADESH TELANGANA"

    return value


gis["STATE_KEY"] = (
    gis["state_name"]
    .apply(state_family)
)

gis["DISTRICT_KEY"] = (
    gis["district"]
    .apply(normalize_text)
)

# ============================================================
# CHECK GIS KEY DUPLICATES
# ============================================================

gis_duplicate_keys = (
    gis.groupby(
        ["STATE_KEY", "DISTRICT_KEY"]
    )
    .size()
)

duplicate_count = (
    gis_duplicate_keys[
        gis_duplicate_keys > 1
    ]
    .shape[0]
)

print(
    "\nDuplicate GIS state/district keys:",
    duplicate_count
)

if duplicate_count > 0:

    print(
        "\nDuplicate GIS keys:"
    )

    print(
        gis_duplicate_keys[
            gis_duplicate_keys > 1
        ]
    )


# ============================================================
# MERGE RISK DATA WITH GIS GEOMETRY
# ============================================================

risk_columns = [
    "STATE_UT",
    "DISTRICT",
    "YEAR",
    "TARGET_YEAR",
    "PREDICTED_WOMEN_CRIME_SHARE_CLIPPED",
    "PREDICTED_RISK_BAND",
    "WOMEN_CRIME_SHARE",
    "TOTAL_IPC",
    "GIS_EXACT_MATCH"
]

risk_for_join = risk_matched[
    [
        "STATE_KEY",
        "DISTRICT_KEY"
    ]
    + [
        column
        for column in risk_columns
        if column in risk_matched.columns
    ]
].copy()


gis_columns = [
    "STATE_KEY",
    "DISTRICT_KEY",
    "state_name",
    "district",
    "stcode",
    "utcode",
    "geometry"
]

gis_for_join = gis[
    [
        column
        for column in gis_columns
        if column in gis.columns
    ]
].copy()


# ============================================================
# MERGE
# ============================================================

risk_layer = gis_for_join.merge(
    risk_for_join,
    on=[
        "STATE_KEY",
        "DISTRICT_KEY"
    ],
    how="inner",
    validate="one_to_one"
)


# ============================================================
# CHECK
# ============================================================

print("\n==============================================")
print("GIS RISK LAYER")
print("==============================================")

print(
    "Risk records available:",
    len(risk_matched)
)

print(
    "GIS polygons joined:",
    len(risk_layer)
)


# ============================================================
# KEEP CLEAN COLUMNS
# ============================================================

output_columns = [
    "state_name",
    "district",
    "STATE_UT",
    "DISTRICT",
    "YEAR",
    "TARGET_YEAR",
    "PREDICTED_WOMEN_CRIME_SHARE_CLIPPED",
    "PREDICTED_RISK_BAND",
    "TOTAL_IPC",
    "GIS_EXACT_MATCH",
    "geometry"
]

output_columns = [
    column
    for column in output_columns
    if column in risk_layer.columns
]

risk_layer = risk_layer[
    output_columns
].copy()


# ============================================================
# SAVE CSV
# ============================================================

csv_output = (
    OUTPUT_DIR
    / "safeher_2014_risk_layer.csv"
)

risk_layer.drop(
    columns=["geometry"]
).to_csv(
    csv_output,
    index=False
)


# ============================================================
# SAVE GEOJSON
# ============================================================

geojson_output = (
    OUTPUT_DIR
    / "safeher_2014_risk_layer.geojson"
)

risk_layer.to_file(
    geojson_output,
    driver="GeoJSON"
)


# ============================================================
# RISK SUMMARY
# ============================================================

print("\n==============================================")
print("RISK BAND SUMMARY")
print("==============================================")

print(
    risk_layer[
        "PREDICTED_RISK_BAND"
    ]
    .value_counts()
    .sort_index()
)


print("\n==============================================")
print("FILES SAVED")
print("==============================================")

print(csv_output)
print(geojson_output)

print(
    "\n2014 GIS risk layer created successfully."
)