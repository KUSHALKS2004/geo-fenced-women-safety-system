import pandas as pd
import geopandas as gpd
from pathlib import Path
import re


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

PREDICTION_FILE = (
    BASE_DIR
    / "models"
    / "dataset_01_02_03_mlp"
    / "mlp_test_predictions.csv"
)

TEST_DATA_FILE = (
    BASE_DIR
    / "data"
    / "master_2001_2014"
    / "temporal_model"
    / "mlp"
    / "test_temporal_2011_2013.csv"
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
# TEXT NORMALIZATION
# ============================================================

def normalize_text(value):

    if pd.isna(value):
        return ""

    value = str(value).upper().strip()

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


# ============================================================
# STATE FAMILY
# ============================================================

def state_family(value):

    value = normalize_text(value)

    # Telangana was part of Andhra Pradesh
    # in the historical reporting structure.
    if value in {
        "ANDHRA PRADESH",
        "TELANGANA"
    }:
        return "ANDHRA PRADESH TELANGANA"

    return value


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading MLP predictions...")

predictions = pd.read_csv(
    PREDICTION_FILE
)

print(
    "Prediction rows:",
    len(predictions)
)


print("\nLoading temporal test dataset...")

test_data = pd.read_csv(
    TEST_DATA_FILE
)

print(
    "Test dataset rows:",
    len(test_data)
)


# ============================================================
# IMPORTANT:
# PREDICTION FILE AND TEST DATA WERE GENERATED
# IN THE SAME TEST-ROW ORDER.
# CREATE EXPLICIT ROW ID TO JOIN THEM.
# ============================================================

if len(predictions) != len(test_data):

    raise ValueError(
        "Prediction and test dataset row counts do not match."
    )


predictions["TEST_ROW_ID"] = range(
    len(predictions)
)

test_data["TEST_ROW_ID"] = range(
    len(test_data)
)


# ============================================================
# JOIN PREDICTION WITH LOCATION
# ============================================================

prediction_data = test_data.merge(
    predictions[
        [
            "TEST_ROW_ID",
            "PREDICTED_WOMEN_CRIME_SHARE",
            "PREDICTED_WOMEN_CRIME_SHARE_CLIPPED",
            "ABSOLUTE_ERROR"
        ]
    ],
    on="TEST_ROW_ID",
    how="left",
    validate="one_to_one"
)


# ============================================================
# SELECT 2013 -> 2014 PREDICTIONS
# ============================================================

prediction_2014 = prediction_data[
    (
        prediction_data["YEAR"] == 2013
    )
    &
    (
        prediction_data["TARGET_YEAR"] == 2014
    )
].copy()


print(
    "\n2013 -> 2014 prediction records:",
    len(prediction_2014)
)


# ============================================================
# NORMALIZED LOCATION KEYS
# ============================================================

prediction_2014["STATE_KEY"] = (
    prediction_2014["STATE_UT"]
    .apply(state_family)
)

prediction_2014["DISTRICT_KEY"] = (
    prediction_2014["DISTRICT"]
    .apply(normalize_text)
)


# ============================================================
# LOAD GIS
# ============================================================

print("\nLoading GIS boundaries...")

gis = gpd.read_file(
    GIS_FILE
)

print(
    "GIS polygons:",
    len(gis)
)

print(
    "Original GIS CRS:",
    gis.crs
)


# ============================================================
# CONVERT GIS TO WGS84
# ============================================================
#
# The NWIC source uses EPSG:7755.
# Leaflet/web maps require WGS84 coordinates.
#
# The conversion is done here so that the GIS data used
# downstream is in EPSG:4326.
# ============================================================

if gis.crs is None:

    raise ValueError(
        "GIS dataset has no CRS information."
    )


gis = gis.to_crs(
    epsg=4326
)


print(
    "Converted GIS CRS:",
    gis.crs
)


# ============================================================
# GIS NORMALIZED LOCATION KEYS
# ============================================================

gis["STATE_KEY"] = (
    gis["state_name"]
    .apply(state_family)
)

gis["DISTRICT_KEY"] = (
    gis["district"]
    .apply(normalize_text)
)


# ============================================================
# EXACT STATE + DISTRICT MATCH
# ============================================================

gis_keys = set(
    zip(
        gis["STATE_KEY"],
        gis["DISTRICT_KEY"]
    )
)


prediction_2014["GIS_EXACT_MATCH"] = (
    prediction_2014.apply(
        lambda row:
            (
                row["STATE_KEY"],
                row["DISTRICT_KEY"]
            )
            in gis_keys,
        axis=1
    )
)


# ============================================================
# MATCH TYPE
# ============================================================

prediction_2014["GIS_MATCH_TYPE"] = (
    prediction_2014["GIS_EXACT_MATCH"]
    .map({
        True: "EXACT",
        False: "UNMATCHED"
    })
)


# ============================================================
# CONSERVATIVE ADMINISTRATIVE PROXY MAPPINGS
# ============================================================
#
# The 2014 crime dataset uses:
#
#   Bengaluru City
#   Bengaluru District
#
# The GIS dataset uses:
#
#   Bengaluru Urban
#   Bengaluru Rural
#
# These are NOT treated as exact textual matches.
# They are explicitly recorded as administrative proxies.
#
# No unrestricted fuzzy matching is performed.
# ============================================================

ADMINISTRATIVE_PROXY_MAPPING = {

    (
        "KARNATAKA",
        "BENGALURU CITY"
    ): (
        "KARNATAKA",
        "BENGALURU URBAN"
    ),

    (
        "KARNATAKA",
        "BENGALURU DISTRICT"
    ): (
        "KARNATAKA",
        "BENGALURU RURAL"
    )

}



# ============================================================
# CREATE GIS MAPPED KEYS FOR EXACT MATCHES
# ============================================================

exact_mask = (
    prediction_2014["GIS_MATCH_TYPE"] == "EXACT"
)

prediction_2014.loc[
    exact_mask,
    "GIS_MAPPED_STATE_KEY"
] = prediction_2014.loc[
    exact_mask,
    "STATE_KEY"
]

prediction_2014.loc[
    exact_mask,
    "GIS_MAPPED_DISTRICT_KEY"
] = prediction_2014.loc[
    exact_mask,
    "DISTRICT_KEY"
]


# ============================================================
# FINAL GIS MATCH FLAG
# ============================================================

prediction_2014["GIS_MAPPED"] = (
    prediction_2014["GIS_MATCH_TYPE"]
    .isin([
        "EXACT",
        "ADMINISTRATIVE_PROXY"
    ])
)


# ============================================================
# RISK BAND
# ============================================================

LOW_MEDIUM_THRESHOLD = 0.050903787688866296

MEDIUM_HIGH_THRESHOLD = 0.08541311974661926


def assign_risk_band(value):

    if value < LOW_MEDIUM_THRESHOLD:

        return "LOW"

    elif value < MEDIUM_HIGH_THRESHOLD:

        return "MEDIUM"

    else:

        return "HIGH"


prediction_2014[
    "PREDICTED_RISK_BAND"
] = prediction_2014[
    "PREDICTED_WOMEN_CRIME_SHARE_CLIPPED"
].apply(
    assign_risk_band
)


# ============================================================
# MATCH SUMMARY
# ============================================================

exact_matches = (
    prediction_2014[
        "GIS_MATCH_TYPE"
    ] == "EXACT"
).sum()


proxy_matches = (
    prediction_2014[
        "GIS_MATCH_TYPE"
    ] == "ADMINISTRATIVE_PROXY"
).sum()


mapped = (
    prediction_2014[
        "GIS_MAPPED"
    ].sum()
)


unmatched = (
    len(prediction_2014)
    - mapped
)


print(
    "\n=============================================="
)

print(
    "2014 PREDICTION GIS MATCHING"
)

print(
    "=============================================="
)


print(
    "2013 -> 2014 predictions :",
    len(prediction_2014)
)


print(
    "Exact GIS matches        :",
    exact_matches
)


print(
    "Administrative proxies   :",
    proxy_matches
)


print(
    "Total GIS mapped         :",
    mapped
)


print(
    "GIS unmatched            :",
    unmatched
)


print(
    "Mapped percentage        :",
    round(
        mapped
        / len(prediction_2014)
        * 100,
        2
    ),
    "%"
)


# ============================================================
# MATCH TYPE SUMMARY
# ============================================================

print(
    "\n=============================================="
)

print(
    "GIS MATCH TYPE SUMMARY"
)

print(
    "=============================================="
)

print(
    prediction_2014[
        "GIS_MATCH_TYPE"
    ]
    .value_counts()
)


# ============================================================
# SAVE PREDICTION DATASET
# ============================================================

prediction_output = (
    OUTPUT_DIR
    / "predicted_2014_risk_by_reporting_unit.csv"
)

prediction_2014.to_csv(
    prediction_output,
    index=False
)


# ============================================================
# SAVE UNMATCHED
# ============================================================

unmatched_output = (
    OUTPUT_DIR
    / "predicted_2014_gis_unmatched.csv"
)

prediction_2014[
    ~prediction_2014["GIS_MAPPED"]
][
    [
        "STATE_UT",
        "DISTRICT",
        "YEAR",
        "TARGET_YEAR",
        "PREDICTED_WOMEN_CRIME_SHARE_CLIPPED",
        "PREDICTED_RISK_BAND"
    ]
].to_csv(
    unmatched_output,
    index=False
)


# ============================================================
# SAVE ADMINISTRATIVE PROXY RECORDS
# ============================================================

proxy_output = (
    OUTPUT_DIR
    / "gis_administrative_proxy_matches_2014.csv"
)

prediction_2014[
    prediction_2014[
        "GIS_MATCH_TYPE"
    ] == "ADMINISTRATIVE_PROXY"
][
    [
        "STATE_UT",
        "DISTRICT",
        "YEAR",
        "TARGET_YEAR",
        "GIS_MAPPED_STATE_KEY",
        "GIS_MAPPED_DISTRICT_KEY",
        "PREDICTED_WOMEN_CRIME_SHARE_CLIPPED",
        "PREDICTED_RISK_BAND"
    ]
].to_csv(
    proxy_output,
    index=False
)


# ============================================================
# DISPLAY RISK SUMMARY
# ============================================================

print(
    "\n=============================================="
)

print(
    "PREDICTED RISK BAND DISTRIBUTION"
)

print(
    "=============================================="
)

print(
    prediction_2014[
        "PREDICTED_RISK_BAND"
    ]
    .value_counts()
    .sort_index()
)


# ============================================================
# DISPLAY ADMINISTRATIVE PROXIES
# ============================================================

print(
    "\n=============================================="
)

print(
    "ADMINISTRATIVE PROXY MATCHES"
)

print(
    "=============================================="
)

proxy_records = prediction_2014[
    prediction_2014[
        "GIS_MATCH_TYPE"
    ] == "ADMINISTRATIVE_PROXY"
][
    [
        "STATE_UT",
        "DISTRICT",
        "GIS_MAPPED_STATE_KEY",
        "GIS_MAPPED_DISTRICT_KEY",
        "PREDICTED_WOMEN_CRIME_SHARE_CLIPPED",
        "PREDICTED_RISK_BAND"
    ]
]


if len(proxy_records) > 0:

    print(
        proxy_records.to_string(
            index=False
        )
    )

else:

    print(
        "No administrative proxy matches found."
    )


# ============================================================
# DISPLAY UNMATCHED
# ============================================================

print(
    "\n=============================================="
)

print(
    "UNMATCHED PREDICTION RECORDS"
)

print(
    "=============================================="
)

print(
    prediction_2014[
        ~prediction_2014["GIS_MAPPED"]
    ][
        [
            "STATE_UT",
            "DISTRICT",
            "PREDICTED_WOMEN_CRIME_SHARE_CLIPPED",
            "PREDICTED_RISK_BAND"
        ]
    ]
    .to_string(index=False)
)


# ============================================================
# FILES SAVED
# ============================================================

print(
    "\n=============================================="
)

print(
    "FILES SAVED"
)

print(
    "=============================================="
)

print(
    prediction_output
)

print(
    unmatched_output
)

print(
    proxy_output
)


print(
    "\n2014 GIS prediction preparation completed."
)