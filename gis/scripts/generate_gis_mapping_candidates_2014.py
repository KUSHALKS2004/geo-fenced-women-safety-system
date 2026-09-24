import pandas as pd
import geopandas as gpd
from pathlib import Path
import re
from difflib import SequenceMatcher


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
# TEXT NORMALIZATION
# ============================================================

def normalize_text(value):
    if pd.isna(value):
        return ""

    value = str(value).upper().strip()

    value = re.sub(r"[^A-Z0-9]+", " ", value)

    value = re.sub(r"\s+", " ", value).strip()

    return value


# ============================================================
# LOAD DATA
# ============================================================

crime = pd.read_csv(CRIME_FILE)

crime_2014 = crime[
    crime["YEAR"] == 2014
].copy()

gis = gpd.read_file(GIS_FILE)


# ============================================================
# NORMALIZED STATE / DISTRICT
# ============================================================

crime_2014["STATE_KEY"] = (
    crime_2014["STATE_UT"]
    .apply(normalize_text)
)

crime_2014["DISTRICT_KEY"] = (
    crime_2014["DISTRICT"]
    .apply(normalize_text)
)

gis["STATE_KEY"] = (
    gis["state_name"]
    .apply(normalize_text)
)

gis["DISTRICT_KEY"] = (
    gis["district"]
    .apply(normalize_text)
)


# ============================================================
# EXACT MATCH CHECK
# ============================================================

gis_keys = set(
    zip(
        gis["STATE_KEY"],
        gis["DISTRICT_KEY"]
    )
)

crime_2014["EXACT_MATCH"] = crime_2014.apply(
    lambda row:
        (
            row["STATE_KEY"],
            row["DISTRICT_KEY"]
        ) in gis_keys,
    axis=1
)


unmatched = crime_2014[
    ~crime_2014["EXACT_MATCH"]
].copy()


# ============================================================
# SIMILARITY FUNCTION
# ============================================================

def similarity(a, b):

    if not a or not b:
        return 0

    return round(
        SequenceMatcher(
            None,
            a,
            b
        ).ratio() * 100,
        2
    )


# ============================================================
# GENERATE CANDIDATES
# ============================================================

results = []


for _, crime_row in unmatched.iterrows():

    state_key = crime_row["STATE_KEY"]
    district_key = crime_row["DISTRICT_KEY"]

    # Only compare with GIS districts
    # from the same state.
    state_gis = gis[
        gis["STATE_KEY"] == state_key
    ].copy()

    candidates = []

    for _, gis_row in state_gis.iterrows():

        score = similarity(
            district_key,
            gis_row["DISTRICT_KEY"]
        )

        candidates.append(
            {
                "GIS_DISTRICT": gis_row["district"],
                "GIS_STATE": gis_row["state_name"],
                "SIMILARITY_SCORE": score
            }
        )

    candidates = sorted(
        candidates,
        key=lambda x: x["SIMILARITY_SCORE"],
        reverse=True
    )

    # Keep top 3 candidates
    top_candidates = candidates[:3]

    for rank, candidate in enumerate(
        top_candidates,
        start=1
    ):

        results.append(
            {
                "CRIME_STATE": crime_row["STATE_UT"],
                "CRIME_DISTRICT": crime_row["DISTRICT"],
                "TOTAL_IPC": crime_row["TOTAL_IPC"],
                "CANDIDATE_RANK": rank,
                "GIS_STATE": candidate["GIS_STATE"],
                "GIS_DISTRICT": candidate["GIS_DISTRICT"],
                "SIMILARITY_SCORE": candidate[
                    "SIMILARITY_SCORE"
                ]
            }
        )


# ============================================================
# SAVE
# ============================================================

candidate_df = pd.DataFrame(results)

output_file = (
    OUTPUT_DIR
    / "gis_mapping_candidates_2014.csv"
)

candidate_df.to_csv(
    output_file,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n==============================================")
print("GIS MAPPING CANDIDATE ANALYSIS")
print("==============================================")

print(
    "Total 2014 crime records :",
    len(crime_2014)
)

print(
    "Exact GIS matches       :",
    crime_2014["EXACT_MATCH"].sum()
)

print(
    "Unmatched records       :",
    len(unmatched)
)

print(
    "\nCandidate file saved to:"
)

print(output_file)


# ============================================================
# DISPLAY TOP CANDIDATES
# ============================================================

print("\n==============================================")
print("TOP GIS CANDIDATES")
print("==============================================")

print(
    candidate_df[
        candidate_df["CANDIDATE_RANK"] == 1
    ]
    .sort_values(
        "SIMILARITY_SCORE",
        ascending=False
    )
    .head(50)
    .to_string(index=False)
)

print("\nGIS candidate generation completed.")