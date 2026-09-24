import pandas as pd
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "master_2001_2014"
    / "master_harmonized_ipc_2001_2014.csv"
)


# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(INPUT_FILE)

print("=" * 80)
print("TEMPORAL REPORTING-UNIT OVERLAP DIAGNOSTIC")
print("2012 → 2013 → 2014")
print("=" * 80)


# ============================================================
# GET YEAR-SPECIFIC RECORDS
# ============================================================

df2012 = df[df["YEAR"] == 2012].copy()
df2013 = df[df["YEAR"] == 2013].copy()
df2014 = df[df["YEAR"] == 2014].copy()


# ============================================================
# BASIC COUNTS
# ============================================================

print("\nRecord counts:")

print("2012:", len(df2012))
print("2013:", len(df2013))
print("2014:", len(df2014))


# ============================================================
# UNIQUE STATE/DISTRICT PAIRS
# ============================================================

keys_2012 = set(
    zip(
        df2012["STATE_UT"].astype(str),
        df2012["DISTRICT"].astype(str)
    )
)

keys_2013 = set(
    zip(
        df2013["STATE_UT"].astype(str),
        df2013["DISTRICT"].astype(str)
    )
)

keys_2014 = set(
    zip(
        df2014["STATE_UT"].astype(str),
        df2014["DISTRICT"].astype(str)
    )
)


# ============================================================
# EXACT OVERLAP
# ============================================================

overlap_2012_2013 = (
    keys_2012 & keys_2013
)

overlap_2013_2014 = (
    keys_2013 & keys_2014
)

print("\n" + "=" * 80)
print("EXACT REPORTING-UNIT OVERLAP")
print("=" * 80)

print(
    "\n2012 → 2013 exact overlap:",
    len(overlap_2012_2013)
)

print(
    "2013 → 2014 exact overlap:",
    len(overlap_2013_2014)
)


# ============================================================
# STATE-LEVEL OVERLAP
# ============================================================

states_2012 = set(
    df2012["STATE_UT"].astype(str)
)

states_2013 = set(
    df2013["STATE_UT"].astype(str)
)

states_2014 = set(
    df2014["STATE_UT"].astype(str)
)

print("\n" + "=" * 80)
print("STATE/UT OVERLAP")
print("=" * 80)

print(
    "\n2012 states:",
    len(states_2012)
)

print(
    "2013 states:",
    len(states_2013)
)

print(
    "2014 states:",
    len(states_2014)
)

print(
    "\nCommon 2012/2013 states:",
    len(states_2012 & states_2013)
)

print(
    "Common 2013/2014 states:",
    len(states_2013 & states_2014)
)


# ============================================================
# REPORTING UNITS THAT DISAPPEAR
# ============================================================

print("\n" + "=" * 80)
print("2012 REPORTING UNITS NOT FOUND IN 2013")
print("=" * 80)

missing_2013 = sorted(
    keys_2012 - keys_2013
)

print(
    "Count:",
    len(missing_2013)
)

for state, district in missing_2013:

    print(
        f"{state} | {district}"
    )


# ============================================================
# REPORTING UNITS THAT APPEAR IN 2013
# ============================================================

print("\n" + "=" * 80)
print("2013 REPORTING UNITS NOT FOUND IN 2012")
print("=" * 80)

new_2013 = sorted(
    keys_2013 - keys_2012
)

print(
    "Count:",
    len(new_2013)
)

for state, district in new_2013:

    print(
        f"{state} | {district}"
    )


# ============================================================
# 2013 → 2014 MISMATCH
# ============================================================

print("\n" + "=" * 80)
print("2013 REPORTING UNITS NOT FOUND IN 2014")
print("=" * 80)

missing_2014 = sorted(
    keys_2013 - keys_2014
)

print(
    "Count:",
    len(missing_2014)
)

for state, district in missing_2014:

    print(
        f"{state} | {district}"
    )


# ============================================================
# 2014 NEW REPORTING UNITS
# ============================================================

print("\n" + "=" * 80)
print("2014 REPORTING UNITS NOT FOUND IN 2013")
print("=" * 80)

new_2014 = sorted(
    keys_2014 - keys_2013
)

print(
    "Count:",
    len(new_2014)
)

for state, district in new_2014:

    print(
        f"{state} | {district}"
    )


# ============================================================
# CASE/WHITESPACE NORMALIZATION TEST
# ============================================================

def normalize_text(value):

    return (
        str(value)
        .strip()
        .upper()
        .replace(".", "")
        .replace(",", "")
        .replace("-", " ")
    )


normalized_2012 = set(
    (
        normalize_text(state),
        normalize_text(district)
    )
    for state, district in keys_2012
)

normalized_2013 = set(
    (
        normalize_text(state),
        normalize_text(district)
    )
    for state, district in keys_2013
)

normalized_2014 = set(
    (
        normalize_text(state),
        normalize_text(district)
    )
    for state, district in keys_2014
)


print("\n" + "=" * 80)
print("NORMALIZED TEXT OVERLAP TEST")
print("=" * 80)

print(
    "\n2012 → 2013 normalized overlap:",
    len(normalized_2012 & normalized_2013)
)

print(
    "2013 → 2014 normalized overlap:",
    len(normalized_2013 & normalized_2014)
)


# ============================================================
# NORMALIZED MATCH EXAMPLES
# ============================================================

print("\n" + "=" * 80)
print("NORMALIZED MATCH EXAMPLES")
print("=" * 80)

normalized_2013_lookup = {}

for state, district in keys_2013:

    key = (
        normalize_text(state),
        normalize_text(district)
    )

    normalized_2013_lookup[key] = (
        state,
        district
    )


count = 0

for state, district in sorted(keys_2012):

    normalized_key = (
        normalize_text(state),
        normalize_text(district)
    )

    if normalized_key in normalized_2013_lookup:

        matched = normalized_2013_lookup[
            normalized_key
        ]

        if (
            state != matched[0]
            or district != matched[1]
        ):

            print(
                f"2012: {state} | {district}"
            )

            print(
                f"2013: {matched[0]} | {matched[1]}"
            )

            print("-" * 60)

            count += 1

            if count >= 30:
                break


print("\nDiagnostic completed.")