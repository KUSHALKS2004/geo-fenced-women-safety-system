import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.cluster import KMeans


# ============================================================
# SAFEHER-AI
# DATASET 01 - FINAL K-MEANS CLUSTERING
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "feature_engineering_01"
    / "kmeans_scaled_features.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "feature_engineering_01"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


CLUSTERED_FILE = (
    OUTPUT_DIR
    / "dataset_01_kmeans_clustered.csv"
)

PROFILE_FILE = (
    OUTPUT_DIR
    / "kmeans_cluster_profiles.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("SAFEHER-AI - FINAL K-MEANS CLUSTERING")
print("=" * 70)

df = pd.read_csv(INPUT_FILE)

print("\nDataset loaded:")
print(f"Rows    : {len(df)}")
print(f"Columns : {len(df.columns)}")


# ============================================================
# FEATURES
# ============================================================

metadata_columns = [
    "RECORD_ID",
    "STATE/UT",
    "DISTRICT",
    "YEAR"
]

feature_columns = [
    column
    for column in df.columns
    if column not in metadata_columns
]

X = df[feature_columns].copy()


print("\nClustering features:")

for feature in feature_columns:
    print(" -", feature)


# ============================================================
# K-MEANS
# ============================================================

K = 2

print("\n" + "-" * 70)
print(f"RUNNING K-MEANS WITH K = {K}")
print("-" * 70)

model = KMeans(
    n_clusters=K,
    random_state=42,
    n_init=20
)

df["CLUSTER"] = model.fit_predict(X)


# ============================================================
# CLUSTER COUNTS
# ============================================================

print("\n" + "-" * 70)
print("CLUSTER SIZES")
print("-" * 70)

cluster_counts = (
    df["CLUSTER"]
    .value_counts()
    .sort_index()
)

print(cluster_counts.to_string())


print("\nCluster percentages:")

cluster_percentages = (
    df["CLUSTER"]
    .value_counts(normalize=True)
    .sort_index()
    * 100
)

for cluster, percentage in cluster_percentages.items():

    print(
        f"Cluster {cluster}: "
        f"{percentage:.2f}%"
    )


# ============================================================
# CLUSTER PROFILE
# ============================================================

print("\n" + "-" * 70)
print("CLUSTER PROFILES")
print("-" * 70)


cluster_profile = (
    df.groupby("CLUSTER")[feature_columns]
    .mean()
)


print(
    cluster_profile.to_string()
)


# ============================================================
# STANDARDIZED CLUSTER PROFILE
# ============================================================

print("\n" + "-" * 70)
print("CLUSTER PROFILE - RELATIVE VALUES")
print("-" * 70)


overall_mean = df[feature_columns].mean()
overall_std = df[feature_columns].std()

profile_z = (
    cluster_profile - overall_mean
) / overall_std


print(
    profile_z.round(3).to_string()
)


# ============================================================
# SAVE PROFILE
# ============================================================

cluster_profile.to_csv(
    PROFILE_FILE
)


# ============================================================
# STATE/UT DISTRIBUTION
# ============================================================

print("\n" + "-" * 70)
print("CLUSTER DISTRIBUTION BY STATE/UT")
print("-" * 70)


state_cluster = pd.crosstab(
    df["STATE/UT"],
    df["CLUSTER"]
)


print(
    state_cluster.to_string()
)


state_cluster.to_csv(
    OUTPUT_DIR
    / "cluster_distribution_by_state.csv"
)


# ============================================================
# TOP RECORDS PER CLUSTER
# ============================================================

print("\n" + "-" * 70)
print("SAMPLE RECORDS FROM EACH CLUSTER")
print("-" * 70)


for cluster in sorted(df["CLUSTER"].unique()):

    print(
        f"\nCluster {cluster}:"
    )

    sample = (
        df[df["CLUSTER"] == cluster]
        [
            [
                "STATE/UT",
                "DISTRICT",
                "YEAR",
                "CLUSTER"
            ]
        ]
        .head(10)
    )

    print(
        sample.to_string(index=False)
    )


# ============================================================
# SAVE CLUSTERED DATASET
# ============================================================

df.to_csv(
    CLUSTERED_FILE,
    index=False
)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("K-MEANS CLUSTERING COMPLETED")
print("=" * 70)

print("\nK:")
print(K)

print("\nClustered dataset:")
print(CLUSTERED_FILE)

print("\nCluster profiles:")
print(PROFILE_FILE)

print("\nIMPORTANT:")
print("Cluster labels are numerical only at this stage.")

print("\nNext step:")
print("Interpret cluster profiles and prepare the temporal risk-prediction dataset.")