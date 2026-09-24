import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)


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
    / "feature_engineering"
)

MODEL_DIR = (
    BASE_DIR
    / "models"
    / "dataset_02_03_kmeans"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 80)
print("FEATURE ENGINEERING + K-MEANS")
print("DATASET 2 + DATASET 3 | 2013–2014")
print("=" * 80)

df = pd.read_csv(INPUT_FILE)

print("\nOriginal shape:", df.shape)


# ============================================================
# BASIC CLEANING
# ============================================================

# Ensure numeric crime columns are numeric
crime_columns = [
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

for col in crime_columns:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    ).fillna(0)


# ============================================================
# RECORD IDENTIFIER
# ============================================================

df["RECORD_ID"] = (
    df["STATE_UT"].astype(str)
    + "_"
    + df["DISTRICT"].astype(str)
    + "_"
    + df["YEAR"].astype(str)
)


# ============================================================
# YEAR NORMALIZATION
# ============================================================

df["YEAR_NORMALIZED"] = (
    df["YEAR"] - df["YEAR"].min()
) / (
    df["YEAR"].max() - df["YEAR"].min()
)


# ============================================================
# ADDITIONAL WOMEN-CRIME FEATURES
# ============================================================

# Violence specifically related to women
df["WOMEN_VIOLENCE"] = (
    df["RAPE"]
    + df["CUSTODIAL_RAPE"]
    + df["ASSAULT_WOMEN"]
    + df["INSULT_WOMEN"]
    + df["CRUELTY_WOMEN"]
)

# Rape proportion among total IPC
df["RAPE_RATE_PROXY"] = np.where(
    df["TOTAL_IPC"] > 0,
    df["RAPE"] / df["TOTAL_IPC"],
    np.nan
)

# Kidnapping proportion among total IPC
df["KIDNAPPING_RATE_PROXY"] = np.where(
    df["TOTAL_IPC"] > 0,
    df["KIDNAPPING_ABDUCTION"] / df["TOTAL_IPC"],
    np.nan
)

# Women crime proportion
df["WOMEN_CRIME_SHARE"] = np.where(
    df["TOTAL_IPC"] > 0,
    df["WOMEN_CRIME_CORE_TOTAL"] / df["TOTAL_IPC"],
    np.nan
)


# ============================================================
# ADDITIONAL CRIME COMPOSITION FEATURES
# ============================================================

df["VIOLENT_CRIME_SHARE"] = np.where(
    df["TOTAL_IPC"] > 0,
    df["VIOLENT_CRIME"] / df["TOTAL_IPC"],
    np.nan
)

df["PROPERTY_CRIME_SHARE"] = np.where(
    df["TOTAL_IPC"] > 0,
    df["PROPERTY_CRIME"] / df["TOTAL_IPC"],
    np.nan
)

df["MURDER_SHARE"] = np.where(
    df["TOTAL_IPC"] > 0,
    df["MURDER"] / df["TOTAL_IPC"],
    np.nan
)

df["ROBBERY_SHARE"] = np.where(
    df["TOTAL_IPC"] > 0,
    df["ROBBERY"] / df["TOTAL_IPC"],
    np.nan
)

df["THEFT_SHARE"] = np.where(
    df["TOTAL_IPC"] > 0,
    df["THEFT"] / df["TOTAL_IPC"],
    np.nan
)

df["BURGLARY_SHARE"] = np.where(
    df["TOTAL_IPC"] > 0,
    df["BURGLARY"] / df["TOTAL_IPC"],
    np.nan
)


# ============================================================
# SERIOUS VIOLENT CRIME
# ============================================================

df["SERIOUS_VIOLENT_CRIME"] = (
    df["MURDER"]
    + df["ATTEMPT_MURDER"]
    + df["CULPABLE_HOMICIDE"]
    + df["RAPE"]
    + df["KIDNAPPING_ABDUCTION"]
    + df["DACOITY"]
    + df["ROBBERY"]
)


# ============================================================
# DATA QUALITY CHECK
# ============================================================

print("\n" + "=" * 80)
print("DATA QUALITY CHECK")
print("=" * 80)

print("\nRows:", len(df))
print("Columns:", len(df.columns))

print("\nMissing values in engineered features:")

engineered_columns = [
    "WOMEN_VIOLENCE",
    "RAPE_RATE_PROXY",
    "KIDNAPPING_RATE_PROXY",
    "WOMEN_CRIME_SHARE",
    "VIOLENT_CRIME_SHARE",
    "PROPERTY_CRIME_SHARE",
    "MURDER_SHARE",
    "ROBBERY_SHARE",
    "THEFT_SHARE",
    "BURGLARY_SHARE",
    "SERIOUS_VIOLENT_CRIME"
]

print(
    df[engineered_columns]
    .isna()
    .sum()
)


# ============================================================
# K-MEANS FEATURE SET
# ============================================================

kmeans_features = [
    "WOMEN_CRIME_CORE_TOTAL",
    "WOMEN_VIOLENCE",
    "RAPE",
    "KIDNAPPING_ABDUCTION",
    "DOWRY_DEATHS",
    "ASSAULT_WOMEN",
    "CRUELTY_WOMEN",
    "VIOLENT_CRIME",
    "SERIOUS_VIOLENT_CRIME",
    "PROPERTY_CRIME",
    "ECONOMIC_OFFENCES",
    "PUBLIC_ORDER_CRIME",
    "TOTAL_IPC"
]

print("\n" + "=" * 80)
print("K-MEANS FEATURES")
print("=" * 80)

for i, feature in enumerate(kmeans_features, start=1):
    print(f"{i:02d}. {feature}")


# ============================================================
# HANDLE ZERO-TOTAL RECORDS
# ============================================================

# For K-Means count-based features, zero-total records are
# legitimate and can remain because all corresponding counts
# are zero.

X = df[kmeans_features].copy()

print("\nK-Means input shape:", X.shape)


# ============================================================
# LOG TRANSFORMATION
# ============================================================

# Crime counts are strongly right-skewed.
# log1p reduces the influence of extremely large reporting units.

X_log = np.log1p(X)


# ============================================================
# STANDARDIZATION
# ============================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X_log)

print("Scaled feature matrix:", X_scaled.shape)


# ============================================================
# K-MEANS EVALUATION
# ============================================================

print("\n" + "=" * 80)
print("K-MEANS EVALUATION")
print("=" * 80)

results = []

for k in range(2, 9):

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=20
    )

    labels = kmeans.fit_predict(X_scaled)

    wcss = kmeans.inertia_

    silhouette = silhouette_score(
        X_scaled,
        labels
    )

    davies_bouldin = davies_bouldin_score(
        X_scaled,
        labels
    )

    calinski_harabasz = calinski_harabasz_score(
        X_scaled,
        labels
    )

    results.append({
        "K": k,
        "WCSS": wcss,
        "Silhouette_Score": silhouette,
        "Davies_Bouldin_Index": davies_bouldin,
        "Calinski_Harabasz_Index": calinski_harabasz
    })

    print(
        f"K={k} | "
        f"WCSS={wcss:.6f} | "
        f"Silhouette={silhouette:.6f} | "
        f"DB={davies_bouldin:.6f} | "
        f"CH={calinski_harabasz:.6f}"
    )


results_df = pd.DataFrame(results)


# ============================================================
# SELECT K USING SILHOUETTE
# ============================================================

best_k = int(
    results_df.loc[
        results_df["Silhouette_Score"].idxmax(),
        "K"
    ]
)

print("\n" + "=" * 80)
print("K SELECTION")
print("=" * 80)

print(
    "Highest silhouette score occurs at K =",
    best_k
)


# ============================================================
# FINAL K-MEANS
# ============================================================

final_kmeans = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=20
)

df["CLUSTER"] = final_kmeans.fit_predict(
    X_scaled
)


# ============================================================
# CLUSTER DISTRIBUTION
# ============================================================

print("\n" + "=" * 80)
print("CLUSTER DISTRIBUTION")
print("=" * 80)

cluster_counts = (
    df["CLUSTER"]
    .value_counts()
    .sort_index()
)

cluster_percentages = (
    cluster_counts
    / len(df)
    * 100
)

cluster_summary = pd.DataFrame({
    "COUNT": cluster_counts,
    "PERCENTAGE": cluster_percentages
})

print(cluster_summary)


# ============================================================
# CLUSTER PROFILE
# ============================================================

print("\n" + "=" * 80)
print("CLUSTER PROFILE")
print("=" * 80)

profile_columns = [
    "TOTAL_IPC",
    "WOMEN_CRIME_CORE_TOTAL",
    "WOMEN_VIOLENCE",
    "RAPE",
    "KIDNAPPING_ABDUCTION",
    "DOWRY_DEATHS",
    "ASSAULT_WOMEN",
    "CRUELTY_WOMEN",
    "VIOLENT_CRIME",
    "PROPERTY_CRIME",
    "ECONOMIC_OFFENCES",
    "PUBLIC_ORDER_CRIME"
]

cluster_profile = (
    df.groupby("CLUSTER")[profile_columns]
    .mean()
)

print(
    cluster_profile.to_string()
)


# ============================================================
# CLUSTER PROFILE — MEDIAN
# ============================================================

cluster_median = (
    df.groupby("CLUSTER")[profile_columns]
    .median()
)

print("\nCluster median profile:")
print(
    cluster_median.to_string()
)


# ============================================================
# CLUSTER DISTRIBUTION BY YEAR
# ============================================================

cluster_by_year = pd.crosstab(
    df["YEAR"],
    df["CLUSTER"]
)

print("\n" + "=" * 80)
print("CLUSTER DISTRIBUTION BY YEAR")
print("=" * 80)

print(cluster_by_year)


# ============================================================
# SAVE DATA
# ============================================================

print("\n" + "=" * 80)
print("SAVING OUTPUTS")
print("=" * 80)


# Full engineered dataset
engineered_file = (
    OUTPUT_DIR
    / "engineered_dataset_02_03_2013_2014.csv"
)

df.to_csv(
    engineered_file,
    index=False
)


# K-Means evaluation
results_file = (
    MODEL_DIR
    / "kmeans_evaluation_dataset_02_03.csv"
)

results_df.to_csv(
    results_file,
    index=False
)


# Cluster profile
profile_file = (
    MODEL_DIR
    / "cluster_profile_dataset_02_03.csv"
)

cluster_profile.to_csv(
    profile_file
)


# Cluster median
median_file = (
    MODEL_DIR
    / "cluster_median_dataset_02_03.csv"
)

cluster_median.to_csv(
    median_file
)


# Cluster distribution
distribution_file = (
    MODEL_DIR
    / "cluster_distribution_dataset_02_03.csv"
)

cluster_summary.to_csv(
    distribution_file
)


# Cluster by year
year_cluster_file = (
    MODEL_DIR
    / "cluster_by_year_dataset_02_03.csv"
)

cluster_by_year.to_csv(
    year_cluster_file
)


# ============================================================
# SAVE SCALER PARAMETERS
# ============================================================

scaler_parameters = pd.DataFrame({
    "FEATURE": kmeans_features,
    "MEAN": scaler.mean_,
    "SCALE": scaler.scale_
})

scaler_parameters.to_csv(
    MODEL_DIR
    / "kmeans_scaler_parameters_dataset_02_03.csv",
    index=False
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\nEngineered dataset:")
print(engineered_file)

print("\nK-Means evaluation:")
print(results_file)

print("\nCluster profile:")
print(profile_file)

print("\nCluster distribution:")
print(distribution_file)

print("\n" + "=" * 80)
print("FEATURE ENGINEERING + K-MEANS COMPLETED")
print("=" * 80)