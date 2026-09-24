import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)


# ============================================================
# SAFEHER-AI
# DATASET 01 - K-MEANS EVALUATION
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


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("SAFEHER-AI - K-MEANS EVALUATION")
print("=" * 70)

df = pd.read_csv(INPUT_FILE)

print("\nDataset loaded:")
print(f"Rows    : {len(df)}")
print(f"Columns : {len(df.columns)}")


# ============================================================
# FEATURE MATRIX
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

X = df[feature_columns]


print("\nNumber of clustering features:", len(feature_columns))

print("\nFeatures:")
for feature in feature_columns:
    print(" -", feature)


# ============================================================
# K VALUES
# ============================================================

k_values = range(2, 9)

results = []


# ============================================================
# EVALUATE K
# ============================================================

print("\n" + "-" * 70)
print("EVALUATING K = 2 TO 8")
print("-" * 70)

for k in k_values:

    print(f"\nRunning K-Means for K = {k}...")

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=20
    )

    labels = model.fit_predict(X)

    inertia = model.inertia_

    silhouette = silhouette_score(
        X,
        labels
    )

    davies_bouldin = davies_bouldin_score(
        X,
        labels
    )

    calinski_harabasz = calinski_harabasz_score(
        X,
        labels
    )

    results.append(
        {
            "K": k,
            "WCSS_INERTIA": inertia,
            "SILHOUETTE_SCORE": silhouette,
            "DAVIES_BOULDIN_INDEX": davies_bouldin,
            "CALINSKI_HARABASZ_SCORE": calinski_harabasz
        }
    )

    print(
        f"WCSS              : {inertia:.2f}"
    )

    print(
        f"Silhouette Score   : {silhouette:.4f}"
    )

    print(
        f"Davies-Bouldin    : {davies_bouldin:.4f}"
    )

    print(
        f"Calinski-Harabasz : {calinski_harabasz:.2f}"
    )


# ============================================================
# RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(results)

print("\n" + "=" * 70)
print("K-MEANS EVALUATION RESULTS")
print("=" * 70)

print(
    results_df.to_string(index=False)
)


# ============================================================
# SAVE RESULTS
# ============================================================

results_file = (
    OUTPUT_DIR
    / "kmeans_evaluation_results.csv"
)

results_df.to_csv(
    results_file,
    index=False
)


# ============================================================
# BEST VALUES BY METRIC
# ============================================================

print("\n" + "-" * 70)
print("METRIC EXTREMES")
print("-" * 70)


best_silhouette = results_df.loc[
    results_df["SILHOUETTE_SCORE"].idxmax()
]

best_davies = results_df.loc[
    results_df["DAVIES_BOULDIN_INDEX"].idxmin()
]

best_calinski = results_df.loc[
    results_df["CALINSKI_HARABASZ_SCORE"].idxmax()
]


print(
    "\nHighest Silhouette Score:"
)

print(
    f"K = {int(best_silhouette['K'])}, "
    f"Score = {best_silhouette['SILHOUETTE_SCORE']:.4f}"
)


print(
    "\nLowest Davies-Bouldin Index:"
)

print(
    f"K = {int(best_davies['K'])}, "
    f"Score = {best_davies['DAVIES_BOULDIN_INDEX']:.4f}"
)


print(
    "\nHighest Calinski-Harabasz Score:"
)

print(
    f"K = {int(best_calinski['K'])}, "
    f"Score = {best_calinski['CALINSKI_HARABASZ_SCORE']:.2f}"
)


# ============================================================
# ELBOW PLOT
# ============================================================

plt.figure(figsize=(9, 5))

plt.plot(
    results_df["K"],
    results_df["WCSS_INERTIA"],
    marker="o"
)

plt.title("K-Means Elbow Curve")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Within-Cluster Sum of Squares (WCSS)")
plt.xticks(list(k_values))
plt.grid(True)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "kmeans_elbow_curve.png",
    dpi=300
)

plt.show()


# ============================================================
# SILHOUETTE PLOT
# ============================================================

plt.figure(figsize=(9, 5))

plt.plot(
    results_df["K"],
    results_df["SILHOUETTE_SCORE"],
    marker="o"
)

plt.title("Silhouette Score by Number of Clusters")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.xticks(list(k_values))
plt.grid(True)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "kmeans_silhouette_scores.png",
    dpi=300
)

plt.show()


# ============================================================
# DAVIES-BOULDIN PLOT
# ============================================================

plt.figure(figsize=(9, 5))

plt.plot(
    results_df["K"],
    results_df["DAVIES_BOULDIN_INDEX"],
    marker="o"
)

plt.title("Davies-Bouldin Index by Number of Clusters")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Davies-Bouldin Index")
plt.xticks(list(k_values))
plt.grid(True)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "kmeans_davies_bouldin.png",
    dpi=300
)

plt.show()


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("K-MEANS EVALUATION COMPLETED")
print("=" * 70)

print("\nResults saved to:")
print(results_file)

print("\nGenerated plots:")
print(" - kmeans_elbow_curve.png")
print(" - kmeans_silhouette_scores.png")
print(" - kmeans_davies_bouldin.png")

print("\nIMPORTANT:")
print("No final cluster assignment has been saved yet.")

print("\nNext step:")
print("Review the metrics and select the clustering configuration.")