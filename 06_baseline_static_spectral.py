"""
SCRIPT 06 — Baseline 1: Static Spectral Clustering
====================================================
Applies standard normalized-cut spectral clustering (Ng, Jordan & Weiss,
2002) to a SINGLE STATIC snapshot of cumulative case counts per district,
weighted by the true administrative adjacency graph.

This baseline deliberately ignores the temporal dimension — that omission
is the core limitation the proposed AST-GL method targets.

Affinity:
    W[i,j] = A[i,j] * exp(-(x[i]-x[j])^2 / 2*sigma^2)
where A is the binary adjacency matrix and x[i] is district i's last
available cumulative confirmed case count.

k is chosen by silhouette-score sweep (k=2..8). The documented result
(silhouette score increases monotonically with k, giving no natural
stopping rule) is itself evidence for parameter sensitivity.

Input:  data/processed/cum_confirmed_matrix_clean.csv
        data/processed/admin_unit_adjacency.json
Output: data/processed/baseline1_static_spectral_clusters.csv
"""

import numpy as np
import pandas as pd
import json
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score

MATRIX_PATH = "data/processed/cum_confirmed_matrix_clean.csv"
ADJ_PATH    = "data/processed/admin_unit_adjacency.json"
OUT_PATH    = "data/processed/baseline1_static_spectral_clusters.csv"

matrix = pd.read_csv(MATRIX_PATH, index_col=0)
matrix = matrix.apply(pd.to_numeric, errors="coerce")

with open(ADJ_PATH) as f:
    adj = json.load(f)

units = list(matrix.index)
N = len(units)
unit_idx = {u: i for i, u in enumerate(units)}

# Binary adjacency matrix
A = np.zeros((N, N))
for u, neighbors in adj.items():
    if u not in unit_idx:
        continue
    for v in neighbors:
        if v in unit_idx:
            A[unit_idx[u], unit_idx[v]] = 1.0
            A[unit_idx[v], unit_idx[u]] = 1.0

# Per-district snapshot: own last available cumulative value
snapshot = matrix.apply(lambda row: row.dropna().iloc[-1]
                        if row.dropna().size else 0.0, axis=1)
case_vals = snapshot.values.astype(float)

# Gaussian affinity gated by adjacency
diff_matrix = np.abs(case_vals[:, None] - case_vals[None, :])
adj_pairs = np.argwhere(np.triu(A, k=1) > 0)
sigma = np.median(diff_matrix[adj_pairs[:, 0], adj_pairs[:, 1]])
sigma = max(sigma, 1e-6)
similarity = np.exp(-(diff_matrix ** 2) / (2 * sigma ** 2))
affinity = A * similarity

# Silhouette sweep
best_k, best_score, best_labels = None, -1, None
print("Silhouette scores by k:")
for k in range(2, 9):
    try:
        sc = SpectralClustering(n_clusters=k, affinity="precomputed",
                                assign_labels="kmeans", random_state=42)
        labels = sc.fit_predict(affinity)
        if len(set(labels)) < 2:
            continue
        norm_aff = affinity / (affinity.max() + 1e-9)
        dist = 1 - norm_aff
        np.fill_diagonal(dist, 0.0)
        score = silhouette_score(dist, labels, metric="precomputed")
        print(f"  k={k}: {score:.4f}")
        if score > best_score:
            best_k, best_score, best_labels = k, score, labels
    except Exception as e:
        print(f"  k={k}: ERROR — {e}")

out_df = pd.DataFrame({
    "admin_unit": units,
    "peak_cum_confirmed_cases": case_vals,
    "cluster": best_labels
}).sort_values(["cluster", "peak_cum_confirmed_cases"], ascending=[True, False])
out_df.to_csv(OUT_PATH, index=False)

print(f"\nBest k = {best_k}  (silhouette = {best_score:.4f})")
print(f"Results → {OUT_PATH}")
print("\nCluster composition:")
for c in sorted(set(best_labels)):
    members = out_df[out_df["cluster"] == c]["admin_unit"].tolist()
    total   = out_df[out_df["cluster"] == c]["peak_cum_confirmed_cases"].sum()
    print(f"  Cluster {c} (n={len(members)}, total cases={total:.0f}): {members}")
