"""
SCRIPT 09 — Ablation: Fixed Temporal Coupling Sweep
=====================================================
Runs the same supra-Laplacian pipeline as Script 08 but replaces the
adaptive coupling with a single fixed global scalar alpha, swept across
7 values: {0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0}.

Metric: mean number of distinct cluster labels a district visits across
the 27-week study period (temporal stability — lower = more stable).

This directly tests the claim that adaptive coupling achieves a better
stability profile than any single fixed value.

NOTE: The adaptive method's actual stability stats are read fresh from
      Script 08's output rather than hard-coded, ensuring consistency.

Input:  data/processed/cum_confirmed_matrix_clean.csv
        data/processed/admin_unit_adjacency.json
        data/processed/wa_adaptive_clusters.csv  (from Script 08)
Output: data/processed/ablation_fixed_coupling_results.csv
"""

import numpy as np
import pandas as pd
import json
from scipy.sparse import lil_matrix, diags
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans

MATRIX_PATH  = "data/processed/cum_confirmed_matrix_clean.csv"
ADJ_PATH     = "data/processed/admin_unit_adjacency.json"
ADAPTIVE_PATH= "data/processed/wa_adaptive_clusters.csv"
OUT_PATH     = "data/processed/ablation_fixed_coupling_results.csv"

N_EIGENVECTORS = 12
K_CLUSTERS     = 7
EPS            = 1e-6

with open(ADJ_PATH) as f:
    adj = json.load(f)
matrix = pd.read_csv(MATRIX_PATH, index_col=0)
matrix.columns = pd.to_datetime(matrix.columns)
matrix = matrix.apply(pd.to_numeric, errors="coerce")
units = list(matrix.index)
N = len(units)
unit_idx = {u: i for i, u in enumerate(units)}

A = np.zeros((N, N))
for u, neighbors in adj.items():
    if u not in unit_idx:
        continue
    for v in neighbors:
        if v in unit_idx:
            A[unit_idx[u], unit_idx[v]] = 1.0
            A[unit_idx[v], unit_idx[u]] = 1.0

weekly = matrix.T.resample("W").max().T.ffill(axis=1)
incident = weekly.diff(axis=1).clip(lower=0).fillna(0)
incident.iloc[:, 0] = weekly.iloc[:, 0].fillna(0)
T = incident.shape[1]
case_array = incident.values.astype(float)


def build_layer_affinities(case_array, A, eps=EPS):
    N, T = case_array.shape
    W_layers = np.zeros((T, N, N))
    adj_pairs = np.argwhere(np.triu(A, k=1) > 0)
    log_case = np.log1p(case_array)
    for t in range(T):
        x_t = log_case[:, t]
        diffs = np.abs(x_t[:, None] - x_t[None, :])
        if len(adj_pairs):
            pair_diffs = diffs[adj_pairs[:, 0], adj_pairs[:, 1]]
            sigma_t = np.median(pair_diffs) if np.median(pair_diffs) > 0 else 1.0
        else:
            sigma_t = 1.0
        sim = np.exp(-(diffs ** 2) / (2 * sigma_t ** 2 + eps))
        W_layers[t] = A * sim
    return W_layers


def build_supra_fixed(W_layers, fixed_alpha):
    T, N, _ = W_layers.shape
    S = lil_matrix((N * T, N * T))
    for t in range(T):
        S[t*N:(t+1)*N, t*N:(t+1)*N] = W_layers[t]
    for t in range(T - 1):
        for i in range(N):
            S[t*N+i, (t+1)*N+i] = fixed_alpha
            S[(t+1)*N+i, t*N+i] = fixed_alpha
    return S.tocsr()


def normalized_laplacian_eigvecs(S, k, eps=EPS):
    degrees = np.array(S.sum(axis=1)).flatten()
    degrees_safe = np.where(degrees > eps, degrees, eps)
    D_inv_sqrt = diags(1.0 / np.sqrt(degrees_safe))
    L_norm = diags(np.ones(S.shape[0])) - D_inv_sqrt @ S @ D_inv_sqrt
    eigvals, eigvecs = eigsh(L_norm, k=k, which="SM")
    return eigvals, eigvecs, degrees


W_layers = build_layer_affinities(case_array, A)
results = []

print("Running fixed-coupling ablation...")
for fixed_alpha in [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
    S = build_supra_fixed(W_layers, fixed_alpha)
    _, eigvecs, _ = normalized_laplacian_eigvecs(S, N_EIGENVECTORS)
    km = KMeans(n_clusters=K_CLUSTERS, random_state=42, n_init=10)
    labels = km.fit_predict(eigvecs)
    label_grid = labels.reshape(T, N).T
    n_distinct = [len(set(label_grid[i])) for i in range(N)]
    mean_s, std_s = np.mean(n_distinct), np.std(n_distinct)
    results.append({"fixed_alpha": fixed_alpha,
                    "mean_clusters_visited_per_unit": mean_s,
                    "std_clusters_visited_per_unit": std_s})
    print(f"  α={fixed_alpha}: mean={mean_s:.3f} (std={std_s:.3f})")

results_df = pd.DataFrame(results)
results_df.to_csv(OUT_PATH, index=False)

# Read adaptive stats fresh from Script 08 output
adaptive_clusters = pd.read_csv(ADAPTIVE_PATH, index_col=0)
n_distinct_adaptive = adaptive_clusters.apply(lambda row: row.nunique(), axis=1)
adaptive_mean = n_distinct_adaptive.mean()
adaptive_std  = n_distinct_adaptive.std()

print(f"\n=== Comparison summary ===")
print(f"Adaptive (from Script 08): mean={adaptive_mean:.3f} (std={adaptive_std:.3f})")
print(results_df.to_string(index=False))
print(f"\nResults → {OUT_PATH}")
