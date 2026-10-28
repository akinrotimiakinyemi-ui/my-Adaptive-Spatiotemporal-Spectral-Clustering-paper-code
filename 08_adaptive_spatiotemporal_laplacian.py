"""
SCRIPT 08 — Proposed Method: Adaptive Spatiotemporal Graph Laplacian
=====================================================================
Implements the AST-GL framework — the paper's core contribution.

PIPELINE:
  1. Per-layer spatial affinity W_t  (log-scale Gaussian kernel on
     incident case counts, gated by true administrative adjacency)
  2. Adaptive temporal coupling α_t[i]  (neighbourhood-rank-stability
     signal: cosine similarity between district i's spatial affinity
     vector to its neighbours at times t and t+1)
  3. Supra-adjacency S  ((N×T) × (N×T) multilayer graph)
  4. Normalized spatiotemporal Laplacian  L_st = I − D^{-½} S D^{-½}
  5. Spectral embedding  (k=12 smallest eigenvectors)
  6. K-means clustering  (k=7, eigengap-motivated)

Key design choices (see manuscript Section 3.4 for full rationale):
  - Log-scale affinities: prevents all high-burden weeks from being
    treated as uniformly similar regardless of relative magnitude
  - α_max = 0.3: keeps temporal coupling subordinate to spatial signal
  - Cosine-similarity coupling: targets genuine spatial reorganisation,
    not raw case-count volatility (earlier design, documented in paper)

Parameters (frozen — NOT re-tuned for DRC cross-validation):
  ALPHA_MAX = 0.3
  N_EIGENVECTORS = 12
  K_CLUSTERS = 7

Input:  data/processed/cum_confirmed_matrix_clean.csv
        data/processed/admin_unit_adjacency.json
Output: data/processed/wa_adaptive_clusters.csv
        data/processed/adaptive_coupling_alpha.csv
"""

import numpy as np
import pandas as pd
import json
from scipy.sparse import lil_matrix, diags
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans

MATRIX_PATH    = "data/processed/cum_confirmed_matrix_clean.csv"
ADJ_PATH       = "data/processed/admin_unit_adjacency.json"
OUT_CLUSTERS   = "data/processed/wa_adaptive_clusters.csv"
OUT_ALPHA      = "data/processed/adaptive_coupling_alpha.csv"

ALPHA_MAX      = 0.3
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

# Weekly incident case series
weekly = matrix.T.resample("W").max().T.ffill(axis=1)
incident = weekly.diff(axis=1).clip(lower=0).fillna(0)
incident.iloc[:, 0] = weekly.iloc[:, 0].fillna(0)
T = incident.shape[1]
case_array = incident.values.astype(float)
print(f"N={N} admin units, T={T} epi-weeks")


def build_layer_affinities(case_array, A, eps=EPS):
    """Per-layer spatial affinity on log-scale incident case counts."""
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


def compute_adaptive_coupling(W_layers, A, alpha_max=ALPHA_MAX, eps=EPS):
    """
    Neighbourhood-rank-stability coupling (the novel contribution).

    For each unit i and transition t→t+1:
      α_t[i] = alpha_max · max(0, cosine_similarity(v_t(i), v_{t+1}(i)))

    where v_t(i) is unit i's vector of spatial affinities to its graph
    neighbours at time t.

    Stable neighbourhood profile → high coupling (temporal smoothing).
    Reorganising neighbourhood    → low coupling (tracks real change).
    """
    T, N, _ = W_layers.shape
    alpha = np.zeros((N, T - 1))
    for i in range(N):
        neighbor_idx = np.where(A[i] > 0)[0]
        if len(neighbor_idx) == 0:
            alpha[i, :] = alpha_max
            continue
        for t in range(T - 1):
            v_t  = W_layers[t,     i, neighbor_idx]
            v_t1 = W_layers[t + 1, i, neighbor_idx]
            nt, nt1 = np.linalg.norm(v_t), np.linalg.norm(v_t1)
            if nt < eps or nt1 < eps:
                cos_sim = 0.5   # degenerate case: default to moderate coupling
            else:
                cos_sim = np.dot(v_t, v_t1) / (nt * nt1)
            alpha[i, t] = alpha_max * max(0.0, cos_sim)
    return alpha


def build_supra_adjacency(W_layers, alpha):
    """Assemble the (N×T)×(N×T) supra-adjacency matrix."""
    T, N, _ = W_layers.shape
    S = lil_matrix((N * T, N * T))
    for t in range(T):
        S[t*N:(t+1)*N, t*N:(t+1)*N] = W_layers[t]
    for t in range(T - 1):
        for i in range(N):
            a = alpha[i, t]
            S[t*N + i, (t+1)*N + i] = a
            S[(t+1)*N + i, t*N + i] = a
    return S.tocsr()


def normalized_laplacian_eigvecs(S, k, eps=EPS):
    """Normalized Laplacian L = I - D^{-½} S D^{-½}, k smallest eigenvecs."""
    degrees = np.array(S.sum(axis=1)).flatten()
    degrees_safe = np.where(degrees > eps, degrees, eps)
    D_inv_sqrt = diags(1.0 / np.sqrt(degrees_safe))
    L_norm = diags(np.ones(S.shape[0])) - D_inv_sqrt @ S @ D_inv_sqrt
    eigvals, eigvecs = eigsh(L_norm, k=k, which="SM")
    return eigvals, eigvecs, degrees


print("Building per-layer spatial affinities...")
W_layers = build_layer_affinities(case_array, A)

print("Computing adaptive neighbourhood-rank-stability coupling...")
alpha = compute_adaptive_coupling(W_layers, A)
print(f"  α: mean={alpha.mean():.4f}, std={alpha.std():.4f}, "
      f"min={alpha.min():.4f}, max={alpha.max():.4f}")

print("Assembling supra-adjacency matrix...")
S = build_supra_adjacency(W_layers, alpha)
print(f"  Supra-graph: {S.shape}, nnz={S.nnz}")

print(f"Computing {N_EIGENVECTORS} smallest eigenvectors of the "
      f"spatiotemporal Laplacian...")
eigvals, eigvecs, degrees = normalized_laplacian_eigvecs(S, N_EIGENVECTORS)
print(f"  Eigenvalues: {np.round(eigvals, 5)}")

print(f"K-means clustering (k={K_CLUSTERS})...")
km = KMeans(n_clusters=K_CLUSTERS, random_state=42, n_init=10)
node_labels = km.fit_predict(eigvecs)
label_grid = node_labels.reshape(T, N).T   # shape (N, T)

cluster_df = pd.DataFrame(label_grid, index=units, columns=incident.columns)
cluster_df.to_csv(OUT_CLUSTERS)

alpha_df = pd.DataFrame(alpha, index=units, columns=incident.columns[:-1])
alpha_df.to_csv(OUT_ALPHA)

# Temporal stability
n_distinct = [len(set(label_grid[i])) for i in range(N)]
print(f"\nTemporal stability: mean={np.mean(n_distinct):.4f} "
      f"std={np.std(n_distinct):.4f} clusters/unit")

# Peak-week cluster assignments
peak_week_idx = case_array.argmax(axis=1)
print("\nTop 15 units by peak weekly cases — cluster at peak week:")
peak_cases = case_array.max(axis=1)
for i in np.argsort(-peak_cases)[:15]:
    c = label_grid[i, peak_week_idx[i]]
    print(f"  {units[i]:45s}  peak={peak_cases[i]:4.0f}  cluster={c}")

print(f"\nCluster labels → {OUT_CLUSTERS}")
print(f"Coupling values → {OUT_ALPHA}")
