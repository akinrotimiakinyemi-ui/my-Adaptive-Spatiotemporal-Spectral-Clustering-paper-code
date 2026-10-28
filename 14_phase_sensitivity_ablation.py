"""
SCRIPT 14 — Phase Sensitivity Ablation
=======================================
Tests whether the optimal fixed coupling value (α) differs between the
EARLY (weeks 1-13, low-case, exploratory) and LATE (weeks 14-27,
established, high-case) phases of the outbreak.

This is the key test for whether a single fixed α can serve the entire
outbreak, or whether different phases genuinely need different coupling —
which is the motivation for an adaptive mechanism.

Finding documented in manuscript (Section 4.2):
  α = 1.0 minimises mean clusters/unit in BOTH phases when evaluated on
  temporal stability alone. This does NOT support the phase-sensitivity
  argument on this metric, and is reported honestly. The transition-
  responsiveness test (Script 15) provides stronger evidence.

Input:  data/processed/cum_confirmed_matrix_clean.csv
        data/processed/admin_unit_adjacency.json
Output: data/processed/phase_sensitivity_ablation.csv
"""

import numpy as np
import pandas as pd
import json
from scipy.sparse import lil_matrix, diags
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans

MATRIX_PATH = "data/processed/cum_confirmed_matrix_clean.csv"
ADJ_PATH    = "data/processed/admin_unit_adjacency.json"
OUT_PATH    = "data/processed/phase_sensitivity_ablation.csv"

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
T_full = incident.shape[1]
case_array_full = incident.values.astype(float)

print(f"Full period: {T_full} weeks ({incident.columns[0].date()} to "
      f"{incident.columns[-1].date()})")
print(f"Early phase: weeks 0-12 ({incident.columns[0].date()} to "
      f"{incident.columns[12].date()})")
print(f"Late phase:  weeks 13-{T_full-1} ({incident.columns[13].date()} to "
      f"{incident.columns[-1].date()})")


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
        W_layers[t] = A * np.exp(-(diffs ** 2) / (2 * sigma_t ** 2 + eps))
    return W_layers


def run_phase(case_array, phase_name):
    N, T = case_array.shape
    W_layers = build_layer_affinities(case_array, A)
    print(f"\n--- {phase_name} (T={T} weeks) ---")
    phase_results = []
    for fixed_alpha in [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
        S = lil_matrix((N * T, N * T))
        for t in range(T):
            S[t*N:(t+1)*N, t*N:(t+1)*N] = W_layers[t]
        for t in range(T - 1):
            for i in range(N):
                S[t*N+i, (t+1)*N+i] = fixed_alpha
                S[(t+1)*N+i, t*N+i] = fixed_alpha
        S = S.tocsr()
        degrees = np.array(S.sum(axis=1)).flatten()
        degrees_safe = np.where(degrees > EPS, degrees, EPS)
        D_inv_sqrt = diags(1.0 / np.sqrt(degrees_safe))
        L_norm = diags(np.ones(S.shape[0])) - D_inv_sqrt @ S @ D_inv_sqrt
        _, eigvecs = eigsh(L_norm, k=N_EIGENVECTORS, which="SM")
        km = KMeans(n_clusters=K_CLUSTERS, random_state=42, n_init=10)
        labels = km.fit_predict(eigvecs)
        label_grid = labels.reshape(T, N).T
        n_distinct = [len(set(label_grid[i])) for i in range(N)]
        mean_s, std_s = np.mean(n_distinct), np.std(n_distinct)
        phase_results.append({"phase": phase_name, "fixed_alpha": fixed_alpha,
                               "mean_clusters_per_unit": mean_s,
                               "std_clusters_per_unit": std_s})
        print(f"  α={fixed_alpha}: mean={mean_s:.3f} (std={std_s:.3f})")
    return pd.DataFrame(phase_results)


early_results = run_phase(case_array_full[:, :13], "EARLY")
late_results  = run_phase(case_array_full[:, 13:], "LATE")

combined = pd.concat([early_results, late_results], ignore_index=True)
combined.to_csv(OUT_PATH, index=False)

best_early = early_results.loc[early_results["mean_clusters_per_unit"].idxmin()]
best_late  = late_results.loc[late_results["mean_clusters_per_unit"].idxmin()]
print(f"\n=== BEST FIXED α PER PHASE ===")
print(f"Early: α={best_early['fixed_alpha']} "
      f"(mean={best_early['mean_clusters_per_unit']:.3f})")
print(f"Late:  α={best_late['fixed_alpha']} "
      f"(mean={best_late['mean_clusters_per_unit']:.3f})")

if best_early["fixed_alpha"] == best_late["fixed_alpha"]:
    print("\n→ Same optimal α in both phases on this metric.")
    print("  Phase-sensitivity argument rests on responsiveness test (Script 15).")
else:
    print(f"\n→ CONFIRMED: optimal α differs between phases "
          f"({best_early['fixed_alpha']} vs {best_late['fixed_alpha']}).")

print(f"\nResults → {OUT_PATH}")
