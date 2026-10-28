"""
SCRIPT 15 — Transition Responsiveness Test
==========================================
Tests whether each method's cluster assignment meaningfully responds
to a REAL, DOCUMENTED structural change in the outbreak, versus
spuriously reassigning during a QUIET CONTROL PERIOD.

Ground-truth event: week of 2014-10-26 — Liberia's weekly incident
cases spike to ~1,330 (vs. low double digits surrounding weeks),
consistent with the documented case-backlog and reclassification
reporting event in late October 2014.

Control period: 2014-07-13 to 2014-07-27 — no known comparable event.

Metric for each method:
  event_reassignments:   # Liberian units changing cluster at event
  control_reassignments: # Liberian units changing cluster at control
  responsiveness_ratio:  (event+1) / (control+1)

The ideal pattern: high event reassignments, zero control reassignments.
This is the central empirical finding (Table 2 in manuscript).

Input:  data/processed/cum_confirmed_matrix_clean.csv
        data/processed/admin_unit_adjacency.json
Output: data/processed/transition_responsiveness_results.csv
        figures/transition_responsiveness.png
"""

import numpy as np
import pandas as pd
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix, diags
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
import os

os.makedirs("figures", exist_ok=True)

MATRIX_PATH = "data/processed/cum_confirmed_matrix_clean.csv"
ADJ_PATH    = "data/processed/admin_unit_adjacency.json"
OUT_CSV     = "data/processed/transition_responsiveness_results.csv"
OUT_FIG     = "figures/transition_responsiveness.png"

N_EIGENVECTORS = 12
K_CLUSTERS     = 7
EPS            = 1e-6
ALPHA_MAX      = 0.3

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
week_cols = list(incident.columns)

EVENT_WEEK_IDX   = next(i for i, d in enumerate(week_cols)
                        if pd.Timestamp("2014-10-26") <= d <= pd.Timestamp("2014-11-02"))
CONTROL_WEEK_IDX = next(i for i, d in enumerate(week_cols)
                        if pd.Timestamp("2014-07-13") <= d <= pd.Timestamp("2014-07-27"))

liberia_idx = [unit_idx[u] for u in units if u.startswith("Liberia")]
print(f"Event week index: {EVENT_WEEK_IDX} ({week_cols[EVENT_WEEK_IDX].date()})")
print(f"Control week index: {CONTROL_WEEK_IDX} ({week_cols[CONTROL_WEEK_IDX].date()})")
print(f"Liberia units monitored: {len(liberia_idx)}")


def build_layer_affinities(case_array, A, eps=EPS):
    N, T = case_array.shape
    W = np.zeros((T, N, N))
    pairs = np.argwhere(np.triu(A, k=1) > 0)
    log_case = np.log1p(case_array)
    for t in range(T):
        x = log_case[:, t]
        diffs = np.abs(x[:, None] - x[None, :])
        sigma = np.median(diffs[pairs[:, 0], pairs[:, 1]]) if len(pairs) else 1.0
        sigma = sigma if sigma > 0 else 1.0
        W[t] = A * np.exp(-(diffs ** 2) / (2 * sigma ** 2 + eps))
    return W


def cluster_full(S, N, T, k_eig, k_clust):
    deg = np.array(S.sum(axis=1)).flatten()
    degsafe = np.where(deg > EPS, deg, EPS)
    Dinv = diags(1.0 / np.sqrt(degsafe))
    L = diags(np.ones(S.shape[0])) - Dinv @ S @ Dinv
    _, eigvecs = eigsh(L, k=k_eig, which="SM")
    labels = KMeans(n_clusters=k_clust, random_state=42, n_init=10).fit_predict(eigvecs)
    return labels.reshape(T, N).T  # (N, T)


def count_reassignments(label_grid, unit_indices, center_idx):
    before = label_grid[unit_indices, center_idx - 1]
    during = label_grid[unit_indices, center_idx]
    after  = label_grid[unit_indices, center_idx + 1]
    return int(np.sum((before != during) | (during != after)))


W_layers = build_layer_affinities(case_array, A)
results = []

# Fixed-coupling settings
for fixed_alpha in [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
    S = lil_matrix((N * T, N * T))
    for t in range(T):
        S[t*N:(t+1)*N, t*N:(t+1)*N] = W_layers[t]
    for t in range(T - 1):
        for i in range(N):
            S[t*N+i, (t+1)*N+i] = fixed_alpha
            S[(t+1)*N+i, t*N+i] = fixed_alpha
    S = S.tocsr()
    label_grid = cluster_full(S, N, T, N_EIGENVECTORS, K_CLUSTERS)
    ev = count_reassignments(label_grid, liberia_idx, EVENT_WEEK_IDX)
    ct = count_reassignments(label_grid, liberia_idx, CONTROL_WEEK_IDX)
    ratio = (ev + 1) / (ct + 1)
    results.append({"method": f"Fixed α={fixed_alpha}",
                    "event_reassignments": ev,
                    "control_reassignments": ct,
                    "responsiveness_ratio": ratio})
    print(f"Fixed α={fixed_alpha}: event={ev}/{len(liberia_idx)}, "
          f"control={ct}/{len(liberia_idx)}, ratio={ratio:.2f}")

# Adaptive coupling
alpha = np.zeros((N, T - 1))
for i in range(N):
    nb = np.where(A[i] > 0)[0]
    if len(nb) == 0:
        alpha[i, :] = ALPHA_MAX
        continue
    for t in range(T - 1):
        v1, v2 = W_layers[t, i, nb], W_layers[t + 1, i, nb]
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        cs = np.dot(v1, v2) / (n1 * n2) if n1 > EPS and n2 > EPS else 0.5
        alpha[i, t] = ALPHA_MAX * max(0.0, cs)

S = lil_matrix((N * T, N * T))
for t in range(T):
    S[t*N:(t+1)*N, t*N:(t+1)*N] = W_layers[t]
for t in range(T - 1):
    for i in range(N):
        a = alpha[i, t]
        S[t*N+i, (t+1)*N+i] = a
        S[(t+1)*N+i, t*N+i] = a
S = S.tocsr()
label_grid_a = cluster_full(S, N, T, N_EIGENVECTORS, K_CLUSTERS)
ev_a = count_reassignments(label_grid_a, liberia_idx, EVENT_WEEK_IDX)
ct_a = count_reassignments(label_grid_a, liberia_idx, CONTROL_WEEK_IDX)
ratio_a = (ev_a + 1) / (ct_a + 1)
results.append({"method": "Adaptive (ours)",
                "event_reassignments": ev_a,
                "control_reassignments": ct_a,
                "responsiveness_ratio": ratio_a})
print(f"Adaptive:  event={ev_a}/{len(liberia_idx)}, "
      f"control={ct_a}/{len(liberia_idx)}, ratio={ratio_a:.2f}")

results_df = pd.DataFrame(results)
results_df.to_csv(OUT_CSV, index=False)

# Figure
fig, ax = plt.subplots(figsize=(11, 6))
x = np.arange(len(results_df))
w = 0.35
ax.bar(x - w/2, results_df["event_reassignments"], w,
       label="During real event (Liberia surge, Oct 26–Nov 2)", color="#DC2626")
ax.bar(x + w/2, results_df["control_reassignments"], w,
       label="During quiet control period (Jul 13–27)", color="#94A3B8")
ax.set_xticks(x)
ax.set_xticklabels(results_df["method"], rotation=30, ha="right")
ax.set_ylabel(f"# Liberian admin units reassigned (of {len(liberia_idx)})")
ax.set_title("Transition Responsiveness: Reassignment During Real Event vs. Quiet Period")
ax.legend()
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(OUT_FIG, dpi=150)

print(f"\n=== SUMMARY (Table 2 in manuscript) ===")
print(results_df.to_string(index=False))
print(f"\nResults → {OUT_CSV}")
print(f"Figure  → {OUT_FIG}")
