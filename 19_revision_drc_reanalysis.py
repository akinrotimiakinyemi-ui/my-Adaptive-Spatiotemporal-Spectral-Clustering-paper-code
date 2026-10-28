"""
SCRIPT 19 — Revision analyses for the DRC North Kivu held-out dataset
=====================================================================
Same frozen parameters as West Africa. Adds (response to reviewers):
  * deterministic eigensolver + 100 k-means initialisations,
  * Baseline 1 (static spectral) and Baseline 2 (graph-adapted
    space-time permutation scan) on the DRC data, so that every
    cross-outbreak comparison in the paper is backed by a result,
  * documented vs burden-based (top-7 by confirmed cases) ground truth.
Inputs are written by Script 16.
"""
import sys, os, json
sys.path.insert(0, "src")
import numpy as np, pandas as pd, networkx as nx
from sklearn.cluster import SpectralClustering
from astgl_core import *

X_df = pd.read_csv("data/processed/drc_incident_matrix.csv", index_col=0)
zones = list(X_df.index); X = X_df.values.astype(float); N, T = X.shape
A = load_adjacency("data/processed/drc_adjacency.json", zones)
for i in range(N):
    if A[i].sum() == 0:
        A[i, i] = 1.0      # Ariwara: self-loop, as in the paper
burden = set(X_df.sum(axis=1).sort_values(ascending=False).index[:7])
GTS = {"documented": DRC_GT_DOCUMENTED, "burden": burden}
json.dump({k: sorted(v) for k, v in GTS.items()},
          open("results/revision_ground_truths_drc.json", "w"), indent=2)

W = layer_affinities(X, A)
alpha = adaptive_alpha(W, A)
print(f"alpha mean={alpha.mean():.3f} sd={alpha.std():.3f}")
vals, vecs, allvals = spectral_embedding(supra_adjacency(W, alpha))
print("12 smallest eigenvalues:", np.round(vals, 5))
gaps = np.diff(vals); print("largest gap after eigenvalue #", int(np.argmax(gaps[1:]) + 2))
rows = []; peak_rows = []
for seed in range(100):
    lab = cluster_embedding(vecs, N, T, seed=seed)
    sm, ssd = stability(lab)
    asg = peak_week_assignment(lab, X, zones)
    peak_rows.append(dict(seed=seed, **{z: int(c) for z, c in asg.items()}))
    for g, gt in GTS.items():
        b = best_matching_cluster(asg, gt)
        rows.append(dict(method="AST-GL", gt=g, seed=seed, stab_mean=sm, stab_sd=ssd,
                         **{k: b[k] for k in ["size", "hits", "purity", "recall", "f1"]},
                         members="; ".join(b["members"])))
    if seed == 42:
        pd.DataFrame(lab, index=zones, columns=X_df.columns).to_csv(
            "results/revision_drc_clusters_seed42.csv")

# Baseline 1: static snapshot of total confirmed cases (k = 8 as for West Africa
# would exceed sensible granularity for 29 zones; use silhouette over 2..8)
from sklearn.metrics import silhouette_score
snap = X.sum(axis=1)
d = np.abs(snap[:, None] - snap[None, :]); pr = np.argwhere(np.triu(A, 1) > 0)
s = max(np.median(d[pr[:, 0], pr[:, 1]]), 1e-6)
aff = A * np.exp(-(d ** 2) / (2 * s ** 2)); np.fill_diagonal(aff, np.maximum(np.diag(aff), 0))
sil = {}
for k in range(2, 9):
    lab = SpectralClustering(n_clusters=k, affinity="precomputed", random_state=42,
                             eigen_solver="lobpcg").fit_predict(aff)
    dist = 1 - aff / (aff.max() + 1e-9); np.fill_diagonal(dist, 0)
    sil[k] = silhouette_score(dist, lab, metric="precomputed")
kb = max(sil, key=sil.get); print("B1 silhouette:", {k: round(v, 3) for k, v in sil.items()}, "-> k =", kb)
for seed in range(100):
    lab = SpectralClustering(n_clusters=kb, affinity="precomputed", random_state=seed,
                             eigen_solver="lobpcg").fit_predict(aff)
    asg = dict(zip(zones, lab))
    for g, gt in GTS.items():
        b = best_matching_cluster(asg, gt)
        rows.append(dict(method="B1-static", gt=g, seed=seed,
                         **{k: b[k] for k in ["size", "hits", "purity", "recall", "f1"]},
                         members="; ".join(b["members"])))

# Baseline 2: graph-adapted space-time permutation scan (same settings as WA)
G = nx.Graph(); G.add_nodes_from(zones)
for i in range(N):
    for j in range(i + 1, N):
        if A[i, j] > 0: G.add_edge(zones[i], zones[j])
cand = set()
for u in zones:
    for r in range(3):
        cand.add(frozenset(nx.single_source_shortest_path_length(G, u, cutoff=r)))
cand = [sorted(z) for z in cand]
wins = [(s0, s0 + L) for L in range(1, 9) for s0 in range(T - L + 1)]
idx = {z: i for i, z in enumerate(zones)}
def scan(arr):
    C = arr.sum(); wk = arr.sum(axis=0); best = (-np.inf, None, None)
    cw = np.concatenate([[0], np.cumsum(wk)])
    for z in cand:
        zt = arr[[idx[u] for u in z]].sum(axis=0); tot = zt.sum()
        if tot <= 0: continue
        cz = np.concatenate([[0], np.cumsum(zt)])
        for a, b in wins:
            c = cz[b] - cz[a]
            if c <= 0: continue
            E = tot * (cw[b] - cw[a]) / C
            if E <= 0 or c <= E: continue
            co, Eo = C - c, C - E
            llr = c * np.log(c / E) + (co * np.log(co / Eo) if co > 0 and Eo > 0 else 0)
            if llr > best[0]: best = (llr, z, (a, b))
    return best
obs = scan(X)
rng = np.random.default_rng(42); shares = X.sum(axis=1) / X.sum(); wk = X.sum(axis=0)
perm = []
for p in range(999):
    arr = np.zeros_like(X)
    for w in range(T):
        if wk[w] > 0: arr[:, w] = rng.multinomial(int(round(wk[w])), shares)
    perm.append(scan(arr)[0])
pval = (np.sum(np.array(perm) >= obs[0]) + 1) / 1000
cols = list(X_df.columns)
print(f"B2 MLC: {obs[1]} weeks {cols[obs[2][0]]}..{cols[obs[2][1]-1]} LLR={obs[0]:.1f} p={pval:.3f}")
for g, gt in GTS.items():
    h = sum(u in gt for u in obs[1]); p_, r_ = h / len(obs[1]), h / len(gt)
    rows.append(dict(method="B2-scan", gt=g, seed=-1, size=len(obs[1]), hits=h,
                     purity=p_, recall=r_, f1=2 * p_ * r_ / (p_ + r_) if p_ + r_ else 0,
                     members="; ".join(obs[1])))
json.dump(dict(mlc=obs[1], llr=obs[0], start=cols[obs[2][0]], end=cols[obs[2][1] - 1],
               p=pval, b1_k=kb, b1_silhouette=sil), open("results/revision_drc_baselines.json", "w"),
          indent=2, default=float)
pd.DataFrame(rows).to_csv("results/revision_drc_eval.csv", index=False)
pd.DataFrame(peak_rows).to_csv("results/revision_drc_peakweek_assignments.csv", index=False)
print("done")
