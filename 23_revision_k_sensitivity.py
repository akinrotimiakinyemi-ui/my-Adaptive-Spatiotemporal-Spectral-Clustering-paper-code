"""SCRIPT 23 — Sensitivity of AST-GL hotspot recovery to the number of clusters k
(primary graph, corrected data; 20 k-means initialisations per k)."""
import sys; sys.path.insert(0, "src")
import numpy as np, pandas as pd
from astgl_core import *
raw = pd.read_csv("data/processed/cum_confirmed_matrix.csv", index_col=0)
raw.columns = pd.to_datetime(raw.columns); raw = raw.apply(pd.to_numeric, errors="coerce")
units = list(raw.index); N = len(units)
A = load_adjacency("data/processed/admin_unit_adjacency_geoboundaries.json", units)
X = weekly_incident(correct_cumulative(raw)[0]).values; T = X.shape[1]
W = layer_affinities(X, A); _, V, _ = spectral_embedding(supra_adjacency(W, adaptive_alpha(W, A)))
rows = []
for k in range(4, 11):
    for s in range(20):
        lab = cluster_embedding(V, N, T, k=k, seed=s)
        b = best_matching_cluster(peak_week_assignment(lab, X, units), WA_GT_DOCUMENTED)
        rows.append(dict(k=k, seed=s, f1=b["f1"], recall=b["recall"], purity=b["purity"],
                         clusters_per_week=np.mean([len(set(lab[:, t])) for t in range(T)])))
df = pd.DataFrame(rows); df.to_csv("results/revision_wa_k_sensitivity.csv", index=False)
print(df.groupby("k")[["f1", "recall", "purity", "clusters_per_week"]].agg(["mean", "std"]).round(3).to_string())
