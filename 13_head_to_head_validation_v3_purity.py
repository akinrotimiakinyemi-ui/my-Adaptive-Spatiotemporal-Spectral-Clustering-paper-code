"""
SCRIPT 13 — Head-to-Head Validation v3: Cluster Purity (FINAL)
===============================================================
The CORRECT and FINAL evaluation used in the manuscript (Table 1).

Metric: best-matching cluster purity + hotspot recall + F1.
For each method, find the single cluster (≥2 members) that maximises
the harmonic mean of:
  - purity:  fraction of that cluster's members that are true hotspots
  - recall:  fraction of ALL ground-truth hotspots in that cluster

This metric credits a method for concentrating documented hotspots
into a single coherent cluster, regardless of how many clusters the
method uses overall — the correct lens for evaluating an unsupervised
partition against a coarse ground truth.

Ground truth (11 districts): WHO/CDC-documented epicentres —
Montserrado, Western Area Urban/Rural, Bombali, Port Loko (Sierra
Leone), Kerouane, Macenta, Mzerekore, Gueckedou (Guinea),
Kailahun (Sierra Leone), Lofa (Liberia).

Input:  data/processed/baseline1_static_spectral_clusters.csv
        data/processed/baseline2_scan_statistic_clusters.json
        data/processed/wa_adaptive_clusters.csv
        data/processed/cum_confirmed_matrix_clean.csv
Output: data/processed/head_to_head_validation_v3_purity.csv
"""

import pandas as pd
import numpy as np
import json

MATRIX_PATH   = "data/processed/cum_confirmed_matrix_clean.csv"
B1_PATH       = "data/processed/baseline1_static_spectral_clusters.csv"
B2_PATH       = "data/processed/baseline2_scan_statistic_clusters.json"
ADAPTIVE_PATH = "data/processed/wa_adaptive_clusters.csv"
OUT_PATH      = "data/processed/head_to_head_validation_v3_purity.csv"

GROUND_TRUTH_HOTSPOTS = {
    "Liberia - Montserrado", "Sierra Leone - Western area urban",
    "Sierra Leone - Western area rural", "Sierra Leone - Bombali",
    "Sierra Leone - Port Loko", "Guinea - Kerouane", "Guinea - Macenta",
    "Guinea - Mzerekore", "Guinea - Gueckedou", "Sierra Leone - Kailahun",
    "Liberia - Lofa",
}
N_GT = len(GROUND_TRUTH_HOTSPOTS)

matrix = pd.read_csv(MATRIX_PATH, index_col=0)
matrix = matrix.apply(pd.to_numeric, errors="coerce")
all_units = list(matrix.index)
peak_cases = matrix.max(axis=1, skipna=True)


def best_matching_cluster(unit_to_cluster, ground_truth, min_size=2):
    """Return stats for the cluster maximising purity×recall F1."""
    cluster_members = {}
    for u, c in unit_to_cluster.items():
        cluster_members.setdefault(c, []).append(u)
    best = None
    for c, members in cluster_members.items():
        if len(members) < min_size:
            continue
        n_hits = sum(1 for m in members if m in ground_truth)
        purity  = n_hits / len(members)
        recall  = n_hits / len(ground_truth)
        f1 = 2 * purity * recall / (purity + recall) if (purity + recall) > 0 else 0.0
        if best is None or f1 > best["f1"]:
            best = {"cluster_id": c, "members": members, "n_hits": n_hits,
                     "purity": purity, "hotspot_recall": recall,
                     "cluster_size": len(members), "f1": f1}
    return best


results = []

# ── Baseline 1 ──────────────────────────────────────────────────────────────
b1 = pd.read_csv(B1_PATH).set_index("admin_unit")["cluster"].to_dict()
best_b1 = best_matching_cluster(b1, GROUND_TRUTH_HOTSPOTS)
print(f"Baseline 1 — best cluster {best_b1['cluster_id']} "
      f"(size={best_b1['cluster_size']}, hits={best_b1['n_hits']}/{N_GT})")
print(f"  purity={best_b1['purity']:.3f}, recall={best_b1['hotspot_recall']:.3f}, "
      f"F1={best_b1['f1']:.3f}")
print(f"  members: {sorted(best_b1['members'])}")
results.append({"method": "Baseline 1: Static Spectral Clustering",
                **{k: v for k, v in best_b1.items() if k != "members"}})

# ── Baseline 2 ──────────────────────────────────────────────────────────────
with open(B2_PATH) as f:
    b2 = json.load(f)
mlc = b2["mlc_units"]
n_hits_b2 = sum(1 for u in mlc if u in GROUND_TRUTH_HOTSPOTS)
purity_b2 = n_hits_b2 / len(mlc)
recall_b2 = n_hits_b2 / N_GT
f1_b2 = 2 * purity_b2 * recall_b2 / (purity_b2 + recall_b2) if (purity_b2 + recall_b2) > 0 else 0.0
print(f"\nBaseline 2 — MLC (size={len(mlc)}, hits={n_hits_b2}/{N_GT})")
print(f"  purity={purity_b2:.3f}, recall={recall_b2:.3f}, F1={f1_b2:.3f}")
print(f"  members: {sorted(mlc)}")
results.append({"method": "Baseline 2: Space-Time Scan Statistic",
                "cluster_id": "MLC", "n_hits": n_hits_b2,
                "purity": purity_b2, "hotspot_recall": recall_b2,
                "cluster_size": len(mlc), "f1": f1_b2})

# ── Adaptive AST-GL ─────────────────────────────────────────────────────────
adaptive = pd.read_csv(ADAPTIVE_PATH, index_col=0)
weekly_cum = pd.read_csv(MATRIX_PATH, index_col=0)
weekly_cum.columns = pd.to_datetime(weekly_cum.columns)
weekly_cum = weekly_cum.apply(pd.to_numeric, errors="coerce")
weekly_res = weekly_cum.T.resample("W").max().T.ffill(axis=1)
incident   = weekly_res.diff(axis=1).clip(lower=0).fillna(0)
incident.iloc[:, 0] = weekly_res.iloc[:, 0].fillna(0)
peak_week_idx = incident.values.argmax(axis=1)

unit_to_cluster_adaptive = {}
for i, unit in enumerate(adaptive.index):
    week_col = adaptive.columns[peak_week_idx[i]]
    unit_to_cluster_adaptive[unit] = adaptive.loc[unit, week_col]

best_a = best_matching_cluster(unit_to_cluster_adaptive, GROUND_TRUTH_HOTSPOTS)
print(f"\nAdaptive AST-GL — best cluster {best_a['cluster_id']} "
      f"(size={best_a['cluster_size']}, hits={best_a['n_hits']}/{N_GT})")
print(f"  purity={best_a['purity']:.3f}, recall={best_a['hotspot_recall']:.3f}, "
      f"F1={best_a['f1']:.3f}")
print(f"  members: {sorted(best_a['members'])}")
results.append({"method": "Proposed: Adaptive AST-GL",
                **{k: v for k, v in best_a.items() if k != "members"}})

results_df = pd.DataFrame(results)
results_df.to_csv(OUT_PATH, index=False)
print(f"\n=== SUMMARY (Table 1 in manuscript) ===")
print(results_df[["method","cluster_size","purity","hotspot_recall","f1"]].to_string(index=False))
print(f"\nResults → {OUT_PATH}")
