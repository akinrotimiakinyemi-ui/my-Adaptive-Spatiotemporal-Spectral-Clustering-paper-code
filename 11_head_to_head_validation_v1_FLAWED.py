"""
SCRIPT 11 — Head-to-Head Validation v1 (Threshold-Based)
=========================================================
NOTE: This script documents the FIRST evaluation design, which was
subsequently found to have a structural flaw and was replaced by
Script 13 (cluster purity). It is retained for full reproducibility
and transparency — the manuscript explicitly reports both the error
and the correction (Section 3.5, Correction 1).

THE FLAW: converting each method's cluster partition into a binary
"hotspot / not hotspot" prediction using a single global case-count
threshold (median peak case count across all 49 districts) produced
misleading results for the proposed method. Because the 2014-2016
outbreak was large and broadly distributed, most clusters' mean case
count exceeded the global median regardless of clustering quality,
flagging 46 of 49 districts as "hotspot" under this rule.

This script is kept as-is so the flaw and its diagnostic output can
be verified. See Script 13 for the corrected evaluation.

Input:  data/processed/cum_confirmed_matrix_clean.csv
        data/processed/baseline1_static_spectral_clusters.csv
        data/processed/baseline2_scan_statistic_clusters.json
        data/processed/wa_adaptive_clusters.csv
Output: data/processed/head_to_head_validation_v1_FLAWED.csv  (for audit)
"""

import pandas as pd
import numpy as np
import json

MATRIX_PATH = "data/processed/cum_confirmed_matrix_clean.csv"
B1_PATH     = "data/processed/baseline1_static_spectral_clusters.csv"
B2_PATH     = "data/processed/baseline2_scan_statistic_clusters.json"
ADAPTIVE_PATH = "data/processed/wa_adaptive_clusters.csv"
OUT_PATH    = "data/processed/head_to_head_validation_v1_FLAWED.csv"

GROUND_TRUTH_HOTSPOTS = {
    "Liberia - Montserrado", "Sierra Leone - Western area urban",
    "Sierra Leone - Western area rural", "Sierra Leone - Bombali",
    "Sierra Leone - Port Loko", "Guinea - Kerouane", "Guinea - Macenta",
    "Guinea - Mzerekore", "Guinea - Gueckedou", "Sierra Leone - Kailahun",
    "Liberia - Lofa",
}

matrix = pd.read_csv(MATRIX_PATH, index_col=0)
matrix = matrix.apply(pd.to_numeric, errors="coerce")
all_units = list(matrix.index)
peak_cases = matrix.max(axis=1, skipna=True)
global_median_peak = peak_cases.median()
print(f"Global median peak case count: {global_median_peak:.1f}")

def precision_recall_f1(predicted, ground_truth):
    tp = len(predicted & ground_truth)
    fp = len(predicted - ground_truth)
    fn = len(ground_truth - predicted)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1, tp, fp, fn

results = []

# Baseline 1
b1 = pd.read_csv(B1_PATH)
cluster_means = b1.groupby("cluster")["peak_cum_confirmed_cases"].mean()
hotspot_clusters_b1 = cluster_means[cluster_means > global_median_peak].index.tolist()
predicted_b1 = set(b1[b1["cluster"].isin(hotspot_clusters_b1)]["admin_unit"])
p, r, f1, tp, fp, fn = precision_recall_f1(predicted_b1, GROUND_TRUTH_HOTSPOTS)
results.append({"method": "Baseline 1: Static Spectral",
                "precision": p, "recall": r, "f1": f1, "tp": tp, "fp": fp, "fn": fn,
                "n_predicted_positive": len(predicted_b1)})
print(f"\nBaseline 1: precision={p:.3f}, recall={r:.3f}, F1={f1:.3f}, "
      f"n_predicted={len(predicted_b1)}")

# Baseline 2
with open(B2_PATH) as f:
    b2 = json.load(f)
predicted_b2 = set(b2["mlc_units"])
p, r, f1, tp, fp, fn = precision_recall_f1(predicted_b2, GROUND_TRUTH_HOTSPOTS)
results.append({"method": "Baseline 2: Scan Statistic",
                "precision": p, "recall": r, "f1": f1, "tp": tp, "fp": fp, "fn": fn,
                "n_predicted_positive": len(predicted_b2)})
print(f"Baseline 2: precision={p:.3f}, recall={r:.3f}, F1={f1:.3f}, "
      f"n_predicted={len(predicted_b2)}")

# Adaptive
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

cluster_peak = {}
for u, c in unit_to_cluster_adaptive.items():
    cluster_peak.setdefault(c, []).append(peak_cases.get(u, 0))
cluster_mean = {c: np.mean(v) for c, v in cluster_peak.items()}
hotspot_clusters_a = [c for c, m in cluster_mean.items() if m > global_median_peak]
predicted_a = {u for u, c in unit_to_cluster_adaptive.items() if c in hotspot_clusters_a}
p, r, f1, tp, fp, fn = precision_recall_f1(predicted_a, GROUND_TRUTH_HOTSPOTS)
results.append({"method": "Adaptive AST-GL",
                "precision": p, "recall": r, "f1": f1, "tp": tp, "fp": fp, "fn": fn,
                "n_predicted_positive": len(predicted_a)})
print(f"Adaptive:   precision={p:.3f}, recall={r:.3f}, F1={f1:.3f}, "
      f"n_predicted={len(predicted_a)}")

print(f"\n*** NOTE: n_predicted={len(predicted_a)} for Adaptive = "
      f"{100*len(predicted_a)/len(all_units):.0f}% of all districts flagged.")
print("*** This is the documented threshold-artifact flaw.")
print("*** See Script 13 (cluster purity) for the corrected evaluation.")

pd.DataFrame(results).to_csv(OUT_PATH, index=False)
print(f"\nFlawed results logged → {OUT_PATH}  (kept for audit trail)")
