"""
SCRIPT 12 — Head-to-Head Validation v2 (ARI/NMI — also corrected)
===================================================================
NOTE: This is the SECOND evaluation design, which was also found to have
a structural limitation and was replaced by Script 13. It is retained for
full reproducibility. The manuscript reports this correction explicitly
(Section 3.5, Correction 1).

THE LIMITATION: Adjusted Rand Index (ARI) and Normalized Mutual
Information (NMI) structurally penalise a fine-grained (k=7 or k=8)
partition against a coarse two-group ground truth, even when individual
clusters are highly pure. A method that correctly splits "Western hotspot
belt" vs. "Eastern hotspot belt" vs. "low-burden periphery" will score
poorly against a ground truth with only 2 groups, even if the clustering
is substantively correct. Manual inspection showed Baseline 1's cluster 3
= {Western Area Urban, Western Area Rural, Bombali, Port Loko} is an exact
4-for-4 match to the documented "western and northwestern Sierra Leone"
hotspot — a result ARI could not credit.

See Script 13 (cluster purity) for the corrected evaluation.

Input:  data/processed/*.csv  (all method output files)
Output: data/processed/head_to_head_validation_v2_ARI_NMI.csv  (for audit)
"""

import pandas as pd
import json
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

MATRIX_PATH   = "data/processed/cum_confirmed_matrix_clean.csv"
B1_PATH       = "data/processed/baseline1_static_spectral_clusters.csv"
B2_PATH       = "data/processed/baseline2_scan_statistic_clusters.json"
ADAPTIVE_PATH = "data/processed/wa_adaptive_clusters.csv"
OUT_PATH      = "data/processed/head_to_head_validation_v2_ARI_NMI.csv"

GROUND_TRUTH_HOTSPOTS = {
    "Liberia - Montserrado", "Sierra Leone - Western area urban",
    "Sierra Leone - Western area rural", "Sierra Leone - Bombali",
    "Sierra Leone - Port Loko", "Guinea - Kerouane", "Guinea - Macenta",
    "Guinea - Mzerekore", "Guinea - Gueckedou", "Sierra Leone - Kailahun",
    "Liberia - Lofa",
}

matrix = pd.read_csv(MATRIX_PATH, index_col=0)
all_units = list(matrix.index)
gt_binary = {u: (1 if u in GROUND_TRUTH_HOTSPOTS else 0) for u in all_units}
results = []

# Baseline 1
b1 = pd.read_csv(B1_PATH).set_index("admin_unit")["cluster"].to_dict()
units_b1 = [u for u in all_units if u in b1]
ari = adjusted_rand_score([gt_binary[u] for u in units_b1], [b1[u] for u in units_b1])
nmi = normalized_mutual_info_score([gt_binary[u] for u in units_b1], [b1[u] for u in units_b1])
results.append({"method": "Baseline 1: Static Spectral", "ARI": ari, "NMI": nmi})
print(f"Baseline 1: ARI={ari:.4f}, NMI={nmi:.4f}")

# Baseline 2 (binary by design — MLC vs. everything else)
with open(B2_PATH) as f:
    b2 = json.load(f)
mlc = set(b2["mlc_units"])
pred_b2 = [1 if u in mlc else 0 for u in all_units]
gt_full  = [gt_binary[u] for u in all_units]
ari = adjusted_rand_score(gt_full, pred_b2)
nmi = normalized_mutual_info_score(gt_full, pred_b2)
results.append({"method": "Baseline 2: Scan Statistic (MLC)", "ARI": ari, "NMI": nmi})
print(f"Baseline 2: ARI={ari:.4f}, NMI={nmi:.4f}")

# Adaptive
adaptive = pd.read_csv(ADAPTIVE_PATH, index_col=0)
weekly_cum = pd.read_csv(MATRIX_PATH, index_col=0)
weekly_cum.columns = pd.to_datetime(weekly_cum.columns)
weekly_cum = weekly_cum.apply(pd.to_numeric, errors="coerce")
weekly_res = weekly_cum.T.resample("W").max().T.ffill(axis=1)
incident   = weekly_res.diff(axis=1).clip(lower=0).fillna(0)
incident.iloc[:, 0] = weekly_res.iloc[:, 0].fillna(0)
peak_week_idx = incident.values.argmax(axis=1)
pred_adaptive = [adaptive.loc[u, adaptive.columns[peak_week_idx[i]]]
                 for i, u in enumerate(adaptive.index)]
gt_a = [gt_binary[u] for u in adaptive.index]
ari = adjusted_rand_score(gt_a, pred_adaptive)
nmi = normalized_mutual_info_score(gt_a, pred_adaptive)
results.append({"method": "Adaptive AST-GL", "ARI": ari, "NMI": nmi})
print(f"Adaptive:   ARI={ari:.4f}, NMI={nmi:.4f}")

print("\n*** NOTE: All ARI values near zero — see manuscript Section 3.5")
print("*** for the documented structural limitation of this metric.")
print("*** See Script 13 (cluster purity) for the corrected evaluation.")

pd.DataFrame(results).to_csv(OUT_PATH, index=False)
print(f"\nResults → {OUT_PATH}")
