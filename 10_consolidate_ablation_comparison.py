"""
SCRIPT 10 — Consolidate Ablation Comparison Table and Figure
=============================================================
Reads the fixed-coupling ablation results (Script 09) and the adaptive
method's cluster output (Script 08), computes the comparison table, and
generates Figure 3 (adaptive vs. fixed coupling stability bar chart).

Input:  data/processed/ablation_fixed_coupling_results.csv
        data/processed/wa_adaptive_clusters.csv
Output: data/processed/comparison_adaptive_vs_fixed.csv
        figures/adaptive_vs_fixed_stability.png
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

os.makedirs("figures", exist_ok=True)

ABLATION_PATH = "data/processed/ablation_fixed_coupling_results.csv"
ADAPTIVE_PATH = "data/processed/wa_adaptive_clusters.csv"
OUT_CSV       = "data/processed/comparison_adaptive_vs_fixed.csv"
OUT_FIG       = "figures/adaptive_vs_fixed_stability.png"

ablation = pd.read_csv(ABLATION_PATH)
adaptive_clusters = pd.read_csv(ADAPTIVE_PATH, index_col=0)
n_distinct = adaptive_clusters.apply(lambda row: row.nunique(), axis=1)
adaptive_mean, adaptive_std = n_distinct.mean(), n_distinct.std()
print(f"Adaptive method (recomputed): mean={adaptive_mean:.4f}, std={adaptive_std:.4f}")

comparison = ablation.rename(columns={
    "mean_clusters_visited_per_unit": "mean_clusters_per_unit",
    "std_clusters_visited_per_unit":  "std_clusters_per_unit",
}).copy()
comparison["method"] = comparison["fixed_alpha"].apply(lambda a: f"Fixed (α={a})")
comparison = comparison[["method", "mean_clusters_per_unit", "std_clusters_per_unit"]]

adaptive_row = pd.DataFrame([{
    "method": "Adaptive (ours)",
    "mean_clusters_per_unit": adaptive_mean,
    "std_clusters_per_unit":  adaptive_std,
}])
full_comparison = pd.concat([comparison, adaptive_row], ignore_index=True)
full_comparison.to_csv(OUT_CSV, index=False)
print(full_comparison.to_string(index=False))

# Bar chart
fig, ax = plt.subplots(figsize=(10, 5.5))
colors = ["#94A3B8"] * (len(full_comparison) - 1) + ["#DC2626"]
bars = ax.bar(
    full_comparison["method"],
    full_comparison["mean_clusters_per_unit"],
    yerr=full_comparison["std_clusters_per_unit"],
    capsize=4, color=colors, edgecolor="black", linewidth=0.6
)
ax.set_ylabel("Mean # distinct clusters visited per admin unit\n(lower = more temporally stable)")
ax.set_title("Temporal Cluster Stability: Adaptive vs. Fixed Coupling\n"
             "(2014 West Africa Ebola outbreak, 49 admin units, 27 weeks)")
ax.tick_params(axis="x", rotation=20)
for bar in bars:
    h = bar.get_height()
    ax.annotate(f"{h:.2f}",
                (bar.get_x() + bar.get_width() / 2, h),
                ha="center", va="bottom", fontsize=9)
fig.tight_layout()
fig.savefig(OUT_FIG, dpi=150)
print(f"\nFigure → {OUT_FIG}")
print(f"Comparison table → {OUT_CSV}")
