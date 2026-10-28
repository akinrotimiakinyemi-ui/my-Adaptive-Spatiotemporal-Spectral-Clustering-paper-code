"""
SCRIPT 03 — Build Cumulative Case Matrix and Sanity Check
==========================================================
Pivots the canonical cumulative-confirmed-case signal into a
(49 admin units) × (137 reporting dates) matrix, then validates
it against known outbreak history.

Expected top districts by peak case count:
  - SL: Montserrado (Liberia), Western Area Urban/Rural, Port Loko, Bombali
  - GN: Macenta, Gueckedou
  - LBR: Montserrado, Margibi

Input:  data/processed/west_africa_canonical_signals.csv
Output: data/processed/cum_confirmed_matrix.csv
        figures/sanity_check_top_districts_CLEAN.png
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

IN_PATH    = "data/processed/west_africa_canonical_signals.csv"
OUT_MATRIX = "data/processed/cum_confirmed_matrix.csv"
FIG_PATH   = "figures/sanity_check_top_districts_CLEAN.png"
os.makedirs("figures", exist_ok=True)

df = pd.read_csv(IN_PATH, parse_dates=["date"])
cc = df[df["signal"] == "cum_confirmed_cases"].copy()
cc["value"] = pd.to_numeric(cc["value"], errors="coerce")
cc = cc.dropna(subset=["value"])
cc["unit_id"] = cc["country"] + " - " + cc["admin_unit"]

cc_agg = cc.groupby(["unit_id", "date"])["value"].max().reset_index()
matrix = cc_agg.pivot(index="unit_id", columns="date", values="value")
matrix = matrix.apply(pd.to_numeric, errors="coerce")
matrix.to_csv(OUT_MATRIX)

peak = matrix.max(axis=1, skipna=True).sort_values(ascending=False)
print(f"Matrix shape: {matrix.shape}")
print(f"\nTop 15 districts by peak cumulative cases:")
print(peak.head(15))

top8 = peak.head(8).index
fig, ax = plt.subplots(figsize=(11, 6))
for unit in top8:
    series = matrix.loc[unit].dropna()
    series = pd.to_numeric(series, errors="coerce").dropna()
    ax.plot(series.index, series.values, marker="o", markersize=3, label=unit)
ax.set_title("Cleaned Cumulative Confirmed Ebola Cases — Top 8 Districts (2014 West Africa)")
ax.set_xlabel("Date")
ax.set_ylabel("Cumulative confirmed cases")
ax.legend(loc="upper left", fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIG_PATH, dpi=150)
print(f"\nSanity-check figure saved: {FIG_PATH}")
