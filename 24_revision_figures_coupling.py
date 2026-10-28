"""SCRIPT 24 — Revised Figures 5 (stability) and 6 (transition responsiveness)."""
import sys; sys.path.insert(0, "src")
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from astgl_core import *

plt.rcParams.update({"font.family": "Liberation Sans", "font.size": 8,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#8a8984", "xtick.color": "#52514e", "ytick.color": "#52514e",
                     "axes.titlesize": 8.5, "axes.titleweight": "bold"})
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#b9b7b1"
order = [f"fixed_{a}" for a in [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]] + ["adaptive"]
names = [f"α = {a}" for a in [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]] + ["Adaptive"]

r = pd.read_csv("results/revision_wa_responsiveness.csv")
r = r[(r.graph == "geo") & (r.data == "full")]
agg = r.groupby("setting").agg(m=("stab_mean", "mean"), s=("stab_mean", "std"),
                               ev=("event", "mean"), ct=("control", "mean")).loc[order]

# ---- Figure 5: stability
fig, ax = plt.subplots(figsize=(6.4, 3.2))
x = np.arange(len(order))
cols = [GREY] * 7 + [BLUE]
ax.bar(x, agg.m, 0.62, color=cols, zorder=2)
ax.errorbar(x, agg.m, yerr=agg.s, fmt="none", ecolor=INK, elinewidth=0.8, capsize=3, zorder=3)
for i, (m, s) in enumerate(zip(agg.m, agg.s)):
    ax.text(i, m + s + 0.08, f"{m:.2f}", ha="center", va="bottom", fontsize=7.4, color=INK)
ax.set_xticks(x); ax.set_xticklabels(names)
ax.set_xlabel("Temporal coupling")
ax.set_ylabel("Distinct clusters visited\nper unit over 27 weeks")
ax.set_ylim(0, agg.m.max() + agg.s.max() + 0.6)
ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
fig.tight_layout()
fig.savefig("figures/revised/fig5_stability.png", dpi=300, facecolor="white")
print(agg.round(3))

# ---- Figure 6: responsiveness
raw = pd.read_csv("data/processed/cum_confirmed_matrix.csv", index_col=0)
raw.columns = pd.to_datetime(raw.columns); raw = raw.apply(pd.to_numeric, errors="coerce")
units = list(raw.index); N = len(units); lib = [i for i, u in enumerate(units) if u.startswith("Liberia")]
A = load_adjacency("data/processed/admin_unit_adjacency_geoboundaries.json", units)
Xdf = weekly_incident(correct_cumulative(raw)[0]); X = Xdf.values; T = X.shape[1]; cols_ = list(Xdf.columns)
W = layer_affinities(X, A); al = adaptive_alpha(W, A)
series = {}
for name, a in [("Adaptive", al), ("α = 0.05", 0.05), ("α = 0.3", 0.3), ("α = 1.0", 1.0)]:
    _, V, _ = spectral_embedding(supra_adjacency(W, a))
    series[name] = [embedding_displacement(V, N, T, lib, t) for t in range(1, T - 1)]
ev_i, ct_i = cols_.index(pd.Timestamp("2014-10-26")), cols_.index(pd.Timestamp("2014-07-13"))
fig = plt.figure(figsize=(7.2, 5.4))
axA = fig.add_axes([0.09, 0.58, 0.68, 0.37])
wk = cols_[1:T - 1]
lc = {"Adaptive": BLUE, "α = 0.05": ORANGE, "α = 0.3": "#1baf7a", "α = 1.0": "#6250d6"}
for name, s in series.items():
    axA.plot(wk, s, color=lc[name], lw=2.0 if name == "Adaptive" else 1.2, marker="o", ms=2.6,
             label=name, zorder=3 if name == "Adaptive" else 2)
for d, lab in [(cols_[ev_i], "event week\n(26 Oct)"), (cols_[ct_i], "control week\n(13 Jul)")]:
    axA.axvline(d, color=MUTED, lw=0.8, ls="--", zorder=1)
    axA.text(d, axA.get_ylim()[1] if False else 0.0, "", fontsize=6)
ymax = max(max(s) for s in series.values())
axA.set_ylim(0, ymax * 1.18)
axA.text(cols_[ev_i], ymax * 1.10, "event week (26 Oct)", ha="center", fontsize=6.8, color=MUTED,
         bbox=dict(fc="white", ec="none", pad=0.5))
axA.text(cols_[ct_i], ymax * 1.10, "control week (13 Jul)", ha="center", fontsize=6.8, color=MUTED,
         bbox=dict(fc="white", ec="none", pad=0.5))
axA.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=7.2)
axA.set_ylabel("Mean embedding change,\nLiberian units")
axA.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
axA.grid(axis="y", color=GRID, lw=0.6)
axA.set_title("A  Week-to-week change in the spectral embedding of Liberia's 14 units (label-free)", loc="left")

axB = fig.add_axes([0.09, 0.08, 0.68, 0.36])
w = 0.38
axB.bar(x - w / 2, agg.ev, w, color=BLUE, label="Event week", zorder=2)
axB.bar(x + w / 2, agg.ct, w, color=GREY, label="Control week", zorder=2)
axB.set_xticks(x); axB.set_xticklabels(names)
axB.set_ylabel("Units reassigned\n(mean of 100 runs)")
axB.set_xlabel("Temporal coupling")
axB.grid(axis="y", color=GRID, lw=0.6, zorder=0)
axB.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=7.2)
axB.set_title("B  Cluster-label reassignments of Liberian units, averaged over 100 k-means initialisations",
              loc="left")
fig.savefig("figures/revised/fig6_responsiveness.png", dpi=300, bbox_inches="tight", facecolor="white")
print("saved; n Liberian units =", len(lib))
