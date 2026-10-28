"""SCRIPT 22 — Revised data-cleaning figure (Fig. 3), DRC epidemic curves (Fig. 8)
and DRC cluster-assignment / adjacency figure (Fig. 9)."""
import sys, json
sys.path.insert(0, "src")
import numpy as np, pandas as pd, networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D
from astgl_core import *

plt.rcParams.update({"font.family": "Liberation Sans", "font.size": 8,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#8a8984", "axes.labelcolor": "#0b0b0b",
                     "xtick.color": "#52514e", "ytick.color": "#52514e",
                     "axes.titlesize": 8.5, "axes.titleweight": "bold"})
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"

# ---------------------------------------------------------------- Fig 3 ----
raw = pd.read_csv("data/processed/cum_confirmed_matrix.csv", index_col=0)
raw.columns = pd.to_datetime(raw.columns); raw = raw.apply(pd.to_numeric, errors="coerce")
clean, log = correct_cumulative(raw)
X = weekly_incident(clean)
top = X.sum(axis=1).sort_values(ascending=False).index[:8]
fig = plt.figure(figsize=(7.2, 5.2))
axA = fig.add_axes([0.08, 0.56, 0.62, 0.40])
for k, u in enumerate(top):
    s = clean.loc[u].dropna()
    axA.plot(s.index, s.values, color=CAT[k], lw=1.5, label=u.replace(" - ", ": ").replace("area", "Area"))
axA.set_ylabel("Cumulative confirmed cases")
axA.grid(axis="y", color=GRID, lw=0.6)
axA.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=7, frameon=False)
axA.set_title("A  Corrected cumulative series, eight highest-burden units", loc="left")
axA.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

def rawpanel(ax, unit, d0, d1, title, note, notexy):
    r = raw.loc[unit].dropna(); c = clean.loc[unit].dropna()
    r = r[(r.index >= d0) & (r.index <= d1)]; c = c[(c.index >= d0) & (c.index <= d1)]
    ax.plot(r.index, r.values, color="#eb6834", lw=1.0, marker="o", ms=2.5, label="As digitised")
    ax.plot(c.index, c.values, color="#2a78d6", lw=1.6, label="Corrected")
    ax.set_title(title, loc="left"); ax.grid(axis="y", color=GRID, lw=0.6)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.annotate(note, xy=notexy[0], xytext=notexy[1], fontsize=6.8, color=INK,
                arrowprops=dict(arrowstyle="-", lw=0.6, color=MUTED))
    ax.set_ylabel("Cumulative confirmed cases")
axB = fig.add_axes([0.08, 0.07, 0.40, 0.36]); axC = fig.add_axes([0.58, 0.07, 0.40, 0.36])
rawpanel(axB, "Sierra Leone - Western area rural", pd.Timestamp("2014-09-20"), pd.Timestamp("2014-10-30"),
         "B  Sierra Leone: Western Area Rural",
         "8 Oct report: 2,593 = national\ntotal entered in district column",
         ((pd.Timestamp("2014-10-08"), 2593), (pd.Timestamp("2014-09-21"), 1700)))
rawpanel(axC, "Liberia - Maryland", pd.Timestamp("2014-10-15"), pd.Timestamp("2014-11-20"),
         "C  Liberia: Maryland County",
         "31 Oct report: row shifted;\nMaryland received\nMontserrado's count (1,496)",
         ((pd.Timestamp("2014-10-31"), 1496), (pd.Timestamp("2014-11-04"), 900)))
axB.legend(fontsize=7, frameon=False, loc="center left")
fig.savefig("figures/revised/fig3_data_cleaning.png", dpi=300, bbox_inches="tight", facecolor="white")
print("fig3 saved")

# ---------------------------------------------------------------- DRC data --
Xd = pd.read_csv("data/processed/drc_incident_matrix.csv", index_col=0)
cols = list(Xd.columns)
dates = [pd.Timestamp("2018-07-29")] + [pd.Timestamp(c) for c in cols[1:]]
zones = list(Xd.index)

# ---------------------------------------------------------------- Fig 8 ----
sel = [("North Kivu - Mabalako", "North Kivu"), ("North Kivu - Beni", "North Kivu"),
       ("North Kivu - Kalunguta", "North Kivu"), ("North Kivu - Katwa", "North Kivu"),
       ("North Kivu - Butembo", "North Kivu"), ("Ituri - Mandima", "Ituri"),
       ("Ituri - Komanda", "Ituri"), ("Ituri - Mambasa", "Ituri")]
fig, axs = plt.subplots(4, 2, figsize=(7.2, 6.0), sharex=True, sharey=True)
for ax, (z, prov) in zip(axs.T.flatten(), sel):
    y = Xd.loc[z].values
    ax.bar(dates, y, width=6, color="#2a78d6" if prov == "North Kivu" else "#eb6834", lw=0)
    ax.set_title(f"{z.split(' - ')[1]} ({prov})", loc="left", fontsize=8)
    ax.grid(axis="y", color=GRID, lw=0.6)
    tot = int(y.sum()); pk = dates[int(np.argmax(y))]
    ax.text(0.98, 0.85, f"total {tot:,}; peak week {pk:%d %b %Y}", transform=ax.transAxes,
            ha="right", fontsize=6.6, color=MUTED)
for ax in axs[:, 0]:
    ax.set_ylabel("New confirmed\ncases per week")
for ax in axs[-1]:
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[8, 11, 2, 5]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax.set_xlim(pd.Timestamp("2018-07-20"), pd.Timestamp("2020-02-05"))
fig.tight_layout(h_pad=0.6)
fig.savefig("figures/revised/fig8_drc_epicurves.png", dpi=300, bbox_inches="tight", facecolor="white")
print("fig8 saved")

# ---------------------------------------------------------------- Fig 9 ----
lab = pd.read_csv("results/revision_drc_clusters_seed42.csv", index_col=0)
pk = Xd.values.argmax(axis=1)
order = sorted(range(len(zones)), key=lambda i: (zones[i].split(" - ")[0], pk[i]))
fig = plt.figure(figsize=(7.2, 6.4))
axA = fig.add_axes([0.17, 0.42, 0.80, 0.54])
K = 7
cmap = ListedColormap(CAT[:K])
M = lab.values[order]
axA.imshow(M, aspect="auto", cmap=cmap, norm=BoundaryNorm(np.arange(-0.5, K), K),
           interpolation="nearest")
for r, i in enumerate(order):
    axA.plot(pk[i], r, marker="o", ms=3.2, mfc="white", mec=INK, mew=0.8)
axA.set_yticks(range(len(order)))
axA.set_yticklabels([zones[i].replace(" - ", ": ") for i in order], fontsize=6.4)
tick = [i for i, d in enumerate(dates) if d.day <= 7 and d.month in (8, 11, 2, 5)]
axA.set_xticks(tick); axA.set_xticklabels([f"{dates[i]:%b\n%Y}" for i in tick], fontsize=6.6)
axA.set_xlabel("Reporting week")
axA.legend(handles=[Line2D([], [], marker="s", ls="", ms=6, color=CAT[c], label=f"Cluster {c}") for c in range(K)]
           + [Line2D([], [], marker="o", ls="", ms=4, mfc="white", mec=INK, label="Zone's peak week")],
           loc="upper left", bbox_to_anchor=(0.0, -0.13), ncol=8, fontsize=6.4, frameon=False,
           handletextpad=0.2, columnspacing=0.8)
axA.set_title("A  AST-GL cluster label of every health zone in every week (reference run)", loc="left")
axA.spines[:].set_visible(False)

# B adjacency graph coloured by peak-week cluster
axB = fig.add_axes([0.03, 0.0, 0.94, 0.30])
adj = json.load(open("data/processed/drc_adjacency.json"))
G = nx.Graph(); G.add_nodes_from(zones)
for u, nb in adj.items():
    for v in nb:
        G.add_edge(u, v)
pos = nx.kamada_kawai_layout(G.subgraph([z for z in zones if G.degree(z) > 0]))
pos["Ituri - Ariwara"] = (max(p[0] for p in pos.values()) + 0.25, max(p[1] for p in pos.values()))
pos = {k: (v[1] * 1.0, v[0] * 1.0) for k, v in pos.items()}  # rotate for width
tot = Xd.sum(axis=1)
for u, v in G.edges():
    cross = u.split(" - ")[0] != v.split(" - ")[0]
    axB.plot(*zip(pos[u], pos[v]), color="#6250d6" if cross else "#b9b7b1", lw=1.2 if cross else 0.7, zorder=1)
for z in zones:
    c = lab.loc[z].values[pk[zones.index(z)]]
    axB.scatter(*pos[z], s=20 + 0.35 * tot[z], color=CAT[c], edgecolor="white", lw=0.8, zorder=3)
    axB.annotate(z.split(" - ")[1], pos[z], (4, 4), textcoords="offset points", fontsize=6,
                 color=INK if z.split(" - ")[0] == "North Kivu" else "#c4501f")
axB.set_title("B  Health-zone adjacency graph (45 edges; purple = province-crossing); "
              "node colour = cluster at peak week, size = total cases; orange labels = Ituri/South Kivu",
              loc="left", fontsize=7.6)
axB.axis("off")
fig.savefig("figures/revised/fig9_drc_clusters.png", dpi=300, bbox_inches="tight", facecolor="white")
print("fig9 saved")
