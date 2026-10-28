"""
SCRIPT 20 — Study-area maps (new Figure 2) and map of AST-GL hotspot
cluster vs ground truth (West Africa). Boundaries: geoBoundaries gbOpen
(Runfola et al., 2020), CC BY 4.0.
"""
import sys, json
sys.path.insert(0, "src")
import numpy as np, pandas as pd, geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib_scalebar.scalebar import ScaleBar
from astgl_core import *

plt.rcParams.update({"font.family": "Liberation Sans", "font.size": 8.5,
                     "axes.titlesize": 9.5, "axes.titleweight": "bold"})
INK, MUTED = "#0b0b0b", "#52514e"

fix = {"Western area rural": "Western Area Rural", "Western area urban": "Western Area Urban",
       "RiverCess": "Rivercess", "Mzerekore": "Nzerekore"}
files = {"Sierra Leone": "data/geo/SLE-ADM2.geojson", "Liberia": "data/geo/LBR-ADM1.geojson",
         "Guinea": "data/geo/GIN-ADM2.geojson"}
G = []
for c, f in files.items():
    g = gpd.read_file(f)[["shapeName", "geometry"]]; g["country"] = c; G.append(g)
G = gpd.GeoDataFrame(pd.concat(G, ignore_index=True), crs=4326).to_crs(32629)
G["geometry"] = G.buffer(0)
nat = gpd.GeoDataFrame(pd.concat([gpd.read_file(f"data/geo/{c}-ADM0.geojson") for c in ["SLE", "LBR", "GIN"]]),
                       crs=4326).to_crs(32629)

raw = pd.read_csv("data/processed/cum_confirmed_matrix.csv", index_col=0)
raw.columns = pd.to_datetime(raw.columns); raw = raw.apply(pd.to_numeric, errors="coerce")
X = weekly_incident(correct_cumulative(raw)[0])
tot = X.sum(axis=1)
inv = {}
for u in X.index:
    c, n = u.split(" - "); inv[(c, fix.get(n, n))] = u
G["unit"] = [inv.get((r.country, r.shapeName)) for r in G.itertuples()]
G["cases"] = G["unit"].map(tot)
G["pt"] = G.representative_point()
pts = {r.unit: r.pt for r in G.itertuples() if r.unit}

adj = json.load(open("data/processed/admin_unit_adjacency_geoboundaries.json"))
edges = {tuple(sorted((u, v))) for u in adj for v in adj[u]}

# ---------------- Figure 2: study area (A: West Africa, B: DRC) --------------
fig = plt.figure(figsize=(7.2, 4.3))
axA = fig.add_axes([0.0, 0.0, 0.60, 1.0]); axB = fig.add_axes([0.62, 0.08, 0.38, 0.86])
inside = G[G.unit.notna()]
outside = G[G.unit.isna()]
outside.plot(ax=axA, color="#ecebe8", edgecolor="white", linewidth=0.4)
inside[inside.cases <= 0].plot(ax=axA, color="#fbf8f4", edgecolor="#d9d7d2", linewidth=0.4)
inside = inside[inside.cases > 0]
inside.plot(ax=axA, column="cases", cmap="Oranges", norm=LogNorm(vmin=1, vmax=tot.max()),
            edgecolor="white", linewidth=0.5,
            legend=True, legend_kwds={"label": "Confirmed cases in study window (log scale)",
                                      "orientation": "horizontal", "shrink": 0.55, "pad": 0.01,
                                      "aspect": 30})
nat.boundary.plot(ax=axA, color=INK, linewidth=0.9)
for a, b in edges:
    pa, pb = pts[a], pts[b]
    cross = a.split(" - ")[0] != b.split(" - ")[0]
    axA.plot([pa.x, pb.x], [pa.y, pb.y], color="#2a78d6" if not cross else "#6250d6",
             lw=0.9 if cross else 0.45, alpha=0.9 if cross else 0.45, zorder=3)
for p in pts.values():
    axA.plot(p.x, p.y, "o", ms=2.2, color="#2a78d6", zorder=4)
key = {"Liberia - Montserrado": (-90e3, -70e3), "Sierra Leone - Western area urban": (-95e3, 10e3),
       "Sierra Leone - Port Loko": (-30e3, 55e3), "Sierra Leone - Bombali": (10e3, 65e3),
       "Sierra Leone - Kailahun": (50e3, 35e3), "Guinea - Gueckedou": (40e3, 55e3),
       "Liberia - Lofa": (60e3, 20e3), "Guinea - Macenta": (75e3, 30e3),
       "Guinea - Kerouane": (55e3, 40e3), "Guinea - Mzerekore": (75e3, -10e3),
       "Sierra Leone - Western area rural": (-90e3, -40e3), "Liberia - Margibi": (70e3, -15e3)}
for u, (dx, dy) in key.items():
    p = pts[u]; name = u.split(" - ")[1].replace("Western area", "W. Area").replace("Mzerekore", "Nzérékoré") \
        .replace("Gueckedou", "Guéckédou").replace("Kerouane", "Kérouané")
    axA.annotate(name, (p.x, p.y), (p.x + dx, p.y + dy), fontsize=7, color=INK,
                 arrowprops=dict(arrowstyle="-", lw=0.5, color=MUTED), zorder=6,
                 bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
_nat = nat.copy(); _nat["c"] = ["SIERRA LEONE", "LIBERIA", "GUINEA"]
for _, r in _nat.iterrows():
    cp = r.geometry.representative_point()
    dx, dy = {"SIERRA LEONE": (-150e3, -170e3), "LIBERIA": (40e3, -190e3), "GUINEA": (120e3, 230e3)}[r.c]
    axA.text(cp.x + dx, cp.y + dy, r.c, fontsize=8, color=MUTED, ha="center", fontweight="bold")
axA.add_artist(ScaleBar(1, location="lower left", length_fraction=0.2, box_alpha=0.7, font_properties={"size": 7}))
axA.annotate("N", xy=(0.93, 0.95), xytext=(0.93, 0.87), xycoords="axes fraction", ha="center",
             fontsize=9, fontweight="bold", arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.2))
axA.legend(handles=[Line2D([], [], color="#2a78d6", lw=0.8, marker="o", ms=3, label="Shared border (within country)"),
                    Line2D([], [], color="#6250d6", lw=1.1, label="Shared border (cross-border)"),
                    Patch(fc="#fbf8f4", ec="#d9d7d2", label="0 confirmed cases"),
                    Patch(fc="#ecebe8", ec="white", label="No surveillance data")],
           loc="upper left", fontsize=6.8, frameon=True, framealpha=0.9)
axA.set_title("A  West Africa 2014: 49 units and adjacency graph", loc="left")
axA.set_axis_off()

# B: DRC locator
c1 = gpd.read_file("data/geo/COD-ADM1.geojson").to_crs(32735)
c2 = gpd.read_file("data/geo/COD-ADM2.geojson").to_crs(32735)
c0 = gpd.read_file("data/geo/COD-ADM0.geojson").to_crs(32735)
prov = c1[c1.shapeName.isin(["North Kivu", "Ituri", "South Kivu"])]
terr = gpd.sjoin(c2.set_geometry(c2.representative_point()), prov[["shapeName", "geometry"]],
                 predicate="within")
terrp = c2.loc[terr.index]
c0.plot(ax=axB, color="#f4f3f1", edgecolor=MUTED, linewidth=0.6)
prov.plot(ax=axB, color="#fde3cf", edgecolor=INK, linewidth=0.8)
terrp.boundary.plot(ax=axB, color="#b9b7b1", linewidth=0.3)
towns = ["Beni", "Butembo", "Bunia", "Goma", "Mambasa", "Mwenga"]
for t in towns:
    r = c2[c2.shapeName == t]
    if len(r):
        p = r.representative_point().iloc[0]
        axB.plot(p.x, p.y, "o", ms=3, color="#e34948", zorder=5)
        off = {"Mambasa": (-60e3, 45e3), "Bunia": (40e3, 45e3), "Butembo": (60e3, -20e3)}.get(t, (60e3, 0))
        axB.annotate(t, (p.x, p.y), (p.x + off[0], p.y + off[1]), fontsize=6.8, va="center",
                     bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85))
for _, r in prov.iterrows():
    p = r.geometry.representative_point()
    dxy = {"Ituri": (-250e3, -60e3), "North Kivu": (-230e3, -40e3), "South Kivu": (-260e3, -90e3)}[r.shapeName]
    axB.text(p.x + dxy[0], p.y + dxy[1], r.shapeName, fontsize=7, color=MUTED, fontweight="bold")
xmin, ymin, xmax, ymax = prov.total_bounds
axB.set_xlim(xmin - 250e3, xmax + 250e3); axB.set_ylim(ymin - 100e3, ymax + 100e3)
axB.add_artist(ScaleBar(1, location="lower left", length_fraction=0.3, box_alpha=0.7, font_properties={"size": 7}))
axB.set_title("B  DRC 2018–2020: affected provinces", loc="left")
axB.set_axis_off()
fig.savefig("figures/revised/fig2_study_area.png", dpi=300, bbox_inches="tight", facecolor="white")
print("fig2 saved")

# ------------- Figure 4: best-matching clusters vs ground truth (AST-GL and Baseline 1)
lab = pd.read_csv("results/revision_wa_clusters_geo_seed42.csv", index_col=0)
asg = peak_week_assignment(lab.values, X.values, list(X.index))
b_a = best_matching_cluster(asg, WA_GT_DOCUMENTED)
ev = pd.read_csv("results/revision_wa_hotspot_eval.csv")
b1row = ev[(ev.graph == "geo") & (ev.data == "full") & (ev.method == "B1-static") & (ev["gt"] == "documented") & (ev.seed == 42)].iloc[0]
b1mem = set(b1row.members.split("; "))
col = {"hit": "#2a78d6", "extra": "#9fc3ef", "miss": "#eb6834", "other": "#f4f3f1"}
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.9))
for ax, mem, title in [(axs[0], set(b_a["members"]), "A  AST-GL (adaptive coupling)"),
                       (axs[1], b1mem, "B  Baseline 1 (static spectral clustering)")]:
    def cat(u, mem=mem):
        if not isinstance(u, str): return None
        if u in mem and u in WA_GT_DOCUMENTED: return "hit"
        if u in mem: return "extra"
        if u in WA_GT_DOCUMENTED: return "miss"
        return "other"
    G["cat"] = G.unit.map(cat)
    G[G.unit.isna()].plot(ax=ax, color="#dcdbd7", edgecolor="white", lw=0.3)
    for k, c in col.items():
        sub = G[G.cat == k]
        if len(sub): sub.plot(ax=ax, color=c, edgecolor="white", lw=0.5)
    nat.boundary.plot(ax=ax, color=INK, lw=0.8)
    n = len(mem); h = len(mem & WA_GT_DOCUMENTED)
    ax.set_title(title, loc="left")
    ax.text(0.02, 0.02, f"cluster size {n}; {h} of 11 documented epicentres", transform=ax.transAxes,
            fontsize=7, color=MUTED)
    ax.set_axis_off()
axs[1].add_artist(ScaleBar(1, location="lower right", length_fraction=0.25, box_alpha=0.7, font_properties={"size": 7}))
fig.legend(handles=[Patch(fc=col["hit"], label="In best-matching cluster and documented"),
                    Patch(fc=col["extra"], label="In best-matching cluster, not documented"),
                    Patch(fc=col["miss"], label="Documented epicentre not captured"),
                    Patch(fc=col["other"], ec="#c9c8c4", label="Other units"),
                    Patch(fc="#dcdbd7", label="No surveillance data")],
           loc="lower center", ncol=3, fontsize=6.8, frameon=False, bbox_to_anchor=(0.5, -0.06))
fig.savefig("figures/revised/fig4_cluster_maps.png", dpi=300, bbox_inches="tight", facecolor="white")
print("fig4 saved", b_a["size"], b_a["hits"])
