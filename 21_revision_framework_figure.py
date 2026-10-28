"""SCRIPT 21 — Revised Figure 1: AST-GL framework schematic (vector, journal style)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

plt.rcParams.update({"font.family": "Liberation Sans", "mathtext.fontset": "dejavusans"})
INK, MUTED = "#0b0b0b", "#52514e"
C = {"data": ("#eaf2fb", "#2a78d6"), "prep": ("#e8f6f0", "#1baf7a"),
     "core": ("#efedfb", "#6250d6"), "eval": ("#fdeee7", "#eb6834")}

fig, ax = plt.subplots(figsize=(7.2, 4.6))
ax.set_xlim(0, 100); ax.set_ylim(-2, 64); ax.axis("off")

def panel(x, y, w, h, key, title):
    fc, ec = C[key]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2",
                                fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h - 2.3, title, ha="center", va="center", fontsize=8.6,
            fontweight="bold", color=INK)

def box(x, y, w, h, text, ec, fs=6.9, bold_first=True):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2,rounding_size=0.6",
                                fc="white", ec=ec, lw=0.8))
    lines = text.split("\n")
    ax.text(x + w / 2, y + h / 2, "\n".join(lines), ha="center", va="center", fontsize=fs,
            color=INK, linespacing=1.35)

def arrow(x0, y0, x1, y1, color=INK, lw=1.2, style="-|>"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=9,
                                 color=color, lw=lw, shrinkA=0, shrinkB=0))

# column geometry
cols = [(1, 19, "data", "1  Data inputs"), (23, 19, "prep", "2  Preprocessing"),
        (45, 31, "core", "3  AST-GL clustering"), (79, 20, "eval", "4  Evaluation")]
for x, w, k, t in cols:
    panel(x, 1.5, w, 60, k, t)
FS = 6.2
e = C["data"][1]
box(2.0, 43, 17, 12.5, "West Africa 2014\nsituation reports\n49 units, 27 weeks\n(development data)", e, fs=FS)
box(2.0, 25.5, 17, 14, "DRC North Kivu\n2018–2020\n29 health zones,\n77 periods\n(held-out data)", e, fs=FS)
box(2.0, 6, 17, 16, "Administrative\nboundaries\n(geoBoundaries;\nDRC: documented\nzone borders)", e, fs=FS)
e = C["prep"][1]
box(24.0, 43, 17, 12.5, "Indicator\nharmonisation and\nanomaly correction", e, fs=FS)
box(24.0, 25.5, 17, 14, "Weekly incident\ncase matrix\n$\\mathbf{X}$ ($N$ units × $T$ weeks)", e, fs=FS)
box(24.0, 6, 17, 16, "Adjacency graph $\\mathbf{A}$\n(units that share\na border)", e, fs=FS)
e = C["core"][1]
box(46.2, 47.5, 28.6, 8, "(a) Weekly affinity $\\mathbf{W}_t$, Eq. (1):\nlog-scale case similarity, restricted to $\\mathbf{A}$", e, fs=FS)
box(46.2, 37.5, 28.6, 8, "(b) Adaptive coupling $\\alpha_t[i]$, Eq. (2):\ncosine similarity of $\\mathbf{v}_t(i)$ and $\\mathbf{v}_{t+1}(i)$", e, fs=FS)
ax.add_patch(FancyBboxPatch((46.2, 14.5), 28.6, 21, boxstyle="round,pad=0.2,rounding_size=0.6",
                            fc="white", ec=e, lw=0.8))
ax.text(60.5, 33.6, "(c) Supra-graph $\\mathbf{S}$, Laplacian $\\mathbf{L}$, Eq. (3)", ha="center",
        va="center", fontsize=FS, color=INK)
base = np.array([[0, 0], [3.2, 0.9], [6.4, 0.2], [2.2, -0.9], [5.2, -1.0]])
edges = [(0, 1), (1, 2), (0, 3), (1, 3), (3, 4), (2, 4), (1, 4)]
layers = [(28.5, "week $t$"), (22.8, "week $t{+}1$"), (17.1, "week $t{+}2$")]
Ps = []
for yo, lab in layers:
    P = base + np.array([48.0, yo]); Ps.append(P)
    for a, b in edges:
        ax.plot(*P[[a, b]].T, color="#9a93e6", lw=0.8, zorder=2)
    ax.text(55.8, yo, lab, ha="left", va="center", fontsize=FS - 0.4, color=MUTED)
for P, Q in zip(Ps[:-1], Ps[1:]):
    for k in range(5):
        ax.plot([P[k, 0], Q[k, 0]], [P[k, 1], Q[k, 1]], color="#eb6834", lw=0.8,
                ls=(0, (2, 1.4)), zorder=1)
for P in Ps:
    ax.scatter(P[:, 0], P[:, 1], s=13, color=e, zorder=3, edgecolors="white", linewidths=0.5)
ax.text(62.0, 24.0, "spatial edges ($\\mathbf{W}_t$)", fontsize=FS - 0.5, color="#6250d6", va="center")
ax.text(62.0, 20.6, "coupling edges $\\alpha_t[i]$", fontsize=FS - 0.5, color="#c4501f", va="center")
box(46.2, 4, 28.6, 8, "(d) Smallest eigenvectors of $\\mathbf{L}$ + $k$-means\n→ spatiotemporal clusters", e, fs=FS)
e = C["eval"][1]
box(80.0, 41, 18, 14.5, "Hotspot recovery\nvs documented and\ncase-burden epicentres;\nbaselines: static spectral\nclustering, scan statistic", e, fs=FS)
box(80.0, 22.5, 18, 15.5, "Coupling analysis\nfixed-$\\alpha$ sweep, stability,\nevent responsiveness;\n100 $k$-means\ninitialisations", e, fs=FS)
box(80.0, 4, 18, 15.5, "Held-out validation\nDRC data, all parameters\nfrozen ($\\alpha_{max}$ = 0.3,\n$k$ = 7, 12 eigenvectors)", e, fs=FS)
for y in (49.2, 32.5, 14):
    arrow(19.3, y, 23.7, y)
arrow(32.5, 42.8, 32.5, 39.8)
arrow(42.3, 31, 44.7, 31, lw=1.6)
for y0, y1 in [(47.3, 45.7), (37.3, 35.7), (14.3, 12.2)]:
    arrow(60.5, y0, 60.5, y1)
arrow(76.3, 31, 78.7, 31, lw=1.6)
fig.savefig("figures/revised/fig1_framework.png", dpi=300, bbox_inches="tight", facecolor="white")
print("saved")
