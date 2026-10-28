"""
SCRIPT 05 — Build Administrative Adjacency Graph
=================================================
Constructs the 49-node, 89-edge administrative border adjacency graph
for Guinea, Liberia, and Sierra Leone, including 9 cross-border edges
encoding the documented 2014 Ebola transmission corridor.

Source: Cross-referenced from Wikipedia district border descriptions,
Wikidata border relations, and regional-grouping membership for Guinea.
See manuscript Section 3.2.2 for the full methodology and limitations.

Output: data/processed/admin_unit_adjacency.json
        figures/adjacency_graph.png
"""

import json
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

os.makedirs("figures", exist_ok=True)

SL_EDGES = [
    ("Kailahun","Kenema"),("Kailahun","Kono"),("Kenema","Bo"),("Kenema","Tonkolili"),
    ("Kenema","Kono"),("Kenema","Pujehun"),("Kono","Koinadugu"),("Kono","Tonkolili"),
    ("Kambia","Port Loko"),("Kambia","Bombali"),("Koinadugu","Bombali"),
    ("Koinadugu","Tonkolili"),("Bombali","Port Loko"),("Bombali","Tonkolili"),
    ("Bombali","Kambia"),("Tonkolili","Port Loko"),("Tonkolili","Bo"),
    ("Tonkolili","Moyamba"),("Port Loko","Western area rural"),("Pujehun","Bo"),
    ("Pujehun","Bonthe"),("Bo","Moyamba"),("Bo","Bonthe"),("Moyamba","Port Loko"),
    ("Moyamba","Bonthe"),("Western area rural","Western area urban"),
    ("Western area rural","Moyamba"),
]
LB_EDGES = [
    ("Bomi","Grand Cape Mount"),("Bomi","Gbarpolu"),("Bomi","Bong"),
    ("Bomi","Montserrado"),("Gbarpolu","Grand Cape Mount"),("Gbarpolu","Bong"),
    ("Gbarpolu","Lofa"),("Grand Cape Mount","Lofa"),("Lofa","Bong"),
    ("Bong","Margibi"),("Bong","Montserrado"),("Bong","Grand Bassa"),
    ("Bong","Nimba"),("Bong","Lofa"),("Montserrado","Margibi"),
    ("Margibi","Grand Bassa"),("Grand Bassa","Nimba"),("Grand Bassa","RiverCess"),
    ("Nimba","RiverCess"),("Nimba","Grand Gedeh"),("RiverCess","Sinoe"),
    ("Grand Gedeh","Sinoe"),("Grand Gedeh","River Gee"),("Sinoe","River Gee"),
    ("River Gee","Maryland"),
]
GN_EDGES = [
    ("Gueckedou","Kissidougou"),("Gueckedou","Macenta"),("Macenta","Beyla"),
    ("Macenta","Yomou"),("Yomou","Lola"),("Lola","Beyla"),
    ("Mzerekore","Macenta"),("Mzerekore","Lola"),("Mzerekore","Yomou"),
    ("Kissidougou","Dabola"),("Dabola","Dinguiraye"),("Kissidougou","Kerouane"),
    ("Kerouane","Kouroussa"),("Kerouane","Beyla"),("Kouroussa","Siguiri"),
    ("Kouroussa","Dabola"),("Dalaba","Pita"),("Dalaba","Telimele"),
    ("Coyah","Dubreka"),("Coyah","Forecariah"),("Dubreka","Boffa"),
    ("Dubreka","Telimele"),("Forecariah","Kindia"),("Kindia","Telimele"),
    ("Kindia","Dalaba"),("Telimele","Boffa"),("Telimele","Pita"),
    ("Conakry","Coyah"),("Conakry","Dubreka"),("Conakry","Forecariah"),
]
CROSS_EDGES = [
    ("Guinea - Gueckedou",   "Sierra Leone - Kailahun"),
    ("Guinea - Gueckedou",   "Liberia - Lofa"),
    ("Sierra Leone - Kailahun", "Liberia - Lofa"),
    ("Guinea - Macenta",     "Liberia - Lofa"),
    ("Guinea - Kissidougou", "Sierra Leone - Kono"),
    ("Sierra Leone - Kambia","Guinea - Forecariah"),
    ("Sierra Leone - Koinadugu","Guinea - Kissidougou"),
    ("Sierra Leone - Pujehun","Liberia - Grand Cape Mount"),
    ("Sierra Leone - Kenema","Liberia - Lofa"),
]

G = nx.Graph()
for u, v in SL_EDGES:
    G.add_edge(f"Sierra Leone - {u}", f"Sierra Leone - {v}", edge_type="domestic")
for u, v in LB_EDGES:
    G.add_edge(f"Liberia - {u}", f"Liberia - {v}", edge_type="domestic")
for u, v in GN_EDGES:
    G.add_edge(f"Guinea - {u}", f"Guinea - {v}", edge_type="domestic")
for u, v in CROSS_EDGES:
    G.add_edge(u, v, edge_type="cross_border")

adj_dict = {node: sorted(list(G.neighbors(node))) for node in G.nodes()}
with open("data/processed/admin_unit_adjacency.json", "w") as f:
    json.dump(adj_dict, f, indent=2)

print(f"Nodes: {G.number_of_nodes()}")
print(f"Edges: {G.number_of_edges()}")
print(f"Cross-border: {sum(1 for _,_,d in G.edges(data=True) if d['edge_type']=='cross_border')}")
print(f"Isolated nodes: {len(list(nx.isolates(G)))}")

# Verify all surveillance units are present
matrix = pd.read_csv("data/processed/cum_confirmed_matrix_clean.csv", index_col=0)
surv_units = set(matrix.index)
graph_units = set(G.nodes())
missing = surv_units - graph_units
print(f"Surveillance units missing from graph: {len(missing)}")
if missing:
    print(f"  {sorted(missing)}")

# Visualize
matrix_peak = matrix.apply(pd.to_numeric, errors="coerce").max(axis=1, skipna=True)
colors = {"Sierra Leone": "#4C72B0", "Liberia": "#DD8452", "Guinea": "#55A868"}
node_colors = [colors.get(n.split(" - ")[0], "#888") for n in G.nodes()]
node_sizes = [200 + 15 * float(matrix_peak.get(n, 0)) ** 0.7 for n in G.nodes()]
pos = nx.spring_layout(G, seed=42, k=0.6)
fig, ax = plt.subplots(figsize=(14, 11))
nx.draw_networkx_edges(G, pos, alpha=0.3, ax=ax)
nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes,
                       alpha=0.85, ax=ax)
labels = {n: n.split(" - ")[1] for n in G.nodes()}
nx.draw_networkx_labels(G, pos, labels=labels, font_size=7, ax=ax)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(facecolor=c, label=k) for k, c in colors.items()],
          loc="upper left", fontsize=10)
ax.set_title("Admin Unit Adjacency Graph (node size = peak confirmed cases)", fontsize=13)
ax.axis("off")
fig.tight_layout()
fig.savefig("figures/adjacency_graph.png", dpi=150)
print("Adjacency graph saved → figures/adjacency_graph.png")
