"""
SCRIPT 18 — Validate the hand-constructed West Africa adjacency graph
against authoritative administrative polygons (geoBoundaries gbOpen,
CC BY 4.0; SLE ADM2, LBR ADM1, GIN ADM2), and write a polygon-derived
adjacency graph for the sensitivity analysis.
Two units are adjacent if their (simplified) polygons share a border of
at least 5 km (boundary of A within 500 m of boundary of B, UTM 29N);
shorter contacts (corner points, digitising slivers) are not counted.
"""
import json, itertools
import geopandas as gpd
import pandas as pd

files = {"Sierra Leone": "data/geo/SLE-ADM2.geojson",
         "Liberia": "data/geo/LBR-ADM1.geojson",
         "Guinea": "data/geo/GIN-ADM2.geojson"}
fix = {"Western area rural": "Western Area Rural", "Western area urban": "Western Area Urban",
       "RiverCess": "Rivercess", "Mzerekore": "Nzerekore"}
gs = []
for c, f in files.items():
    g = gpd.read_file(f)[["shapeName", "geometry"]]
    g["country"] = c
    gs.append(g)
G = pd.concat(gs, ignore_index=True)
G = gpd.GeoDataFrame(G, geometry="geometry", crs=4326).to_crs(32629)
G["geometry"] = G.buffer(0)
hand = json.load(open("data/processed/admin_unit_adjacency.json"))
units = sorted(hand)
def key(u):
    c, n = u.split(" - ")
    return c, fix.get(n, n)
lookup = {(r.country, r.shapeName): r.geometry for r in G.itertuples()}
missing = [u for u in units if key(u) not in lookup]
print("units without polygon:", missing)
geo_edges = set()
for a, b in itertools.combinations(units, 2):
    ga, gb = lookup[key(a)], lookup[key(b)]
    if ga.distance(gb) > 500:
        continue
    shared_km = ga.boundary.buffer(500).intersection(gb.boundary).length / 1000
    if shared_km >= 5.0:
        geo_edges.add(tuple(sorted((a, b))))
hand_edges = {tuple(sorted((u, v))) for u, nb in hand.items() for v in nb if u != v}
both = hand_edges & geo_edges
only_hand = sorted(hand_edges - geo_edges)
only_geo = sorted(geo_edges - hand_edges)
print(f"hand edges={len(hand_edges)} geo edges={len(geo_edges)} agree={len(both)}")
print("in hand graph but NOT sharing a polygon border:")
for e in only_hand: print("   ", e)
print("sharing a polygon border but NOT in hand graph:")
for e in only_geo: print("   ", e)
adj = {u: [] for u in units}
for a, b in geo_edges:
    adj[a].append(b); adj[b].append(a)
json.dump(adj, open("data/processed/admin_unit_adjacency_geoboundaries.json", "w"), indent=2)
pd.DataFrame([{"u": a, "v": b, "in_hand": (a, b) in hand_edges, "in_geo": (a, b) in geo_edges}
              for a, b in sorted(hand_edges | geo_edges)]).to_csv(
    "results/revision_adjacency_comparison.csv", index=False)
iso = [u for u in units if not adj[u]]
print("isolated in geo graph:", iso)
