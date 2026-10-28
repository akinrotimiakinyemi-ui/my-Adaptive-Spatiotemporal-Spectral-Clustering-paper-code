"""
SCRIPT 07 — Baseline 2: Space-Time Permutation Scan Statistic
==============================================================
Implements the graph-adapted space-time permutation scan statistic
(Kulldorff et al., 2005), the algorithm underlying SaTScan — the
field's de facto gold-standard outbreak detection tool.

Key adaptations:
  - Spatial zones = k-hop neighbourhoods on the true adjacency graph
    (NOT Euclidean circles), following Aamodt et al. (2006).
  - Log-space likelihood ratio computation to avoid float64 overflow
    when case counts are large.
  - Permutation null = multinomial reallocation of weekly totals across
    units, proportional to each unit's overall case share.

Parameters:
  MAX_GRAPH_RADIUS = 2   (hops)
  MAX_TIME_WINDOW_WEEKS = 8
  N_PERMUTATIONS = 999

Runtime: ~90 seconds on a standard laptop.

Input:  data/processed/cum_confirmed_matrix_clean.csv
        data/processed/admin_unit_adjacency.json
Output: data/processed/baseline2_scan_statistic_clusters.json
"""

import numpy as np
import pandas as pd
import json
import networkx as nx

MATRIX_PATH = "data/processed/cum_confirmed_matrix_clean.csv"
ADJ_PATH    = "data/processed/admin_unit_adjacency.json"
OUT_PATH    = "data/processed/baseline2_scan_statistic_clusters.json"

MAX_GRAPH_RADIUS      = 2
MAX_TIME_WINDOW_WEEKS = 8
N_PERMUTATIONS        = 999
RNG_SEED              = 42

with open(ADJ_PATH) as f:
    adj = json.load(f)
G = nx.Graph()
for u, neighbors in adj.items():
    for v in neighbors:
        G.add_edge(u, v)

matrix = pd.read_csv(MATRIX_PATH, index_col=0)
matrix.columns = pd.to_datetime(matrix.columns)
matrix = matrix.apply(pd.to_numeric, errors="coerce")
units = list(matrix.index)

# Weekly incident series from cumulative
weekly = matrix.T.resample("W").max().T
weekly = weekly.ffill(axis=1)
incident = weekly.diff(axis=1).clip(lower=0).fillna(0)
incident.iloc[:, 0] = weekly.iloc[:, 0].fillna(0)
T = incident.shape[1]
case_array = incident.values.astype(float)
week_cols = list(incident.columns)
print(f"Incident matrix: {len(units)} units × {T} weeks")
print(f"Total incident cases: {case_array.sum():.0f}")


def build_candidate_zones(G, units, max_radius):
    zones = set()
    for u in units:
        if u not in G:
            zones.add(frozenset([u]))
            continue
        for r in range(max_radius + 1):
            zone = nx.single_source_shortest_path_length(G, u, cutoff=r)
            zones.add(frozenset(zone.keys()))
    return [frozenset(z) for z in zones]


def build_candidate_windows(n_weeks, max_window):
    windows = []
    for length in range(1, max_window + 1):
        for start in range(n_weeks - length + 1):
            windows.append((start, start + length))
    return windows


def compute_log_glr(c_in, E_in, C):
    if E_in <= 0 or c_in <= E_in:
        return 0.0
    c_out, E_out = C - c_in, C - E_in
    if E_out <= 0 or c_out < 0:
        return 0.0
    log_glr = c_in * np.log(c_in / E_in)
    if c_out > 0:
        log_glr += c_out * np.log(c_out / E_out)
    return log_glr


def scan_max_log_glr(case_array, units, unit_idx, zones, windows, C):
    best_log_glr, best_zone, best_window = -np.inf, None, None
    zone_member_idx = [tuple(unit_idx[u] for u in z if u in unit_idx) for z in zones]
    all_week_totals = case_array.sum(axis=0)
    for zi, member_idx in enumerate(zone_member_idx):
        if not member_idx:
            continue
        zone_week_totals = case_array[list(member_idx), :].sum(axis=0)
        zone_total = zone_week_totals.sum()
        if zone_total <= 0:
            continue
        for (start, end) in windows:
            c_in = zone_week_totals[start:end].sum()
            if c_in <= 0:
                continue
            E_in = (zone_total / C) * (all_week_totals[start:end].sum() / C) * C
            log_glr = compute_log_glr(c_in, E_in, C)
            if log_glr > best_log_glr:
                best_log_glr, best_zone, best_window = log_glr, zones[zi], (start, end)
    return best_log_glr, best_zone, best_window


unit_idx = {u: i for i, u in enumerate(units)}
C = case_array.sum()
zones   = build_candidate_zones(G, units, MAX_GRAPH_RADIUS)
windows = build_candidate_windows(T, MAX_TIME_WINDOW_WEEKS)
print(f"Candidate zones: {len(zones)}, windows: {len(windows)}, "
      f"total cylinders: {len(zones)*len(windows)}")

obs_log_glr, obs_zone, obs_window = scan_max_log_glr(
    case_array, units, unit_idx, zones, windows, C)

print(f"\nObserved MLC log-GLR: {obs_log_glr:.4f}")
print(f"MLC zone ({len(obs_zone)} units): {sorted(obs_zone)}")
print(f"MLC window: {week_cols[obs_window[0]].date()} to "
      f"{week_cols[min(obs_window[1], T-1)].date()}")

# Monte Carlo permutation test
rng = np.random.default_rng(RNG_SEED)
unit_shares = case_array.sum(axis=1) / C
perm_log_glrs = np.zeros(N_PERMUTATIONS)
week_totals = case_array.sum(axis=0)

print(f"\nRunning {N_PERMUTATIONS} permutations...")
for p in range(N_PERMUTATIONS):
    perm_array = np.zeros_like(case_array)
    for w in range(T):
        total_w = week_totals[w]
        if total_w <= 0:
            continue
        perm_array[:, w] = rng.multinomial(int(round(total_w)), unit_shares)
    perm_log_glr, _, _ = scan_max_log_glr(
        perm_array, units, unit_idx, zones, windows, C)
    perm_log_glrs[p] = perm_log_glr
    if (p + 1) % 100 == 0:
        print(f"  {p+1}/{N_PERMUTATIONS} permutations done")

p_value = (np.sum(perm_log_glrs >= obs_log_glr) + 1) / (N_PERMUTATIONS + 1)
print(f"\nMonte Carlo p-value (n={N_PERMUTATIONS}): {p_value:.4f}")

result = {
    "mlc_units": sorted(obs_zone),
    "mlc_log_glr": float(obs_log_glr),
    "mlc_week_start": str(week_cols[obs_window[0]].date()),
    "mlc_week_end": str(week_cols[min(obs_window[1], T-1)].date()),
    "p_value": float(p_value),
    "n_permutations": N_PERMUTATIONS,
}
with open(OUT_PATH, "w") as f:
    json.dump(result, f, indent=2)
print(f"Results → {OUT_PATH}")
