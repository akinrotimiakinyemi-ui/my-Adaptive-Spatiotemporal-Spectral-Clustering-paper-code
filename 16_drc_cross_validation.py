"""
SCRIPT 16 — DRC North Kivu Cross-Validation Pipeline
=====================================================
Runs the complete AST-GL framework on the 2018-2020 DRC North Kivu
Ebola outbreak dataset — a GENUINE OUT-OF-SAMPLE test. ALL parameters
(ALPHA_MAX, N_EIGENVECTORS, K_CLUSTERS, the coupling formula) are
carried forward UNCHANGED from the West Africa design. Nothing is
re-tuned on this dataset.

Dataset: Andersen Lab / TSRI compilation from WHO situation reports
  Source: github.com/andersen-lab/ebola-drc-epidemiology
  29 health zones (North Kivu, Ituri, South Kivu provinces)
  77 weekly epi-periods, Aug 2018 – Jun 2020
  ~3,417 new confirmed cases (weekly incidence, wide-format matrix)

Ground-truth hotspot zones (7, sourced from WHO situation reports and
Aruna et al. 2019 MMWR, which documents 77.1% of all cases coming from
6 health zones):
  North Kivu: Mabalako, Beni, Butembo, Katwa, Kalunguta
  Ituri:      Mandima, Komanda

Pipeline:
  1. Parse wide-format epi-week matrix
  2. Build DRC health-zone adjacency graph (29 nodes, 45 edges)
  3. Run AST-GL (unchanged parameters)
  4. Evaluate: best-matching cluster purity / recall / F1
  5. Cross-outbreak comparison table

Input:  data/raw/drc_north_kivu/dataonly_ebolaDRC.csv
Output: data/processed/drc_incident_matrix.csv
        data/processed/drc_adjacency.json
        data/processed/drc_adaptive_clusters.csv
        data/processed/cross_outbreak_comparison.csv
        figures/drc_epidemic_curve.png
        figures/cross_outbreak_comparison.png
"""

import numpy as np
import pandas as pd
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.sparse import lil_matrix, diags
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
import os

os.makedirs("figures", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

RAW_DRC    = "data/raw/drc_north_kivu/dataonly_ebolaDRC.csv"
OUT_MATRIX = "data/processed/drc_incident_matrix.csv"
OUT_ADJ    = "data/processed/drc_adjacency.json"
OUT_CLUST  = "data/processed/drc_adaptive_clusters.csv"
OUT_COMP   = "data/processed/cross_outbreak_comparison.csv"

# Frozen parameters — same as West Africa
ALPHA_MAX      = 0.3
N_EIGENVECTORS = 12
K_CLUSTERS     = 7
EPS            = 1e-6

# ─── STEP 1: Parse DRC wide-format epi-week matrix ──────────────────────────
print("=== STEP 1: Parsing DRC North Kivu dataset ===")

raw = pd.read_csv(RAW_DRC, header=None)


def parse_week_start(s):
    if not isinstance(s, str):
        return None
    part = s.strip().split(" - ")[0].strip()
    try:
        return pd.to_datetime(part, format="%d %b %Y", errors="coerce")
    except Exception:
        return None


week_labels_raw = raw.iloc[0, 3:].tolist()
week_dates = [parse_week_start(w) for w in week_labels_raw]

week_col_labels = []
for i, d in enumerate(week_dates):
    try:
        if d is not None and not pd.isnull(d):
            week_col_labels.append(d.strftime("%Y-%m-%d"))
        else:
            week_col_labels.append(f"week_{i}")
    except Exception:
        week_col_labels.append(f"week_{i}")

records = []
current_province = None
for _, row in raw.iloc[2:].iterrows():
    vals = row.tolist()
    cell0 = str(vals[0]).strip()
    if cell0.lower() not in ("nan", "") and str(vals[1]) != "Total New Cases by Week":
        current_province = cell0
    if str(vals[1]).strip() == "Total New Cases by Week":
        continue
    location = str(vals[1]).strip()
    case_def = str(vals[2]).strip().replace(" ", "")
    if case_def not in ("Confirmed", "Probable"):
        continue
    weekly = []
    for i in range(len(week_dates)):
        v = vals[3 + i] if 3 + i < len(vals) else None
        try:
            weekly.append(float(v) if str(v).strip() not in ("nan", "") else 0.0)
        except Exception:
            weekly.append(0.0)
    records.append({"province": current_province, "health_zone": location,
                    "case_def": case_def,
                    "province_zone": f"{current_province} - {location}",
                    "weekly": weekly})

# Confirmed-only incident case matrix
confirmed = [r for r in records if r["case_def"] == "Confirmed"]
zones = sorted(set(r["province_zone"] for r in confirmed))
N = len(zones)
T = len(week_dates)
zone_idx = {z: i for i, z in enumerate(zones)}

case_array = np.zeros((N, T))
for r in confirmed:
    i = zone_idx[r["province_zone"]]
    case_array[i, :] = np.array(r["weekly"])
case_array = np.clip(case_array, 0, None)

incident_df = pd.DataFrame(case_array, index=zones, columns=week_col_labels)
incident_df.to_csv(OUT_MATRIX)

print(f"Health zones: {N}")
print(f"Epi-weeks:    {T}")
print(f"Total new confirmed cases: {case_array.sum():.0f}")
print(f"Date range: {week_col_labels[0]} to {week_col_labels[-1]}")
print("Top 10 zones by peak weekly cases:")
peak = case_array.max(axis=1)
for z in np.argsort(-peak)[:10]:
    print(f"  {zones[z]:45s}  peak={peak[z]:.0f}")

# ─── STEP 2: DRC adjacency graph ────────────────────────────────────────────
print("\n=== STEP 2: Building DRC health-zone adjacency graph ===")

NK, IT, SK = "North Kivu", "Ituri", "South Kivu"
DRC_EDGES = [
    (f"{NK} - Mabalako", f"{NK} - Beni"),
    (f"{NK} - Mabalako", f"{NK} - Butembo"),
    (f"{NK} - Mabalako", f"{NK} - Kalunguta"),
    (f"{NK} - Mabalako", f"{NK} - Oicha"),
    (f"{NK} - Beni",     f"{NK} - Butembo"),
    (f"{NK} - Beni",     f"{NK} - Kalunguta"),
    (f"{NK} - Beni",     f"{NK} - Oicha"),
    (f"{NK} - Beni",     f"{NK} - Mutwanga"),
    (f"{NK} - Beni",     f"{NK} - Biena"),
    (f"{NK} - Butembo",  f"{NK} - Katwa"),
    (f"{NK} - Butembo",  f"{NK} - Vuhovi"),
    (f"{NK} - Butembo",  f"{NK} - Musienene"),
    (f"{NK} - Butembo",  f"{NK} - Masereka"),
    (f"{NK} - Katwa",    f"{NK} - Vuhovi"),
    (f"{NK} - Katwa",    f"{NK} - Musienene"),
    (f"{NK} - Kalunguta",f"{NK} - Oicha"),
    (f"{NK} - Kalunguta",f"{NK} - Mutwanga"),
    (f"{NK} - Oicha",    f"{NK} - Mutwanga"),
    (f"{NK} - Mabalako", f"{IT} - Mandima"),
    (f"{NK} - Beni",     f"{IT} - Mandima"),
    (f"{NK} - Beni",     f"{IT} - Komanda"),
    (f"{NK} - Oicha",    f"{IT} - Komanda"),
    (f"{IT} - Mandima",  f"{IT} - Komanda"),
    (f"{IT} - Mandima",  f"{IT} - Mambasa"),
    (f"{IT} - Mandima",  f"{IT} - Lolwa"),
    (f"{IT} - Komanda",  f"{IT} - Mambasa"),
    (f"{IT} - Komanda",  f"{IT} - Bunia"),
    (f"{IT} - Mambasa",  f"{IT} - Lolwa"),
    (f"{IT} - Bunia",    f"{IT} - Nyankunde"),
    (f"{IT} - Bunia",    f"{IT} - Tchomia"),
    (f"{IT} - Bunia",    f"{IT} - Rwampara"),
    (f"{IT} - Nyankunde",f"{IT} - Tchomia"),
    (f"{NK} - Lubero",   f"{NK} - Masereka"),
    (f"{NK} - Lubero",   f"{NK} - Kayna"),
    (f"{NK} - Lubero",   f"{NK} - Alimbongo"),
    (f"{NK} - Lubero",   f"{NK} - Pinga"),
    (f"{NK} - Lubero",   f"{NK} - Manguredjipa"),
    (f"{NK} - Masereka", f"{NK} - Kayna"),
    (f"{NK} - Kayna",    f"{NK} - Alimbongo"),
    (f"{NK} - Kyondo",   f"{NK} - Butembo"),
    (f"{NK} - Kyondo",   f"{NK} - Vuhovi"),
    (f"{NK} - Vuhovi",   f"{NK} - Musienene"),
    (f"{NK} - Goma",     f"{NK} - Nyiragongo"),
    (f"{NK} - Goma",     f"{NK} - Lubero"),
    (f"{SK} - Mwenga",   f"{NK} - Butembo"),
]

A_drc = np.zeros((N, N))
for u, v in DRC_EDGES:
    if u in zone_idx and v in zone_idx:
        i, j = zone_idx[u], zone_idx[v]
        A_drc[i, j] = 1.0
        A_drc[j, i] = 1.0

# Isolated nodes get self-loop
for i in range(N):
    if A_drc[i].sum() == 0:
        print(f"  WARNING: {zones[i]} isolated — adding self-loop")
        A_drc[i, i] = 1.0

adj_dict = {zones[i]: [zones[j] for j in range(N)
                        if A_drc[i, j] > 0 and i != j] for i in range(N)}
with open(OUT_ADJ, "w") as f:
    json.dump(adj_dict, f, indent=2)

n_edges = int((A_drc > 0).sum()) // 2
print(f"DRC graph: {N} nodes, {n_edges} edges")

# ─── STEP 3: AST-GL (unchanged parameters) ──────────────────────────────────
print("\n=== STEP 3: Running AST-GL (unchanged West Africa parameters) ===")


def build_layer_affinities(case_array, A, eps=EPS):
    N, T = case_array.shape
    W = np.zeros((T, N, N))
    pairs = np.argwhere(np.triu(A, k=1) > 0)
    log_case = np.log1p(case_array)
    for t in range(T):
        x = log_case[:, t]
        diffs = np.abs(x[:, None] - x[None, :])
        sigma = (np.median(diffs[pairs[:, 0], pairs[:, 1]])
                 if len(pairs) else 1.0)
        sigma = sigma if sigma > 0 else 1.0
        W[t] = A * np.exp(-(diffs ** 2) / (2 * sigma ** 2 + eps))
    return W


def compute_adaptive_coupling(W_layers, A, alpha_max=ALPHA_MAX, eps=EPS):
    T, N, _ = W_layers.shape
    alpha = np.zeros((N, T - 1))
    for i in range(N):
        nb = np.where(A[i] > 0)[0]
        if len(nb) == 0:
            alpha[i, :] = alpha_max
            continue
        for t in range(T - 1):
            v1, v2 = W_layers[t, i, nb], W_layers[t + 1, i, nb]
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            cs = np.dot(v1, v2) / (n1 * n2) if n1 > eps and n2 > eps else 0.5
            alpha[i, t] = alpha_max * max(0.0, cs)
    return alpha


W_layers = build_layer_affinities(case_array, A_drc)
alpha = compute_adaptive_coupling(W_layers, A_drc)
print(f"α: mean={alpha.mean():.4f}, std={alpha.std():.4f}")

S = lil_matrix((N * T, N * T))
for t in range(T):
    S[t*N:(t+1)*N, t*N:(t+1)*N] = W_layers[t]
for t in range(T - 1):
    for i in range(N):
        a = alpha[i, t]
        S[t*N+i, (t+1)*N+i] = a
        S[(t+1)*N+i, t*N+i] = a
S = S.tocsr()

degrees = np.array(S.sum(axis=1)).flatten()
degrees_safe = np.where(degrees > EPS, degrees, EPS)
D_inv_sqrt = diags(1.0 / np.sqrt(degrees_safe))
L_norm = diags(np.ones(S.shape[0])) - D_inv_sqrt @ S @ D_inv_sqrt
eigvals, eigvecs = eigsh(L_norm, k=N_EIGENVECTORS, which="SM")
print(f"Eigenvalues (first 8): {np.round(eigvals[:8], 5)}")

km = KMeans(n_clusters=K_CLUSTERS, random_state=42, n_init=10)
node_labels = km.fit_predict(eigvecs)
label_grid = node_labels.reshape(T, N).T   # (N, T)

cluster_df = pd.DataFrame(label_grid, index=zones, columns=week_col_labels)
cluster_df.to_csv(OUT_CLUST)
print(f"Cluster labels → {OUT_CLUST}")

n_distinct = [len(set(label_grid[i])) for i in range(N)]
print(f"Temporal stability: mean={np.mean(n_distinct):.4f}, "
      f"std={np.std(n_distinct):.4f}")

# ─── STEP 4: Evaluation against ground-truth hotspot zones ──────────────────
print("\n=== STEP 4: Hotspot evaluation ===")

DRC_GT = {
    "North Kivu - Mabalako", "North Kivu - Beni",
    "North Kivu - Butembo",  "North Kivu - Katwa",
    "North Kivu - Kalunguta","Ituri - Mandima", "Ituri - Komanda",
}
N_GT_DRC = len(DRC_GT)

print("Ground-truth hotspot zones — peak case validation:")
for gt in sorted(DRC_GT):
    if gt in zone_idx:
        print(f"  {gt:45s}  peak={peak[zone_idx[gt]]:.0f}")
    else:
        print(f"  {gt} — NOT FOUND in dataset (check zone name)")

peak_week_idx = case_array.argmax(axis=1)
unit_to_cluster = {zones[i]: label_grid[i, peak_week_idx[i]] for i in range(N)}

cluster_members = {}
for u, c in unit_to_cluster.items():
    cluster_members.setdefault(c, []).append(u)

best = None
for c, members in cluster_members.items():
    if len(members) < 2:
        continue
    n_hits = sum(1 for m in members if m in DRC_GT)
    purity = n_hits / len(members)
    recall = n_hits / N_GT_DRC
    f1 = 2 * purity * recall / (purity + recall) if (purity + recall) > 0 else 0.0
    if best is None or f1 > best["f1"]:
        best = {"cluster_id": c, "members": members, "n_hits": n_hits,
                "purity": purity, "recall": recall,
                "cluster_size": len(members), "f1": f1}

print(f"\nBest-matching cluster: {best['cluster_id']} "
      f"(size={best['cluster_size']}, hits={best['n_hits']}/{N_GT_DRC})")
print(f"  purity={best['purity']:.3f}, recall={best['recall']:.3f}, "
      f"F1={best['f1']:.3f}")
print(f"  members: {sorted(best['members'])}")

# ─── STEP 5: Cross-outbreak comparison ──────────────────────────────────────
print("\n=== STEP 5: Cross-outbreak comparison (Table 5 in manuscript) ===")

# West Africa results — read fresh from Script 13 output if available,
# otherwise use the paper-reported values as fallback.
try:
    wa_purity = pd.read_csv(
        "data/processed/head_to_head_validation_v3_purity.csv")
    wa_row = wa_purity[wa_purity["method"].str.contains("Adaptive")]
    wa_purity_val = float(wa_row["purity"].values[0])
    wa_recall_val = float(wa_row["hotspot_recall"].values[0])
    wa_f1_val     = float(wa_row["f1"].values[0])
except Exception:
    print("  (Script 13 output not found — using paper-reported WA values)")
    wa_purity_val, wa_recall_val, wa_f1_val = 0.417, 0.909, 0.571

try:
    wa_clusters = pd.read_csv("data/processed/wa_adaptive_clusters.csv", index_col=0)
    wa_stab = wa_clusters.apply(lambda r: r.nunique(), axis=1)
    wa_stab_mean, wa_stab_std = wa_stab.mean(), wa_stab.std()
except Exception:
    wa_stab_mean, wa_stab_std = 3.878, 0.479

comparison = pd.DataFrame([
    {"outbreak": "West Africa 2014-16", "admin_units": 49, "epi_weeks": 27,
     "astgl_purity": wa_purity_val, "astgl_recall": wa_recall_val,
     "astgl_f1": wa_f1_val, "stability_mean": wa_stab_mean,
     "stability_std": wa_stab_std, "params_retuned": "N/A (design set)"},
    {"outbreak": "DRC North Kivu 2018-20", "admin_units": N, "epi_weeks": T,
     "astgl_purity": best["purity"], "astgl_recall": best["recall"],
     "astgl_f1": best["f1"], "stability_mean": np.mean(n_distinct),
     "stability_std": np.std(n_distinct), "params_retuned": "None"},
])
comparison.to_csv(OUT_COMP, index=False)
print(comparison.to_string(index=False))

# ─── STEP 6: Figures ────────────────────────────────────────────────────────
print("\n=== STEP 6: Generating figures ===")

# Epidemic curve
top8_idx = np.argsort(case_array.sum(axis=1))[-8:][::-1]
x = np.arange(T)
fig, ax = plt.subplots(figsize=(13, 6))
for zi in top8_idx:
    label = zones[zi].split(" - ")[1]
    ax.plot(x, case_array[zi], marker="o", markersize=2.5, label=label)
step = 8
xtick_pos = list(range(0, T, step))
ax.set_xticks(xtick_pos)
ax.set_xticklabels([week_col_labels[i] for i in xtick_pos], rotation=30, ha="right", fontsize=8)
ax.set_title("Weekly New Confirmed Ebola Cases — Top 8 Health Zones\n"
             "DRC North Kivu Outbreak, 2018–2020")
ax.set_xlabel("Epi-week start date")
ax.set_ylabel("New confirmed cases")
ax.legend(loc="upper left", fontsize=8, ncol=2)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("figures/drc_epidemic_curve.png", dpi=150)
print("  figures/drc_epidemic_curve.png saved")

# Cross-outbreak comparison figure
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("AST-GL Cross-Outbreak Generalisation: West Africa vs. DRC North Kivu",
             fontsize=12, fontweight="bold")

# Panel A: metric comparison
ax = axes[0]
metrics    = ["Purity", "Recall", "F1"]
wa_vals    = [wa_purity_val, wa_recall_val, wa_f1_val]
drc_vals   = [best["purity"], best["recall"], best["f1"]]
xpos = np.arange(len(metrics))
w = 0.35
bars1 = ax.bar(xpos - w/2, wa_vals,  w, label="West Africa 2014-16",
               color="#4C72B0", alpha=0.85)
bars2 = ax.bar(xpos + w/2, drc_vals, w, label="DRC North Kivu 2018-20",
               color="#DD8452", alpha=0.85)
ax.set_xticks(xpos)
ax.set_xticklabels(metrics)
ax.set_ylim(0, 1.05)
ax.set_ylabel("Score")
ax.set_title("A. Hotspot Recovery (no parameter re-tuning for DRC)")
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)
for bar in bars1:
    ax.annotate(f"{bar.get_height():.2f}",
                (bar.get_x() + bar.get_width()/2, bar.get_height()),
                ha="center", va="bottom", fontsize=9)
for bar in bars2:
    ax.annotate(f"{bar.get_height():.2f}",
                (bar.get_x() + bar.get_width()/2, bar.get_height()),
                ha="center", va="bottom", fontsize=9)

# Panel B: dataset scale
ax = axes[1]
datasets = ["West Africa\n2014-16", "DRC North Kivu\n2018-20"]
units_n  = [49, N]
weeks_n  = [27, T]
ax2 = ax.twinx()
ax.bar( [0 - 0.2, 1 - 0.2], units_n, 0.35, label="Admin units", color="#55A868", alpha=0.85)
ax2.bar([0 + 0.2, 1 + 0.2], weeks_n, 0.35, label="Epi-weeks",   color="#C44E52", alpha=0.85)
ax.set_xticks([0, 1])
ax.set_xticklabels(datasets)
ax.set_ylabel("Admin units",  color="#55A868")
ax2.set_ylabel("Epi-weeks",   color="#C44E52")
ax.set_title("B. Dataset Scale")
h1 = mpatches.Patch(color="#55A868", label="Admin units")
h2 = mpatches.Patch(color="#C44E52", label="Epi-weeks")
ax.legend(handles=[h1, h2], fontsize=9, loc="upper right")

fig.tight_layout()
fig.savefig("figures/cross_outbreak_comparison.png", dpi=150)
print("  figures/cross_outbreak_comparison.png saved")
print("\nDRC cross-validation complete.")
