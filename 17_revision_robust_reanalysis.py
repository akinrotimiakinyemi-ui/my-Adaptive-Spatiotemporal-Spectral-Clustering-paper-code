"""
SCRIPT 17 — Revision analyses, West Africa (response to reviewers)
==================================================================
Definitive West Africa results for the revised manuscript.

Changes relative to Scripts 06-15 (all reported in the manuscript):
  * Primary adjacency graph = shared-border graph derived from
    geoBoundaries polygons (Script 18). The hand-constructed graph used in
    the submitted manuscript is retained as a sensitivity analysis.
  * Exact, deterministic eigensolver (numpy.linalg.eigh) instead of ARPACK.
  * Every k-means-dependent quantity is reported over 100 initialisations
    (seeds 0-99); seed 42 is the reference run shown in tables/figures.
  * Label-free responsiveness: spectral-embedding displacement at the
    event week vs the control week (deterministic).
  * Data-cleaning variants (Reviewer 2): full / no_spike / raw.
  * Two ground truths (Reviewer 2): documented / burden (top-11 by cases).
  * Phase analysis (early vs late) repeated on the primary graph.
Outputs: results/revision_wa_*.csv / .json
"""
import sys, os, json, time
sys.path.insert(0, "src")
import numpy as np
import pandas as pd
from sklearn.cluster import SpectralClustering
from astgl_core import *

os.makedirs("results", exist_ok=True)
SEEDS = list(range(100))
FIXED = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
EVENT, CONTROL = pd.Timestamp("2014-10-26"), pd.Timestamp("2014-07-13")

raw = pd.read_csv("data/processed/cum_confirmed_matrix.csv", index_col=0)
raw.columns = pd.to_datetime(raw.columns)
raw = raw.apply(pd.to_numeric, errors="coerce")
units = list(raw.index)
N = len(units)
lib = [i for i, u in enumerate(units) if u.startswith("Liberia")]

variants = {"full": correct_cumulative(raw, True, True)[0],
            "no_spike": correct_cumulative(raw, False, True)[0],
            "raw": raw.copy()}
graphs = {"geo": "data/processed/admin_unit_adjacency_geoboundaries.json",
          "hand": "data/processed/admin_unit_adjacency.json"}

X_full = weekly_incident(variants["full"])
burden_gt = set(X_full.sum(axis=1).sort_values(ascending=False).index[:11])
GTS = {"documented": WA_GT_DOCUMENTED, "burden": burden_gt}
json.dump({k: sorted(v) for k, v in GTS.items()},
          open("results/revision_ground_truths_wa.json", "w"), indent=2)

rows_eval, rows_resp, rows_emb, rows_phase, scan_out = [], [], [], [], {}
for gname, gpath in graphs.items():
    A = load_adjacency(gpath, units)
    for vname, cum in variants.items():
        Xdf = weekly_incident(cum)
        X = Xdf.values.astype(float)
        T = X.shape[1]
        cols = list(Xdf.columns)
        ev_i, ct_i = cols.index(EVENT), cols.index(CONTROL)
        W = layer_affinities(X, A)
        alpha = adaptive_alpha(W, A)
        if vname == "full":
            pd.DataFrame(alpha, index=units, columns=cols[:-1]).to_csv(
                f"results/revision_wa_alpha_{gname}.csv")
        settings = {"adaptive": alpha, **{f"fixed_{a}": a for a in FIXED}}
        for sname, al in settings.items():
            t0 = time.time()
            vals, vecs, _ = spectral_embedding(supra_adjacency(W, al))
            if vname == "full" and sname == "adaptive":
                np.save(f"results/revision_wa_eigvals_{gname}.npy", vals)
            de = embedding_displacement(vecs, N, T, lib, ev_i)
            dc = embedding_displacement(vecs, N, T, lib, ct_i)
            allw = [embedding_displacement(vecs, N, T, lib, t) for t in range(1, T - 1)]
            rows_emb.append(dict(graph=gname, data=vname, setting=sname,
                                 disp_event=de, disp_control=dc, disp_ratio=de / dc,
                                 event_rank=int((np.array(allw) >= de).sum()),
                                 n_weeks=len(allw), disp_median=float(np.median(allw))))
            for seed in SEEDS:
                lab = cluster_embedding(vecs, N, T, seed=seed)
                if gname == "geo" and vname == "full" and sname == "adaptive" and seed == 42:
                    pd.DataFrame(lab, index=units, columns=cols).to_csv(
                        "results/revision_wa_clusters_geo_seed42.csv")
                st_m, st_sd = stability(lab)
                ev, ct = reassignments(lab, lib, ev_i), reassignments(lab, lib, ct_i)
                rows_resp.append(dict(graph=gname, data=vname, setting=sname, seed=seed,
                                      event=ev, control=ct, ratio=ratio(ev, ct),
                                      stab_mean=st_m, stab_sd=st_sd))
                if sname == "adaptive":
                    asg = peak_week_assignment(lab, X, units)
                    for gtn, gt in GTS.items():
                        b = best_matching_cluster(asg, gt)
                        rows_eval.append(dict(graph=gname, data=vname, method="AST-GL",
                                              gt=gtn, seed=seed, size=b["size"],
                                              hits=b["hits"], purity=b["purity"],
                                              recall=b["recall"], f1=b["f1"],
                                              members="; ".join(b["members"])))
            # phase analysis (stability within early / late halves), primary data only
            if vname == "full":
                for ph, (a0, a1) in {"early": (0, 13), "late": (13, T)}.items():
                    Wp = W[a0:a1]
                    alp = al if np.isscalar(al) else al[:, a0:a1 - 1]
                    _, vp, _ = spectral_embedding(supra_adjacency(Wp, alp))
                    sm = [stability(cluster_embedding(vp, N, a1 - a0, seed=s))[0]
                          for s in SEEDS[:20]]
                    rows_phase.append(dict(graph=gname, setting=sname, phase=ph,
                                           stab_mean=np.mean(sm), stab_sd=np.std(sm, ddof=1)))
            print(f"[{gname}/{vname}] {sname:10s} {time.time()-t0:5.1f}s", flush=True)

        # Baseline 1: static spectral clustering on last cumulative snapshot (k=8)
        snap = cum.apply(lambda r: r.dropna().iloc[-1] if r.dropna().size else 0.0,
                         axis=1).values.astype(float)
        d = np.abs(snap[:, None] - snap[None, :])
        pr = np.argwhere(np.triu(A, 1) > 0)
        s = max(np.median(d[pr[:, 0], pr[:, 1]]), 1e-6)
        aff = A * np.exp(-(d ** 2) / (2 * s ** 2))
        for seed in SEEDS:
            lab = SpectralClustering(n_clusters=8, affinity="precomputed",
                                     assign_labels="kmeans", random_state=seed,
                                     eigen_solver="lobpcg").fit_predict(aff)
            asg = dict(zip(units, lab))
            for gtn, gt in GTS.items():
                b = best_matching_cluster(asg, gt)
                rows_eval.append(dict(graph=gname, data=vname, method="B1-static",
                                      gt=gtn, seed=seed, size=b["size"], hits=b["hits"],
                                      purity=b["purity"], recall=b["recall"], f1=b["f1"],
                                      members="; ".join(b["members"])))
        # Baseline 2: scan statistic (permutation p-value for the clean data only)
        nperm = 999 if vname == "full" else 0
        llr, zone, (a, b_), p = scan_mlc(X, units, A, n_perm=nperm)
        scan_out[f"{gname}/{vname}"] = dict(mlc=zone, llr=llr, start=str(cols[a].date()),
                                            end=str(cols[b_ - 1].date()), p=p)
        for gtn, gt in GTS.items():
            h = sum(u in gt for u in zone)
            pp, rr = h / len(zone), h / len(gt)
            rows_eval.append(dict(graph=gname, data=vname, method="B2-scan", gt=gtn,
                                  seed=-1, size=len(zone), hits=h, purity=pp, recall=rr,
                                  f1=2 * pp * rr / (pp + rr) if pp + rr else 0.0,
                                  members="; ".join(sorted(zone))))
        print(f"[{gname}/{vname}] scan MLC {zone} {scan_out[f'{gname}/{vname}']}", flush=True)

pd.DataFrame(rows_eval).to_csv("results/revision_wa_hotspot_eval.csv", index=False)
pd.DataFrame(rows_resp).to_csv("results/revision_wa_responsiveness.csv", index=False)
pd.DataFrame(rows_emb).to_csv("results/revision_wa_embedding_responsiveness.csv", index=False)
pd.DataFrame(rows_phase).to_csv("results/revision_wa_phase.csv", index=False)
json.dump(scan_out, open("results/revision_wa_scan.json", "w"), indent=2, default=float)
print("done")
