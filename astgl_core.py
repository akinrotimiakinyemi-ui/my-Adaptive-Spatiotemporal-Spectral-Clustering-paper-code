"""
astgl_core.py — shared, deterministic implementation of AST-GL and the
evaluation protocol used in the revised manuscript.

Revision note (response to reviewers): the original scripts computed the
spectral embedding with ARPACK (scipy.sparse.linalg.eigsh, which="SM").
ARPACK starts from a random vector, and when eigenvalues are close or
repeated the returned eigenvectors are an arbitrary rotation of the
eigenspace. k-means on such vectors can give different partitions on
different machines/library versions. All revised analyses therefore use
an exact dense symmetric eigensolver (numpy.linalg.eigh), which is
deterministic, and report robustness over many k-means initialisations.
"""
import json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

EPS = 1e-6
ALPHA_MAX = 0.3
N_EIGENVECTORS = 12
K_CLUSTERS = 7

WA_GT_DOCUMENTED = {
    "Liberia - Montserrado", "Sierra Leone - Western area urban",
    "Sierra Leone - Western area rural", "Sierra Leone - Bombali",
    "Sierra Leone - Port Loko", "Guinea - Kerouane", "Guinea - Macenta",
    "Guinea - Mzerekore", "Guinea - Gueckedou", "Sierra Leone - Kailahun",
    "Liberia - Lofa",
}
# Six health zones named individually in WHO Disease Outbreak News
# (26 June 2020, Fig. 1); five of them are also the five most-affected
# zones in Aruna et al. (2019, MMWR 68:1162-5). Komanda, used in the
# submitted manuscript, is not named in either source and was removed.
DRC_GT_DOCUMENTED = {
    "North Kivu - Mabalako", "North Kivu - Beni", "North Kivu - Butembo",
    "North Kivu - Katwa", "North Kivu - Kalunguta", "Ituri - Mandima",
}


# ----------------------------------------------------------------- data --
def load_adjacency(path, units):
    with open(path) as f:
        adj = json.load(f)
    idx = {u: i for i, u in enumerate(units)}
    A = np.zeros((len(units), len(units)))
    for u, nbs in adj.items():
        if u not in idx:
            continue
        for v in nbs:
            if v in idx and v != u:
                A[idx[u], idx[v]] = A[idx[v], idx[u]] = 1.0
    return A


def correct_cumulative(matrix, spikes=True, monotone=True, spike_ratio=3.0):
    """Two-stage cleaning of a cumulative-count matrix (units x dates)."""
    m = matrix.copy()
    log = []
    if spikes:
        for u in m.index:
            valid = m.loc[u].dropna()
            if len(valid) < 3:
                continue
            vals, dates = valid.values, valid.index
            for i in range(1, len(vals) - 1):
                p, c, n = vals[i - 1], vals[i], vals[i + 1]
                if p <= 0 or n <= 0:
                    continue
                if c >= spike_ratio * p and c >= spike_ratio * n:
                    m.loc[u, dates[i]] = (p + n) / 2
                    log.append((u, dates[i], c, (p + n) / 2, "isolated_spike"))
    if monotone:
        for u in m.index:
            valid = m.loc[u].dropna()
            rm = -np.inf
            corr = valid.copy()
            for d in valid.index:
                if valid[d] < rm:
                    log.append((u, d, valid[d], rm, "non_monotonic_drop"))
                    corr[d] = rm
                else:
                    rm = valid[d]
            m.loc[u, valid.index] = corr.values
    return m, log


def weekly_incident(cum_matrix):
    weekly = cum_matrix.T.resample("W").max().T.ffill(axis=1)
    inc = weekly.diff(axis=1).clip(lower=0).fillna(0)
    inc.iloc[:, 0] = weekly.iloc[:, 0].fillna(0)
    return inc


# -------------------------------------------------------------- AST-GL --
def layer_affinities(X, A):
    """X: (N,T) incident counts. Returns W: (T,N,N)."""
    N, T = X.shape
    W = np.zeros((T, N, N))
    pairs = np.argwhere(np.triu(A, k=1) > 0)
    L = np.log1p(X)
    for t in range(T):
        x = L[:, t]
        d = np.abs(x[:, None] - x[None, :])
        s = np.median(d[pairs[:, 0], pairs[:, 1]]) if len(pairs) else 1.0
        s = s if s > 0 else 1.0
        W[t] = A * np.exp(-(d ** 2) / (2 * s ** 2 + EPS))
    return W


def adaptive_alpha(W, A, alpha_max=ALPHA_MAX):
    T, N, _ = W.shape
    alpha = np.zeros((N, T - 1))
    for i in range(N):
        nb = np.where(A[i] > 0)[0]
        nb = nb[nb != i]
        if len(nb) == 0:
            alpha[i, :] = alpha_max
            continue
        for t in range(T - 1):
            v1, v2 = W[t, i, nb], W[t + 1, i, nb]
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            cs = v1 @ v2 / (n1 * n2) if (n1 > EPS and n2 > EPS) else 0.5
            alpha[i, t] = alpha_max * max(0.0, cs)
    return alpha


def supra_adjacency(W, alpha):
    """alpha: scalar (fixed coupling) or (N, T-1) array (adaptive)."""
    T, N, _ = W.shape
    S = np.zeros((N * T, N * T))
    for t in range(T):
        S[t * N:(t + 1) * N, t * N:(t + 1) * N] = W[t]
    for t in range(T - 1):
        for i in range(N):
            a = alpha if np.isscalar(alpha) else alpha[i, t]
            S[t * N + i, (t + 1) * N + i] = a
            S[(t + 1) * N + i, t * N + i] = a
    return S


def spectral_embedding(S, k_eig=N_EIGENVECTORS):
    deg = S.sum(axis=1)
    dinv = 1.0 / np.sqrt(np.where(deg > EPS, deg, EPS))
    L = np.eye(S.shape[0]) - (dinv[:, None] * S * dinv[None, :])
    L = (L + L.T) / 2
    vals, vecs = np.linalg.eigh(L)          # exact, deterministic
    return vals[:k_eig], vecs[:, :k_eig], vals


def cluster_embedding(vecs, N, T, k=K_CLUSTERS, seed=42):
    lab = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(vecs)
    return lab.reshape(T, N).T                # (N, T)


# ---------------------------------------------------------- evaluation --
def best_matching_cluster(unit_to_cluster, gt, min_size=2):
    groups = {}
    for u, c in unit_to_cluster.items():
        groups.setdefault(c, []).append(u)
    best = None
    for c, mem in groups.items():
        if len(mem) < min_size:
            continue
        h = sum(m in gt for m in mem)
        p, r = h / len(mem), h / len(gt)
        f1 = 2 * p * r / (p + r) if p + r > 0 else 0.0
        if best is None or f1 > best["f1"]:
            best = dict(cluster=c, members=sorted(mem), hits=h, size=len(mem),
                        purity=p, recall=r, f1=f1)
    return best


def peak_week_assignment(labels, X, units):
    pk = X.argmax(axis=1)
    return {u: labels[i, pk[i]] for i, u in enumerate(units)}


def stability(labels):
    n = np.array([len(set(r)) for r in labels])
    return n.mean(), n.std(ddof=1)


def reassignments(labels, idx, centre):
    b, d, a = labels[idx, centre - 1], labels[idx, centre], labels[idx, centre + 1]
    return int(np.sum((b != d) | (d != a)))


def ratio(ev, ct):
    return (ev + 1) / (ct + 1)


def embedding_displacement(vecs, N, T, idx, centre):
    """Label-free responsiveness: mean (over units in idx) of the larger of
    the two consecutive-layer displacements of the row-normalised spectral
    embedding around week `centre`. Invariant to k-means initialisation."""
    E = vecs.reshape(T, N, -1)
    E = E / np.maximum(np.linalg.norm(E, axis=2, keepdims=True), 1e-12)
    d1 = np.linalg.norm(E[centre, idx] - E[centre - 1, idx], axis=1)
    d2 = np.linalg.norm(E[centre + 1, idx] - E[centre, idx], axis=1)
    return float(np.mean(np.maximum(d1, d2)))


def scan_mlc(X, units, A, max_radius=2, max_window=8, n_perm=0, seed=42):
    """Graph-adapted space-time permutation scan statistic (Kulldorff 2005)
    with k-hop adjacency zones. Returns (llr, zone, (start, end), p)."""
    import networkx as nx
    N, T = X.shape
    G = nx.Graph(); G.add_nodes_from(range(N))
    for i in range(N):
        for j in range(i + 1, N):
            if A[i, j] > 0:
                G.add_edge(i, j)
    zones = set()
    for u in range(N):
        for r in range(max_radius + 1):
            zones.add(frozenset(nx.single_source_shortest_path_length(G, u, cutoff=r)))
    zones = [sorted(z) for z in zones]
    wins = [(a, a + L) for L in range(1, max_window + 1) for a in range(T - L + 1)]

    def scan(arr):
        C = arr.sum(); cw = np.concatenate([[0], np.cumsum(arr.sum(axis=0))])
        best = (-np.inf, None, None)
        for z in zones:
            zt = arr[z].sum(axis=0); tot = zt.sum()
            if tot <= 0:
                continue
            cz = np.concatenate([[0], np.cumsum(zt)])
            for a, b in wins:
                c = cz[b] - cz[a]
                if c <= 0:
                    continue
                E = tot * (cw[b] - cw[a]) / C
                if E <= 0 or c <= E:
                    continue
                co, Eo = C - c, C - E
                llr = c * np.log(c / E) + (co * np.log(co / Eo) if co > 0 and Eo > 0 else 0)
                if llr > best[0]:
                    best = (llr, z, (a, b))
        return best

    obs = scan(X)
    p = np.nan
    if n_perm:
        rng = np.random.default_rng(seed)
        sh = X.sum(axis=1) / X.sum(); wk = X.sum(axis=0)
        perm = []
        for _ in range(n_perm):
            arr = np.zeros_like(X)
            for w in range(T):
                if wk[w] > 0:
                    arr[:, w] = rng.multinomial(int(round(wk[w])), sh)
            perm.append(scan(arr)[0])
        p = (np.sum(np.array(perm) >= obs[0]) + 1) / (n_perm + 1)
    return obs[0], [units[i] for i in obs[1]], obs[2], p
