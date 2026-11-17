"""Graph utilities on top of networkx: build graphs from dependence matrices, minimum spanning trees
(Mantegna 1999), centralities and communities."""
from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd


def from_matrix(W: pd.DataFrame, directed: bool = False, threshold: float = 0.0, absolute: bool = True) -> nx.Graph:
    """Edges where |W_ij| > threshold. For directed graphs W[i, j] is read as an edge j -> i (receiver rows),
    matching the GFEVD and Granger conventions used elsewhere."""
    G = nx.DiGraph() if directed else nx.Graph()
    names = list(W.index)
    G.add_nodes_from(names)
    A = W.to_numpy(dtype=float)
    for a, i in enumerate(names):
        for b, j in enumerate(names):
            if a == b or (not directed and b <= a):
                continue
            w = A[a, b]
            if not np.isfinite(w):
                continue
            if (abs(w) if absolute else w) > threshold:
                if directed:
                    G.add_edge(j, i, weight=abs(w) if absolute else w, signed=w)
                else:
                    G.add_edge(i, j, weight=abs(w) if absolute else w, signed=w)
    return G


def mst(corr: pd.DataFrame) -> nx.Graph:
    """Minimum spanning tree on the metric d_ij = sqrt(2 (1 - rho_ij))."""
    names = list(corr.index)
    C = corr.to_numpy()
    G = nx.Graph()
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            G.add_edge(names[a], names[b], distance=float(np.sqrt(max(2 * (1 - C[a, b]), 0.0))), corr=float(C[a, b]))
    return nx.minimum_spanning_tree(G, weight="distance")


def centralities(G: nx.Graph) -> pd.DataFrame:
    w = "weight"
    out = {"degree": dict(G.degree()), "strength": dict(G.degree(weight=w))}
    if G.is_directed():
        out["in_strength"] = dict(G.in_degree(weight=w))
        out["out_strength"] = dict(G.out_degree(weight=w))
        try:
            out["pagerank"] = nx.pagerank(G, weight=w)
        except nx.PowerIterationFailedConvergence:
            pass
    else:
        try:
            out["eigenvector"] = nx.eigenvector_centrality_numpy(G, weight=w)
        except Exception:  # disconnected / degenerate graphs
            pass
    H = G.copy()
    for u, v, d in H.edges(data=True):
        d["dist"] = 1.0 / d[w] if d.get(w, 0) > 0 else 1e9
    out["betweenness"] = nx.betweenness_centrality(H, weight="dist")
    return pd.DataFrame(out)


def communities(G: nx.Graph, seed: int = 0) -> dict[str, int]:
    H = G.to_undirected() if G.is_directed() else G
    parts = nx.community.louvain_communities(H, weight="weight", seed=seed)
    return {n: k for k, part in enumerate(parts) for n in part}
